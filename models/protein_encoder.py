# models/protein_encoder.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel
import numpy as np

class ProteinEncoder:
    """
    Encode protein sequences using ESM2 transformer model
    ESM2 is pretrained on evolutionary data for protein understanding
    """
    
    def __init__(self, model_name="facebook/esm2_t12_35M_UR50D", device=None):
        """
        Args:
            model_name: ESM2 model variant
                - esm2_t12_35M_UR50D: 35M params (fast, good for MVP)
                - esm2_t33_650M_UR50D: 650M params (better accuracy)
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading ESM2 model: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.embed_dim = self.model.config.hidden_size
        print(f"✅ Protein encoder ready. Embedding dim: {self.embed_dim}")
    
    def encode(self, sequences, batch_size=4):
        """
        Encode protein sequences to embeddings
        
        Args:
            sequences: List of protein sequence strings or single string
            batch_size: Sequences per batch
            
        Returns:
            numpy array of shape (n_sequences, embed_dim)
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        all_embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            
            # Tokenize proteins
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)

# Test
if __name__ == "__main__":
    encoder = ProteinEncoder()
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    
    embeddings = encoder.encode(test_seq)
    print(f"Input length: {len(test_seq)} amino acids")
    print(f"Output shape: {embeddings.shape}")
    print(f"Sample values: {embeddings[0, :5]}")
