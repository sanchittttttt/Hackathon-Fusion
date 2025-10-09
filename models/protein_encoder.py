# models/protein_encoder.py
import torch
from transformers import AutoTokenizer, EsmModel
import numpy as np

class ProteinEncoder:
    def __init__(self, model_name="facebook/esm2_t12_35M_UR50D"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embed_dim = self.model.config.hidden_size
        print(f"✅ Protein encoder ready (dim={self.embed_dim})")
    
    def encode(self, sequences):
        if isinstance(sequences, str):
            sequences = [sequences]
        
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
