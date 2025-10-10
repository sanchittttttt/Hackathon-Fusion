import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models.protein_encoder import ProteinEncoder
from models.molecule_encoder import MoleculeEncoder
from models.binding_predictor import BindingMLP
from metrics import pKd_to_Kd, Kd_to_Ki, Kd_to_IC50, Kd_to_EC50, pKd_to_DeltaG
from utils.interpretability import get_top_fingerprint_weights
from utils.phase1_readiness import phase1_readiness_score
from utils.molecular_highlight import highlight_important_substructures

def predict_with_confidence_ranking(proteins, smiles, model_path="models/saved_models/binding_model_v2.pt", n_samples=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    protein_encoder = ProteinEncoder()
    molecule_encoder = MoleculeEncoder()
    
    model = BindingMLP().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    protein_features = torch.tensor(protein_encoder.encode(proteins), dtype=torch.float32).to(device)
    molecule_features = torch.tensor(molecule_encoder.encode(smiles), dtype=torch.float32).to(device)
    
    mean_pred, std_pred = model.mc_dropout_predict(protein_features, molecule_features, n_samples)
    
    mean_pred = mean_pred.cpu().detach().numpy().flatten()
    std_pred = std_pred.cpu().detach().numpy().flatten()
    conf_score = 1 - (std_pred - std_pred.min()) / (std_pred.max() - std_pred.min() + 1e-8)
    
    kd_values = pKd_to_Kd(mean_pred)
    ki_values = Kd_to_Ki(kd_values)
    ic50_values = Kd_to_IC50(kd_values)
    ec50_values = Kd_to_EC50(kd_values)
    deltaG_values = pKd_to_DeltaG(mean_pred)
    
    top_bits, top_scores = get_top_fingerprint_weights(model, top_k=20)
    
    # Plot interpretability bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_bits)), top_scores, tick_label=top_bits)
    plt.xlabel('Fingerprint Bit Index')
    plt.ylabel('Importance Score')
    plt.title('Top Important Molecular Fingerprint Bits')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Highlight important substructures in molecules
    for i, smi in enumerate(smiles):
        important_bits = top_bits  # You could refine which bits are most relevant per molecule
        print(f"Highlighting molecular substructures for molecule {i+1}")
        highlight_important_substructures(smi, important_bits)

    # Compute phase 1 readiness for each molecule
    readiness_scores = np.array([phase1_readiness_score(s) for s in smiles])

    return mean_pred, conf_score, kd_values, ki_values, ic50_values, ec50_values, deltaG_values, top_bits, top_scores, readiness_scores


def run_pipeline(proteins: list[str], smiles: list[str], n_samples: int = 30, top_k: int = 20, 
                 model_path: str = "models/saved_models/binding_model.pt", out_dir: str = "outputs", 
                 make_highlights: bool = False) -> dict:
    """
    Pure function that runs the complete prediction pipeline and returns structured data.
    
    Args:
        proteins: List of protein FASTA sequences
        smiles: List of SMILES strings
        n_samples: Number of MC dropout samples for uncertainty estimation
        top_k: Number of top fingerprint bits to extract
        model_path: Path to the trained model
        out_dir: Output directory for results and highlights
        make_highlights: Whether to generate substructure highlight images
        
    Returns:
        dict: Contains 'results_df', 'top_bits', 'top_scores', 'highlight_paths', 'summary_metrics'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize encoders and model
    protein_encoder = ProteinEncoder()
    molecule_encoder = MoleculeEncoder()
    
    model = BindingMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Encode inputs
    protein_features = torch.tensor(protein_encoder.encode(proteins), dtype=torch.float32).to(device)
    molecule_features = torch.tensor(molecule_encoder.encode(smiles), dtype=torch.float32).to(device)
    
    # Handle broadcasting if needed
    if len(proteins) == 1 and len(smiles) > 1:
        protein_features = protein_features.repeat(len(smiles), 1)
    elif len(smiles) == 1 and len(proteins) > 1:
        molecule_features = molecule_features.repeat(len(proteins), 1)
    
    # Get predictions with uncertainty
    mean_pred, std_pred = model.mc_dropout_predict(protein_features, molecule_features, n_samples)
    
    mean_pred = mean_pred.cpu().detach().numpy()
    std_pred = std_pred.cpu().detach().numpy()
    conf_score = 1 - (std_pred - std_pred.min()) / (std_pred.max() - std_pred.min() + 1e-8)
    
    # Convert to various metrics
    kd_values = pKd_to_Kd(mean_pred)
    ki_values = Kd_to_Ki(kd_values)
    ic50_values = Kd_to_IC50(kd_values)
    ec50_values = Kd_to_EC50(kd_values)
    deltaG_values = pKd_to_DeltaG(mean_pred)
    
    # Get top fingerprint bits
    top_bits, top_scores = get_top_fingerprint_weights(model, top_k=top_k)
    
    # Compute phase 1 readiness scores
    readiness_scores = np.array([phase1_readiness_score(s) for s in smiles])
    
    # Create results dataframe
    results_data = []
    for i, (protein, smi) in enumerate(zip(proteins, smiles)):
        results_data.append({
            'protein': protein[:50] + "..." if len(protein) > 50 else protein,
            'smiles': smi,
            'mean_pKd': mean_pred[i],
            'confidence': conf_score[i],
            'Kd_nM': kd_values[i] * 1e9,  # Convert to nM
            'Ki_nM': ki_values[i] * 1e9,
            'IC50_nM': ic50_values[i] * 1e9,
            'EC50_nM': ec50_values[i] * 1e9,
            'DeltaG_kcal_mol': deltaG_values[i],
            'phase1_readiness': readiness_scores[i],
            'binding_probability': 1 / (1 + np.exp(-(mean_pred[i] - 5)))  # Sigmoid transform for probability
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Generate highlight images if requested
    highlight_paths = []
    if make_highlights:
        highlight_dir = Path(out_dir) / "highlights"
        highlight_dir.mkdir(parents=True, exist_ok=True)
        
        for i, smi in enumerate(smiles):
            try:
                output_path = highlight_dir / f"molecule_{i}_{smi[:10]}_highlighted.png"
                highlight_path = highlight_important_substructures(smi, top_bits, str(output_path))
                highlight_paths.append(highlight_path)
            except Exception as e:
                print(f"Warning: Could not generate highlight for molecule {i}: {e}")
                highlight_paths.append(None)
    
    # Summary metrics
    summary_metrics = {
        'best_pKd': mean_pred.max(),
        'median_confidence': np.median(conf_score),
        'avg_phase1_readiness': np.mean(readiness_scores),
        'total_molecules': len(smiles),
        'total_proteins': len(set(proteins))
    }
    
    return {
        'results_df': results_df,
        'top_bits': top_bits,
        'top_scores': top_scores,
        'highlight_paths': highlight_paths,
        'summary_metrics': summary_metrics
    }

# Example usage
if __name__ == "__main__":
    proteins = [
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSEDVPMVLVGNKCDLSAKSKWGKFGKKFKDVSVV",
        "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEDRLRVPRGSDIAGTTSTL"
    ]
    smiles = [
        "CC1=CC=C(C=C1)C2=NC=NC=N2",
        "CC(C)CC[C@@H](C)[C@@H](N1CC[C@H](CNC(=O)C2=CC=CC=C2)C[C@@H]1C(=O)O[C@@H]3CCN(CC3)C(=O)C4=CC=C(C=C4)S(=O)(=O)N)O"
    ]
    results = predict_with_confidence_ranking(proteins, smiles)
    print("Results:", results)
