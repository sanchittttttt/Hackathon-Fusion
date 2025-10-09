# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from models.protein_encoder import ProteinEncoder
from models.molecule_encoder import MoleculeEncoder
from models.binding_predictor import BindingMLP

class BindingDataset(Dataset):
    def __init__(self, protein_features, molecule_features, labels):
        self.protein = torch.tensor(protein_features, dtype=torch.float32)
        self.molecule = torch.tensor(molecule_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.protein[idx], self.molecule[idx], self.labels[idx]

print("\n" + "="*60)
print("TRAINING BINDING AFFINITY PREDICTOR")
print("="*60 + "\n")

# Load data
df = pd.read_csv("data/raw/davis.csv")
print(f"Loaded {len(df)} samples")

# Extract features
print("\nExtracting features...")
protein_encoder = ProteinEncoder()
molecule_encoder = MoleculeEncoder()

protein_features = protein_encoder.encode(df['Target_Sequence'].tolist())
molecule_features = molecule_encoder.encode(df['Ligand_SMILES'].tolist())

# Convert to pKd (log scale)
labels = -np.log10(df['Kd_nM'].values * 1e-9)
print(f"pKd range: {labels.min():.2f} - {labels.max():.2f}")

# Split data
train_idx, test_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)

train_dataset = BindingDataset(protein_features[train_idx], molecule_features[train_idx], labels[train_idx])
test_dataset = BindingDataset(protein_features[test_idx], molecule_features[test_idx], labels[test_idx])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Train model
model = BindingMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nTraining...")
for epoch in range(50):
    model.train()
    train_loss = 0
    for protein, molecule, label in train_loader:
        output = model(protein, molecule)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for protein, molecule, label in test_loader:
                output = model(protein, molecule)
                test_loss += criterion(output, label).item()
        
        print(f"Epoch {epoch+1:2d} | Train: {train_loss/len(train_loader):.4f} | Test: {test_loss/len(test_loader):.4f}")

# Save model
Path("models/saved_models").mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), "models/saved_models/binding_model.pt")
print("\n✅ Model saved to models/saved_models/binding_model.pt")
print("="*60 + "\n")
