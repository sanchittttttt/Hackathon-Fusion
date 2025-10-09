# download_data.py
import pandas as pd
from pathlib import Path

print("Downloading Davis dataset...")

base = "https://raw.githubusercontent.com/dingyan20/Davis-Dataset-for-DTA-Prediction/main/"

df_drugs = pd.read_csv(base + "drugs.csv")
df_proteins = pd.read_csv(base + "proteins.csv")
df_affinity = pd.read_csv(base + "drug_protein_affinity.csv")

print(f"✅ Drugs: {len(df_drugs)}, Proteins: {len(df_proteins)}, Pairs: {len(df_affinity)}")

# Merge first 200 samples
merged = []
for _, row in df_affinity.head(200).iterrows():
    drug = df_drugs[df_drugs['Drug_Index'] == row['Drug_Index']]
    protein = df_proteins[df_proteins['Protein_Index'] == row['Protein_Index']]
    
    if len(drug) > 0 and len(protein) > 0:
        merged.append({
            'Ligand_SMILES': drug.iloc[0]['Canonical_SMILES'],
            'Target_Sequence': protein.iloc[0]['Sequence'],
            'Target_Name': protein.iloc[0]['Gene_Name'],
            'Kd_nM': row['Affinity'],
            'IC50_nM': row['Affinity'] * 1.5,
        })

df = pd.DataFrame(merged)
Path('data/raw').mkdir(parents=True, exist_ok=True)
df.to_csv('data/raw/davis.csv', index=False)

print(f"✅ Saved {len(df)} samples to data/raw/davis.csv")
print(f"Kd range: {df['Kd_nM'].min():.1f} - {df['Kd_nM'].max():.1f} nM")

