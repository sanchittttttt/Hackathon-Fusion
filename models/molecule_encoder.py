# models/molecule_encoder.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

class MoleculeEncoder:
    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits
        print(f"✅ Molecule encoder ready (bits={n_bits})")
    
    def encode(self, smiles_list):
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        fps = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(self.n_bits))
        
        return np.array(fps)
