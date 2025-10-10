from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
from pathlib import Path
import os

# Try to import drawing modules, but handle gracefully if they fail
try:
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import SimilarityMaps
    import matplotlib.pyplot as plt
    DRAWING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Molecular drawing not available: {e}")
    DRAWING_AVAILABLE = False


def highlight_important_substructures(smiles, important_bits, output_path=None, radius=2):
    """
    Highlights molecular substructures responsible for important fingerprint bits.
    
    Args:
        smiles (str): SMILES string of molecule.
        important_bits (list[int]): List of important fingerprint bit indices.
        output_path (str): Path to save the highlighted image. If None, generates a default path.
        radius (int): Radius used for Morgan fingerprint generation.
        
    Returns:
        str: Path to the saved highlighted image, or None if drawing is not available.
    """
    if not DRAWING_AVAILABLE:
        print("Warning: Molecular drawing not available on this system. Skipping highlight generation.")
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Use RDKit's Morgan fingerprint with bitInfo
    bitInfo = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, bitInfo=bitInfo)
    
    atoms_to_highlight = set()
    for bit in important_bits:
        if bit in bitInfo:
            env_list = bitInfo[bit]
            # Each env is (atomId, radius), add involved atoms to highlight
            for atom_id, rad in env_list:
                env_atoms = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_id)
                atoms_to_highlight.update(env_atoms)
                atoms_to_highlight.add(atom_id)  # Add center atom as well
    
    highlight_atoms = list(atoms_to_highlight)
    highlight_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in highlight_atoms and bond.GetEndAtomIdx() in highlight_atoms:
            highlight_bonds.append(bond.GetIdx())

    # Generate output path if not provided
    if output_path is None:
        # Create safe filename from SMILES
        safe_smiles = "".join(c for c in smiles if c.isalnum() or c in ('-', '_'))[:20]
        output_dir = Path("outputs/highlights")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{safe_smiles}_highlighted.png"
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        drawer = Draw.MolDraw2DCairo(500, 300)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightBonds=highlight_bonds)
        drawer.FinishDrawing()
        img = drawer.GetDrawingText()
        
        with open(output_path, "wb") as f:
            f.write(img)
        
        return str(output_path)
    except Exception as e:
        print(f"Warning: Failed to generate molecular highlight: {e}")
        return None


# Example usage
if __name__ == "__main__":
    example_smiles = "CC1=CC=C(C=C1)C2=NC=NC=N2"  # Staurosporine fragment
    important_fingerprint_bits = [1380, 208, 579]  # example bits, replace with your top bits

    highlight_important_substructures(example_smiles, important_fingerprint_bits)
