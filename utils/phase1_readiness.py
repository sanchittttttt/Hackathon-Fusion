from rdkit import Chem
from rdkit.Chem import Descriptors

def lipinski_rule_of_five(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0  # Invalid molecule 

    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    
    score = 0
    if mol_wt < 500:
        score += 1
    if logp < 5:
        score += 1
    if h_donors < 5:
        score += 1
    if h_acceptors < 10:
        score += 1
    
    # Max score = 4 (all rules passed)
    return score / 4  # normalize to 0-1


def simple_toxicity_alert(smiles):
    # A placeholder function: flags molecules with nitro groups as toxic (example)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 1  # Highly toxic if invalid

    # Search for nitro group '[N+](=O)[O-]'
    patt = Chem.MolFromSmarts('[N+](=O)[O-]')
    if mol.HasSubstructMatch(patt):
        return 0  # Toxic
    return 1  # Non-toxic


def phase1_readiness_score(smiles):
    drug_like = lipinski_rule_of_five(smiles)
    toxicity = simple_toxicity_alert(smiles)
    
    # Simple normalized ADMET heuristics can be added here later
    # For now, sum of drug_likeness and toxicity weighted equally
    readiness_score = (drug_like + toxicity) / 2
    return readiness_score
