# test_imports.py
import torch
import transformers
from rdkit import Chem
import pandas as pd
import numpy as np
import streamlit as st

print("✅ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test RDKit
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(O)=O")
print(f"RDKit working: {mol is not None}")
