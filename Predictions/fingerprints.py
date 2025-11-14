# Functions to generate fingerprints from SMILES using RDKit.

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


# Function to generate Morgan fingerprints
def generate_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=17284)
    return None

# Function to generate MACCS keys
def generate_maccs_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return MACCSkeys.GenMACCSKeys(mol)
    return None