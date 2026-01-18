import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit import RDLogger

# Disable warnings for poorly formatted SMILES (they will be removed)
RDLogger.DisableLog('rdApp.warning')

# Import new dataset SMILES and logS
df = pd.read_csv('wk6/01_aqsol_db_raw.csv')
df.columns = df.columns.str.strip()
df = df[['SMILES', 'Solubility']].rename(columns={'Solubility': 'logS'})

# Get mol objects from SMILES; parse issues
def mol_from_smiles(s):
    if not isinstance(s, str) or not s.strip():
        return None
    return Chem.MolFromSmiles(s)
df["mol"] = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)
df = df[df["mol"].notna()].copy()

# Remove inorganic molecules
def is_organic(mol):
    if mol is None:
        return False
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
df = df[df["mol"].apply(is_organic)].copy()

# Compute descriptiors for both models
def aromatic_proportion(mol):
    if mol is None:
        return float("nan")
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return 0.0
    aro = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    return aro / heavy
def compute_desc_rf(mol):
    return pd.Series({
        "HBA": Lipinski.NumHAcceptors(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "LogP": Descriptors.MolLogP(mol),
        "MolWt": Descriptors.MolWt(mol),
        "AroProp": aromatic_proportion(mol)})
def compute_desc_mlp(mol):
    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "AroProp": aromatic_proportion(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "BertzCT": Descriptors.BertzCT(mol)})

# Finalize dataframes for each model
desc_rf = df["mol"].apply(compute_desc_rf)
df_rf = pd.concat([df.drop(columns=["mol"]), desc_rf], axis=1)
desc_mlp = df["mol"].apply(compute_desc_mlp)
df_mlp = pd.concat([df.drop(columns=["mol"]), desc_mlp], axis=1)

# Prep models
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)