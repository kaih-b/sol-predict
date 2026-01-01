from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

# Identifies final set of molecular descriptors which are most pertinent to solubility
esol_descriptors = ['MolWt', 'LogP', 'TPSA', 'RotB', 'AroProp']
# The factors which most effect solubility are size, polarity, hydrophobicity, flexibility, & aromaticity
# MolWt -> size; LogP -> hydrophobicity; TPSA -> polarity; RotB -> flexibility; AroProp -> aromaticity

# Calculates ESOL descriptors for any given SMILES representaion of a molecule
def get_esol_descriptors(SMILES):
    """
    Calculate the above ESOL descriptors for a given SMILES string.

    Parameters:
    SMILES (str): The SMILES representation for the molecule.

    Returns:
    dict: A dictionary containing the calculated ESOL descriptors.
    """
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        return None
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotB': Descriptors.NumRotatableBonds(mol),
        "AroProp": (sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / mol.GetNumAtoms())
        }
    return descriptors

# Reads in SMILES and logS data from previous week's dataset
df = pd.read_csv('wk2/delaney_dataset.csv')
df.rename(columns={'measured log(solubility:mol/L)': 'logS', 'ESOL predicted log(solubility:mol/L)': 'ESOL logS'}, inplace=True)
smiles_df = df[['SMILES', 'logS']]

# Applies ESOL descriptor calculation to each SMILES string and stores in a new dataframe; removes any invalid entries
descriptor_series = smiles_df['SMILES'].apply(get_esol_descriptors)
valid_idx = descriptor_series.notnull()
descriptor_df = pd.DataFrame(descriptor_series[valid_idx].tolist())

# Checks for missing SMILES or calculation failures (all should be 0); describes the resulting dataframe
print(descriptor_df.isnull().sum())
print(descriptor_df.describe())

# Concatenates descriptor dataframe with original logS values for final dataset
X = descriptor_df[esol_descriptors]
y = smiles_df.loc[valid_idx, 'logS'].reset_index(drop=True)
final_df = pd.concat([X, y], axis=1)

# Exports final dataframe to CSV
final_df.to_csv('wk3/final_features.csv', index=False)