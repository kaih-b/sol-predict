import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

df = pd.read_csv('delaney_dataset.csv')
df.rename(columns={'measured log(solubility:mol/L)': 'logS', 'ESOL predicted log(solubility:mol/L)': 'ESOL logS'}, inplace=True)

# Convert SMILES to RDKit Molecule objects, removing any rows with invalid SMILES strings
df['MOL'] = df['SMILES'].apply(lambda X: Chem.MolFromSmiles(X) if pd.notna(X) else None)
df = df[df['MOL'].notna()]

# Add columns for various molecular descriptors
df['MW'] = df['MOL'].apply(Descriptors.MolWt)
df['logP'] = df['MOL'].apply(Descriptors.MolLogP)
df['HBD'] = df['MOL'].apply(Descriptors.NumHDonors)
df['HBA'] = df['MOL'].apply(Descriptors.NumHAcceptors)
df['TPSA'] = df['MOL'].apply(Descriptors.TPSA)
df['RotB'] = df['MOL'].apply(Descriptors.NumRotatableBonds)

# Exports new dataframe to CSV
df.to_csv("esol_descriptors.csv", index=False)