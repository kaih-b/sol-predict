import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

# Load cleaned descriptor data
df = pd.read_csv('wk4/cleaned_descriptors.csv')

# Convert SMILES to RDKit Mol objects and filter out invalid entries
df['MOL'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
df = df[df['MOL'].notna()]

# Function to add new descriptors
def add_descriptors(df):
    df['HBD'] = df['MOL'].apply(Lipinski.NumHDonors)
    df['HBA'] = df['MOL'].apply(Lipinski.NumHAcceptors)
    df['RingCount'] = df['MOL'].apply(Lipinski.RingCount)
    df['FractionCSP3'] = df['MOL'].apply(rdMolDescriptors.CalcFractionCSP3)
    df['HeavyAtomCount'] = df['MOL'].apply(Lipinski.HeavyAtomCount)
    df['BertzCT'] = df['MOL'].apply(Descriptors.BertzCT)
    return df

# Add new descriptors to the dataframe
df = add_descriptors(df)
df_expanded = df.drop(columns=['MOL'])

# Save the expanded dataframe to a new CSV file
df_expanded.to_csv('wk4/expanded_descriptors.csv', index=False)