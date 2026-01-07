import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

df = pd.read('cleaned_descriptors.csv')

target_col = df['logS']
base_descriptors = [c for c in df.columns if c not in ['SMILES', 'logS']]

df['MOL'] = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(s) if pd.notna(s) else None)
df = df[df['MOL'].notna()]

def add_descriptors(df):
    df['HBD'] = df['MOL'].apply(Lipinski.NumHDonors)
    df['HBA'] = df['MOL'].apply(Lipinski.NumHAcceptors)
    df['RingCount'] = df['MOL'].apply(Lipinski.RingCount)
    df['FractionCSP3'] = df['MOL'].apply(rdMolDescriptors.CalcFractionCSP3)
    df['HeavyAtomCount'] = df['MOL'].apply(Lipinski.HeavyAtomCount)
    df['BertzCT'] = df['MOL'].apply(Descriptors.BertzCT)
    return df

df = add_descriptors(df)
df_expanded = df.drop(columns=['MOL'])

df_expanded.to_csv('wk4/expanded_descriptors.csv', index=False)