import pandas as pd
import numpy as np

# Read in the frozen ESOL descriptors
df = pd.read_csv('wk3/final_features.csv')

# Identify the target and descriptor columns
target_col = 'logS'
descriptor_cols = ['MolWt', 'LogP', 'TPSA', 'RotB', 'AroProp']

# Separate the features and target variable
X = df[descriptor_cols].copy()
y = df[target_col].copy()

# Calculate and print the variances of each descriptor
variances = X.var()
print(variances)

# Identify and print descriptors with near-zero variance
near_zero_var_threshold = 1e-4
near_zero_var = variances[variances < near_zero_var_threshold]
print('Near-zero variance descriptors:')
print(near_zero_var)

# Calculate and print the correlation matrix of the descriptors
corr_matrix = X.corr().abs()
print(corr_matrix)

# Identify and print highly correlated descriptor pairs (correlation > 0.9)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [(col, row, upper.loc[row, col])
    for col in upper.columns
    for row in upper.index
    if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.9]
print('Highly correlated pairs (|r| > 0.9): ')
for a, b, r in high_corr_pairs:
    print(a, b, r)

# For high correlation, we can drop one of the descriptors in each pair
to_drop_corr = set()

for col1, col2, r in high_corr_pairs:
    # compute mean absolute correlation of each with all others
    mean_corr1 = corr_matrix[col1].mean()
    mean_corr2 = corr_matrix[col2].mean()
    # drop the one that has a higher mean correlation with others
    drop = col1 if mean_corr1 > mean_corr2 else col2
    to_drop_corr.add(drop)
print('Dropping due to high correlation:', to_drop_corr)

# Drop the identified descriptors from the dataset
to_drop_var = set(near_zero_var.index)
to_drop = list(to_drop_var.union(to_drop_corr))
print('Final descriptors to drop:', to_drop)
X_clean = X.drop(columns=to_drop)
print('Columns kept:', X_clean.columns.tolist())

# Based on the analysis, we can decide to drop descriptors if needed
# For these descriptors and this sample, no descriptors are dropped (minimum variance is 1e-1 and max correlation is 0.473)
# This is a good indication that the descriptors used are appropriate for modeling (non-redundant, independent, and changing across data)
# However, further analysis is needed with other data to validate this conclusion

# Save the 'cleaned' descriptor dataset (same as original in this case, with added SMILES from original dataset)
SMILES = pd.read_csv('wk2/delaney_dataset.csv')['SMILES']
X_clean.insert(0, 'SMILES', SMILES)
X_clean.insert(1, 'logS', y)
X_clean.to_csv('wk4/cleaned_descriptors.csv', index=False)