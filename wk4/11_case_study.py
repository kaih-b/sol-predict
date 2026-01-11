import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Recreate train-test split
df = pd.read_csv('wk4/final_descriptors.csv')
target_col = 'logS'
descriptor_cols = [c for c in df.columns if c not in [target_col, 'SMILES']]
X = df[descriptor_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Load optimized random forest model
with open('wk4/rf_best_params.json', 'r') as f:
    best_params = json.load(f)
rf_tuned = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
rf_tuned.fit(X_train, y_train)

# Compute predicted logS and residuals
y_pred = rf_tuned.predict(X_test)
residuals = y_test - y_pred
abs_errs = np.abs(residuals)

# Select the worst-case  molecule for case study
worst_pos = int(np.argmax(abs_errs))
worst_index = y_test.index[worst_pos]
row = df.loc[worst_index]

# Calculate metrics for worst-case molecule
SMILES = row['SMILES']
MolWt = row['MolWt']
BertzCT = row['BertzCT']
logP = row['LogP']
logS = float(y_test[worst_index])
pred_logS = float(y_pred[worst_pos])
resid = float(residuals[worst_index])
abs_err = float(abs_errs[worst_index])

# Create a dataframe for worst-case molecule; save to csv
data = {'SMILES': SMILES,
        'MolWt': MolWt,
        'BertzCT': BertzCT,
        'logP': logP,
        'exp_logS': logS,
        'pred_logS': pred_logS,
        'residual': resid,
        'abs_error': abs_err}
case_df = pd.DataFrame([data])
case_df.to_csv('wk4/worst_case_study.csv', index=False)

# Visualize and save worst-case study molecule
exp_val = case_df['exp_logS'].iloc[0]
pred_val = case_df['pred_logS'].iloc[0]
abs_error = case_df['abs_error'].iloc[0]
labels = ['Experimental logS', 'Predicted logS']
values = [exp_val, pred_val]
plt.bar(labels, values)
plt.ylabel('logS')
plt.title('Case Study Molecule: Experimental vs Predicted logS')
# Annotate error
plt.text(1.0, max(values) * 1.1, f'|Error| = {abs_error:.3f}', ha='center')
plt.tight_layout()
plt.savefig('exports/case_study.png', dpi=300)
plt.close()