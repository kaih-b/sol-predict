import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

seed = 42

# Import and organize frozen final descriptors
df = pd.read_csv('wk4/final_descriptors.csv')
target_col = 'logS'
descriptor_cols = [c for c in df.columns if c not in [target_col, 'SMILES']]
X = df[descriptor_cols]
y = df[target_col]

# Recreate train-test-val split (80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, random_state = seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = seed)

# Combine training and validation set (this is the true training set for the RF)
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

# Load optimized RF model
with open('wk4/rf_best_params.json', 'r') as f:
    best_params = json.load(f)
rf_tuned = RandomForestRegressor(random_state = 42, n_job = -1, **best_params)
rf_tuned.fit(X_train, y_train)

# Fit model
rf_tuned.fit(X_train_full, y_train_full)
y_test_pred = rf_tuned.predict(X_test)

# Gather test metrics
test_rmse = root_mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
resid = y_test - y_test_pred
abs_err = np.abs(resid)

# Summarize metrics
report = (
    '----- Tuned RF Model w/ 80/10/10 Split -----\n'
    f'RMSE: {test_rmse:.3f}\n'
    f'R2: {test_r2:.3f}')
print(report)

# Plot residuals (carryover from wk4/08_residual_diagnostic.py)
# Visualize and save residuals plot
plt.figure()
plt.hist(resid, bins=30)
plt.axvline(0.0, linestyle='--', color='black')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.title('Test Residual Distribution')
plt.tight_layout()
plt.savefig('wk5/04_rf_rerun_resid_dist.png', dpi=300)
plt.close()

# Visualize and save residuals vs predicted values plot
plt.figure()
plt.scatter(y_test_pred, resid, alpha=0.5, s=10)
plt.axhline(0.0, linestyle='--', color='black')
plt.xlabel('Predicted logS')
plt.ylabel('Residual')
plt.title('Residuals vs Predicted (Test Set)')
plt.tight_layout()
plt.savefig('wk5/04_rf_rerun_resid_vs_predicted.png', dpi=300)
plt.close()

# RMSE is slightly higher, but still very near to its original value, 
# indicating that the holdout set does not have a substantial effect on performance (as hoped)