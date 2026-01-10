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

# Compute predictions and residuals
y_test_pred = rf_tuned.predict(X_test)
residuals = y_test - y_test_pred
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
residuals = residuals.reset_index(drop=True)

# Visualize and save residuals plot
plt.figure()
plt.hist(residuals, bins=30)
plt.axvline(0.0, linestyle='--', color='black')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.title('Test Residual Distribution')
plt.tight_layout()
plt.savefig('wk4/residuals_histogram.png', dpi=300)
plt.close()

# Visualize and save residuals vs predicted values plot
plt.figure()
plt.scatter(y_test_pred, residuals, alpha=0.5, s=10)
plt.axhline(0.0, linestyle='--', color='black')
plt.xlabel('Predicted logS')
plt.ylabel('Residual')
plt.title('Residuals vs Predicted (Test Set)')
plt.tight_layout()
plt.savefig('wk4/residuals_vs_predicted.png', dpi=300)
plt.close()

# Identify top descriptors as previously determined
top_descriptors = ['LogP', 'BertzCT', 'MolWt']

# Visualize and save residual plot for each top descriptor
for desc in top_descriptors:
    x_vals = X_test_reset[desc]
    plt.figure()
    plt.scatter(x_vals, residuals, alpha=0.5, s=10)
    plt.axhline(0.0, linestyle='--', color='black')
    plt.xlabel(desc)
    plt.ylabel('Residual')
    plt.title(f'Residuals vs {desc} (Test Set)')
    plt.tight_layout()
    out_path = f'wk4/residuals_vs_{desc}.png'
    plt.savefig(out_path, dpi=300)
    plt.close()

# Initialize summary list
summary_rows = []

# Save correlation values between residuals and each top descriptor
for desc in top_descriptors:
    x_vals = X_test_reset[desc].values
    res_vals = residuals.values
    # Use numpy.corrcoef for Pearson linear correlation (identifies basic trends between top features and residuals)
    corr_matrix = np.corrcoef(x_vals, res_vals)
    corr = corr_matrix[0, 1]
    summary_rows.append({'descriptor': desc, 'residual_corr': corr})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('wk4/residuals_feature_correlation.csv', index=False)