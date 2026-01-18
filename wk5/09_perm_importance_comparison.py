import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in perm importances
df_rf = pd.read_csv('wk4/rf_perm_importance.csv')
df_mlp = pd.read_csv('wk5/08_mlp_perm_importance.csv')

# Differentiate models
df_rf = df_rf[['descriptor', 'delta_RMSE']].rename(columns={'delta_RMSE': 'RF'})
df_mlp = df_mlp[['descriptor', 'delta_rmse_mean']].rename(columns={'delta_rmse_mean': 'MLP'})

# Merge model results and fill NaNs
df_merged = df_mlp.merge(df_rf, on='descriptor', how='left')
df_merged = df_merged.sort_values('MLP', ascending=True)
df_merged = df_merged.fillna(0)

# Initialize figure and setup plot parameters
y = np.arange(len(df_merged))
bar_height = 0.35
plt.figure(figsize=(10, 5))

# Plot bars for each model
plt.barh(
    y - bar_height/2,
    df_merged['RF'],
    height=bar_height,
    label='Random Forest')
plt.barh(
    y + bar_height/2,
    df_merged['MLP'],
    height=bar_height,
    label='Expanded MLP')

# Add labels and export
plt.yticks(y, df_merged['descriptor'])
plt.xlabel('ΔRMSE')
plt.title('Permutation Importance')
plt.legend()
plt.tight_layout()
plt.savefig('exports/permutation_importance_comparison.png', dpi=300)
plt.close()