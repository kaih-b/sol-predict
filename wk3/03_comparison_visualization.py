import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the CSV files
lin = pd.read_csv('wk3/linear_regression_results.csv')  # adjust name if needed
dt = pd.read_csv('wk3/decision_tree_results.csv')
rf = pd.read_csv('wk3/random_forest_results.csv')

# Concatenate the dataframes
results = pd.concat([lin, dt, rf], ignore_index=True)
results = results.replace({np.nan: None})

# Reorganize the concatenated dataframe
results = results[['model',
    'test_rmse',
    'test_r2',
    'train_rmse',
    'train_r2',
    'max_depth',
    'n_estimators']]

# Save the combined results to a new CSV file
results.to_csv('wk3/model_comparison.csv', index=False)

# Visualize and save the comparison of models based on Test RMSE
plt.bar(results['model'], results['test_rmse'])
plt.ylabel('Test RMSE')
plt.title('Model Test RMSE Comparison')
plt.tight_layout()
plt.savefig('wk3/rmse_comparison.png')
plt.show()

# Visualize and save the comparison of models based on Test R²
plt.bar(results['model'], results['test_r2'])
plt.ylabel('Test R²')
plt.title('Model Test R² Comparison')
plt.tight_layout()
plt.savefig('wk3/r2_comparison.png')
plt.show()