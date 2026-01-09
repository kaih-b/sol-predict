import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error

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

# Compute baseline predictions and RMSE
baseline_preds = rf_tuned.predict(X_test)
baseline_rmse = root_mean_squared_error(y_test, baseline_preds)
print(f'Baseline test RMSE: {baseline_rmse:.4f}')

# Compute permutation importances (very similar setup to hyperparameter tuning; parameters matched with previous weeks)
perm_result = permutation_importance(
    rf_tuned,
    X_test,
    y_test,
    scoring='neg_root_mean_squared_error',
    n_repeats=20,
    random_state=42,
    n_jobs=-1)
perm_importances = perm_result.importances_mean
perm_importances_std = perm_result.importances_std

# Export permutation importance RMSE and standard deviation into a dataframe
perm_df = pd.DataFrame({
    'descriptor': descriptor_cols,
    'perm_importance': perm_importances,
    'perm_importance_std': perm_importances_std})

# Sort and output the permutation results based on RMSE impact
perm_df.sort_values('perm_importance', ascending=False, inplace=True)
print(perm_df.head(7))

# Export results to CSV
perm_df.to_csv('wk4/rf_perm_importance.csv', index=False)

# Read in and save a full CSV with both descriptor and permutation importances
gini_df = pd.read_csv('wk4/rf_gini_importance.csv')
comparison_df = gini_df.merge(perm_df, on='descriptor', how='inner')
comparison_df.sort_values('perm_importance', ascending=False, inplace=True)
comparison_df.to_csv('wk4/rf_importance_comparison.csv', index=False)