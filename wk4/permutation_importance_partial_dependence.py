import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
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
delta_rmse = perm_result.importances_mean
delta_rmse_std = perm_result.importances_std

# Export permutation importance RMSE and standard deviation into a dataframe
perm_df = pd.DataFrame({
    'descriptor': descriptor_cols,
    'delta_RMSE': delta_rmse,
    'delta_RMSE_std': delta_rmse_std})

# Sort and output the permutation results based on RMSE impact
perm_df.sort_values('delta_RMSE', ascending=False, inplace=True)
print(perm_df.head(7))

# Export results to CSV
perm_df.to_csv('wk4/rf_perm_importance.csv', index=False)

# Read in and save a full CSV with both descriptor and permutation importances
gini_df = pd.read_csv('wk4/rf_gini_importance.csv')
comparison_df = gini_df.merge(perm_df, on='descriptor', how='inner')
comparison_df.sort_values('delta_RMSE', ascending=False, inplace=True)
comparison_df.to_csv('wk4/rf_importance_comparison.csv', index=False)

# Create and save a bar plot of the descriptors ranked by delta RMSE
plt.figure()
plt.bar(x=range(len(perm_df)), height=perm_df['delta_RMSE'])
plt.xticks(ticks=range(len(perm_df)), labels=perm_df['descriptor']),
plt.ylabel('ΔRMSE (permutation importance)')
plt.title('Descriptors Ranked by Permutation Importance (Test Set)')
plt.tight_layout()
plt.savefig('wk4/rf_perm_importance_plot.png')
plt.close()

# Choose top 3 descriptors to inspect partial dependence
top_pdp_descriptors = perm_df.head(3)['descriptor'].tolist()

# Generate plot of partial dependence based on training data
disp = PartialDependenceDisplay.from_estimator(
    rf_tuned,
    X_train,
    features=top_pdp_descriptors,
    feature_names=descriptor_cols)
fig = disp.figure_
fig.tight_layout(rect=(0, 0, 1, 0.95))   # make room for title
plt.suptitle('Partial Dependence Plot for Top 3 Descriptors')
plt.savefig('wk4/rf_pdp.png', dpi=300) # increase quality to better observe partial dependence behavior
plt.close()