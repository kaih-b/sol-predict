import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('wk4/final_descriptors.csv')
target_col = 'logS'
descriptor_cols = [c for c in df.columns if c not in [target_col, 'SMILES']]

# Prepare X and y datasets for RF model
X = df[descriptor_cols]
y = df[target_col]

# Recreate train/test split with the same parameters as previous weeks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Load best hyperparameters
with open('wk4/rf_best_params.json', 'r') as f:
    best_params = json.load(f)

# Load optimized random forest model
rf_tuned = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
rf_tuned.fit(X_train, y_train)

# Use Gini to assess which descriptor reduces error (e.g. predicts solubility) most directly
importances = rf_tuned.feature_importances_
importance_df = pd.DataFrame({
    'descriptor': descriptor_cols,
    'gini_importance': importances})

# Sort and print descriptor importance ranking
importance_df.sort_values('gini_importance', ascending=False, inplace=True)
print('Descriptors ranked by Gini importance:')
print(importance_df.head(7))

# Save the ranked importances to CSV
importance_df.to_csv('wk4/rf_gini_importance.csv', index=False)