import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

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

# Compute train metrics for context
y_train_pred = rf_tuned.predict(X_train)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Compute test metrics as primary
y_test_pred = rf_tuned.predict(X_test)
test_rmse = root_mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Initialize CV metrics
cv_rmse_mean = 0.7133379883371542
cv_rmse_std = 0.06460845605593817

# Build a one-row metrics table
metrics_row = {'Model': 'Tuned_RF',
    'Descriptors': 'final_descriptors',
    'CV_RMSE_mean': cv_rmse_mean,
    'CV_RMSE_std': cv_rmse_std,
    'Train_RMSE': train_rmse,
    'Train_R2': train_r2,
    'Test_RMSE': test_rmse,
    'Test_R2': test_r2,
    'n_train': len(X_train),
    'n_test': len(X_test),
    'random_seed': 42}
metrics_df = pd.DataFrame([metrics_row])

# Save tuned RF metrics to CSV
metrics_df.to_csv('wk4/rf_final_metrics.csv', index=False)