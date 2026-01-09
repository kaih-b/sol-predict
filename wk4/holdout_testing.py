import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

df = pd.read_csv('wk4/final_descriptors.csv')
target_col = 'logS'

# Prepare X and y datasets for RF model
X = df.drop(columns=[target_col, 'SMILES'])
y = df[target_col]

# Recreate train/test split with the same parameters as previous weeks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Load best hyperparameters
with open('wk4/rf_best_params.json', 'r') as f:
    best_params = json.load(f)

# Load optimized random forest model
rf_tuned = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
rf_tuned.fit(X_train, y_train)

# Train metrics
y_train_pred = rf_tuned.predict(X_train)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Test metrics
y_test_pred = rf_tuned.predict(X_test)
test_rmse = root_mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Save and output train and test metrics for the tuned model on the holdout data
metrics = {
    'train_RMSE': f'{float(train_rmse):.4f}',
    'train_R2': f'{float(train_r2):.4f}',
    'test_RMSE': f'{float(test_rmse):.4f}',
    'test_R2': f'{float(test_r2):.4f}',
    'cv_mean_RMSE': 0.7133}
print(metrics)

# Because test_RMSE < CV_RMSE, there is no indication of overfitting!
# The difference indicates that the 20% test data happens to be easier than the CV folds
# Because the difference falls within 2 standard deviations, there is no evidence of other error