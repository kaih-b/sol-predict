import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from model_evaluator import evaluate_model

# Load the training and testing datasets, making sure to squeeze y data into a series from a dataframe
X_train = pd.read_csv('wk3/X_train.csv')
X_test = pd.read_csv('wk3/X_test.csv')
y_train = pd.read_csv('wk3/y_train.csv').squeeze()
y_test = pd.read_csv('wk3/y_test.csv').squeeze()

# Creates and fits the Random Forest Regressor model, using all CPU cores
model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
# Fit the model on the training data
model.fit(X_train, y_train)

# Evaluate model
rmse, r2 = evaluate_model(model, X_test, y_test)
train_rmse, train_r2 = evaluate_model(model, X_train, y_train)

# Output evaluation metrics, including training set performance to assess overfitting
print('Random Forest Regression Results')
print(f'Test RMSE: {rmse:.3f}')
print(f'Test R²: {r2:.3f}')
print(f'Train RMSE: {train_rmse:.3f}')
print(f'Train R²: {train_r2:.3f}')

# Output model parameters for transparency
print('Model parameters:', model.get_params())

# Save results to csv
results = {'model': 'RandomForestRegressor',
    'max_depth': None,
    'test_rmse': rmse,
    'test_r2': r2,
    'train_rmse': train_rmse,
    'train_r2': train_r2,
    'n_estimators': 200}
pd.DataFrame([results]).to_csv('wk3/random_forest_results.csv',index=False)

# Save feature importances to csv
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importances.to_csv('wk3/rf_feature_importances.csv')