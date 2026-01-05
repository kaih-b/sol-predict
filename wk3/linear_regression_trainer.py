import pandas as pd
from sklearn.linear_model import LinearRegression
from model_evaluator import evaluate_model

# Load the training and testing datasets, making sure to squeeze y data into a series from a dataframe
X_train = pd.read_csv('wk3/X_train.csv')
X_test = pd.read_csv('wk3/X_test.csv')
y_train = pd.read_csv('wk3/y_train.csv').squeeze()
y_test = pd.read_csv('wk3/y_test.csv').squeeze()

# Initialize baseline linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Evaluate model
rmse, r2 = evaluate_model(model, X_test, y_test)
train_rmse, train_r2 = evaluate_model(model, X_train, y_train)

# Output evaluation metrics, including training set performance to assess overfitting
print('Linear Regression Results')
print(f'Test RMSE: {rmse:.3f}')
print(f'Test R²: {r2:.3f}')
print(f'Train RMSE: {train_rmse:.3f}')
print(f'Train R²: {train_r2:.3f}')

# Output model parameters for transparency
print('Model parameters:', model.get_params())

# Save results to csv
results = {'model': 'LinearRegression',
    'max_depth': 5,
    'test_rmse': rmse,
    'test_r2': r2,
    'train_rmse': train_rmse,
    'train_r2': train_r2}

pd.DataFrame([results]).to_csv('wk3/linear_regression_results.csv', index=False)