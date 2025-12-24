import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('wk2/esol_descriptors.csv')

# Define X and y variables
X = df[['MW', 'logP', 'TPSA', 'HBD', 'HBA', 'RotB']]
y = df['logS']

# Split the dataset into training and testing sets, with 80% training and 20% testing and a standard random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model using the training data
# We split the dataset so that we can evaluate the efficacy of the model on unseen data
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Collect the coefficients for later analysis, and places them in a dataframe for easy comparison
coef_df = pd.DataFrame({'Descriptor': X.columns, 'Coefficient': model.coef_}).sort_values(by='Coefficient', key=abs, ascending=False)
print(coef_df)

# Evaluate the model using R² and RMSE
r2_model = r2_score(y_test, y_pred)
rmse_model = np.sqrt(mean_squared_error(y_test, y_pred))

# Creates ESOL baseline to compare with linear regression model
esol_test = df.loc[y_test.index, 'ESOL logS']
r2_esol = r2_score(y_test, esol_test)
rmse_esol = np.sqrt(mean_squared_error(y_test, esol_test))

# Prints each model's R² and RMSE for comparison
print('\nRegression Model')
print(f'R² = {r2_model:.2f}, RMSE = {rmse_model:.2f}')

print('ESOL Baseline')
print(f'R² = {r2_esol:.2f}, RMSE = {rmse_esol:.2f}\n')

# Plotting the predicted vs actual logS values for both models
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Regression Model')
plt.scatter(y_test, esol_test, alpha=0.5, label='ESOL')

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--')

plt.xlabel('logS')
plt.ylabel('Predicted logS')
plt.title('Predicted vs Actual Solubility')
plt.legend()
plt.tight_layout()
plt.show()

# Tableau style plot of the elements for end-stage analysis
results_df = df.loc[y_test.index].copy()
results_df['Predicted_logS_LR'] = y_pred
results_df['Residual_LR'] = results_df['Predicted_logS_LR'] - results_df['logS']
results_df.to_csv('wk2/esol_linear_regression_results.csv', index=False)