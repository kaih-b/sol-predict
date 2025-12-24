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

# Create and plot the correlation between these descriptors and logS
corr_df = X.copy()
corr_df['logS'] = y

plt.figure(figsize=(8,6))
sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='Blues', square=True)
plt.title('Correlation Heatmap of Molecular Descriptors for logS')
plt.show()

# Takeways
# logP (-0.83) and MW (-0.64) have the strongest inverse correlation with logS,
# while HBD (0.22) and TPSA (0.13) have weak positive correlations.
# In terms of interrlation between descriptors, TPSA and HBA have a strong positive correlation (0.90), 
# followed by TPSA and HBD (0.76). HBA and HBD have a moderate positive correlation (0.58).
# MW moderately correlates with HBA (0.55), TPSA, and logP (both 0.47).
# MW and logS have a moderate negative correlation (-0.64), while logP has a moderate negative correlation
# with HBD (-0.52) and TPSA (-0.46). All other correlations are relatively weak (abs value < 0.4).

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

# Prints model's R² and RMSE for comparison
print('Regression Model')
print(f'R² = {r2_model:.2f}, RMSE = {rmse_model:.2f}\n')