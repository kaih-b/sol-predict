import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

df = pd.read_csv('wk3/esol_final_features.csv')

# Splits  dataframe into target variable (y) and frozen feature set (x)
X = df.drop(columns=['logS'])
y = df['logS']

# Split the dataset into training and testing sets, with 80% training and 20% testing and a standard random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluates a given model using RMSE and R², assessing both deviation from true values and overall correlation
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, r2

# Seperates and exports the training and testing datasets to CSV files for later use
X_train.to_csv('wk3/X_train.csv', index=False)
X_test.to_csv('wk3/X_test.csv', index=False)
y_train.to_csv('wk3/y_train.csv', index=False)
y_test.to_csv('wk3/y_test.csv', index=False)