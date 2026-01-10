import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np

# Load datasets and define target
X_base_df = pd.read_csv('wk4/cleaned_descriptors.csv')
X_expanded_df = pd.read_csv('wk4/expanded_descriptors.csv')
target_col = 'logS'

# Prepare X and y datasets for RF model
def get_Xy(df):
    X = df.drop(columns=[target_col, 'SMILES'])
    y = df[target_col]
    return X, y

# Split data into training and testing sets
X_base, y_base = get_Xy(X_base_df)
X_expanded, y_expanded = get_Xy(X_expanded_df)

# Define train-test split and ensure same y train and test sets for base and expanded datasets
X_train_base, X_test_base, y_train, y_test = train_test_split(X_base, y_base, test_size=0.2, random_state=42)
X_train_expanded = X_expanded.loc[X_train_base.index]
X_test_expanded = X_expanded.loc[X_test_base.index]

# Function to train and evaluate RF model, maintaining consistent parameters
def evaluate_rf(X_train, X_test, y_train, y_test, random_state=42):
    rf = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1) # Same parameters as wk3
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

# Train and evaluate base and expanded descriptor sets
rmse_base, r2_base = evaluate_rf(X_train_base, X_test_base, y_train, y_test)
rmse_expanded, r2_expanded = evaluate_rf(X_train_expanded, X_test_expanded, y_train, y_test)

# Print initial results
print(f'Base descriptors RMSE: {rmse_base}; R²: {r2_base}')
print(f'Expanded descriptors RMSE: {rmse_expanded}; R²: {r2_expanded}')

# Descriptor testing: remove one added feature at a time from expanded set
base_cols = set(X_base.columns)
expanded_cols = set(X_expanded.columns)
added_features = list(expanded_cols - base_cols)
print('Added features:', added_features)

# Create a results dataframe to store descriptor testing results
results = []
rmse_all, r2_all = rmse_expanded, r2_expanded

# Evaluate performance without each added feature
for feature in added_features:
    cols_without_feature = [c for c in X_expanded.columns if c != feature]
    X_train_minus = X_train_expanded[cols_without_feature]
    X_test_minus = X_test_expanded[cols_without_feature]

    rmse_minus, r2_minus = evaluate_rf(X_train_minus, X_test_minus, y_train, y_test)
    
    results.append({
        'Feature_removed': feature,
        'RMSE with all features': rmse_all,
        'R² with all features': r2_all,
        'RMSE without feature': rmse_minus,
        'R² without feature': r2_minus,
        'Delta RMSE': rmse_minus - rmse_all,    # Positive means removal hurt performance
        'Delta R²': r2_minus - r2_all})     # Negative means removal hurt performance

# Print and save descriptor testing results
descriptor_tests = pd.DataFrame(results)
print(descriptor_tests)
descriptor_tests.to_csv('wk4/descriptor_test_results.csv', index=False)

# Determine which extra features to keep based on performance impact (0.005 RMSE threshold to ensure meaningful change)
kept_extra_features = descriptor_tests[descriptor_tests['Delta RMSE'] > 0.005]['Feature_removed'].tolist()
print('Extra features kept:', kept_extra_features)

# Reevaluate model with final features
final_cols = list(base_cols.union(kept_extra_features))
X_final = X_expanded[final_cols]

# Train-test split for final feature set
rmse_final, r2_final = evaluate_rf(X_train_expanded[final_cols], X_test_expanded[final_cols], y_train, y_test)
print(f'Final feature set RMSE: {rmse_final}; R²: {r2_final}')

# Save final descriptors to CSV
final_feature_cols = list(X_final.columns)
cols_to_keep = final_feature_cols + [target_col]
if 'SMILES' in X_expanded_df.columns:
    cols_to_keep = ['SMILES'] + cols_to_keep
final_df = X_expanded_df[cols_to_keep].copy()
final_df.to_csv('wk4/final_descriptors.csv', index=False)

# At this stage, the features are ready for hyperparameter tuning and further model optimization