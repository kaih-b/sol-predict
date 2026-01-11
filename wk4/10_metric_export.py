import pandas as pd

# Read in model comparisons and drop extra metrics
df_old = pd.read_csv('wk3/model_comparison.csv')
df_old.columns = df_old.columns.str.lower()
df_new = pd.read_csv('wk4/rf_final_metrics.csv')
df_new.columns = df_new.columns.str.lower()
keep_metrics = ['model', 'test_rmse', 'test_r2']
df_old_keep = [c for c in df_old.columns if c in keep_metrics]
df_new_keep = [c for c in df_new.columns if c in keep_metrics]
df_old = df_old[df_old_keep]
df_new = df_new[df_new_keep]

# Concatenate old and new model metrics; export to csv
df = pd.concat([df_old, df_new], ignore_index=True)
df = df.rename(columns={'model': 'Model', 'test_rmse': 'Test RMSE', 'test_r2': 'Test R²'})
name_map = {'LinearRegression': 'Linear Regression', 'DecisionTreeRegressor': 'Decision Tree', 
            'RandomForestRegressor': 'Random Forest (Base)', 'Tuned_RF': 'Random Forest (Tuned)'}
df['Model'] = df['Model'].replace(name_map)
df.sort_values('Test RMSE', inplace=True)
df.to_csv('tableau/inputs/model_comparison.csv', index=False)

# Clean up and export gini importances
df_imp = pd.read_csv('wk4/rf_gini_importance.csv')
df_imp = df_imp.rename(columns={'descriptor': 'Descriptor', 'gini_importance': 'Importance'})
df_imp.to_csv('tableau/inputs/rf_gini_importance.csv', index=False)

# Clean up and export permutation importances
df_perm = pd.read_csv('wk4/rf_perm_importance.csv')
df_perm.drop('delta_RMSE_std', axis=1, inplace=True)
df_perm = df_perm.rename(columns={'descriptor': 'Descriptor', 'delta_RMSE': 'ΔRMSE'})
df_perm.to_csv('tableau/inputs/rf_perm_importance.csv', index=False)