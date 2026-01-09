if __name__ == '__main__':
    import json
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV, KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error

    df = pd.read_csv('wk4/final_descriptors.csv')
    target_col = 'logS'

    # Prepare X and y datasets for RF model
    X = df.drop(columns=[target_col, 'SMILES'])
    y = df[target_col]

    # Recreate or train/test split with the same parameters as previous weeks
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Initialize a random forest regressor with the same parameters as previous weeks
    rf = RandomForestRegressor(random_state = 42, n_jobs = -1)

    # Define which hyperparameters to tune and choose base values for each
    param_grid = {
        'n_estimators': [100, 200, 400],
        'max_depth': [10, 20, 40],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4]}

    # Cross-validate with 5 folds, shuffling for randomness and specifying random_state to standardize across runs
    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)

    # Set up a grid search to assess all hyperparameter permutations (verbose enabled to help estimate grid search runtime)
    grid_search = GridSearchCV(
        estimator = rf,
        param_grid = param_grid,
        cv = cv, # use the 5-fold cross-validation established above
        scoring = 'neg_root_mean_squared_error', # negative because sklearn tries to maximize, so we want the maximum negative value e.g. the lowest absolute value
        n_jobs = -1,
        return_train_score = True, # track training scores to avoid overfitting
        verbose = 1)

    # Run the grid search on the training set
    grid_search.fit(X_train, y_train)

    # Extract results into a dataframe and convert RMSE back to positive for interpretability
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results['mean_test_RMSE'] = -cv_results['mean_test_score']
    cv_results['std_test_RMSE'] = cv_results['std_test_score']

    # Keeps only useful columns for analysis
    cols_to_keep = [
        'param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf',
        'mean_test_RMSE', 'std_test_RMSE', 'mean_train_score', 'std_train_score']
    cv_results_subset = cv_results[cols_to_keep].copy()

    # Sorts remaining columns based on RMSE score to place the best-performing permutations first
    cv_results_subset.sort_values('mean_test_RMSE', inplace=True)

    # Save hyperparameter results to csv
    cv_results_subset.to_csv('wk4/rf_cv_results.csv', index=False)

    # Convert the best parameters list to a .json file for easier interpretation and loading into later scripts
    best_params = grid_search.best_params_
    with open('wk4/rf_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    # Retrain best model on full training set, evaluate on test set, and get training set performance for comparison
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    y_train_pred = best_model.predict(X_train)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)

    # Output accuracy metrics for the highest-performing hyperparameter combination
    print(f'Best params: {best_params}')
    print(f'Best CV RMSE: {-grid_search.best_score_}')
    print(f'Training RMSE: {train_rmse}')
    print(f'Test RMSE: {test_rmse}')