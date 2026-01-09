# Random Forest Hyperparameter Tuning

## Hyperparameter Grid Searched

Performed a hyperparameter grid search using 5-fold cross-validation on the following training split:

- **n_estimators**: [100, 200, 400]
- **max_depth**: [10, 20, 40]
- **min_samples_split**: [2, 4, 8]
- **min_samples_leaf**: [1, 2, 4]

## Motivation

Five-fold cross-validation has the following benefits over a single 80/20 train-test split:

- Averages performance across multiple train-test splits, reducing error variance.
- Provides a standard deviation across folds, allowing for analysis on how dependent the model's performance is on subsets across a dataset
- Reduces risk of misrepresenting overall performance from a single "lucky" or "unlucky" split.

In total, this corresponds to 3<sup>4</sup> = 81 hyperparameter combintaions, each evaluated across 5 folds for the 7 final descriptors chosen.

## Best Hyperparameters (via mean CV RMSE)

- **n_estimators** = 200
- **max_depth** = 20
- **min_samples_split** = 2
- **min_samples_leaf** = 2

**Mean CV RMSE** = 0.7133
**Standard Deviation Across Folds** = 0.0646

These hyperparameters will be used from now on as they best predicted solubility with no clear sign of overfitting (reasonable standard deviation across folds). They are the basis for evaluating this tuned random forest model against other models.