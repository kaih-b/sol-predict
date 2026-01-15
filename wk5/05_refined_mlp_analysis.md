# MLP Refining

## Process

This step included refining the MLP by introducing three new variables:

- **Scaler** - RF models are scale-invariant, so adding a scaler to the MLP allows for better generalization of model performance comparison. Scaling prevents the MLP from either over-interpreting one or more descriptor because its numerical values are large (e.g. MolWt, BertzCT)
- **Additional Configurations** - this hyperparameter sweep considered the following new configurations:
    - `hidden_sizes = (32, 16)`
    - `hidden_sizes = (256, 128)`
    - `hidden_sizes = (64, 32, 16, 8)`
    - `dropout_p = 0.0`
    - `dropout_p = 0.1`
    - `learning_rate = 2e-3`
    - `weight_decay = 0`
    - `weight_decay = 1e-5`
    - `weight_decay = 1e-4`
- **Extended Descriptors** - previous testing indicated that the MLP may benefit materially from additional features, so each configuration was tested both for the base descriptor set and the expanded, unpruned descriptor set used at the beginning of Week 4. It includes the following additional descriptors:
    - Hydrogen Bond Donors (HBD)
    - Ring Count
    - Ratio of sp3-hybridized Carbons to Total Carbons (CSP3)
    - Heavy Atom Count

## Results

The configuration with the lowest validation loss, `val_mse = 0.308`, and `test_rmse = 0.636` was the following:

- `n_features = 11` (expanded descriptor set)
- `hidden_sizes = (256, 128)`
- `dropout_p = 0.1`
- `learning_rate = 2e-3`
- `weight_decay = 1e-5`

This is consistent with previous interpretations of baseline MLP models. 

The configuration from the base descriptor set with the lowest validation loss, `val_mse = 0.363`, and `test_rmse = 0.642` was the following:

- `n_features = 7`
- `hidden_sizes = (64, 32, 16, 8)`
- `dropout_p = 0.0`
- `learning_rate = 2e-3`
- `weight_decay = 1e-3`

### Interpretation

Expanded descriptors lowered training error substantially, but failed to improve generalization (`test_rmse`). This is evidence that the MLP is aided in predicting training set solubility by additional features, but fails to generalize that prediction to unseen data. With these results, there is no need to revert to the expanded descriptor set.

Between the two descriptor sets, best model performance also changed. Extended best uses some dropout (`0.1`) and very light weight decay (`1e-5`) with a wide model `(256–128)`, while base best uses no dropout but moderately heavy weight decay (`1e-3`) with a deeper, smaller network `(64–32–16–8)`.

This is consistent with intuition. With more features, the model benefits from more capacity (wider and lighter decay) while with fewer features, the model benefits from stronger strinkage (heavy decay) to improvce generalization.

### Next Steps

Before axing the extended descriptor model entirely, we will perform multi-seed comparisons for both model's performances and compute mean and standard deviation for test RMSE across both models. If there remains no material performance benefit, we will accept the `(64-32-16-8)_0.0,adam_lr2e-3_wd1e-3` as the MLP for comparison against RF performance.