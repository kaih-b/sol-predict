# SolPredict Final Results

## 1. Background & Summary

Aqueous solubility plays a central role in drug discovery, affecting absorption, distribution, and formulation decisions. Experimentally measuring solubility is costly and time-consuming, motivating the use of predictive computational models to predict compound solubility early in development. Machine learning approaches to solubility prediction typically rely on molecular descriptors that encode size, polarity, lipophilicity, functional group composition, and more. Classical models such as Random Forests are well-suited to tabular descriptor data and often serve as strong baselines. Neural networks, while more sensitive to hyperparameters and initialization, offer increased representational capacity and may capture nonlinear interactions among descriptors. In this project, I developed, tuned, and compared machine learning models for predicting aqueous solubility (`logS`) from molecular descriptors using the Delaney ESOL dataset, with an emphasis on fair model comparison, stability across random seeds, and underlying chemistry-based interpretability.

Three main models were tuned and evaluated: a Random Forest via scikit-learn with a baseline descriptor-set, a multilayer perceptron (MLP) via PyTorch using the same base descriptor-set, and an MLP using an expanded descriptor-set. Across 25 random-seed trials, the expanded MLP achieved both the lowest mean RMSE and the lowest variance, indicating generalization and training stability superior to the Random Forest and base MLP. While the Random Forest occasionally produced strong individual predictions, its performance was less stable across seeds.

![Final Model Comparison](exports/updated_model_comparison.png)

### 1.1 Key Takeaways

- Neural networks materially outperform Random Forests under a matched evaluation protocol for this task
- Expanding the descriptor set improves stability moreso than peak performance
- Performance differences are larger than expected seed-level noise, supporting meaningful model comparisons

This project is exploratory in nature and prioritizes methodological logic and learning over production-grade reproducibility, and results should be interpreted accordingly.

## 2. Data & Experimental Setup

### 2.1 Dataset

Models were trained on the Delaney ESOL dataset, with the target variable defined as `logS` (logarithm of aqueous solubility). Dataset preprocessing was kept minimal to avoid unnecessary complications or introducing additional modeling assumptions.

### 2.2 Train/Validation/Test Splits

Data were split into training, validation, and test sets using a fixed 80/10/10 split. The same split was reused across all models to ensure comparability (RF trained on the 80/10 train-validation segment, so all testing was done against the same remaining 10 for each model).

### 2.3 Evaluation Metrics

Root Mean Squared Error (RMSE) was used as the primary evaluation metric, with R² reported as a secondary diagnostic. Tuned hyperparameters were determined based on minimum RMSE, keeping the train/validation/test split consistent throughout. To assess robustness, each final model configuration was evaluated across 25 random seeds. Final performance is reported as the mean and standard deviation of RMSE across seeds, rather than relying on a single run.

## 3. Features & Models

### 3.1 Descriptors

Two sets of descriptors were used in the final models:
1. **Base**: comprised of seven descriptors capturing size, polarity, complexity, aromaticity, and lipophilicity.
2. **Expanded**: comprised of eleven descriptors intended to provide richer chemical signal for the NN model.

### 3.2 Models

1. **Random Forest**: The Random Forest model serves as a strong baseline for tabular chemical data. It is more insensitive to feature scaling and can capture nonlinear effects, but struggles to extrapolate smoothly across a wider chemical space. 
    
    Tuned Hyperparameters: `max_depth = 20`, `min_samples_leaf = 2`, `min_samples_split = 2`, `n_estimators = 200`.

2. **Base MLP**: A neural network trained on the base descriptor set using an Adam optimizer. 
    
    Tuned Hyperparamaters: `hidden_sizes = (64, 32, 16, 8)`, `dropout_p = 0.0`, `learning_rate = 2e-3`, `weight_decay = 1e-3`
    
3. **Expanded MLP**: Trained on the expanded descriptor set. Tests whether additional feature information improves learning stability and generalization.

    Tuned Hyperparameters: `hidden_sizes = (256, 128)`, `dropout_p = 0.1`, `learning_rate = 2e-3`, `weight_decay = 1e-5`

## 5. Results

### 5.1 Seed-Tested Performance

Across repeated seed evaluations, the expanded-descriptor MLP achieved the lowest mean RMSE and lowest standard deviation, indicating both improved accuracy and consistency. The base MLP outperformed the Random Forest on average, suggesting that even modest neural architectures can be competitive when properly tuned.

| Model | Features | Mean RMSE | RMSE STD |
|-------|----------|------|------|
| Tuned Random Forest | tuned | 0.705 | 0.085 |
| Base MLP | tuned | 0.676 | 0.069 |
| Expanded MLP | expanded | 0.648 | 0.055 |

### 5.2 Residual and Error Analysis

Residual distributions for all models were approximately centered around zero, indicating low systematic bias. However, the expanded MLP showed tighter clustering and shorter tails, consistent with its lower RMSE variance. Error analysis suggests that descriptor expansion improves robustness rather than simply optimizing best-case performance.

![Residual Comparison by Model](exports/rf_mlp_residual_comparison.png)

### 5.3 MLP Learning Behavior

Learning curves for the neural networks showed stable convergence with minor overfitting. The expanded-descriptor MLP exhibited smoother validation loss trajectories, suggesting improved signaling during training, but also appears to have mildly greater overfitting.

![Base MLP Learning Curves](wk5/07_MLP_base_learning_curve.png)
![Expanded MLP Learning Curves](wk5/07_MLP_ext_learning_curve.png)

## 6. Basic Model Interpretation

### Random Forest

Gini-based and permutation analysis identified `logP`, a metric for lipophilicity, as by far the greatest predictor for solubility. `BertzCT`, a proxy for molecular complexity, `MolWt`, and `TPSA` (total polar surface area) followed.

![Gini Importance RF](exports/rf_gini_importance.png)
![Permutation Importance RF](exports/rf_perm_importance.png)

### MLP Models

Permutation analysis for the expanded MLP model revealed somewhat overlapping but different importance patterns compared to the Random Forest. The expanded MLP appeared to distribute importance across a broader set of descriptors, consistent with its improved stability. As with the Random Forest, these findings reflect model sensitivity rather than mechanistic insight.

![Permutation Importance RF vs MLP](exports/permutation_importance_comparison.png)