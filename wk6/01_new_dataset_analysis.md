# External Validation on AqSolDB

The AqSolDB test set differs substantially from Delaney ESOL, and the effects are clear in the residual diagnostics and bin-stratified metrics.

`test_rmse_mlp = 1.168`

`test_rmse_rf = 1.208`

Though this generalization beyond the ESOL molecules (1.71x RMSE for RF, 1.80x RMSE for MLP, as extracted from RMSEs across 25 seeds shown in `wk4/06_seed_testing_summary.csv`) is relatively poor, further analysis indicates why these differences are present and where the model **is** able to generalize. 

**NOTE:** To more directly compared RMSE performace across datasets, the AqSolDB models should be surveyed across seeds, but this simplified version works as a basic diagnostic for where the models succeed and fail with completely new data.

## Summary

External performance degrades primarily due to domain shift: AqSolDB contains large and complex molecules with an extreme solubility distribution, whereas ESOL primarily contains smaller, drug-like molecules. This shift had substantial effects on the models' abilities to generalize.

- Bin-stratified metrics confirm size & complexity dependent error growth (e.g. MLP RMSE rises from ~1.0 at ≤20 heavy atoms to ~2.1 at 40–60)
- Both RF and MLP struggle in the very low-solubility tail (RMSE ~ 2.36 for logS in (-10, -8]), implying that chemistry with these extreme cases is just more difficult to predict
- Residual diagnostics across descriptors indicate the main failure mode is extrapolation in new regions, not a single-feature bias
- For the model to generalize better across AqSolDB's domain, additional features and a much broader training set would likely be advantageous

## Error Increases Systematically with Molecular Size & Complexity

Stratifying by HeavyAtomCount and BertzCT shows a clear decline in MLP performance:

### MLP vs HeavyAtomCount
- ≤20 heavy atoms: RMSE 0.98, R² 0.80 (n=706)
- 20–40: RMSE 1.48, R² 0.52 (n=216)
- 40–60: RMSE 2.08, R² 0.44 (n=33)

### MLP vs BertzCT
- 0–500: RMSE 1.05, R² 0.71 (n=612)
- 500–1000: RMSE 1.17, R² 0.72 (n=267)
- 1000–1500: RMSE 1.94, R² 0.33 (n=53)

RF follows a similar pattern (does not use HeavyAtomCount as a descriptor, MolWt is highly correlated and show similar issues in the residual diagnostic below) and degrades even more in the 1000–1500 complexity band (RMSE 2.11, R² 0.21). The takeaway is not the exact bins but the trend: as molecules become larger and more complex, both models lose predictive stability. This is consistent with the claim that the model fails to generalize beyond the ESOL domain.

## Breakdown in Low-Solubility Distribution Tail

Binning by logS value shows very large errors in the low-solubility tail for both models:

- (-8, -6]: RMSE ~ 1.68 (MLP/RF)
- (-10, -8]: RMSE ~ 2.36 (MLP/RF), n=28

Within these logS bins, the actual logS variance is small (std ~ 0.42–0.56), so RMSE values even above 1 imply the models are missing signals, and these errors are very costly.

AqSolDB contains a broader and more extreme solubility range than ESOL, and the very low logS tail likely reflects a combination of (i) out-of-distribution chemistry relative to the ESOL training domain and (ii) greater variance common in compiled experimental databases (e.g. differing experimental conditions or measurement protocols), both of which can increase error in the extreme tail.

## Residual Plot Diagnostics

Across the residual plots (shown in `wk6/dataset_figures`), the dominant pattern is:

- Dense central regions (typical drug-like ranges) show residuals roughly centered near 0 with moderate spread
- Sparse extremes show larger and more scattered residuals (high leverage points)

Specific signals from representative plots:

### MolWt

![MolWt MLP Residual Plot for AqSolDB Dataset](dataset_figures/residuals_vs_MolWt_MLP.png)
![MolWt RF Residual Plot for AqSolDB Dataset](dataset_figures/residuals_vs_MolWt_RF.png)

MAqSolDB includes rare very high-molecular-weight compounds (approaching ~2000), and these high-leverage points are associated with larger residual magnitudes.

### TPSA

![TPSA MLP Residual Plot for AqSolDB Dataset](dataset_figures/residuals_vs_TPSA_MLP.png)
![TPSA RF Residual Plot for AqSolDB Dataset](dataset_figures/residuals_vs_TPSA_RF.png)

Also heavy-tailed (rare points with very high TPSA). Errors appear more variable where data are sparse, suggesting limited training coverage in highly polar molecules.

### logP

![LogP MLP Residual Plot for AqSolDB Dataset](dataset_figures/residuals_vs_LogP_MLP.png)
![LogP RF Residual Plot for AqSolDB Dataset](dataset_figures/residuals_vs_LogP_RF.png)

Residuals remain roughly centered in the mid-range, but tails include scattered large errors, again pointing to extrapolation error over descriptor bias.

Overall, these plots do not suggest a single descriptor is “wrong.” Instead, they indicate that the dominant failure mode is extrapolation in sparsely populated regions of feature space (e.g., very large/complex molecules and extreme tail values), where the ESOL-trained models have limited training coverage and residual variance increases. 

## Where is the Model Most Useful?

Given the AqSolDB results, the RF/MLP models are most reliable when applied to molecules similar to the Delaney ESOL domain. This includes small-to-moderate sized, organic, “drug-like” compounds where the descriptor ranges overlap the training distribution.

Recommended operating domain: lower size/complexity regime, where both error and fit quality are substantially better.

- HeavyAtomCount ≤ 20: MLP RMSE ≈ 0.98, R² ≈ 0.80 (n=706)
- BertzCT ≤ 1000: MLP RMSE ≈ 1.05–1.17, R² ≈ 0.71–0.72 (n=879 combined)

In practice, this corresponds to typical small molecules (often similar to drug-like scale), and avoids the high-leverage regions where performance degrades (HeavyAtomCount > 40 for MLP; very high BertzCT; very low logS tail). Practically, this means the model is well-suited for in-domain screening and prioritization of ESOL-like small molecules, a common application for chemical engineering workflows or applications in pharmaceutical R&D.