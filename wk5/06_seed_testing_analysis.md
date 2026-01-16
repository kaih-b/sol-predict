# Seed Testing Takeaways

## Results Table

| model    | mean     | std      | min      | max      |
| -------- | -------- | -------- | -------- | -------- |
| MLP_base | 0.675997 | 0.068553 | 0.568504 | 0.862101 |
| MLP_ext  | 0.648387 | 0.054642 | 0.557117 | 0.783067 |
| RF       | 0.704947 | 0.084738 | 0.518960 | 0.888231 |

## Interpretation

Across 25 random seeds, `MLP_ext` achieves the lowest mean test RMSE and the lowest standard deviation, indicating both better average performance and greater stability than the other models. Its worst-case (maximum) RMSE is also noticeably lower than that of `MLP_base` and `RF`, even though the RF attains the single best run (lowest minimum RMSE) on one seed. 

The differences in mean RMSE between models are substantially larger than the variability implied by the standard deviations, so it is reasonable to treat these differences as robust rather than noise from seed choice. On this dataset, `MLP_base` also materially outperforms `RF`, suggesting that a well-configured MLP can outperform a tuned RF, even on tabular molecular descriptor data.

## Next Steps
- Plot error distributions for `RF`, `MLP_base`, and `MLP_ext`.
- Plot training and validation curves for `MLP_base` and `MLP_ext` to assess convergence and overfitting