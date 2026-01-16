import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in per-seed and per-point results
seed_df = pd.read_csv('wk5/06_seed_testing_results.csv')
preds_df = pd.read_csv('wk5/06_per_iteration_preds.csv')
curves_df = 

# Define abs err and create a function to easily output error metrics for each model
preds_df["abs_err"] = preds_df["residual"].abs()
def arrays_for(model_name):
    dfm = preds_df[preds_df["model"] == model_name]
    y_true = dfm["y_exp"].to_numpy(dtype=float)
    resid = dfm["residual"].to_numpy(dtype=float)
    abs_err = dfm["abs_err"].to_numpy(dtype=float)
    return y_true, resid, abs_err

# Collect error metrics for each model
y_rf, resid_rf, abs_rf = arrays_for("RF")
y_mlp, resid_mlp, abs_mlp = arrays_for("MLP_base")
y_ext, resid_ext, abs_ext = arrays_for("MLP_ext")

# Collect matplotlib endpoints and all residuals
resid_all = np.concatenate([resid_rf, resid_mlp, resid_ext])
lim = np.quantile(np.abs(resid_all), 0.99)
bins = np.linspace(-lim, lim, 60)

# Visualize residual plot for each model
plt.figure()
plt.hist(resid_rf, bins=bins, alpha=0.5, label="RF")
plt.hist(resid_mlp, bins=bins, alpha=0.5, label="MLP_base")
plt.hist(resid_ext, bins=bins, alpha=0.5, label="MLP_ext")
plt.axvline(0.0, linewidth=1)
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Residual Histogram")
plt.legend()
plt.tight_layout()
plt.savefig('exports/rf_mlp_residual_comparison.png', dpi=300)
plt.close()

# Visualize and save residual vs true logS plot for each model
def resid_vs_true_plot(y_true, resid, outpath):
    plt.figure()
    plt.scatter(y_true, resid, s=10, alpha=0.3)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Experimental logS")
    plt.ylabel("Residual")
    plt.title('Residual vs True logS')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
resid_vs_true_plot(y_rf, resid_rf, "wk5/07_rf_resid_vs_exp.png")
resid_vs_true_plot(y_mlp, resid_mlp, "wk5/07_mlp_base_resid_vs_exp")
resid_vs_true_plot(y_ext, resid_ext, 'wk5/07_mlp_ext_resid_vs_exp')

# Define and save CDF info for each model
def cdf_xy(abs_err):
    s = np.sort(abs_err)
    p = np.arange(1, len(s) + 1) / len(s)
    return s, p
x_rf, p_rf = cdf_xy(abs_rf)
x_mlp, p_mlp = cdf_xy(abs_mlp)
x_ext, p_ext = cdf_xy(abs_ext)

# Plot CDFs for each model
plt.figure()
plt.plot(x_rf, p_rf, label="RF")
plt.plot(x_mlp, p_mlp, label="MLP_base")
plt.plot(x_ext, p_ext, label="MLP_ext")
plt.xlabel("|Residual|")
plt.ylabel("CDF")
plt.title("Absolute Error CDF")
plt.legend()
plt.tight_layout()
plt.savefig("exports/rf_mlp_abs_err_cdf", dpi=300)
plt.close()

# Summarize error distributions for each of the three models
def summarize_errors(resid, abs_err):
    return {
        "resid_mean": float(np.mean(resid)),
        "resid_std": float(np.std(resid)),
        "median_abs_err": float(np.quantile(abs_err, 0.50)),
        "p90_abs_err": float(np.quantile(abs_err, 0.90)),
        "p95_abs_err": float(np.quantile(abs_err, 0.95))}

# Construct dataframe
summary_rows = [
    {"model": "RF", **summarize_errors(resid_rf, abs_rf)},
    {"model": "MLP_base", **summarize_errors(resid_mlp, abs_mlp)},
    {"model": "MLP_ext", **summarize_errors(resid_ext, abs_ext)}]
summary_df = pd.DataFrame(summary_rows)

# Add RMSE metrics and export df
rmse_stats = (seed_df.groupby("model")["test_rmse"].agg(rmse_mean="mean", rmse_std="std", rmse_min="min", rmse_max="max").reset_index())
summary_df = summary_df.merge(rmse_stats, on="model", how="left")
summary_df.to_csv("wk5/07_error_dist_summary.csv", index=False)