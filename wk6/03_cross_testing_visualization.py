import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('wk6/02_db_test_summary.csv')

# Prep bar labels
df['perm'] = df['train'] + ' → ' + df['test']
order = ['AqSolDB → AqSolDB', 'AqSolDB → ESOL', 'ESOL → AqSolDB', 'ESOL → ESOL']
df['perm'] = pd.Categorical(df['perm'], categories=order, ordered=True)
df['model'] = pd.Categorical(df['model'], categories=['MLP', 'RF'], ordered=True)

# Gather metrics
means = df.pivot(index='perm', columns='model', values='mean_rmse').loc[order]
stds  = df.pivot(index='perm', columns='model', values='std_rmse').loc[order]

# Organize figures and initialize bar objects with mean and std shown
x = np.arange(len(order))
w = 0.38
plt.figure(figsize=(10,5))
bars_mlp = plt.bar(x - w/2, means['MLP'], w, yerr=stds['MLP'], capsize=4, label='MLP')
bars_rf  = plt.bar(x + w/2, means['RF'],  w, yerr=stds['RF'],  capsize=4, label='RF')

# Label plot 
plt.xticks(x, order)
plt.ylabel('RMSE')
plt.title('Train/Test Permutations')
plt.legend()
plt.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.5)

# Add headroom for labels
top = float(np.nanmax([means['MLP'].max() + stds['MLP'].max(),
                       means['RF'].max()  + stds['RF'].max()]))
plt.ylim(0, 2.5)

# Add labels
def add_labels(bar_container, model_name):
    for i, b in enumerate(bar_container):
        perm = order[i]
        rmse = float(means.loc[perm, model_name])
        sd = float(stds.loc[perm, model_name])

        y = rmse + sd
        plt.annotate(
            f'{rmse:.3f}\n(±{sd:.3f})',
            xy=(b.get_x() + b.get_width()/2, y),
            xytext=(0, 6),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=10)
add_labels(bars_mlp, 'MLP')
add_labels(bars_rf, 'RF')

# Tighten layout and save
plt.tight_layout()
plt.savefig('exports/dataset_performance_comparison.png', dpi=300)
plt.close()