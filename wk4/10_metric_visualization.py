import pandas as pd
import matplotlib.pyplot as plt

# Compare and save model evaluation metrics
df = pd.read_csv('exports/model_comparison.csv')
df = df.sort_values('Test RMSE', ascending=True)
plt.figure(figsize=(8, 4))
plt.bar(df['Model'], df['Test RMSE'])
plt.ylabel('Test RMSE')
plt.title('Model Comparison')
for i, (rmse, r2) in enumerate(zip(df['Test RMSE'], df['Test R²'])):
    plt.text(i, rmse, f'{rmse:.3f}\n(R²={r2:.3f})', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.xticks(fontsize=10)
plt.ylim(0, (1.2))
plt.savefig('exports/model_comparison.png', dpi=300)
plt.close()

# Compare and save descriptor importances
df_desc = pd.read_csv('exports/rf_gini_importance.csv')
df_desc = df_desc.sort_values('Importance', ascending=True)
plt.figure(figsize=(7, 4))
plt.barh(df_desc['Descriptor'], df_desc['Importance'])
plt.xlabel('Gini Importance')
plt.title(f'Random Forest Feature Importance (Gini)')
plt.tight_layout()
plt.savefig('exports/rf_gini_importance.png', dpi=300)
plt.close()

# Compare and save permutation importances
df_perm = pd.read_csv('exports/rf_perm_importance.csv')
df_perm = df_perm.sort_values('ΔRMSE', ascending=True)
plt.figure(figsize=(7, 4))
plt.barh(df_perm['Descriptor'], df_perm['ΔRMSE'])
plt.xlabel('ΔRMSE')
plt.title(f'Random Forest Permutation Importance')
plt.tight_layout()
plt.savefig('exports/rf_perm_importance.png', dpi=300)
plt.close()