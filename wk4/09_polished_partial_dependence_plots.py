import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# Recreate train-test split
df = pd.read_csv('wk4/final_descriptors.csv')
target_col = 'logS'
descriptor_cols = [c for c in df.columns if c not in [target_col, 'SMILES']]
X = df[descriptor_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Load optimized random forest model
with open('wk4/rf_best_params.json', 'r') as f:
    best_params = json.load(f)
rf_tuned = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
rf_tuned.fit(X_train, y_train)

# Identify descriptors for partial dependence analysis (as before)
pdp_descriptors = ['LogP', 'BertzCT', 'MolWt']

# Plot and save partial dependences for each descriptor
for desc in pdp_descriptors:
    plt.figure()
    # Create partial dependence plot
    PartialDependenceDisplay.from_estimator(rf_tuned, X_train, features=[desc])
    # Plot partial dependence with Matplotlib
    plt.title(f'Partial Dependence of logS on {desc}')
    plt.tight_layout()
    out_path = f'exports/pdp_{desc}.png'
    plt.savefig(out_path, dpi=300)
    plt.close()