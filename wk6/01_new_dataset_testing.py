import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit import RDLogger
import matplotlib.pyplot as plt

# Disable warnings for poorly formatted SMILES (they will be removed)
RDLogger.DisableLog('rdApp.warning')

# Import new dataset SMILES and logS
df = pd.read_csv('wk6/01_aqsol_db_raw.csv')
df.columns = df.columns.str.strip()
df = df[['SMILES', 'Solubility']].rename(columns={'Solubility': 'logS'})

# Get mol objects from SMILES; parse issues
def mol_from_smiles(s):
    if not isinstance(s, str) or not s.strip():
        return None
    return Chem.MolFromSmiles(s)
df['mol'] = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)
df = df[df['mol'].notna()].copy()

# Remove inorganic molecules
def is_organic(mol):
    if mol is None:
        return False
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
df = df[df['mol'].apply(is_organic)].copy()

# Compute descriptiors for both models
def aromatic_proportion(mol):
    if mol is None:
        return float('nan')
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return 0.0
    aro = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    return aro / heavy
def compute_desc_rf(mol):
    return pd.Series({
        'HBA': Lipinski.NumHAcceptors(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'RotB': Lipinski.NumRotatableBonds(mol),
        'LogP': Descriptors.MolLogP(mol),
        'MolWt': Descriptors.MolWt(mol),
        'AroProp': aromatic_proportion(mol)})
def compute_desc_mlp(mol):
    return pd.Series({
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'RotB': Lipinski.NumRotatableBonds(mol),
        'AroProp': aromatic_proportion(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'RingCount': rdMolDescriptors.CalcNumRings(mol),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
        'HeavyAtomCount': mol.GetNumHeavyAtoms(),
        'BertzCT': Descriptors.BertzCT(mol)})

# Finalize dataframes for each model
desc_rf = df['mol'].apply(compute_desc_rf)
df_rf = pd.concat([df.drop(columns=['mol']), desc_rf], axis=1)
desc_mlp = df['mol'].apply(compute_desc_mlp)
df_mlp = pd.concat([df.drop(columns=['mol']), desc_mlp], axis=1)

# Prep models
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

best_params_mlp = {'n_features': 11, 'hidden_sizes': (256, 128), 'dropout_p': 0.1, 'learning_rate': 2e-3, 'weight_decay': 1e-5}

target_col = 'logS'
descriptor_cols_rf = [c for c in df_rf.columns if c not in [target_col, 'SMILES']]
descriptor_cols_mlp = [c for c in df_mlp.columns if c not in [target_col, 'SMILES']]
X_rf = df_rf[descriptor_cols_rf]
y_rf = df_rf[target_col]
X_mlp = df_mlp[descriptor_cols_mlp]
y_mlp = df_mlp[target_col]

# Train RF
X_train_rf, X_temp_rf, y_train_rf, y_temp_rf = train_test_split(X_rf, y_rf, test_size = 0.2, random_state = seed)
X_val_rf, X_test_rf, y_val_rf, y_test_rf = train_test_split(X_temp_rf, y_temp_rf, test_size = 0.5, random_state = seed)
X_train_full = np.concatenate([X_train_rf, X_val_rf])
y_train_full = np.concatenate([y_train_rf, y_val_rf])
with open('wk4/rf_best_params.json', 'r') as f:
    best_params_rf = json.load(f)
rf = RandomForestRegressor(random_state = seed, n_jobs = -1, **best_params_rf)
rf.fit(X_train_full, y_train_full)
y_test_pred_rf = rf.predict(X_test_rf)
test_rmse_rf = root_mean_squared_error(y_test_rf, y_test_pred_rf)

# Helpers for MLP
class SolubilityDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor
        self.y = y_tensor
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPRegressor(nn.Module):
    def __init__(self, n_features, hidden_sizes=(64, 32), dropout_p=0.2):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        tot_train_loss = 0.0
        n_train = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_func(preds, y_batch)
            loss.backward()
            optimizer.step()
            bs = X_batch.size(0)
            tot_train_loss += loss.item() * bs
            n_train += bs
        avg_train_loss = tot_train_loss / n_train

        model.eval()
        tot_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = loss_func(preds, y_batch)
                bs = X_batch.size(0)
                tot_val_loss += loss.item() * bs
                n_val += bs
        avg_val_loss = tot_val_loss / n_val

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            sd = model.state_dict()
            best_state = {k: v.detach().clone() for k, v in sd.items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return train_losses, val_losses

X_train_mlp, X_temp_mlp, y_train_mlp, y_temp_mlp = train_test_split(X_mlp, y_mlp, test_size=0.2, random_state=seed)
X_val_mlp, X_test_mlp, y_val_mlp, y_test_mlp = train_test_split(X_temp_mlp, y_temp_mlp, test_size=0.5, random_state=seed)
scaler_mlp = StandardScaler()
scaler_mlp.fit(X_train_mlp)
X_train_mlp_s = scaler_mlp.transform(X_train_mlp).astype(np.float32)
X_val_mlp_s = scaler_mlp.transform(X_val_mlp).astype(np.float32)
X_test_mlp_s = scaler_mlp.transform(X_test_mlp).astype(np.float32)
X_train_mlp_s_t = torch.from_numpy(X_train_mlp_s)
X_val_mlp_s_t = torch.from_numpy(X_val_mlp_s)
X_test_mlp_s_t = torch.from_numpy(X_test_mlp_s)
y_train_mlp_t = torch.from_numpy(y_train_mlp.values.astype(np.float32).reshape(-1, 1))
y_val_mlp_t = torch.from_numpy(y_val_mlp.values.astype(np.float32).reshape(-1, 1))
y_test_mlp_t = torch.from_numpy(y_test_mlp.values.astype(np.float32).reshape(-1, 1))
train_ds_mlp = SolubilityDataset(X_train_mlp_s_t, y_train_mlp_t)
val_ds_mlp = SolubilityDataset(X_val_mlp_s_t, y_val_mlp_t)
test_ds_mlp = SolubilityDataset(X_test_mlp_s_t, y_test_mlp_t)
batch_size = 64
train_loader_mlp = DataLoader(train_ds_mlp, batch_size=batch_size, shuffle=True)
val_loader_mlp = DataLoader(val_ds_mlp, batch_size=batch_size, shuffle=False)
test_loader_mlp = DataLoader(test_ds_mlp, batch_size=batch_size, shuffle=False)

# Train MLP
num_epochs = 100
loss_func = nn.MSELoss()
mlp = MLPRegressor(n_features=best_params_mlp['n_features'], hidden_sizes=best_params_mlp['hidden_sizes'], dropout_p=best_params_mlp['dropout_p'])
optimizer = torch.optim.Adam(mlp.parameters(), lr=best_params_mlp['learning_rate'], weight_decay=best_params_mlp['weight_decay'])
train_losses, val_losses = train_model(mlp, train_loader_mlp, val_loader_mlp, loss_func, optimizer, num_epochs)

def predict(model, X_t):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_t).numpy().reshape(-1)
    return y_pred
y_test_pred_mlp = predict(mlp, X_test_mlp_s_t)
test_rmse_mlp = root_mean_squared_error(y_test_mlp.to_numpy(dtype=float), y_test_pred_mlp)

# See how well the model generalizes on a much larger dataset
# print(test_rmse_rf) # 1.208
# print(test_rmse_mlp) # 1.168
# Both MUCH (~2x) the magnitude of RMSE for the Delaney dataset. Code from here on seeks to explore that discrepancy

# Comparison with Delaney
df_old = pd.read_csv('wk4/final_descriptors.csv')
df_old.columns = df_old.columns.str.strip()
y_old = df_old['logS'].to_numpy(dtype=float)
y_new = df_mlp['logS'].to_numpy(dtype=float)

# Visualize the datasets compared with each other
plt.figure()
plt.hist(y_old, bins=40, alpha=0.5, label='Dataset 1 (Delaney ESOL)')
plt.hist(y_new, bins=40, alpha=0.5, label='Dataset 2 (AqSolDB)')
plt.xlabel('logS')
plt.ylabel('Count')
plt.title('logS Distribution by Dataset')
plt.legend()
plt.tight_layout()
plt.savefig('wk6/dataset_figures/logs_dist_by_dataset.png', dpi=300)
plt.close()

# Plot residuals for each descriptor in each dataset to identify if any of them are particularly problematic in the new dataset
def plot_residuals_vs(x, residuals, descriptor, model_name):
    plt.figure()
    plt.scatter(x, residuals, s=12, alpha=0.5)
    plt.axhline(0.0)
    plt.xlabel(descriptor)
    plt.ylabel('Residual')
    plt.title(f'Residuals vs {descriptor} ({model_name})')
    plt.tight_layout()
    plt.savefig(f'wk6/dataset_figures/residuals_vs_{descriptor}_{model_name}.png', dpi=300)
    plt.close()
resid_rf = (y_test_rf.to_numpy(dtype=float) - y_test_pred_rf)
resid_mlp = (y_test_mlp.to_numpy(dtype=float) - y_test_pred_mlp)
for desc in descriptor_cols_mlp:
    if desc in X_test_rf.columns:
        plot_residuals_vs(X_test_rf[desc].to_numpy(dtype=float), resid_rf, desc, 'RF')
    plot_residuals_vs(X_test_mlp[desc].to_numpy(dtype=float), resid_mlp, desc, 'MLP')

# Metrics: bins to see how performance changes with different descriptors (evaluating ranges outside the ESOL scope)
def rmse(y_true, y_pred):
    return root_mean_squared_error(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
def r2(y_true, y_pred):
    return r2_score(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))

# Gives RMSE with bins
def rmse_by_bins(y_true, y_pred, descriptor, bins, labels=None, title=''):
    # Analysis table; each row is one sample
    dfb = pd.DataFrame({
        'y_true': np.asarray(y_true, dtype=float),
        'y_pred': np.asarray(y_pred, dtype=float),
        'x': np.asarray(descriptor, dtype=float),
    }).dropna() # clean NaNs

    # Drop all rows into bins
    dfb['bin'] = pd.cut(dfb['x'], bins=bins, labels=labels, include_lowest=True)
    out = []
    # Compute metrics within each bin
    for b, g in dfb.groupby('bin', observed=True):
        if len(g) < 20: # skip small sample-size bins (unstable)
            continue
        out.append({
            'bin': str(b),
            'n': len(g),
            'rmse': rmse(g['y_true'], g['y_pred']),
            'r2': r2(g['y_true'], g['y_pred']),
            'y_true_mean': float(g['y_true'].mean()),
            'y_true_std': float(g['y_true'].std())})
    # Save to df for later use
    out_df = pd.DataFrame(out).sort_values('bin')
    print(f'\nRMSE by bins: {title}')
    print(out_df.to_string(index=False))
    return out_df

# Heavy Atom Count
heavy_bins = [0, 20, 40, 60, 80, 200]
rmse_by_bins(
    y_test_mlp.to_numpy(dtype=float),
    y_test_pred_mlp,
    X_test_mlp['HeavyAtomCount'].to_numpy(dtype=float),
    bins=heavy_bins,
    title='MLP vs HeavyAtomCount')

# BertzCT
bertz_bins = [0, 500, 1000, 1500, 2500, 6000]
rmse_by_bins(
    y_test_mlp.to_numpy(dtype=float),
    y_test_pred_mlp,
    X_test_mlp['BertzCT'].to_numpy(dtype=float),
    bins=bertz_bins,
    title='MLP vs BertzCT')
rmse_by_bins(
    y_test_rf.to_numpy(dtype=float),
    y_test_pred_rf, 
    X_test_rf['BertzCT'].to_numpy(dtype=float),
    bins=bertz_bins,
    title='RF vs BertzCT')

# By logS (hard cases)
logS_bins = [-20, -10, -8, -6, -4, -2, 0, 5]
rmse_by_bins(
    y_test_mlp.to_numpy(dtype=float),
    y_test_pred_mlp,
    y_test_mlp.to_numpy(dtype=float),
    bins=logS_bins,
    title='MLP by logS bin')
rmse_by_bins(
    y_test_rf.to_numpy(dtype=float),
    y_test_pred_rf,
    y_test_rf.to_numpy(dtype=float),
    bins=logS_bins,
    title='RF by logS bin')