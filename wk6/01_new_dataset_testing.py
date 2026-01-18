import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit import RDLogger

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
df["mol"] = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)
df = df[df["mol"].notna()].copy()

# Remove inorganic molecules
def is_organic(mol):
    if mol is None:
        return False
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
df = df[df["mol"].apply(is_organic)].copy()

# Compute descriptiors for both models
def aromatic_proportion(mol):
    if mol is None:
        return float("nan")
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return 0.0
    aro = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    return aro / heavy
def compute_desc_rf(mol):
    return pd.Series({
        "HBA": Lipinski.NumHAcceptors(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "LogP": Descriptors.MolLogP(mol),
        "MolWt": Descriptors.MolWt(mol),
        "AroProp": aromatic_proportion(mol)})
def compute_desc_mlp(mol):
    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "AroProp": aromatic_proportion(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "BertzCT": Descriptors.BertzCT(mol)})

# Finalize dataframes for each model
desc_rf = df["mol"].apply(compute_desc_rf)
df_rf = pd.concat([df.drop(columns=["mol"]), desc_rf], axis=1)
desc_mlp = df["mol"].apply(compute_desc_mlp)
df_mlp = pd.concat([df.drop(columns=["mol"]), desc_mlp], axis=1)

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
        yhat = model(X_t).numpy().reshape(-1)
    return yhat
y_test_pred_mlp = predict(mlp, X_test_mlp_s_t)
test_rmse_mlp = root_mean_squared_error(y_test_mlp.to_numpy(dtype=float), y_test_pred_mlp)

print(test_rmse_rf)
print(test_rmse_mlp)