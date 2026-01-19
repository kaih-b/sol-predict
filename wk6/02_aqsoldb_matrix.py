import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

df_rf_esol  = pd.read_csv('wk4/final_descriptors.csv')
df_mlp_esol = pd.read_csv('wk4/expanded_descriptors.csv')
df_rf_aq    = pd.read_csv('wk6/01_aqsoldb_rf_descriptors.csv')
df_mlp_aq   = pd.read_csv('wk6/01_aqsoldb_mlp_descriptors.csv')

target_col = 'logS'
descriptor_cols_rf  = [c for c in df_rf_esol.columns  if c not in [target_col, 'SMILES']]
descriptor_cols_mlp = [c for c in df_mlp_esol.columns if c not in [target_col, 'SMILES']]

X_rf_esol  = df_rf_esol[descriptor_cols_rf]
y_rf_esol  = df_rf_esol[target_col]
X_mlp_esol = df_mlp_esol[descriptor_cols_mlp]
y_mlp_esol = df_mlp_esol[target_col]

X_rf_aq  = df_rf_aq[descriptor_cols_rf]
X_mlp_aq = df_mlp_aq[descriptor_cols_mlp]
y_rf_aq  = df_rf_aq[target_col]
y_mlp_aq = df_mlp_aq[target_col]

def split_80_10_10(X, y, seed=42):
    # 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    return X_train, y_train, X_val, y_val, X_test, y_test

# ESOL -> AqSolDB (preserve 80/10/10 on ESOL; external test = full AqSolDB)
# RF (ESOL internal 80/10/10; keep internal test too, in case you want it)
X_rf_train, y_rf_train, X_rf_val, y_rf_val, X_rf_test_in, y_rf_test_in = split_80_10_10(X_rf_esol, y_rf_esol, seed=seed)

# For RF, train on train+val
X_rf_train_full = pd.concat([X_rf_train, X_rf_val], axis=0)
y_rf_train_full = pd.concat([y_rf_train, y_rf_val], axis=0)

# External (cross-dataset) test
X_rf_test = X_rf_aq
y_rf_test = y_rf_aq

# MLP (ESOL internal 80/10/10; keep val separate for training)
X_mlp_train, y_mlp_train, X_mlp_val, y_mlp_val, X_mlp_test_in, y_mlp_test_in = split_80_10_10(X_mlp_esol, y_mlp_esol, seed=seed)

# External (cross-dataset) test
X_mlp_test = X_mlp_aq
y_mlp_test = y_mlp_aq

# AqSolDB -> ESOL (preserve 80/10/10 on AqSolDB; external test = full ESOL)
# RF (AqSolDB internal 80/10/10; keep internal test too)
X_rf_train_rev, y_rf_train_rev, X_rf_val_rev, y_rf_val_rev, X_rf_test_in_rev, y_rf_test_in_rev = split_80_10_10(X_rf_aq, y_rf_aq, seed=seed)

# For RF, train on train+val
X_rf_train_full_rev = pd.concat([X_rf_train_rev, X_rf_val_rev], axis=0)
y_rf_train_full_rev = pd.concat([y_rf_train_rev, y_rf_val_rev], axis=0)

# External (cross-dataset) test
X_rf_test_rev = X_rf_esol
y_rf_test_rev = y_rf_esol

# MLP (AqSolDB internal 80/10/10; keep val separate)
X_mlp_train_rev, y_mlp_train_rev, X_mlp_val_rev, y_mlp_val_rev, X_mlp_test_in_rev, y_mlp_test_in_rev = split_80_10_10(X_mlp_aq, y_mlp_aq, seed=seed)

# External (cross-dataset) test
X_mlp_test_rev = X_mlp_esol
y_mlp_test_rev = y_mlp_esol

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

# RF model cross-evaluation
def train_rf_cross(X_train, y_train, X_test, y_test):
    with open('wk4/rf_best_params.json', 'r') as f:
        best_params = json.load(f)
    rf = RandomForestRegressor(random_state=seed, n_jobs=-1, **best_params)
    rf.fit(X_train, y_train)
    y_test_pred = rf.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    return test_rmse, y_test_pred

# MLP model cross-evaluation
def train_mlp_cross(X_train_df, y_train_ser, X_test_df, y_test_ser):
    batch_size = 64
    num_epochs = 100
    loss_func = nn.MSELoss()
    best_params = {
        'n_features': X_train_df.shape[1],
        'hidden_sizes': (256, 128),
        'dropout_p': 0.1,
        'learning_rate': 2e-3,
        'weight_decay': 1e-5}

    X_tr, X_val, y_tr, y_val = train_test_split(X_train_df, y_train_ser, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    scaler.fit(X_tr)

    X_tr  = torch.from_numpy(scaler.transform(X_tr).astype(np.float32))
    X_val = torch.from_numpy(scaler.transform(X_val).astype(np.float32))
    X_te  = torch.from_numpy(scaler.transform(X_test_df).astype(np.float32))

    y_tr  = torch.from_numpy(y_tr.values.astype(np.float32).reshape(-1, 1))
    y_val = torch.from_numpy(y_val.values.astype(np.float32).reshape(-1, 1))
    y_te  = torch.from_numpy(y_test_ser.values.astype(np.float32).reshape(-1, 1))

    train_loader = DataLoader(SolubilityDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SolubilityDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = MLPRegressor(
        n_features=best_params['n_features'],
        hidden_sizes=best_params['hidden_sizes'],
        dropout_p=best_params['dropout_p'])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'])

    train_losses, val_losses = train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs)

    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_te).numpy().reshape(-1)

    test_rmse = root_mean_squared_error(y_te.numpy().reshape(-1), y_test_pred)
    return y_test_pred, test_rmse, train_losses, val_losses

# RANDOM FOREST
# 1) ESOL -> ESOL (internal test split)
rmse_rf_e2e, pred_rf_e2e = train_rf_cross(
    X_train=X_rf_train_full,    # ESOL train+val (matches your original behavior)
    y_train=y_rf_train_full,
    X_test=X_rf_test_in,        # ESOL internal 10% test
    y_test=y_rf_test_in
)
print("RF ESOL -> ESOL (internal) RMSE:", rmse_rf_e2e)

# 2) ESOL -> AqSolDB (cross)
rmse_rf_e2a, pred_rf_e2a = train_rf_cross(
    X_train=X_rf_train_full,
    y_train=y_rf_train_full,
    X_test=X_rf_test,           # full AqSolDB
    y_test=y_rf_test
)
print("RF ESOL -> AqSolDB (cross) RMSE:", rmse_rf_e2a)

# 3) AqSolDB -> AqSolDB (internal test split)
rmse_rf_a2a, pred_rf_a2a = train_rf_cross(
    X_train=X_rf_train_full_rev,   # AqSol train+val
    y_train=y_rf_train_full_rev,
    X_test=X_rf_test_in_rev,       # AqSol internal 10% test
    y_test=y_rf_test_in_rev
)
print("RF AqSolDB -> AqSolDB (internal) RMSE:", rmse_rf_a2a)

# 4) AqSolDB -> ESOL (cross)
rmse_rf_a2e, pred_rf_a2e = train_rf_cross(
    X_train=X_rf_train_full_rev,
    y_train=y_rf_train_full_rev,
    X_test=X_rf_test_rev,          # full ESOL
    y_test=y_rf_test_rev
)
print("RF AqSolDB -> ESOL (cross) RMSE:", rmse_rf_a2e)

# MLP
pred_mlp_e2e, rmse_mlp_e2e, tr_loss_e2e, val_loss_e2e = train_mlp_cross(
    X_train_df=X_mlp_esol,
    y_train_ser=y_mlp_esol,
    X_test_df=X_mlp_test_in,
    y_test_ser=y_mlp_test_in
)
print("MLP ESOL -> ESOL (internal) RMSE:", rmse_mlp_e2e)

# 2) ESOL -> AqSolDB (cross)
pred_mlp_e2a, rmse_mlp_e2a, tr_loss_e2a, val_loss_e2a = train_mlp_cross(
    X_train_df=X_mlp_esol,
    y_train_ser=y_mlp_esol,
    X_test_df=X_mlp_aq,
    y_test_ser=y_mlp_aq
)
print("MLP ESOL -> AqSolDB (cross) RMSE:", rmse_mlp_e2a)

# 3) AqSolDB -> AqSolDB (internal test split)
pred_mlp_a2a, rmse_mlp_a2a, tr_loss_a2a, val_loss_a2a = train_mlp_cross(
    X_train_df=X_mlp_aq,
    y_train_ser=y_mlp_aq,
    X_test_df=X_mlp_test_in_rev,
    y_test_ser=y_mlp_test_in_rev
)
print("MLP AqSolDB -> AqSolDB (internal) RMSE:", rmse_mlp_a2a)

# 4) AqSolDB -> ESOL (cross)
pred_mlp_a2e, rmse_mlp_a2e, tr_loss_a2e, val_loss_a2e = train_mlp_cross(
    X_train_df=X_mlp_aq,
    y_train_ser=y_mlp_aq,
    X_test_df=X_mlp_esol,
    y_test_ser=y_mlp_esol
)
print("MLP AqSolDB -> ESOL (cross) RMSE:", rmse_mlp_a2e)