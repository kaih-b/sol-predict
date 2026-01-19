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

# Establish test seeds
seeds = list(range(25))

# Read in descriptors from each dataset and for each model
df_rf_esol  = pd.read_csv('wk4/final_descriptors.csv')
df_mlp_esol = pd.read_csv('wk4/expanded_descriptors.csv')
df_rf_aq    = pd.read_csv('wk6/01_aqsoldb_rf_descriptors.csv')
df_mlp_aq   = pd.read_csv('wk6/01_aqsoldb_mlp_descriptors.csv')

# Split up columns
target_col = 'logS'
descriptor_cols_rf  = [c for c in df_rf_esol.columns  if c not in [target_col, 'SMILES']]
descriptor_cols_mlp = [c for c in df_mlp_esol.columns if c not in [target_col, 'SMILES']]

# Split up input and target for each dataset
X_rf_esol  = df_rf_esol[descriptor_cols_rf]
y_rf_esol  = df_rf_esol[target_col]
X_mlp_esol = df_mlp_esol[descriptor_cols_mlp]
y_mlp_esol = df_mlp_esol[target_col]
X_rf_aq  = df_rf_aq[descriptor_cols_rf]
X_mlp_aq = df_mlp_aq[descriptor_cols_mlp]
y_rf_aq  = df_rf_aq[target_col]
y_mlp_aq = df_mlp_aq[target_col]

# Helper function to replicate splits across tests
def split_80_10_10(X, y, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    return X_train, y_train, X_val, y_val, X_test, y_test

# MLP helpers (carryover)
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

# RF model evaluation
def RF_eval(X_train, y_train, X_test, y_test, seed=42):
    with open('wk4/rf_best_params.json', 'r') as f:
        best_params = json.load(f)
    rf = RandomForestRegressor(random_state=seed, n_jobs=-1, **best_params)
    rf.fit(X_train, y_train)
    y_test_pred = rf.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    return test_rmse, y_test_pred

# MLP model evaluation
def MLP_eval(X_train_df, y_train_ser, X_test_df, y_test_ser, seed=42):
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

    X_tr = torch.from_numpy(scaler.transform(X_tr).astype(np.float32))
    X_val = torch.from_numpy(scaler.transform(X_val).astype(np.float32))
    X_te = torch.from_numpy(scaler.transform(X_test_df).astype(np.float32))

    y_tr = torch.from_numpy(y_tr.values.astype(np.float32).reshape(-1, 1))
    y_val = torch.from_numpy(y_val.values.astype(np.float32).reshape(-1, 1))
    y_te = torch.from_numpy(y_test_ser.values.astype(np.float32).reshape(-1, 1))

    train_loader = DataLoader(SolubilityDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SolubilityDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

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

# Test on (same) 25 random seed for apples-to-apples mean and std RMSE comparison between datasets
rows = []
for seed in seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # RF train and test set from ESOL data
    X_rf_train, y_rf_train, X_rf_val, y_rf_val, X_rf_test_esol, y_rf_test_esol = split_80_10_10(X_rf_esol, y_rf_esol, seed=seed)
    X_rf_train_esol = pd.concat([X_rf_train, X_rf_val], axis=0)
    y_rf_train_esol = pd.concat([y_rf_train, y_rf_val], axis=0)

    # RF train and test set from AqSolDB data
    X_rf_train_rev, y_rf_train_rev, X_rf_val_rev, y_rf_val_rev, X_rf_test_aq, y_rf_test_aq = split_80_10_10(X_rf_aq, y_rf_aq, seed=seed)
    X_rf_train_aq = pd.concat([X_rf_train_rev, X_rf_val_rev], axis=0)
    y_rf_train_aq = pd.concat([y_rf_train_rev, y_rf_val_rev], axis=0)

    # MLP train and test set from ESOL data
    X_mlp_train_esol, y_mlp_train_esol, _, _, X_mlp_test_esol, y_mlp_test_esol = split_80_10_10(X_mlp_esol, y_mlp_esol, seed=seed)

    # MLP train and test set from AqSolDB data
    X_mlp_train_aq, y_mlp_train_aq, _, _, X_mlp_test_aq, y_mlp_test_aq = split_80_10_10(X_mlp_aq, y_mlp_aq, seed=seed)

    # RFs
    # ESOL train -> ESOL test
    rmse_rf_e2e, pred_rf_e2e = RF_eval(
        X_train=X_rf_train_esol,
        y_train=y_rf_train_esol,
        X_test=X_rf_test_esol,
        y_test=y_rf_test_esol,
        seed=seed)
    # 2) ESOL train -> AqSolDB test
    rmse_rf_e2a, pred_rf_e2a = RF_eval(
        X_train=X_rf_train_esol,
        y_train=y_rf_train_esol,
        X_test=X_rf_test_aq,
        y_test=y_rf_test_aq,
        seed=seed)
    # 3) AqSolDB train -> AqSolDB test
    rmse_rf_a2a, pred_rf_a2a = RF_eval(
        X_train=X_rf_train_aq,
        y_train=y_rf_train_aq,
        X_test=X_rf_test_aq,
        y_test=y_rf_test_aq,
        seed=seed)
    # 4) AqSolDB train -> ESOL test
    rmse_rf_a2e, pred_rf_a2e = RF_eval(
        X_train=X_rf_train_aq,
        y_train=y_rf_train_aq,
        X_test=X_rf_test_esol,
        y_test=y_rf_test_esol,
        seed=seed)
    # MLPs
    # ESOL train -> ESOL test
    pred_mlp_e2e, rmse_mlp_e2e, tr_loss_e2e, val_loss_e2e = MLP_eval(
        X_train_df=X_mlp_train_esol,
        y_train_ser=y_mlp_train_esol,
        X_test_df=X_mlp_test_esol,
        y_test_ser=y_mlp_test_esol,
        seed=seed)
    # 2) ESOL train -> AqSolDB test
    pred_mlp_e2a, rmse_mlp_e2a, tr_loss_e2a, val_loss_e2a = MLP_eval(
        X_train_df=X_mlp_train_esol,
        y_train_ser=y_mlp_train_esol,
        X_test_df=X_mlp_test_aq,
        y_test_ser=y_mlp_test_aq,
        seed=seed)
    # 3) AqSolDB train -> AqSolDB test
    pred_mlp_a2a, rmse_mlp_a2a, tr_loss_a2a, val_loss_a2a = MLP_eval(
        X_train_df=X_mlp_train_aq,
        y_train_ser=y_mlp_train_aq,
        X_test_df=X_mlp_test_aq,
        y_test_ser=y_mlp_test_aq,
        seed=seed)
    # 4) AqSolDB train -> ESOL test
    pred_mlp_a2e, rmse_mlp_a2e, tr_loss_a2e, val_loss_a2e = MLP_eval(
        X_train_df=X_mlp_train_aq,
        y_train_ser=y_mlp_train_aq,
        X_test_df=X_mlp_test_esol,
        y_test_ser=y_mlp_test_esol,
        seed=seed)
    
    # Append each result to a table for aggregation, df conversion, and export
    rows.extend([
        {'model': 'RF', 'train': 'ESOL', 'test': 'ESOL', 'seed': seed, 'rmse': float(rmse_rf_e2e)},
        {'model': 'RF', 'train': 'ESOL', 'test': 'AqSolDB', 'seed': seed, 'rmse': float(rmse_rf_e2a)},
        {'model': 'RF', 'train': 'AqSolDB', 'test': 'AqSolDB', 'seed': seed, 'rmse': float(rmse_rf_a2a)},
        {'model': 'RF', 'train': 'AqSolDB', 'test': 'ESOL', 'seed': seed, 'rmse': float(rmse_rf_a2e)},
        {'model': 'MLP', 'train': 'ESOL', 'test': 'ESOL', 'seed': seed, 'rmse': float(rmse_mlp_e2e)},
        {'model': 'MLP', 'train': 'ESOL', 'test': 'AqSolDB', 'seed': seed, 'rmse': float(rmse_mlp_e2a)},
        {'model': 'MLP', 'train': 'AqSolDB','test': 'AqSolDB', 'seed': seed, 'rmse': float(rmse_mlp_a2a)},
        {'model': 'MLP', 'train': 'AqSolDB','test': 'ESOL', 'seed': seed, 'rmse': float(rmse_mlp_a2e)},])

# Organize and aggregate results into df
results = pd.DataFrame(rows)
summary = (results.groupby(['model', 'train', 'test'], as_index=False).agg(mean_rmse=('rmse', 'mean'), std_rmse=('rmse', 'std'), n=('rmse', 'count')))

# Save summary and individual seed results to csv
results.to_csv("wk6/02_db_test_per_seed.csv", index=False)
summary.to_csv("wk6/02_db_test_summary.csv", index=False)