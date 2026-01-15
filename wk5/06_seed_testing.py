import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Define seeds to test and best models 
seeds = [0, 1, 2, 3, 4]
best = {'n_features': 7, 'hidden_sizes': (64, 32, 16, 8), 'dropout_p': 0.0, 'learning_rate': 2e-3, 'weight_decay': 1e-3}
best_ext = {'n_features': 11, 'hidden_sizes': (256, 128), 'dropout_p': 0.1, 'learning_rate': 2e-3, 'weight_decay': 1e-5}

# Read in info
df = pd.read_csv('wk4/final_descriptors.csv')
target_col = 'logS'
descriptor_cols = [c for c in df.columns if c not in [target_col, 'SMILES']]
X = df[descriptor_cols]
y = df[target_col]
df_ext = pd.read_csv('wk4/expanded_descriptors.csv')
descriptor_cols_ext = [c for c in df_ext.columns if c not in [target_col, 'SMILES']]
X_ext = df_ext[descriptor_cols_ext]
y_ext = df_ext[target_col]

# Define MLP classes and RMSE function
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
    return train_losses, val_losses, best_state, best_val_loss

def evaluate_rmse(model, data_loader, loss_func):
    model.eval()
    tot_loss, n = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            preds = model(X_batch)
            loss = loss_func(preds, y_batch)
            bs = X_batch.size(0)
            tot_loss += loss.item() * bs
            n += bs
    mse = tot_loss / n
    return float(np.sqrt(mse))

results = {} # key: model (RF or MLP)

# RF Loop
for seed in seeds:
    # Split train/test/vals for each seed
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, random_state = seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = seed)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Create and fit RF model
    with open('wk4/rf_best_params.json', 'r') as f:
        best_params = json.load(f)
    rf_tuned = RandomForestRegressor(random_state = seed, n_jobs = -1, **best_params)
    rf_tuned.fit(X_train, y_train)
    rf_tuned.fit(X_train_full, y_train_full)

    # Get test metrics
    y_test_pred_rf = rf_tuned.predict(X_test)
    test_rmse_rf = root_mean_squared_error(y_test, y_test_pred_rf)
    results['RF'] = {'seed': seed, 'test_rmse': test_rmse_rf}

# MLP Loop (base descriptors)
for seed in seeds:
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Split train/test/vals for each seed
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, random_state = seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = seed)
    y_train_np = y_train.values.astype(np.float32).reshape(-1, 1)
    y_val_np = y_val.values.astype(np.float32).reshape(-1, 1)
    y_test_np = y_test.values.astype(np.float32).reshape(-1, 1)
    y_train_t = torch.from_numpy(y_train_np)
    y_val_t = torch.from_numpy(y_val_np)
    y_test_t = torch.from_numpy(y_test_np)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    train_ds = SolubilityDataset(X_train_s, y_train_t)
    val_ds = SolubilityDataset(X_val_s, y_val_t)
    test_ds = SolubilityDataset(X_test_s, y_test_t)
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    n_features = X_train_s.shape[1]
    num_epochs = 100
    loss_func = nn.MSELoss()

    model = MLPRegressor(n_features=best['n_features'], hidden_sizes=best['hidden_sizes'], dropout_p=best['dropout_p'])
    optimizer = torch.optim.Adam(model.parameters(), lr=best['learning_rate'], weight_decay=best['weight_decay'])

    train_losses, val_losses = train_model(model=model, train_loader=train_loader, val_loader=val_loader, loss_func=loss_func, optimizer=optimizer, num_epochs=num_epochs)
    mlp_rmse = evaluate_rmse(model, test_loader, loss_func)

    results['MLP_base'] = {'seed': seed, 'test_rmse': test_rmse_rf}

print(results)