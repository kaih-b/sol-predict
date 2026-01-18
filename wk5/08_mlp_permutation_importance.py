import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

# Revert to baseline seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Select model for interpretation
model_name = 'MLP_ext'
best_params = {'n_features': 11, 'hidden_sizes': (256, 128), 'dropout_p': 0.1, 'learning_rate': 2e-3, 'weight_decay': 1e-5}
df = pd.read_csv('wk4/expanded_descriptors.csv')
target_col = 'logS'
descriptor_cols = [c for c in df.columns if c not in [target_col, 'SMILES']]
X = df[descriptor_cols]
y = df[target_col]

# Define helper classes and model trainers (carryover)
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

# Split and scale (carryover)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train).astype(np.float32)
X_val_s = scaler.transform(X_val).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)
X_train_s_t = torch.from_numpy(X_train_s)
X_val_s_t = torch.from_numpy(X_val_s)
X_test_s_t = torch.from_numpy(X_test_s)
y_train_t = torch.from_numpy(y_train.values.astype(np.float32).reshape(-1, 1))
y_val_t = torch.from_numpy(y_val.values.astype(np.float32).reshape(-1, 1))
y_test_t = torch.from_numpy(y_test.values.astype(np.float32).reshape(-1, 1))
train_ds = SolubilityDataset(X_train_s_t, y_train_t)
val_ds = SolubilityDataset(X_val_s_t, y_val_t)
test_ds = SolubilityDataset(X_test_s_t, y_test_t)
batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Train model
num_epochs = 100
loss_func = nn.MSELoss()
model = MLPRegressor(n_features=best_params['n_features'], hidden_sizes=best_params['hidden_sizes'], dropout_p=best_params['dropout_p'])
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
train_losses, val_losses = train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs)

# Define model predictor helper func
def predict(model, X_t):
    model.eval()
    with torch.no_grad():
        yhat = model(X_t).numpy().reshape(-1)
    return yhat

# Baseline RMSE metrics (base perm)
yhat_test = predict(model, X_test_s_t)
baseline_rmse = root_mean_squared_error(y_test.to_numpy(dtype=float), yhat_test)

# Match repeats from RF testing
n_repeats = 20
rng = np.random.default_rng(seed)

rows = []
for j, feat in enumerate(descriptor_cols):
    deltas = []
    # Repeat permutations 20x (same as RF)
    for r in range(n_repeats):
        Xp = X_test_s_t.detach().clone()
        # Create a random perm (shuffle samples)
        perm_idx = rng.permutation(Xp.shape[0])
        # Identifies the current descriptor's impact on y-hat (predicted y)
        Xp[:, j] = Xp[perm_idx, j]
        # Make the prediction and determine its RMSE (to compare to full model)
        yhat_p = predict(model, Xp)
        rmse_p = root_mean_squared_error(y_test.to_numpy(dtype=float), yhat_p)
        deltas.append(rmse_p - baseline_rmse)
    rows.append({
        'descriptor': feat,
        'delta_rmse_mean': float(np.mean(deltas)),
        'delta_rmse_std': float(np.std(deltas, ddof=1))})

# Save permutation importance to csv
perm_df = pd.DataFrame(rows).sort_values('delta_rmse_mean', ascending=False).reset_index(drop=True)
perm_df['baseline_rmse'] = float(baseline_rmse)
perm_df.to_csv('wk5/08_mlp_perm_importance.csv', index=False)