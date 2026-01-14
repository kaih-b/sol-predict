import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Tweak model trainer from previous day
##########

df = pd.read_csv('wk4/final_descriptors.csv')
target_col = 'logS'
descriptor_cols = [c for c in df.columns if c not in [target_col, 'SMILES']]
X = df[descriptor_cols]
y = df[target_col]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, random_state = seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
X_train_np = X_train.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32).reshape(-1, 1)
y_val_np = y_val.values.astype(np.float32).reshape(-1, 1)
y_test_np = y_test.values.astype(np.float32).reshape(-1, 1)
X_train_t = torch.from_numpy(X_train_np)
X_val_t = torch.from_numpy(X_val_np)
X_test_t = torch.from_numpy(X_test_np)
y_train_t = torch.from_numpy(y_train_np)
y_val_t = torch.from_numpy(y_val_np)
y_test_t = torch.from_numpy(y_test_np)

class SolubilityDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor
        self.y = y_tensor
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SolubilityDataset(X_train_t, y_train_t)
val_ds = SolubilityDataset(X_val_t, y_val_t)
test_ds = SolubilityDataset(X_test_t, y_test_t)

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

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
    # instantiate performance trackers
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None

    # model trainer
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
            best_state = {k: v.detach().clone() for k, v in sd.items()} # new: stores an independent best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, best_state, best_val_loss # add outputs for best values

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

##########

# Collect constant hyperparameters
n_features = X_train_t.shape[1]
weight_decay = 1e-4
num_epochs = 100
loss_func = nn.MSELoss()

# Instantiate configurations for the model and optimizer
model_configs = [
    ('(64-32)_0.2', (64, 32), 0.2),
    ('(128-64)_0.2', (128, 64), 0.2),
    ('(64-32-16)_0.2', (64, 32, 16), 0.2),
    ('(64-32)_0.4', (64, 32), 0.4),
    ('(128-64)_0.4', (128, 64), 0.4),
    ('(64-32-16)_0.4', (64, 32, 16), 0.4)]
opt_configs = {
    'adam_lr1e-3': {'lr': 1e-3, 'weight_decay': weight_decay},
    'adam_lr5e-4': {'lr': 5e-4, 'weight_decay': weight_decay}}

results = {}  # key: (model_name, opt_name)

# Train a model for each combination
for model_name, hidden, drop in model_configs:
    for opt_name, opt_params in opt_configs.items():
        model = MLPRegressor(n_features=n_features, hidden_sizes=hidden, dropout_p=drop)
        optimizer = torch.optim.Adam(model.parameters(), **opt_params)
        train_curve, val_curve, best_state, best_val = train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs)
        # Store results in a dictionary with all params
        results[(model_name, opt_name)] = {
            'hidden': hidden,
            'dropout': drop,
            'opt': opt_params,
            'train_curve': train_curve,
            'val_curve': val_curve,
            'best_val': best_val,
            'best_state': best_state}

# Pick best by validation loss
best_by_val = min(results, key=lambda k: results[k]['best_val'])
best_val = results[best_by_val]['best_val']
print(f'Best by val: {best_by_val}\nVal: {best_val:.3f}')

# Evaluate RMSE correctly by loading best state into a fresh model
best_rmse = float('inf')
best_by_rmse = None
for key, rec in results.items():
    model_name, opt_name = key
    model = MLPRegressor(n_features=n_features, hidden_sizes=rec['hidden'], dropout_p=rec['dropout'])
    model.load_state_dict(rec['best_state'])
    rmse = evaluate_rmse(model, test_loader, loss_func)
    rec['test_rmse'] = rmse
    if rmse < best_rmse:
        best_rmse = rmse
        best_by_rmse = key
print(f'Best by RMSE: {best_by_rmse}\nRMSE: {best_rmse:.3f}')

# Save full results table
rows = []
for (mname, oname), rec in results.items():
    rows.append({
        'model': mname,
        'optimizer': oname,
        'hidden': str(rec['hidden']),
        'dropout': rec['dropout'],
        'lr': rec['opt']['lr'],
        'weight_decay': rec['opt']['weight_decay'],
        'best_val_mse': rec['best_val'],
        'test_rmse': rec['test_rmse']})
res_df = pd.DataFrame(rows).sort_values(['test_rmse', 'best_val_mse'])
res_df.to_csv('wk5/03_mlp_hyperparam_sweep_results.csv', index=False)

# Visualize best config's training/validation curve
best_key = best_by_val

# Get the results for the training and validation curves
train_curve = results[best_key]['train_curve']
val_curve   = results[best_key]['val_curve']

# Plot and save the curves
plt.figure()
plt.plot(train_curve, label='Train loss (MSE)')
plt.plot(val_curve, label='Val loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss Curves (best by val): {best_key[0]} + {best_key[1]}')
plt.legend()
plt.grid(alpha = 0.5)
plt.savefig('wk5/03_mlp_hyperparam_sweep_train_val_curve.png', dpi=300)
plt.close()