import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Reload model from 02_pytorch_mlp.py to tinker with hyperparameters
##########
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Import and organize frozen final descriptors
df = pd.read_csv('wk4/final_descriptors.csv')
target_col = 'logS'
descriptor_cols = [c for c in df.columns if c not in [target_col, 'SMILES']]
X = df[descriptor_cols]
y = df[target_col]

# Recreate train-test split (with temp holdout)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, random_state = seed)

# Split temp set into test and validation (monitor generalization; determine when to stop training; hyperparameter tuning)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

# Convert split to float32 to prepare for torch implementation
X_train_np = X_train.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32).reshape(-1, 1)
y_val_np = y_val.values.astype(np.float32).reshape(-1, 1)
y_test_np = y_test.values.astype(np.float32).reshape(-1, 1)

# Convert np to tensor
X_train_t = torch.from_numpy(X_train_np)
X_val_t = torch.from_numpy(X_val_np)
X_test_t = torch.from_numpy(X_test_np)
y_train_t = torch.from_numpy(y_train_np)
y_val_t = torch.from_numpy(y_val_np)
y_test_t = torch.from_numpy(y_test_np)

# Create solubility dataset subclass from torch dataset class (so that torch treats it as a valid dataset and for better interaction with DataLoader)
class SolubilityDataset(Dataset):
    # Store the input matrix and target vector as class atrributes
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor
        self.y = y_tensor
    # Tells PyTorch how many samples there are --> allows it to determine when it has iterated through an epoch
    def __len__(self):
        return self.X.shape[0]
    # Defines how to fetch a sample via its index (DataLoader groups these into batches) 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Instantiate dataset objects
train_ds = SolubilityDataset(X_train_t, y_train_t)
val_ds = SolubilityDataset(X_val_t, y_val_t)
test_ds = SolubilityDataset(X_test_t, y_test_t)

# Create dataloader (creates batches)
batch_size = 64 # pretty standard starting point; will be tweaked later on
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) # shuffles for training
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)  # allows for reproducibility (batch composition does not change)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Define MLP model!!
class MLPRegressor(nn.Module):
    def __init__(self, n_features, hidden_sizes=(64, 32), dropout_p=0.2): # hard-code base values from baseline model to keep as keyword args
        super().__init__()
        # tracks parameters so nn torch object behaves correctly
        layers = []
        in_dim = n_features

        # Creates hidden layers
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h)) # weights and biases
            layers.append(nn.ReLU()) # ReLU (0 for negatives, positives stay same; alternative to sigmoid)
            layers.append(nn.Dropout(p=dropout_p)) # dropout set 
            in_dim = h

        # Adds the final output layer (input with size of last hidden layer; output with size 1 neuron)
        layers.append(nn.Linear(in_dim, 1))

        # Builds the model so that the layers are used sequentially from input -> hidden -> output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Define a function to train and validate the model
def train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs):
    # instantiate performance (loss) trackers
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    # model trainer
    for epoch in range(num_epochs):
        model.train()
        tot_train_loss = 0.0
        n_train = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch) # forward pass
            loss = loss_func(preds, y_batch) # compute loss
            loss.backward() # backpropagate
            optimizer.step() # update weights

            inst_batch_size = X_batch.size(0) # final batch may be smaller than batch_size
            tot_train_loss += loss.item() * inst_batch_size # loss.item() is the mean loss over this batch; multiply by batch size to get total loss for this batch
            n_train += inst_batch_size

        avg_train_loss = tot_train_loss / n_train # after all batches have run, finds average loss across training set

        # model validator
        model.eval()
        tot_val_loss = 0.0
        n_val = 0

        with torch.no_grad(): # reduces computational burden (no gradients b/c no backpropagation)
            for X_batch, y_batch in val_loader: # repeat logic from train set with no backpropagation
                preds = model(X_batch)
                loss = loss_func(preds, y_batch)

                inst_batch_size = X_batch.size(0)
                tot_val_loss += loss.item() * inst_batch_size
                n_val += inst_batch_size

        avg_val_loss = tot_val_loss / n_val

        # store train and val losses across epochs
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # track best model based on minimum validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
    # load best weights into model (use best epoch, not the final epoch)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

# Evaluate RMSE (apples-to-apples comparison with tuned RF)
def evaluate_rmse(model, data_loader, loss_func):
    model.eval()
    tot_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            preds = model(X_batch)
            loss = loss_func(preds, y_batch) # MSE average across batch
            batch_size = X_batch.size(0)
            tot_loss += loss.item() * batch_size # batch mean MSE * batch size = total SE for this batch
            n_samples += batch_size

    mse = tot_loss / n_samples
    rmse = np.sqrt(mse)
    return rmse

##########

# Collect constant hyperparameters
n_features = 7
weight_decay = 1e-4
num_epochs = 100
loss_func = nn.MSELoss()

# Instantiate various models
model_configs = [
    ('h64-32_d0.2', (64, 32),     0.2),
    ('h128-64_d0.2', (128, 64),   0.2),
    ('h64-32-16_d0.2', (64, 32, 16), 0.2),
    ('h64-32_d0.4', (64, 32),     0.4),
    ('h128-64_d0.4', (128, 64),   0.4),
    ('h64-32-16_d0.4', (64, 32, 16), 0.4)]
models = {}
for name, hidden, drop in model_configs:
    models[name] = MLPRegressor(n_features=n_features, hidden_sizes=hidden, dropout_p=drop)

opt_configs = {
    'adam_lr1e-3': {'lr': 1e-3, 'weight_decay': weight_decay},
    'adam_lr5e-4': {'lr': 5e-4, 'weight_decay': weight_decay}}

losses = {}

for model_name, model in models.items():
    for opt_name, params in opt_configs.items():
        optimizer = torch.optim.Adam(model.parameters(), **params)

        train, val = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            num_epochs=num_epochs
        )

        losses[(model_name, opt_name)] = {'train': train, 'val': val, 'model': model}

# Define loss (cost) function and optimizer (MSE & Adam)
loss_func = nn.MSELoss()
optimizers_1 = []
optimizers_2 = []
for model_name, model in models.items():
    optimizers_1.append(torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay))
    optimizers_2.append(torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=weight_decay))

train_losses = []
val_losses = []

# Call training loop (predefined default hyperparameters); store RMSE
train_losses_set1, val_losses_set1,  = [], []
for model, opt in zip(models.values(), optimizers_1):
    train, val = train_model(model, train_loader, val_loader, loss_func, opt, num_epochs)
    train_losses_set1.append(train)
    val_losses_set1.append(val)

train_losses_set2, val_losses_set2 = [], []
for model, opt in zip(models.values(), optimizers_2):
    train, val = train_model(model, train_loader, val_loader, loss_func, opt, num_epochs)
    train_losses_set2.append(train)
    val_losses_set2.append(val)

best_params = None
best_val = float('inf')
for key in losses:  # key is (model_name, opt_name)
    final_val = min(losses[key]['val'])
    if final_val < best_val:
        best_val = final_val
        best_params = key
print('Best params:', best_params)
print('Best val loss:', best_val)

best_key = None
best_rmse = float('inf')
for (model_name, opt_name), rec in losses.items():
    rmse = evaluate_rmse(rec['model'], test_loader, loss_func)
    if rmse < best_rmse:
        best_rmse = rmse
        best_key = (model_name, opt_name)
print('Lowest RMSE combo:', best_key)
print('Lowest RMSE:', best_rmse)