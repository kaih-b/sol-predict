import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Maintain random seed from RF models (here for convenience)
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

# Verify batch shapes and datatypes (sanity check)
xb, yb = next(iter(train_loader))
print('Train batch X shape:', xb.shape)
print('Train batch y shape:', yb.shape)
print('Train batch X dtype:', xb.dtype)
print('Train batch y dtype:', yb.dtype)
# Output (all as expected)
# Train batch X shape: torch.Size([64, 7])
# Train batch y shape: torch.Size([64, 1])
# Train batch X dtype: torch.float32
# Train batch y dtype: torch.float32
xv, yv = next(iter(val_loader))
print('Val batch X shape:', xv.shape)
print('Val batch y shape:', yv.shape)
print('Val batch X dtype:', xv.dtype)
print('Val batch y dtype:', yv.dtype)
# Output (all as expected)
# Val batch X shape: torch.Size([64, 7])
# Val batch y shape: torch.Size([64, 1])
# Val batch X dtype: torch.float32
# Val batch y dtype: torch.float32

# Data Review
descriptor_count = X_train_t.shape[1]
print('descriptor_count:', descriptor_count)
print('Train set:', len(train_ds), 'samples')
print('Val set:  ', len(val_ds), 'samples')
print('Test set: ', len(test_ds), 'samples')
print('Train batches per epoch:', len(train_loader)) # Batches per epoch (how many iterations the training loop will run each epoch)
print('Val batches per epoch:', len(val_loader))
print('Test batches:', len(test_loader))
# Output
# descriptor_count: 7
# Train set: 915 samples
# Val set:   114 samples
# Test set:  115 samples
# Train batches per epoch: 15
# Val batches per epoch: 2
# Test batches: 2

# From here, we can begin training the MLP

# Basic hyperparameter choices (batch_size already defined)
n_features = X_train_t.shape[1] # this is just the number of final descriptors (7)
hidden_sizes = (64, 32)
dropout_p = 0.3
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 100

# Define MLP model!!
class MLPRegressor(nn.Module):
    def __init__(self, n_features, hidden_sizes=hidden_sizes, dropout_p=dropout_p):
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
    
# Instantiate MLP
model = MLPRegressor(n_features=n_features, hidden_sizes=hidden_sizes, dropout_p=dropout_p)

# Define loss (cost) function and optimizer (MSE & Adam)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

        # print results for each epoch
        print(f'Epoch {epoch+1}/{num_epochs}\n -Train MSE: {avg_train_loss:.4f}\n -Val MSE: {avg_val_loss:.4f}')

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

# Call training loop (predefined default hyperparameters); store RMSE
train_losses, val_losses = train_model(model=model, train_loader=train_loader, val_loader=val_loader, loss_func=loss_func, optimizer=optimizer, num_epochs=num_epochs)
mlp_rmse = evaluate_rmse(model, test_loader, loss_func)


# Compare results with case study molecule
case = pd.read_csv('wk4/worst_case_study.csv')
rf_metrics = pd.read_csv('wk4/rf_final_metrics.csv')
rf_rmse = rf_metrics['Test_RMSE'].iloc[0]

row = case.iloc[0]
case_smiles = row['SMILES']
row_df = df[df['SMILES'] == case_smiles].iloc[0] # update row to capture all seven descriptors (only top 3 saved in worst_case_study.csv)

# Save RF results for case study molecule
exp_logS = float(row['exp_logS'])
rf_pred_logS = float(row['pred_logS']) # RF prediction from wk4, taken from csv
rf_error = float(row['residual'])
rf_abs = float(row['abs_error'])

# Build input for MLP with same full descriptor set as RF training
x_case_np = row_df[descriptor_cols].values.astype('float32')
x_case = torch.tensor(x_case_np).unsqueeze(0) # fix shape

# Run the MLP model
model.eval()
with torch.no_grad():
    mlp_pred_logS = model(x_case).item()
mlp_error = mlp_pred_logS - exp_logS
mlp_abs = abs(mlp_error)

# Report metrics comparing the models
report = ('---- Worst-Case Study Molecule -----\n'
    f'\nSMILES: {case_smiles}'
    f'\nExperimental logS: {exp_logS:.3f}\n'
    f'\nRF predicted logS: {rf_pred_logS:.3f}'
    f'\nRF error: {rf_error:+.3f}\n'
    f'\nMLP predicted logS: {mlp_pred_logS:.3f}'
    f'\nMLP error: {mlp_error:+.3f}\n'
    f'\n|RF error| = {rf_abs:.3f}'
    f'\n|MLP error| = {mlp_abs:.3f}\n')
RMSE_report = (f'\nRF RMSE: {rf_rmse:.3f}\n'
               f'MLP RMSE: {mlp_rmse:.3f}')

# Save metrics to a markdown
with open('wk5/02_rf_vs_baseline_mlp.md', 'w') as f:
    f.write(report)
    if rf_abs > mlp_abs:
        f.write(f'\nBaseline MLP outperforms tuned RF for worst-case study molecule by {(rf_abs - mlp_abs):.3f}\n')
    elif mlp_abs > rf_abs:
        f.write(f'\nTuned RF outperforms baseline MLP for worst-case study molecule by {(mlp_abs - rf_abs):.3f}\n')
    else:
        f.write('\nBaseline MLP and tuned RF perform the same on the worst-case study molecule\n')
    f.write(RMSE_report)

# Log and save baseline MLP metrics for future comparison
results_df = pd.DataFrame([{
    'Model': 'Baseline MLP',
    'Descriptors': 'final_descriptors',
    'Test_RMSE': mlp_rmse,
    'n_train': len(train_ds),
    'n_test': len(test_ds)}])
results_df.to_csv('wk5/02_mlp_baseline_metrics.csv', index=False)