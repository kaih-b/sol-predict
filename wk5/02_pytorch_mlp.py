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
    # Defines how to fetch one sample via its index
    # This builds batches 
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
            layers.append(nn.ReLU()) # sigmoid alternative ReLU (0 for negatives, positives stay same)
            layers.append(nn.Dropout(p=dropout_p)) # dropout set 
            in_dim = h 

        # Adds the final output layer (input with size of last hidden layer; output wuth size 1 neuron)
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
            tot_train_loss += loss.item() * inst_batch_size # adds the average loss across batch (loss.item()) multiplied by the batch_size
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
            best_state_dict = model.state_dict()

        # print results for each epoch
        print(f'Epoch {epoch+1}/{num_epochs}\n -Train MSE: {avg_train_loss:.4f}\n -Val MSE: {avg_val_loss:.4f}')

    # load best weights into model (use best epoch, not the final epoch)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

# Call training loop (predefined default hyperparameters)
train_losses, val_losses = train_model(model=model, train_loader=train_loader, val_loader=val_loader, loss_func=loss_func, optimizer=optimizer, num_epochs=num_epochs)
print(np.min(train_losses), np.min(val_losses))

# Compare results with case study molecule
case_SMILES = 'O=C1c2ccccc2C(=O)c3ccccc13'
case_index = None

row = df[df['SMILES'] == case_SMILES].iloc[0]
label = f'SMILES: {case_SMILES}'

