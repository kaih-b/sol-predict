import torch
from torch.utils.data import Dataset, DataLoader
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
print("Val batch X shape:", xv.shape)
print("Val batch y shape:", yv.shape)
print("Val batch X dtype:", xv.dtype)
print("Val batch y dtype:", yv.dtype)
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