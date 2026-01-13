import torch

# Create a tensor tracking all operations of x
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Simple function: each element squared, summed into a single number (gradient is defined)
y = (x ** 2).sum()

# Backpropagate
y.backward()

# Inspect gradient behavior
print(x.grad)
# Expected output: tensor([2., 4., 6.])
# .backward() computes the gradients, so d/dx (x**2) = 2x, so each component of the original tensor is multiplied by 2
# This is the simplest example of how neural networks learn via PyTorch!
# In real models, the x is model parameters

# Random test data
torch.manual_seed(0)
N = 100
x = torch.randn(N, 1) # shape matches linear regression
y = 3 * x + 0.1 * torch.randn(N, 1)

# Define model (first arg weight, second arg bias)
model = torch.nn.Linear(1, 1)

# Cost function (aka loss function) as MSE
cost_fn = torch.nn.MSELoss()

# Adam optimizer (built-in for torch)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(200):
    optimizer.zero_grad() # clear gradients
    y_pred = model(x) # forward pass
    cost = cost_fn(y_pred, y) # compute cost function
    cost.backward() # compute gradients (backpropogation)
    optimizer.step() # update parameters
    # visualize progress (loss decreases)
    if epoch % 20 == 0:
        print(f'Epoch {epoch}: loss = {cost.item():.4f}')

# Output
# Epoch 0: loss = 9.2104
# Epoch 20: loss = 1.0613
# Epoch 40: loss = 0.0151
# Epoch 60: loss = 0.0293
# Epoch 80: loss = 0.0086
# Epoch 100: loss = 0.0086
# Epoch 120: loss = 0.0084
# Epoch 140: loss = 0.0084
# Epoch 160: loss = 0.0084
# Epoch 180: loss = 0.0084

# Check weights and bias
print(model.weight.item(), model.bias.item())

# Intution:
# What does requires_grad=True do?
# tracks all operations of x; allows for backpropagation (gradients in x.grad after y.backward())

# Why must the loss be scalar?
# .backward() computes a gradient from a single scalar input, so the loss must be scalar for backpropagation

# Why do we zero gradients each step?
# python accumulates gradients, so for each step we have to clear it
# makes updates to the model factually correct

# What does the optimizer actually update?
# biases and weights within the model; changes their .data properties using gradients in x.grad