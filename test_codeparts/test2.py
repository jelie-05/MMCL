import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

# Define the custom MSE loss function
class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, input, target):
        loss = torch.mean((input - target) ** 2)
        return loss

# Instantiate the model, loss function, and optimizer
model = SimpleNN()
criterion = CustomMSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data for demonstration
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
targets = torch.tensor([[3.0], [7.0]])

# Forward pass: Compute predicted outputs by passing inputs to the model
outputs = model(inputs)

# Compute the loss
loss = criterion(outputs, targets)
print('Loss:', loss.item())

# Backward pass: Compute gradient of the loss with respect to model parameters
loss.backward()

# Print gradients
print('Gradients:')
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f'{name}: {param.grad}')

# Update model parameters
optimizer.step()

# Zero the gradients after updating
optimizer.zero_grad()

# Perform another forward pass to see the updated loss
outputs = model(inputs)
loss = criterion(outputs, targets)
print('Updated Loss:', loss.item())
