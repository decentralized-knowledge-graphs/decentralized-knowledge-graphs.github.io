import torch

# Define the input data
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Define the model architecture
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

# Define the loss function
criterion = torch.nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    # Forward pass
    y_pred = model(x_data)
    
    # Compute the loss
    loss = criterion(y_pred, y_data)
    
    # Zero the gradients and backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update the parameters
    optimizer.step()
    
    # Print the loss at every 100th epoch
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    y_test = model(torch.tensor([[5.0]]))
    print(f'Prediction for x=5: {y_test.item():.4f}')
