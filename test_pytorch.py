import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("ATLAS - PyTorch Environment Verification")
print("=" * 60)

# Basic PyTorch info
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

# Test tensor operations
print("\n" + "=" * 60)
print("Tensor Operations Test")
print("=" * 60)

# Create traffic-like data
# Simulating 4 directions + pedestrian count = 5 features
traffic_data = torch.tensor([
    [12, 8, 5, 3, 2],   # Time step 1
    [15, 10, 4, 2, 3],  # Time step 2
    [18, 12, 3, 1, 4],  # Time step 3
    [22, 15, 2, 0, 5]   # Time step 4
], dtype=torch.float32)

print(f"Traffic Data Tensor Shape: {traffic_data.shape}")
print(f"Traffic Data:\n{traffic_data}")

# Test basic neural network for traffic prediction
class TrafficPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 10)  # 5 features in, 10 hidden
        self.layer2 = nn.Linear(10, 5)   # 5 features out
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

# Create model instance
model = TrafficPredictor()
print(f"\nModel Architecture: {model}")

# Test forward pass
with torch.no_grad():
    sample_input = traffic_data[0:1]  # First time step
    prediction = model(sample_input)
    print(f"\nSample Input: {sample_input}")
    print(f"Model Prediction: {prediction}")

# Test gradient computation
print("\n" + "=" * 60)
print("Gradient Computation Test")
print("=" * 60)

# Simple loss calculation
target = torch.tensor([[14, 9, 6, 4, 3]], dtype=torch.float32)
criterion = nn.MSELoss()

# Enable gradient tracking
sample_input.requires_grad_(True)
prediction = model(sample_input)
loss = criterion(prediction, target)

# Backward pass
loss.backward()
print(f"Loss: {loss.item():.4f}")
print(f"Gradients for first layer weights:\n{model.layer1.weight.grad}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - PyTorch Environment Ready!")
print("=" * 60)
