"""
ATLAS Traffic AI - Base Foundation
"""
import torch
import torch.nn as nn
import numpy as np

class TrafficState:
    """Represents intersection state"""
    def __init__(self):
        self.north_cars = 0
        self.south_cars = 0
        self.east_cars = 0
        self.west_cars = 0
        self.north_peds = 0
        self.south_peds = 0
        self.east_peds = 0
        self.west_peds = 0
        self.emergency = 0
    
    def to_tensor(self):
        return torch.tensor([
            self.north_cars, self.south_cars, self.east_cars, self.west_cars,
            self.north_peds, self.south_peds, self.east_peds, self.west_peds,
            self.emergency
        ], dtype=torch.float32)

class BaseTrafficAI(nn.Module):
    """Simple neural network for traffic control"""
    def __init__(self, input_size=9, hidden_size=64, output_size=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def choose_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(7)
        with torch.no_grad():
            state_tensor = state if isinstance(state, torch.Tensor) else torch.FloatTensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = self.forward(state_tensor)
            return torch.argmax(q_values).item()

if __name__ == "__main__":
    # Quick test
    model = BaseTrafficAI()
    test_state = torch.randn(9)
    action = model.choose_action(test_state, epsilon=0.0)
    print(f"Model created. Test action: {action}")
