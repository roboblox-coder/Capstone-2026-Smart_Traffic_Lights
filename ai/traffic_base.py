"""
Base neural network model for traffic AI.
Supports both a single hidden size (for backward compatibility) and a list of hidden sizes.
"""

import torch
import torch.nn as nn

class BaseTrafficAI(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Args:
            input_size: number of state features.
            hidden_sizes: list of integers, sizes of hidden layers.
            output_size: number of actions.
        """
        super().__init__()
        layers = []
        prev = input_size
        # If hidden_sizes is a single integer, convert to list
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def choose_action(self, state, epsilon=0.1):
        """Epsilon‑greedy action selection."""
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.network[-1].out_features, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return torch.argmax(q_values).item()
