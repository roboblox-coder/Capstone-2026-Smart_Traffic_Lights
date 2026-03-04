"""
DQN model with configurable hidden layers.
"""

import torch
import torch.nn as nn
import numpy as np

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
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def choose_action(self, state, epsilon=0.1):
        """
        Epsilon-greedy action selection.
        state can be numpy array or torch tensor.
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.network[-1].out_features)
        else:
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state)
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                q_values = self.forward(state)
                return torch.argmax(q_values).item()
