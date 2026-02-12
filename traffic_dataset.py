"""
Traffic dataset utilities
"""
import torch
from torch.utils.data import Dataset
import numpy as np

class TrafficDataset(Dataset):
    """Synthetic dataset for initial development"""
    def __init__(self, num_samples=1000):
        self.data = []
        for _ in range(num_samples):
            hour = np.random.randint(0, 24)
            if 7 <= hour <= 9:
                base = np.random.poisson(15)
                peds = np.random.poisson(8)
            elif 16 <= hour <= 18:
                base = np.random.poisson(12)
                peds = np.random.poisson(10)
            else:
                base = np.random.poisson(5)
                peds = np.random.poisson(3)
            
            features = [
                base + np.random.randint(-2, 3),  # north
                base + np.random.randint(-2, 3),  # south
                base//2 + np.random.randint(-1, 2),  # east
                base//2 + np.random.randint(-1, 2),  # west
                peds//4,  # north peds
                peds//4,  # south peds
                peds//4,  # east peds
                peds//4,  # west peds
                1 if np.random.random() < 0.01 else 0  # emergency
            ]
            self.data.append(torch.FloatTensor(features))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def state_to_tensor(state_dict):
    """Convert state dict to tensor"""
    return torch.FloatTensor([
        state_dict['north_cars'], state_dict['south_cars'],
        state_dict['east_cars'], state_dict['west_cars'],
        state_dict['north_peds'], state_dict['south_peds'],
        state_dict['east_peds'], state_dict['west_peds'],
        state_dict['emergency']
    ])
