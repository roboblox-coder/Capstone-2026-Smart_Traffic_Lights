"""
Minimal training script – run this to verify everything works
"""
import torch
import torch.optim as optim
import numpy as np
from src.models.traffic_base import BaseTrafficAI, TrafficState
from src.data.traffic_dataset import state_to_tensor

def random_traffic_state():
    """Generate a random traffic state for testing"""
    state = TrafficState()
    state.north_cars = np.random.randint(0, 20)
    state.south_cars = np.random.randint(0, 20)
    state.east_cars = np.random.randint(0, 15)
    state.west_cars = np.random.randint(0, 15)
    state.north_peds = np.random.randint(0, 8)
    state.south_peds = np.random.randint(0, 8)
    state.east_peds = np.random.randint(0, 8)
    state.west_peds = np.random.randint(0, 8)
    state.emergency = 1 if np.random.random() < 0.1 else 0
    return state

def main():
    print("="*50)
    print("ATLAS AI - Quick Training Test")
    print("="*50)
    
    # Create model
    model = BaseTrafficAI()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop (fake simulation)
    for episode in range(10):
        state = random_traffic_state()
        total_reward = 0
        
        for step in range(20):  # 20 steps per episode
            state_tensor = state_to_tensor(state.__dict__)
            action = model.choose_action(state_tensor, epsilon=0.3)
            
            # Fake transition
            next_state = random_traffic_state()
            reward = np.random.randn() * 5  # dummy reward
            total_reward += reward
            
            # Dummy training step (no real update)
            q_values = model(state_tensor.unsqueeze(0))
            loss = q_values.mean()  # placeholder
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
        
        print(f"Episode {episode+1:2d} | Total reward: {total_reward:7.2f}")
    
    print("\n✅ Training test complete!")
    print("\nNow your team can extend this with:")
    print("  • Real reward functions")
    print("  • Experience replay")
    print("  • Target networks")
    print("  • Traffic simulation (SUMO)")
    print("="*50)

if __name__ == "__main__":
    main()
