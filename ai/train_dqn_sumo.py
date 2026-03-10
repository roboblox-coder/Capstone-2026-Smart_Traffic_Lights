"""
Train DQN agent on SUMO simulation via TraCI.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from traffic_base import BaseTrafficAI
from sumo_env import SumoTrafficEnv

# --- CONFIGURATION (edit these) ---
SUMO_CFG = "sim.sumocfg"
TLS_ID = "3153556582"          # Replace with actual traffic light ID
USE_GUI = False                   # Set to True to watch training (slower)
EPISODES = 20
TIME_LIMIT = 1800                  # seconds per episode
MODEL_SAVE_PATH = "ai/dqn_sumo.pth"

def train_dqn_sumo(env, model, episodes=EPISODES, steps_per_episode=TIME_LIMIT,
                   gamma=0.99, lr=0.001, epsilon_start=0.9, epsilon_end=0.05,
                   epsilon_decay=200, batch_size=32, replay_capacity=10000):
    """Train DQN on SUMO environment."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    replay_buffer = []
    epsilon = epsilon_start

    episode_rewards = []
    episode_losses = []

    for ep in range(episodes):
        state = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        total_reward = 0
        losses = []

        for step in range(steps_per_episode):
            # Epsilon‑greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(env.action_size)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            replay_buffer.append((
                state_tensor.squeeze(0).numpy(),
                action,
                reward,
                next_state_tensor.squeeze(0).numpy(),
                done
            ))
            if len(replay_buffer) > replay_capacity:
                replay_buffer.pop(0)

            if len(replay_buffer) >= batch_size:
                indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                batch = [replay_buffer[i] for i in indices]

                states = torch.FloatTensor([b[0] for b in batch])
                actions = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
                rewards = torch.FloatTensor([b[2] for b in batch])
                next_states = torch.FloatTensor([b[3] for b in batch])
                dones = torch.FloatTensor([b[4] for b in batch])

                current_q = model(states).gather(1, actions).squeeze(1)

                with torch.no_grad():
                    max_next_q = model(next_states).max(1)[0]
                    target_q = rewards + gamma * max_next_q * (1 - dones)

                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            state_tensor = next_state_tensor
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon_end, epsilon_start * np.exp(-ep / epsilon_decay))

        episode_rewards.append(total_reward)
        avg_loss = np.mean(losses) if losses else 0
        episode_losses.append(avg_loss)

        print(f"Episode {ep+1}/{episodes} | Total Reward: {total_reward:.2f} | "
              f"Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

    # Plot training curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(episode_rewards)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1,2,2)
    plt.plot(episode_losses)
    plt.title("Average Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("ai/training_curve.png")
    plt.show()

    return model

if __name__ == "__main__":
    # Create directories if needed
    os.makedirs("ai", exist_ok=True)

    # Create environment
    env = SumoTrafficEnv(SUMO_CFG, TLS_ID, time_limit=TIME_LIMIT, use_gui=USE_GUI)

    # Create DQN model (using two hidden layers of 64 each)
    model = BaseTrafficAI(input_size=env.state_size,
                          hidden_sizes=[64, 64],
                          output_size=env.action_size)

    print(f"State size: {env.state_size}, Action size: {env.action_size}")

    # Train
    trained_model = train_dqn_sumo(env, model, episodes=EPISODES,
                                    steps_per_episode=TIME_LIMIT)

    # Save model
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    env.stop_simulation()
