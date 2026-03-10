"""
Train DQN agent on SUMO simulation via TraCI.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.models.traffic_base import BaseTrafficAI
from src.sumo_env import SumoTrafficEnv
import os

def train_dqn_sumo(env, model, episodes=50, steps_per_episode=3600,
                   gamma=0.99, lr=0.001, epsilon_start=0.9, epsilon_end=0.05,
                   epsilon_decay=200, batch_size=32, replay_capacity=10000):
    """Train DQN on SUMO environment."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    replay_buffer = []  # simple list (could use deque for efficiency)
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

            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # Store transition in replay buffer
            replay_buffer.append((
                state_tensor.squeeze(0).numpy(),
                action,
                reward,
                next_state_tensor.squeeze(0).numpy(),
                done
            ))
            if len(replay_buffer) > replay_capacity:
                replay_buffer.pop(0)

            # Training step (if enough samples)
            if len(replay_buffer) >= batch_size:
                # Sample random mini‑batch
                indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                batch = [replay_buffer[i] for i in indices]

                states = torch.FloatTensor([b[0] for b in batch])
                actions = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
                rewards = torch.FloatTensor([b[2] for b in batch])
                next_states = torch.FloatTensor([b[3] for b in batch])
                dones = torch.FloatTensor([b[4] for b in batch])

                # Current Q values
                current_q = model(states).gather(1, actions).squeeze(1)

                # Target Q values (using target network would be better, but simple for now)
                with torch.no_grad():
                    max_next_q = model(next_states).max(1)[0]
                    target_q = rewards + gamma * max_next_q * (1 - dones)

                # Compute loss and update
                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            state_tensor = next_state_tensor
            total_reward += reward

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_start * np.exp(-ep / epsilon_decay))

        # Record episode stats
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
    plt.savefig("docs/figures/dqn_sumo_training.png")
    plt.show()

    return model

if __name__ == "__main__":
    # --- CONFIGURATION ---
    SUMO_CFG = "osm.sumocfg"          # Path to your SUMO config file
    TLS_ID = "cluster_3648303878_5019372006_7625804229_7625804230"             # Traffic light ID (change after verifying!)
    USE_GUI = True                     # Set to True to watch the simulation
    EPISODES = 20                      # Number of episodes
    TIME_LIMIT = 1800                   # Seconds per episode (e.g., 30 minutes)
    MODEL_SAVE_PATH = "models/dqn_sumo.pth"

    # Create directories if needed
    os.makedirs("models", exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)

    # Create environment
    env = SumoTrafficEnv(SUMO_CFG, TLS_ID, time_limit=TIME_LIMIT, use_gui=USE_GUI)

    # Create DQN model (input size = state_size, output = action_size)
    model = BaseTrafficAI(input_size=env.state_size,
                          hidden_size=64,
                          output_size=env.action_size)

    print(f"State size: {env.state_size}, Action size: {env.action_size}")

    # Train
    trained_model = train_dqn_sumo(env, model, episodes=EPISODES,
                                    steps_per_episode=TIME_LIMIT)

    # Save model
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    env.stop_simulation()
