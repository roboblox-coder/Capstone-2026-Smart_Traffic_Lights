import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from traffic_base import BaseTrafficAI
from sumo_env import SumoTrafficEnv

# === CONFIG ===
SUMO_CFG = "sim.sumocfg"
TLS_ID = "3153556582"          # Change to your intersection
USE_GUI = False
EPISODES = 5
TIME_LIMIT = 300
MODEL_SAVE_PATH = "ai/dqn_sumo.pth"
# ==============

def train():
    env = SumoTrafficEnv(SUMO_CFG, TLS_ID, time_limit=TIME_LIMIT, use_gui=USE_GUI)
    model = BaseTrafficAI(input_size=env.state_size, hidden_sizes=[64, 64], output_size=env.action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    replay_buffer = []
    epsilon = 0.9
    epsilon_end = 0.05
    epsilon_decay = 200
    gamma = 0.99
    batch_size = 32

    rewards = []
    losses = []

    for ep in range(EPISODES):
        state = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        total_reward = 0
        ep_losses = []

        for step in range(TIME_LIMIT):
            if np.random.random() < epsilon:
                action = np.random.randint(env.action_size)
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()

            next_state, reward, done, _ = env.step(action)
            next_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            replay_buffer.append((
                state_tensor.squeeze(0).numpy(),
                action,
                reward,
                next_tensor.squeeze(0).numpy(),
                done
            ))
            if len(replay_buffer) > 10000:
                replay_buffer.pop(0)

            if len(replay_buffer) >= batch_size:
                indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                batch = [replay_buffer[i] for i in indices]

                states = torch.FloatTensor([b[0] for b in batch])
                actions = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
                rewards_b = torch.FloatTensor([b[2] for b in batch])
                next_states = torch.FloatTensor([b[3] for b in batch])
                dones = torch.FloatTensor([b[4] for b in batch])

                current_q = model(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    max_next_q = model(next_states).max(1)[0]
                    target_q = rewards_b + gamma * max_next_q * (1 - dones)

                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_losses.append(loss.item())

            state_tensor = next_tensor
            total_reward += reward
            if done:
                break

        epsilon = max(epsilon_end, epsilon * np.exp(-ep / epsilon_decay))
        rewards.append(total_reward)
        avg_loss = np.mean(ep_losses) if ep_losses else 0
        losses.append(avg_loss)
        print(f"Episode {ep+1}: Reward={total_reward:.0f}, Loss={avg_loss:.2f}")

    plt.plot(rewards)
    plt.title("Training Rewards")
    plt.savefig("ai/training_curve.png")
    plt.show()

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    env.stop_simulation()

if __name__ == "__main__":
    os.makedirs("ai", exist_ok=True)
    train()
