"""
Run a trained DQN agent on SUMO with PyGame visualization.
"""

import pygame
import torch
import numpy as np
import sys
import os

from traffic_base import BaseTrafficAI
from sumo_env import SumoTrafficEnv

# --- Configuration (edit these as needed) ---
SUMO_CFG = "sim.sumocfg"
TLS_ID = "cluster_3648303878_5019372006_7625804229_7625804230"          # Replace with actual traffic light ID
MODEL_PATH = "ai/dqn_sumo.pth"
USE_GUI = True
TIME_LIMIT = 1800                 # seconds

# PyGame settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
SCALE = 2.0
BACKGROUND_COLOR = (50, 50, 50)
VEHICLE_COLOR = (255, 0, 0)
VEHICLE_SIZE = (10, 5)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("ATLAS AI - SUMO Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    env = SumoTrafficEnv(SUMO_CFG, TLS_ID, time_limit=TIME_LIMIT, use_gui=USE_GUI)

    model = BaseTrafficAI(input_size=env.state_size, hidden_sizes=[64, 64], output_size=env.action_size)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("No trained model found – using random actions.")

    state = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    running = True
    step_count = 0
    total_reward = 0

    while running and step_count < TIME_LIMIT:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if os.path.exists(MODEL_PATH):
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(env.action_size)

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1

        screen.fill(BACKGROUND_COLOR)

        import traci
        vehicles = traci.vehicle.getIDList()
        for veh_id in vehicles:
            x, y = traci.vehicle.getPosition(veh_id)
            px = int(x * SCALE)
            py = int(SCREEN_HEIGHT//2 - y * SCALE)
            pygame.draw.rect(screen, VEHICLE_COLOR, (px, py, VEHICLE_SIZE[0], VEHICLE_SIZE[1]))

        info_text = f"Step: {step_count} | Reward: {total_reward:.1f} | Action: {action}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        clock.tick(60)

        state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        if done:
            break

    print(f"Episode finished after {step_count} steps. Total reward: {total_reward:.2f}")
    env.stop_simulation()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
