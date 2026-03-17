import pygame
import torch
import numpy as np
import sys
import os
from traffic_base import BaseTrafficAI
from sumo_env import SumoTrafficEnv

# === CONFIG ===
SUMO_CFG = "sim.sumocfg"
TLS_ID = "3153556582"          # Change to your intersection
MODEL_PATH = "ai/dqn_sumo.pth"
USE_GUI = True
TIME_LIMIT = 1800
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 600
SCALE = 2.0
# ==============

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
    else:
        print("No trained model – using random actions.")

    state = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    running = True
    step = 0
    total_reward = 0

    while running and step < TIME_LIMIT:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if os.path.exists(MODEL_PATH):
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
        else:
            action = np.random.randint(env.action_size)

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1

        screen.fill((50,50,50))
        import traci
        for vid in traci.vehicle.getIDList():
            x,y = traci.vehicle.getPosition(vid)
            px = int(x*SCALE)
            py = SCREEN_HEIGHT//2 - int(y*SCALE)
            pygame.draw.rect(screen, (255,0,0), (px, py, 10, 5))

        info = f"Step {step} | Reward {total_reward:.1f} | Action {action}"
        screen.blit(font.render(info, True, (255,255,255)), (10,10))
        pygame.display.flip()
        clock.tick(60)

        state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        if done:
            break

    env.stop_simulation()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
