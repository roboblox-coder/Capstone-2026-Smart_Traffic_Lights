import pygame
import torch
import numpy as np
import sys
import os
from traffic_base import BaseTrafficAI
from sumo_env import SumoTrafficEnv

#*Make sure you have a trained model at ai/dqn_sumo.pth. If not, the script falls back to random actions.*#

# === CONFIGURATION – NOW MULTIPLE INTERSECTIONS ===
SUMO_CFG = "sim.sumocfg"
TLS_IDS = [
    "3153556582",                          # Main intersection
    "cluster_3648303878_5019372006_7625804229_7625804230",  # Bellevue Wy & NE 8th
    "cluster_5019371995_53129841",          # 106th Ave NE & NE 8th
    "4270714542",                           # 108th Ave NE & NE 8th
]
# Use same model for all (if they have same state/action dimensions)
MODEL_PATH = "ai/dqn_sumo.pth"
STATE_SIZE = 16      # Must match training
ACTION_SIZE = 8
USE_GUI = True
TIME_LIMIT = 3600
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 600
SCALE = 2.0
# ===================================================

def get_lanes_for_tls(tls_id):
    """Helper to get unique lanes for a traffic light (needed for state)."""
    import traci
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    return list(dict.fromkeys(lanes))

def build_state(tls_id, lanes_dict):
    """Build state vector for one traffic light."""
    import traci
    lanes = lanes_dict[tls_id]
    state = []
    for lane in lanes:
        queue = traci.lane.getLastStepHaltingNumber(lane)
        waiting = traci.lane.getWaitingTime(lane)
        state.extend([queue, waiting])
    if len(state) != STATE_SIZE:
        state = state[:STATE_SIZE] + [0] * (STATE_SIZE - len(state))
    return state

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("ATLAS AI - Multi-Intersection Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    # Create environment (still needed for simulation start/stop)
    # We'll use the first TLS_ID for env, but we control all manually in loop
    env = SumoTrafficEnv(SUMO_CFG, TLS_IDS[0], time_limit=TIME_LIMIT, use_gui=USE_GUI)

    # Load model (shared across all intersections)
    model = BaseTrafficAI(input_size=STATE_SIZE, hidden_sizes=[64, 64], output_size=ACTION_SIZE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("No trained model – using random actions for all intersections.")

    # Start simulation
    env.start_simulation()
    import traci

    # Pre-fetch lane lists for each TLS (they don't change)
    lanes_dict = {tls_id: get_lanes_for_tls(tls_id) for tls_id in TLS_IDS}

    step = 0
    total_reward = 0
    running = True

    while running and step < TIME_LIMIT:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Control each intersection
        for tls_id in TLS_IDS:
            if os.path.exists(MODEL_PATH):
                state = build_state(tls_id, lanes_dict)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()
            else:
                action = np.random.randint(ACTION_SIZE)
            traci.trafficlight.setPhase(tls_id, action)

        # Advance simulation (step once for all)
        traci.simulationStep()
        step += 1

        # Compute reward for first intersection (optional, just for display)
        lanes = lanes_dict[TLS_IDS[0]]
        step_reward = -sum(traci.lane.getWaitingTime(l) for l in lanes)
        total_reward += step_reward

        # Render vehicles
        screen.fill((50,50,50))
        for vid in traci.vehicle.getIDList():
            x,y = traci.vehicle.getPosition(vid)
            px = int(x*SCALE)
            py = SCREEN_HEIGHT//2 - int(y*SCALE)
            pygame.draw.rect(screen, (255,0,0), (px, py, 10, 5))

        info = f"Step {step} | Total Reward {total_reward:.1f}"
        screen.blit(font.render(info, True, (255,255,255)), (10,10))
        pygame.display.flip()
        clock.tick(60)

    env.stop_simulation()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

