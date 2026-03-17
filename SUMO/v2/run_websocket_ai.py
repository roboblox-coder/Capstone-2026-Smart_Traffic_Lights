import os
import sys
import time
import torch
import traci
from websocket_server import SimulationWebSocketServer
from ai.traffic_base import BaseTrafficAI

# ===== CONFIGURATION =====
SUMO_CONFIG = "sim.sumocfg"
TLS_ID = "3153556582"          # <-- YOUR TRAFFIC LIGHT ID
MODEL_PATH = "ai/dqn_sumo.pth"
STATE_SIZE = 16                 # <-- lanes * 2 (adjust if needed)
ACTION_SIZE = 8                  # <-- number of phases
MAX_STEPS = 3600                 # 1 hour at 1 step/s
USE_GUI = False                  # set True to see SUMO window (slower)
# =========================

print("Starting WebSocket server...")
ws = SimulationWebSocketServer()
ws.start()
time.sleep(2)

print("Loading AI model...")
device = torch.device('cpu')
model = BaseTrafficAI(input_size=STATE_SIZE, hidden_sizes=[64, 64], output_size=ACTION_SIZE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded model from {MODEL_PATH}")
else:
    print(f"⚠️ Model not found – using fallback (action 0).")
    model = None

print("Starting SUMO...")
sumo_binary = "sumo-gui" if USE_GUI else "sumo"
cmd = [sumo_binary, "-c", SUMO_CONFIG]   # NO extra flags
print(f"Running command: {cmd}")
try:
    traci.start(cmd)
except Exception as e:
    print(f"❌ TraCI start failed: {e}")
    ws.stop()
    sys.exit(1)

step = 0
no_vehicle_counter = 0

try:
    while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
        traci.simulationStep()

        if model is not None:
            lanes = traci.trafficlight.getControlledLanes(TLS_ID)
            unique_lanes = list(dict.fromkeys(lanes))
            state = []
            for lane in unique_lanes:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                waiting = traci.lane.getWaitingTime(lane)
                state.extend([queue, waiting])
            if len(state) != STATE_SIZE:
                state = state[:STATE_SIZE] + [0] * (STATE_SIZE - len(state))
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()
            traci.trafficlight.setPhase(TLS_ID, action)

        vehicles = []
        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
            vehicles.append({
                "id": vid,
                "x": round(x, 2),
                "y": round(y, 2),
                "speed": round(traci.vehicle.getSpeed(vid), 2),
                "angle": round(traci.vehicle.getAngle(vid), 2),
                "edge": traci.vehicle.getRoadID(vid),
            })

        traffic_lights = []
        for tl_id in traci.trafficlight.getIDList():
            traffic_lights.append({
                "id": tl_id,
                "phase": traci.trafficlight.getPhase(tl_id),
                "state": traci.trafficlight.getRedYellowGreenState(tl_id),
            })

        ws.broadcast({
            "type": "step",
            "step": step,
            "vehicles": vehicles,
            "trafficLights": traffic_lights
        })

        step += 1
        if step % 10 == 0:
            active = traci.vehicle.getIDCount()
            print(f"Step {step}: {active} vehicles")

        active = traci.vehicle.getIDCount()
        if active == 0:
            no_vehicle_counter += 1
            if no_vehicle_counter > 100:
                print("No vehicles – ending early.")
                break
        else:
            no_vehicle_counter = 0

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    traci.close()
    ws.stop()
    print("Done.")
