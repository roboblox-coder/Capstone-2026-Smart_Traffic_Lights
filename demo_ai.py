import traci
import torch
import time
from websocket_server import SimulationWebSocketServer
from ai.traffic_base import BaseTrafficAI

# Configuration – adjust these to match your training
TLS_ID = "3153556582"
MODEL_PATH = "ai/dqn_sumo.pth"
STATE_SIZE = 16      # lanes * 2
ACTION_SIZE = 8       # number of phases
MAX_STEPS = 120       # short for demo

# Start WebSocket server
ws = SimulationWebSocketServer()
ws.start()
time.sleep(2)

# Load the trained model
device = torch.device('cpu')
model = BaseTrafficAI(input_size=STATE_SIZE, hidden_sizes=[64, 64], output_size=ACTION_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully.")

# Start SUMO (no extra flags, just like test_ws.py)
traci.start(["sumo", "-c", "sim.sumocfg"])

step = 0
try:
    while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
        traci.simulationStep()

        # --- AI decision ---
        lanes = traci.trafficlight.getControlledLanes(TLS_ID)
        unique_lanes = list(dict.fromkeys(lanes))
        state = []
        for lane in unique_lanes:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            waiting = traci.lane.getWaitingTime(lane)
            state.extend([queue, waiting])
        # Pad/truncate to expected size
        if len(state) != STATE_SIZE:
            state = state[:STATE_SIZE] + [0] * (STATE_SIZE - len(state))
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
        traci.trafficlight.setPhase(TLS_ID, action)

        # --- Broadcast vehicle positions ---
        vehicles = []
        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
            vehicles.append({"id": vid, "x": round(x,2), "y": round(y,2)})
        ws.broadcast({"step": step, "vehicles": vehicles})

        step += 1
        if step % 10 == 0:
            print(f"Step {step}, vehicles: {len(vehicles)}")

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    traci.close()
    ws.stop()
    print("Done.")
