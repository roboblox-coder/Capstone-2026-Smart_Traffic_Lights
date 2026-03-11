import traci
from websocket_server import SimulationWebSocketServer
import time

print("Starting WebSocket server...")
ws = SimulationWebSocketServer()
ws.start()
time.sleep(2)

print("Starting SUMO...")
traci.start(["sumo", "-c", "sim.sumocfg", "--no-step-log"])

step = 0
while step < 100:
    traci.simulationStep()
    # Send dummy vehicle list (you could also fetch real vehicles)
    ws.broadcast({"step": step, "vehicles": [], "trafficLights": []})
    step += 1
    if step % 10 == 0:
        print(f"Step {step}")
    time.sleep(0.1)

traci.close()
ws.stop()
print("Done.")
