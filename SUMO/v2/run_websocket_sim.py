"""
WebSocket-Enabled TraCI Simulation
===================================
Runs SUMO via TraCI while broadcasting live vehicle / traffic-light
data to all connected WebSocket clients every simulation step.

Clients can send JSON commands to interact with the simulation
(e.g. change a traffic-light phase).

Start via:  python run.py  →  option 5
"""

import os
import sys
import subprocess
import json
import time

# ── SUMO setup ───────────────────────────────────────────────────────
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    try:
        import sumolib as _sl
        sumo_pkg = os.path.dirname(os.path.dirname(_sl.__file__))
        sys.path.append(os.path.join(sumo_pkg, 'tools'))
    except ImportError:
        sys.exit("❌ Error: Set SUMO_HOME env var or install sumolib")

import traci
from sumolib import checkBinary

# ── Local imports ────────────────────────────────────────────────────
from websocket_server import SimulationWebSocketServer

# ── Configuration ────────────────────────────────────────────────────
SUMO_BINARY = checkBinary('sumo')          # headless; change to 'sumo-gui' if desired
SUMO_CONFIG = "sim.sumocfg"
WS_HOST     = "localhost"
WS_PORT     = 8765
MAX_STEPS   = 3600                         # 1 hour at 1 step/s

BASE_FOLDER = "WebSocket_Simulation_Data"

# ── Output folder ────────────────────────────────────────────────────
counter = 1
while True:
    folder_name = f"{BASE_FOLDER}_{counter}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✅ Created output folder: {folder_name}")
        break
    counter += 1

fcd_xml   = os.path.join(folder_name, "fcd.xml")
queue_xml = os.path.join(folder_name, "queue.xml")

# ── Start WebSocket server ───────────────────────────────────────────
ws_server = SimulationWebSocketServer(host=WS_HOST, port=WS_PORT)
ws_server.start()

# Give frontends a moment to connect before the sim starts
print("⏳ Waiting 2 s for clients to connect …")
time.sleep(2)
print(f"   {ws_server.client_count} client(s) connected.\n")

# ── Start TraCI ──────────────────────────────────────────────────────
sumo_cmd = [
    SUMO_BINARY,
    "-c", SUMO_CONFIG,
    "--fcd-output", fcd_xml,
    "--queue-output", queue_xml,
    "--no-step-log", "true",
    "--time-to-teleport", "-1",
    "--tls.actuated.jam-threshold", "30",
]

print("🚗 Starting TraCI + WebSocket Simulation …")
try:
    traci.start(sumo_cmd)
except Exception as e:
    ws_server.stop()
    sys.exit(f"❌ Failed to start TraCI: {e}")

# ── Simulation loop ─────────────────────────────────────────────────
step = 0
try:
    while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
        traci.simulationStep()

        # ── Gather vehicle data ──────────────────────────────────
        vehicles = []
        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
            vehicles.append({
                "id":    vid,
                "x":     round(x, 2),
                "y":     round(y, 2),
                "speed": round(traci.vehicle.getSpeed(vid), 2),
                "angle": round(traci.vehicle.getAngle(vid), 2),
                "edge":  traci.vehicle.getRoadID(vid),
            })

        # ── Gather traffic-light data ────────────────────────────
        traffic_lights = []
        for tl_id in traci.trafficlight.getIDList():
            traffic_lights.append({
                "id":    tl_id,
                "phase": traci.trafficlight.getPhase(tl_id),
                "state": traci.trafficlight.getRedYellowGreenState(tl_id),
            })

        # ── Broadcast to all WebSocket clients ───────────────────
        ws_server.broadcast({
            "type":          "step",
            "step":          step,
            "vehicles":      vehicles,
            "trafficLights": traffic_lights,
        })

        # ── Process incoming commands from clients ───────────────
        for cmd in ws_server.get_pending_commands():
            action = cmd.get("action")
            try:
                if action == "setPhase":
                    traci.trafficlight.setPhase(cmd["tlsId"], int(cmd["phase"]))
                    print(f"   🤖 setPhase({cmd['tlsId']}, {cmd['phase']})")

                elif action == "setSpeed":
                    traci.vehicle.setSpeed(cmd["vehId"], float(cmd["speed"]))

                elif action == "pause":
                    print("   ⏸️  Pause requested — press Enter to resume …")
                    # input()  # disabled for headless operation

                else:
                    print(f"   ⚠️  Unknown command: {action}")

            except Exception as exc:
                print(f"   ❌ Command error: {exc}")
                ws_server.broadcast({
                    "type": "error",
                    "message": str(exc),
                })

        step += 1

        # Progress indicator
        if step % 100 == 0:
            active = traci.vehicle.getIDCount()
            clients = ws_server.client_count
            print(f"   Step {step}: {active} vehicles, {clients} client(s)", end="\r")

except KeyboardInterrupt:
    print("\n\n⏹️  Interrupted by user.")

finally:
    sys.stdout.write("\n")
    print("🛑 Closing TraCI …")
    traci.close()
    ws_server.stop()
    sys.stdout.flush()

# ── XML → CSV conversion ────────────────────────────────────────────
print("🔄 Converting XML → CSV …")
xml2csv_script = os.path.join(sys.path[-1], "xml", "xml2csv.py")

def convert_to_csv(xml_file):
    if os.path.exists(xml_file) and os.path.exists(xml2csv_script):
        print(f"   Processing {xml_file} …")
        subprocess.run([sys.executable, xml2csv_script, xml_file])
    else:
        print(f"   ⚠️  Could not process {xml_file}.")

convert_to_csv(fcd_xml)
convert_to_csv(queue_xml)

print(f"🎉 DONE! Data saved to '{folder_name}'.")
