import os
import sys
import subprocess

# --- 1. SET UP SUMO HOME AND TRACI ---
# Auto-detect SUMO Tools path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    # Auto-detect from installed sumolib package
    try:
        import sumolib as _sl
        sumo_pkg = os.path.dirname(os.path.dirname(_sl.__file__))
        sys.path.append(os.path.join(sumo_pkg, 'tools'))
    except ImportError:
        sys.exit("❌ Error: Set SUMO_HOME env var or install sumolib (pip install sumolib)")

import traci
from sumolib import checkBinary

# --- CONFIGURATION ---
sumoBinary = checkBinary('sumo') # use 'sumo-gui' instead if you want to see the simulation!
sumoConfig = "sim.sumocfg"

base_folder_name = "TraCI_Simulation_Data"

# --- 2. FIND NEXT OUTPUT FOLDER ---
counter = 1
while True:
    folder_name = f"{base_folder_name}_{counter}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✅ Created output folder: {folder_name}")
        break
    counter += 1

fcd_xml = os.path.join(folder_name, "fcd.xml")
queue_xml = os.path.join(folder_name, "queue.xml")

# --- 3. START TRACI ---
print("🚗 Starting TraCI Simulation...")
# We use command line flags to override the config file outputs just like run_auto_sim.py
sumoCmd = [
    sumoBinary, 
    "-c", sumoConfig,
    "--fcd-output", fcd_xml,
    "--queue-output", queue_xml,
    "--no-step-log", "true", # Reduces console spam
    "--time-to-teleport", "-1", # Optional: prevents vehicles from teleporting if stuck, you might want this for RL
    "--tls.actuated.jam-threshold", "30",
]

try:
    traci.start(sumoCmd)
except Exception as e:
    sys.exit(f"❌ Failed to start TraCI: {e}")

# --- 4. SIMULATION LOOP (AI HOOKS GO HERE) ---
step = 0
try:
    # The condition `traci.simulation.getMinExpectedNumber() > 0` runs until the network is empty.
    # We also limit it to 3600 steps to match the sim.sumocfg duration, unless all cars leave sooner.
    while traci.simulation.getMinExpectedNumber() > 0 and step < 3600:
        traci.simulationStep()
        
        # --- 🤖 AI OBSERVATION AREA 🤖 ---
        # Read data from the simulation here. Examples:
        # vehicles = traci.vehicle.getIDList()
        # for veh_id in vehicles:
        #     speed = traci.vehicle.getSpeed(veh_id)
        #     edge = traci.vehicle.getRoadID(veh_id)
        
        # Optionally, get info about traffic lights:
        # tl_ids = traci.trafficlight.getIDList()
        # for tl in tl_ids:
        #     current_phase = traci.trafficlight.getPhase(tl)
        
        # --- 🤖 AI ACTION AREA 🤖 ---
        # Execute actions based on observations. Example:
        # if step % 30 == 0:  # Every 30 seconds
        #     traci.trafficlight.setPhase("my_tls_id", 0) 
        
        step += 1
        
        # Just to show it's running:
        if step % 100 == 0:
            active_vehicles = traci.vehicle.getIDCount()
            print(f"Step {step}: {active_vehicles} vehicles active.", end='\r')

finally:
    # Always close the connection safely
    sys.stdout.write("\n")
    print("🛑 Closing TraCI Connection...")
    traci.close()
    sys.stdout.flush()

# --- 5. CONVERT XML TO CSV (Optional, using existing script) ---
print("🔄 Converting XML to CSV...")
xml2csv_script = os.path.join(sys.path[-1], "xml", "xml2csv.py") # Usually in tools/xml/

def convert_to_csv(xml_file):
    if os.path.exists(xml_file) and os.path.exists(xml2csv_script):
        print(f"   Processing {xml_file}...")
        subprocess.run([sys.executable, xml2csv_script, xml_file])
    else:
        print(f"   ⚠️ Warning: Could not process {xml_file}.")

convert_to_csv(fcd_xml)
convert_to_csv(queue_xml)

print(f"🎉 DONE! TraCI Simulation complete. Data saved to '{folder_name}'.")
