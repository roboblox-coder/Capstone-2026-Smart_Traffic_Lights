import os
import subprocess
import sys

# --- CONFIGURATION ---
# Base name for your output folders
base_folder_name = "Simulation_Data"

# Auto-detect SUMO tools path
SUMO_HOME = os.environ.get("SUMO_HOME")
if not SUMO_HOME:
    try:
        import sumolib as _sl
        SUMO_HOME = os.path.dirname(os.path.dirname(_sl.__file__))
    except ImportError:
        sys.exit("❌ Error: Set SUMO_HOME or install sumolib (pip install sumolib)")
tools_path = os.path.join(SUMO_HOME, "tools")
xml2csv_script = os.path.join(tools_path, "xml", "xml2csv.py")

# --- 1. FIND THE NEXT FOLDER NUMBER ---
counter = 1
while True:
    folder_name = f"{base_folder_name}_{counter}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✅ Created output folder: {folder_name}")
        break
    counter += 1

# --- 2. DEFINE OUTPUT PATHS ---
# We will tell SUMO to save the files directly into this new folder
fcd_xml = os.path.join(folder_name, "fcd.xml")
queue_xml = os.path.join(folder_name, "queue.xml")

# --- 3. RUN SUMO ---
print("🚗 Starting SUMO Simulation...")
# We use command line flags to override the config file outputs!
from sumolib import checkBinary
sumo_cmd = [
    checkBinary('sumo'),
    "-c", "sim.sumocfg",
    "--fcd-output", fcd_xml,
    "--queue-output", queue_xml,
    "--tls.actuated.jam-threshold", "30",
]

try:
    subprocess.run(sumo_cmd, check=True)
    print("✅ Simulation complete.")
except subprocess.CalledProcessError:
    print("❌ Error running SUMO. Check your config file.")
    sys.exit(1)

# --- 4. CONVERT XML TO CSV ---
def convert_to_csv(xml_file):
    if not os.path.isfile(xml2csv_script):
        print(f"   ⚠️ xml2csv.py not found at {xml2csv_script}, skipping conversion.")
        return
    if os.path.exists(xml_file):
        print(f"   Processing {xml_file}...")
        subprocess.run([sys.executable, xml2csv_script, xml_file])
    else:
        print(f"   ⚠️ Warning: {xml_file} was not generated.")

convert_to_csv(fcd_xml)
convert_to_csv(queue_xml)

print(f"🎉 DONE! Check the folder '{folder_name}' for your CSV files.")
