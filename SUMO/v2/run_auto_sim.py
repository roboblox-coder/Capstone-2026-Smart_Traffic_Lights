import os
import subprocess
import sys

# --- CONFIGURATION ---
# Base name for your output folders
base_folder_name = "Simulation_Data"

# Path to your SUMO tools (Using your specific path from previous screenshots)
# If this fails, we will try to detect it automatically.
tools_path = r"C:\Users\Duke3\AppData\Local\Programs\Python\Python313\Lib\site-packages\sumo\tools"
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
sumo_cmd = [
    "sumo", 
    "-c", "sim.sumocfg",
    "--fcd-output", fcd_xml,
    "--queue-output", queue_xml
]

try:
    subprocess.run(sumo_cmd, check=True)
    print("✅ Simulation complete.")
except subprocess.CalledProcessError:
    print("❌ Error running SUMO. Check your config file.")
    sys.exit(1)

# --- 4. CONVERT XML TO CSV ---
print("🔄 Converting XML to CSV...")

# Helper function to run the conversion
def convert_to_csv(xml_file):
    if os.path.exists(xml_file):
        print(f"   Processing {xml_file}...")
        subprocess.run([sys.executable, xml2csv_script, xml_file])
    else:
        print(f"   ⚠️ Warning: {xml_file} was not generated.")

convert_to_csv(fcd_xml)
convert_to_csv(queue_xml)

print(f"🎉 DONE! Check the folder '{folder_name}' for your CSV files.")