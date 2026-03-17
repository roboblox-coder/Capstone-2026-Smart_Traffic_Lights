🚦 ATLAS SUMO Web Demo – Quick Start
This guide will help you set up and run the 3D traffic simulation website using the files from the main branch.

1. Prerequisites
Python 3.10+ – Download from python.org

SUMO (Simulation of Urban MObility) – Download from sumo.dlr.de

After installing, set the environment variable SUMO_HOME to the installation folder (e.g., C:\Program Files (x86)\Eclipse\Sumo)

Add %SUMO_HOME%\bin to your system PATH

2. Get the Code
Open Git Bash and run:

bash
# Clone the repository (this downloads all files)
cd ~/OneDrive/Documents/College/   # or any folder you prefer
git clone https://github.com/roboblox-coder/Capstone-2026-Smart_Traffic_Lights.git
cd Capstone-2026-Smart_Traffic_Lights
git checkout main   # ensure you're on the main branch
Now you have the complete project. All the files you need are inside SUMO/v2/.

3. Set Up the Python Environment
bash
# Go into the v2 folder
cd SUMO/v2

# Create a virtual environment (isolates dependencies)
python -m venv .venv

# Activate it (Git Bash on Windows)
source .venv/Scripts/activate
4. Install Required Packages
bash
pip install traci sumolib websockets
(No AI packages needed for the base demo; they are only required if you later add the AI agent.)

5. Run the WebSocket Simulation
bash
python run_websocket_sim.py
This command starts the SUMO simulation and a WebSocket server that broadcasts vehicle positions. Keep this terminal window open.

6. Open the 3D Frontend
Open your web browser (Chrome, Firefox, or Edge)

Navigate to the following file (you can also double‑click it in File Explorer):

text
C:\Users\YourName\OneDrive\Documents\College\Capstone-2026-Smart_Traffic_Lights\SUMO\v2\frontend\index.html
You should see a 3D city scene with moving vehicles. The traffic lights will operate using a default fixed‑time program (no AI) – this proves the simulation and website are working.

Troubleshooting
"Could not connect to TraCI" – Make sure SUMO is installed and SUMO_HOME is set correctly.

Browser shows "Disconnected" – Wait a few seconds; the WebSocket server may still be starting. Check the terminal for error messages.

Large simulation output files – The script creates folders like WebSocket_Simulation_Data_1. These are temporary and can be deleted; they are already ignored by Git.

