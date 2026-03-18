# 🚦 SUMO Traffic Simulation — NE 8th St Corridor (Bellevue)

## Prerequisites

| Tool | Download |
|------|----------|
| **SUMO** (desktop app) | [sumo.dlr.de/docs/Downloads.php](https://sumo.dlr.de/docs/Downloads.php) |
| **Python 3.10+** | [python.org/downloads](https://www.python.org/downloads/) |

After installing SUMO, set the **`SUMO_HOME`** environment variable so the scripts can find SUMO's built-in tools:

- **Windows** — add a System Environment Variable:  
  `SUMO_HOME = C:\Program Files (x86)\Eclipse\Sumo`  *(adjust to your install path)*
- **macOS/Linux** — add to your shell profile (`~/.bashrc` / `~/.zshrc`):  
  ```bash
  export SUMO_HOME="/usr/share/sumo"
  ```

> **Tip:** If `SUMO_HOME` is not set, the scripts will try to auto-detect the path from the installed `sumolib` Python package. Setting `SUMO_HOME` is still recommended.

---

## Setup

```bash
# 1. Clone / download this repo
git clone <repo-url>
cd SUMO

# 2. Create & activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Simulation

```bash
cd v2
python run.py
```

This opens an interactive menu:

| Option | Mode | Description |
|--------|------|-------------|
| **1** | Harvard Real-Life | Generates routes from Harvard traffic data, then opens SUMO-GUI |
| **2** | Synthetic Traffic | Runs with pre-built synthetic routes in SUMO-GUI |
| **3** | Headless Export | Runs SUMO headlessly, exports FCD & queue data → CSV |
| **4** | TraCI | Runs via TraCI for programmatic / AI control |
| **5** | WebSocket | TraCI + live WebSocket feed for a Three.js frontend |

> **Option 5** starts a WebSocket server on `ws://localhost:8765`.  
> Open `v2/frontend/index.html` in a browser to see vehicles move in real time.

---

## PyTorch Installation Guide

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



## Project Structure

```
SUMO/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore
└── v2/
    ├── run.py                # 🚀 Main launcher (start here)
    ├── real_life_simulation.py  # Converts Harvard Excel → SUMO routes
    ├── run_auto_sim.py       # Headless sim + CSV export
    ├── run_traci_sim.py      # TraCI-based sim with AI hooks
    ├── run_websocket_sim.py  # TraCI + WebSocket broadcaster
    ├── websocket_server.py   # Reusable WS server module
    ├── sim.sumocfg           # SUMO config file
    ├── NE_8th_St_Corridor.net.xml  # Road network
    ├── harvard_simulation.rou.xml  # Routes from Harvard data
    ├── synthetic.rou.xml     # Synthetic routes
    ├── Real_intersection_data/     # Harvard Excel source files
    ├── ai/                   # This AI helps you set up and run the 3D traffic simulation website using the files from the main branch.
        ├── sumo_env
        ├── traffic_base
        ├── train_dqn_sumo
        └── visualize_sumo_ai
    └── frontend/
        └── index.html (simulation.html on the website)  # Three.js live viewer (no build step)
```
