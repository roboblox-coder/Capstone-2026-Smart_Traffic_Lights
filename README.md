# Capstone-2026-Smart_Traffic_Lights
Repository &amp; workspace for everything related to BC's Team 5 Capstone project.

# Project Name: "SYNAPTIX Project: ATLAS"

- Brief description of your project.
"Project: ATLAS" is our attempt to utilize an AI algorithm paired with a system of cameras as a plan to improve the quality of traffic at intersections for pedestrians, bicyclists, late-night drivers, and emergency vehicles alike.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Design Details](#design-details)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Provide a concise introduction to your project. Explain what problem it solves and why it's useful.

Project: ATLAS is an affordable, privacy-first traffic management solution designed to modernize urban intersections without the high costs of industrial sensors or the security risks of mass data storage. By utilizing low-cost camera systems and real-time AI, ATLAS eliminates unnecessary idling for drivers and creates a safer environment for pedestrians and emergency vehicles. Our "zero-retention" approach ensures that while the system is highly intelligent, it remains invisible to data breaches—processing what it sees in the moment and discarding the footage instantly to protect citizen privacy.

## Features

List the key features of your project. Use bullet points for clarity.

- Feature 1: AI ALgorithm using Three.js, SUMO (Simulation of Urban MObility) & PyTorch AI
- Feature 2: Cameras

- ... (AI expanded version):

- Cost-Efficient Perception: Replaces expensive LiDAR and inductive loops with high-performance, budget-friendly cameras to lower installation barriers for any city.
- Privacy-by-Design (Stateless Processing): Operates entirely on real-time data streams with no video storage, ensuring no personal identifiable information (PII) is ever at risk of being stolen or leaked.
- Dynamic AI Sequencing: A PyTorch-driven algorithm that prioritizes road users based on immediate demand, clearing intersections for late-night drivers and emergency "green waves."
- Vulnerable Road User (VRU) Safety: Real-time detection of bicyclists and pedestrians to adjust signal timings dynamically, preventing accidents before they happen.

## Design Details

Explain the high-level design decisions and architecture of your project. Include diagrams or code snippets if necessary.

- Hardware (Low-Cost Vision): The system utilizes standard-definition wide-angle cameras. These provide sufficient data for AI classification while keeping hardware costs significantly lower than industrial-grade traffic sensors.

- AI Engine (PyTorch): We use a lightweight PyTorch model optimized for the "edge." The model performs object detection (identifying cars vs. pedestrians) and immediately converts these into numerical data points (e.g., "3 cars in Lane A, 1 pedestrian at Crosswalk B").

- Simulation & Logic (Three.js & SUMO): The SUMO simulation environment acts as the central logic controller. It ingests the numerical data points to simulate the most efficient signal phase, testing and executing timing changes in a virtual-to-physical loop. This simulation will then be "re-drawn" by Three.js into a 3d simulation in order to make the visual aspects of the simulation more visually appealing.

- Ephemeral Data Pipeline: To ensure privacy, the system follows an In-Flight Processing model. Once the AI extracts the necessary metadata (object count and position), the raw video buffer is overwritten. No footage is saved to a hard drive or sent to a central cloud server.



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


# Simulation Website:
https://synaptix-website-773415276572.us-west1.run.app/ 



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
    ├── e2output.xml
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



#### The following is a W.I.P. since this project is still in its extremely early phases. These sections will be updated and refined as time goes on and development on project ATLAS continues.
### Example Code:

```python
# Code snippet or example to showcase design principles

Installation
Provide instructions on how to install your project. Include any dependencies or prerequisites.
Requires Python and PyGame to run.

# Installation steps
$ git clone https://github.com/your-username/your-repo.git
$ cd your-repo
$ npm install  # or any other relevant command

Configuration
Explain how users can configure your project. If applicable, include details about configuration files.

Example Configuration:
# Configuration file example
key: value

Usage
Provide examples and instructions on how users can use your project. Include code snippets or command-line examples.

Example Usage:
# Example command or usage

Contributing
Explain how others can contribute to your project. Include guidelines for pull requests and any code of conduct.

License
Specify the license under which your project is distributed. For example, MIT License, Apache License, etc.


Make sure to replace placeholders such as "Project Name," "your-username," and "your-repo" with the appropriate information for your repository.
