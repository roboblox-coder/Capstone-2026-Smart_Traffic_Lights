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

---

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
    ├── sim.sumocfg           # SUMO config file
    ├── NE_8th_St_Corridor.net.xml  # Road network
    ├── harvard_simulation.rou.xml  # Routes from Harvard data
    ├── synthetic.rou.xml     # Synthetic routes
    └── Real_intersection_data/     # Harvard Excel source files
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
