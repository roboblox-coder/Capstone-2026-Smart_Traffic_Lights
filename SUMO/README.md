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
