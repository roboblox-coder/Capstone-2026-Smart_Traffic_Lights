# SYNAPTIX — Project: ATLAS
### Automated Traffic Logic And Synchronization

Repository & workspace for Bellevue College Team 5's capstone project.

"Project: ATLAS" is our attempt to utilize an AI algorithm paired with a
system of cameras as a plan to improve the quality of traffic at
intersections for pedestrians, bicyclists, late-night drivers, and
emergency vehicles alike.

ATLAS is an affordable, privacy-first traffic management solution
designed to modernize urban intersections without the high costs of
industrial sensors or the security risks of mass data storage. By
utilizing low-cost camera systems and real-time AI, ATLAS eliminates
unnecessary idling for drivers and creates a safer environment for
pedestrians and emergency vehicles. Our "zero-retention" approach
ensures that while the system is highly intelligent, it remains
invisible to data breaches — processing what it sees in the moment and
discarding the footage instantly to protect citizen privacy.

**Live demo site:** https://synaptix-website-773415276572.us-west1.run.app/

## Features

- **Cost-efficient perception:** replaces expensive LiDAR and inductive
  loops with high-performance, budget-friendly cameras to lower
  installation barriers for any city.
- **Privacy-by-design (stateless processing):** operates entirely on
  real-time data streams with no video storage, so no personally
  identifiable information is ever at risk.
- **Dynamic AI sequencing:** a PyTorch-driven reinforcement-learning
  policy that prioritizes road users based on immediate demand.
- **Vulnerable road user safety:** real-time detection of bicyclists
  and pedestrians to adjust signal timings dynamically.

## Architecture

- **Hardware (low-cost vision):** standard-definition wide-angle
  cameras provide sufficient data for AI classification at a fraction
  of industrial sensor cost.
- **AI engine (PyTorch):** a parameter-shared FRAP Double-DQN controls
  all 12 signalized intersections of a calibrated digital twin of
  Bellevue's NE 8th St corridor (built from real intersection counts).
- **Simulation & logic (SUMO + Three.js):** SUMO (Simulation of Urban
  MObility) runs the corridor; a WebSocket bridge streams live vehicle
  and signal state into a Three.js 3D viewer in the browser.
- **Ephemeral data pipeline:** once the AI extracts numerical metadata
  (counts and positions), the raw video buffer is overwritten — nothing
  is written to disk or sent to a cloud.

## Results (simulation, verified)

Measured on the calibrated NE 8th St corridor (12 lights), 5 paired
random seeds, identical demand for every controller:

- **vs conventional fixed-time signals** (what most arterials run): the
  RL controller cuts average wait per vehicle by **~21–35%** and edges
  throughput (**+3–10%**), winning 4–5 of 5 seeds depending on model.
- **vs SUMO's adaptive (actuated) controller:** the best model
  (`SUMO/v2/ai/v3/model_di2_best.pth`) also shows **~8% lower wait
  (4/5 seeds)**; this single-checkpoint result is provisional pending
  re-validation (see the V4 re-test notes).
- Throughput remains capacity-bound on this over-saturated corridor —
  the win is delay, not volume.

Full tables, reproduction commands, and honest caveats:
[`SUMO/v2/ai/v3/RESULTS.md`](SUMO/v2/ai/v3/RESULTS.md).

## Quickstart

Prerequisites: **Python 3.10+** and **SUMO**
([download](https://sumo.dlr.de/docs/Downloads.php)). Set `SUMO_HOME`
to the SUMO install folder (e.g. `C:\Program Files (x86)\Eclipse\Sumo`)
and add `%SUMO_HOME%\bin` to `PATH`. If `SUMO_HOME` is unset, the
scripts try to auto-detect it from the `sumolib` package.

```bash
git clone https://github.com/DukeSchnepf/AI-Traffic.git
cd AI-Traffic/SUMO/v2

python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
pip install traci sumolib websockets # base demo
pip install torch numpy              # add these for the AI

# Base demo (no AI): SUMO + WebSocket bridge
python run_websocket_sim.py
# then open frontend/index.html in a browser

# AI-controlled demo and training/eval workflows:
# see RUN_AI.md in this folder
```

The interactive launcher `python run.py` offers more modes
(Harvard-data routes, synthetic traffic, headless CSV export, TraCI).

Evaluate the AI against the baselines (fixed-time, SUMO actuated) on
identical seeds:

```bash
python ai/eval_network.py --sumo-cfg sim_calibrated.sumocfg \
  --v3-ckpt ai/v3/model_di2_best.pth --episodes 5 --time-limit 1200 \
  --decision-interval 2 --yellow-time 5 --fixed-green-seconds 25
```

## Repository layout

```
.
├── README.md
├── docs/
│   ├── design/      # design specs + implementation plans (V2, V3)
│   └── archive/     # early prototypes and superseded plans
└── SUMO/
    └── v2/
        ├── run.py                    # interactive launcher
        ├── run_websocket_sim.py      # live demo, SUMO-native signals
        ├── run_websocket_ai.py       # live demo, AI-controlled signals
        ├── RUN_AI.md                 # AI quickstart
        ├── sim.sumocfg / sim_calibrated.sumocfg
        ├── NE_8th_St_Corridor.net.xml
        ├── Real_intersection_data/   # source traffic counts
        ├── sumo_calibration/         # demand calibration + report
        ├── frontend/index.html       # Three.js 3D viewer
        └── ai/
            ├── sumo_env.py           # single-light RL environment
            ├── multi_env.py          # 12-light corridor environment
            ├── eval_network.py       # AI vs baselines, paired seeds
            ├── v2/                   # MAPPO/GAT line (superseded)
            └── v3/                   # FRAP-DQN line (current)
                ├── train_frap_dqn.py
                ├── model_best.pth        # committed eval default
                ├── model_di2_best.pth    # production best
                └── RESULTS.md            # verified numbers + caveats
```

Key documents: results & claims
([RESULTS.md](SUMO/v2/ai/v3/RESULTS.md)), decision history
([DECISIONS_V1_V2_V3.md](SUMO/v2/ai/DECISIONS_V1_V2_V3.md)), training
audit ([TRAINING_REVIEW.md](SUMO/v2/ai/TRAINING_REVIEW.md)), current
status & next steps ([HANDOFF.md](SUMO/v2/ai/HANDOFF.md)).

## Team

Duke Schnepf, Cory Maccini, Nikolaus Henkel, Ryan Liang —
supervised by Dr. Sara Farag.
