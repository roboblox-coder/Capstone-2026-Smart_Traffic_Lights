🚦 ATLAS AI Module – Quick Start Guide
This guide covers how to set up and use the AI traffic‑light controller developed by Ryan.
All necessary files are located in the v2/ai/ folder of the team repository.

🔧 Prerequisites
Python 3.10+

SUMO (either installed from sumo.dlr.de or via pip install eclipse-sumo).
If using the installer, set the environment variable SUMO_HOME (e.g., C:\Program Files (x86)\Eclipse\Sumo).

Virtual environment (recommended)

🛠️ Setup
Activate your virtual environment from the v2 folder:

bash
source .venv/Scripts/activate      # Windows Git Bash
# or .venv\Scripts\activate on cmd
Install required packages:

bash
pip install torch torchvision torchaudio matplotlib traci sumolib pygame
Verify SUMO connection:

bash
python -c "import traci; traci.start(['sumo', '-c', 'sim.sumocfg', '--no-step-log']); traci.close(); print('✅ OK')"
🎮 Training the AI
Option A – Train with SUMO (realistic)
Choose a traffic light ID
Open the network in SUMO‑GUI:

bash
sumo-gui -n NE_8th_St_Corridor.net.xml
Click on an intersection; the tlLogic ID appears in the bottom‑left panel.
Example IDs: 3153556582, cluster_1804700800_53129739, etc.

Edit ai/train_dqn_sumo.py – set your chosen ID and adjust training parameters:

python
TLS_ID = "3153556582"            # your chosen ID
USE_GUI = True                    # watch the simulation (optional)
EPISODES = 20                     # number of episodes
TIME_LIMIT = 1800                 # seconds per episode
Start training:

bash
python ai/train_dqn_sumo.py
The model is saved as ai/dqn_sumo.pth after training.

A plot of rewards and loss will be shown/saved.

Option B – Train on Synthetic Data (no SUMO, faster)
For quick DQN prototyping without SUMO, use the Bellevue dataset:

python
# train_bellevue.py (place in ai/ folder)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from traffic_base import BaseTrafficAI

# Load Bellevue data (adjust path as needed)
df = pd.read_csv("../../data/synthetic/bellevue_traffic.csv")
features = df[['Hour of Day', 'Is Peak Hour', 'Cycle Length Sec',
               'Green Split Ratio', 'Vehicles Per Hour', 'Saturation Degree']].values
targets = df['Est Wait Time Sec'].values

# Normalize
mean, std = features.mean(axis=0), features.std(axis=0) + 1e-8
features = (features - mean) / std

X = torch.FloatTensor(features)
y = torch.FloatTensor(targets).unsqueeze(1)

model = BaseTrafficAI(input_size=6, hidden_sizes=[64, 64], output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(50):
    perm = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        idx = perm[i:i+32]
        xb, yb = X[idx], y[idx]
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, loss = {loss.item():.4f}")

torch.save(model.state_dict(), "ai/bellevue_predictor.pth")
This trains a predictor for wait times – a good warm‑up before SUMO training.

👀 Visualizing a Trained Agent
After training (or even without a model), watch the AI control the intersection:

Update ai/visualize_sumo_ai.py – set the same TLS_ID as in training.

Run the visualizer:

bash
python ai/visualize_sumo_ai.py
PyGame window opens with vehicles moving.

If ai/dqn_sumo.pth exists, the agent uses it; otherwise random actions.

Top‑left shows step count, cumulative reward, and chosen action.

🔍 Key Performance Indicators (KPIs)
Track these metrics to evaluate the agent:

Average Wait Time (seconds) – lower is better.

Maximum Queue Length (vehicles) – lower is better.

Throughput (vehicles per hour) – higher is better.

You can log these by extending the environment or using the provided evaluation scripts (e.g., baseline_metrics.py, ai_metrics.py).

