# Quickstart — Run the V3 Traffic-Signal AI

A reinforcement-learning controller (FRAP Double-DQN) for the 12-light
NE 8th St corridor in Bellevue, trained on a SUMO simulation calibrated
from real Harvard/Bellevue traffic counts. **It verifiably beats
conventional fixed-time signal control.**

> Run everything from `SUMO/v2/`.

## 1. Setup (one time)

```bash
pip install -r requirements.txt
# SUMO must be on PATH or SUMO_HOME set. Verify everything imports:
python -c "import torch, numpy, traci, sumolib; print('ok')"
```

## 2. See it work in 2 minutes — the verifiable result

Run the AI against conventional fixed-time signals and SUMO's adaptive
controller, on 5 paired seeds:

```bash
python ai/eval_network.py --sumo-cfg sim_calibrated.sumocfg \
    --v3-ckpt ai/v3/model_best.pth \
    --episodes 5 --time-limit 1200 --fixed-green-seconds 25
```

You'll see a table. The headline (mean of 5 seeds):

| controller | throughput | wait/veh |
|---|---:|---:|
| fixed-time (conventional) | 1,721 | 8,777 |
| **V3 FRAP-DQN (this AI)** | **1,779** | **6,947** |
| SUMO actuated (strong baseline) | 2,078 | 5,504 |

**V3 beats conventional fixed-time by +3.4% throughput and −20.9% wait.**
(It does not beat SUMO's actuated baseline — see `ai/v3/RESULTS.md` for
why that's the saturated-corridor ceiling.)

## 3. See the AI's logs as charts — proof it's working

```bash
python ai/v3/make_v3_plots.py
# writes ai/v3/plots/comparison.png  + ai/v3/plots/learning_curve.png
```

- **`ai/v3/plots/comparison.png`** — the win vs fixed-time, as bars.
- **`ai/v3/plots/learning_curve.png`** — the agent's eval delay +
  throughput across training episodes (it learns; the spikes are the
  documented DQN instability).

Both PNGs are committed to the repo, so they also render directly on
GitHub under `SUMO/v2/ai/v3/plots/`.

## 4. Watch it drive traffic live (browser)

```bash
python run_websocket_ai.py          # starts SUMO + WebSocket server
# then open frontend/index.html in a browser
```

- Vehicles move in a 3D view; the 12 lights are driven by the AI.
- **Model dropdown (top bar):** switch the active controller —
  **V1 · DQN / V2 · MAPPO / V3 · FRAP-DQN** — *live, mid-simulation*.
- **AI Decisions panel (sidebar):** a live scrolling log of every phase
  change the active model makes (`step · light → green slot`), and a
  `⟳ switched to …` line when you change models.

All loaded models are selectable; the demo defaults to V3.

## 5. Train it yourself (~25 min for a quick model)

```bash
python ai/v3/train_frap_dqn.py \
    --episodes 30 --reward-mode combined \
    --eval-seeds 1042 1043 1044 1045 1046 \
    --out-dir ai/runs/my_v3

# then eval your model:
python ai/eval_network.py --sumo-cfg sim_calibrated.sumocfg \
    --v3-ckpt ai/runs/my_v3/checkpoints/best.pth --episodes 5 \
    --fixed-green-seconds 25
```

Key knobs (see `ai/v3/RESULTS.md` for what each was found to do):
`--reward-mode combined` (avoids the gridlock loophole), `--tau 0.005`
(Polyak soft target updates, reduces collapse), `--reward-beta` (wait vs
throughput balance), `--gamma`, `--lr`.

## 6. Verify the core logic without SUMO (~5 s)

```bash
python -m ai.v3.tests.test_frap_q   # per-phase Q discrimination + agent
```

## Where everything is

| | |
|---|---|
| Verifiable result + numbers | `ai/v3/RESULTS.md` |
| Full V1→V2→V3 decision history | `ai/DECISIONS_V1_V2_V3.md` |
| Plan to beat native | `ai/v3/PLAN_BEAT_NATIVE.md` |
| Charts (committed) | `ai/v3/plots/` |
| AI code | `ai/v3/` (`frap_q_net.py`, `frap_dqn_agent.py`, `train_frap_dqn.py`) |
| Eval harness | `ai/eval_network.py` |
