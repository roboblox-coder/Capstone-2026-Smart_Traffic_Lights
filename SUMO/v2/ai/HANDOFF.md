# AI pipeline — what's left

The Double-DQN controller is wired into the live WebSocket simulation
and the frontend AI badge updates correctly. The smoke test passes,
a short training run produces a `best.pth` checkpoint, and the
frontend reports `AI: active slot=N Δ=…` in green over a live socket.

The remaining work is the long training run and the post-training
analysis. Deadline is **Wednesday**.

---

## What's still to do

### 1. Real training run (≥ 20 episodes)

The 5-episode smoke run is enough to prove the pipeline; it is *not*
enough for the agent to learn anything useful. At 5 episodes the
policy collapses to a single green slot and `Δ` stops incrementing.

```bash
cd SUMO/v2
python ai/train_dqn_sumo.py --episodes 30 --time-limit 1200
```

Roughly ~25–35 minutes wall-clock on CPU. Watch the `wait_mean`
column shrink across episodes — that's the signal the agent is
learning. Reward can look noisy because of the switch penalty;
trust waiting time, not reward, for the trend.

### 2. Plot the curves

```bash
python ai/make_plots.py
```

Writes PNGs into `ai/logs/`. Sanity check: `wait_mean` trends down
and `loss` stays bounded.

### 3. Head-to-head vs baselines (optional)

```bash
python ai/eval.py --episodes 3
```

Compares the trained DQN vs fixed-cycle vs greedy-actuated on the
same intersection. Useful if you want a number to put in slides.

### 4. Live verification, end to end

After the long training run completes:

```bash
python run_websocket_ai.py
# then open SUMO/v2/frontend/index.html in a browser
```

The browser should show:
- `● Connected` badge (green)
- `AI: active slot=N Δ=…` badge with `Δ` **incrementing** over time
  (the trained policy actually switches phases now)
- vehicles moving in the 3D viewer

---

## Run order (full reference)

```bash
cd SUMO/v2

# Environment (Windows note: set SUMO_HOME if running outside this shell;
# the pip-installed `sumo` package lives at
# <python>/Lib/site-packages/sumo — that's what SUMO_HOME should point at).
python -c "import torch, numpy, traci, sumolib, websockets; print('ok')"

# 0. smoke test (≈ 5 SUMO-seconds of agent activity)
python ai/sanity_check.py

# 1. real training
python ai/train_dqn_sumo.py --episodes 30 --time-limit 1200

# 2. plots
python ai/make_plots.py

# 3. live run
python run_websocket_ai.py
# open SUMO/v2/frontend/index.html

# 4. optional baseline comparison
python ai/eval.py --episodes 3
```

---

## File map

```
SUMO/
├── .gitignore                          last.pth and logs/ are intentionally ignored
└── v2/
    ├── websocket_server.py             threaded WS server, ASCII-safe prints
    ├── run_websocket_ai.py             live runner — loads best.pth, broadcasts ai{...}
    ├── frontend/index.html             AI badge reads msg.ai
    └── ai/
        ├── __init__.py
        ├── README.md                   user-facing quickstart
        ├── HANDOFF.md                  this file
        ├── dqn_agent.py                Double-DQN: online/target net, replay, save/load
        ├── eval.py                     DQN vs fixed-cycle vs actuated
        ├── make_plots.py               reward / wait / loss / epsilon curves
        ├── sanity_check.py             60-step end-to-end smoke test
        ├── sumo_env.py                 TraCI env: yellow + min-green enforced internally
        ├── train_dqn_sumo.py           training driver, writes best.pth + last.pth + CSV
        └── traffic_base.py             MLP definition shared by online/target — do not edit
```

Target intersection: `TLS_ID = "3153556582"` (set in both
`train_dqn_sumo.py` defaults and at the top of `run_websocket_ai.py`).
It has 8 phases: 4 green slots (0, 2, 4, 6) and 4 yellows.

---

## Known risks

1. **Light control uses `setRedYellowGreenState`, never `setPhase`.**
   This is deliberate — once SUMO sees an explicit state string it
   switches the TLS into a one-phase override program where
   `setPhase(tls, 6)` raises *phase index 6 is not in the allowed
   range [0,0]*. Don't "simplify" the env back to `setPhase`.
2. **TraCI labels are load-bearing.** `sumo_env.py` uses labelled
   connections so the probe and the training session don't share one
   global TraCI. If you see *unexpected keyword 'label'*, the SUMO
   build is too old — upgrade it; do not strip the labels.
3. **TLS id might not match the active network.** The default
   `3153556582` is the right one for `NE_8th_St_Corridor.net.xml`.
   If you regenerate the network, list the new IDs:
   ```bash
   python -c "import traci, sumolib; traci.start([sumolib.checkBinary('sumo'), '-c','sim.sumocfg','--start','--quit-on-end']); print(traci.trafficlight.getIDList()); traci.close()"
   ```
   …then pass `--tls-id ...` to the trainer and update the `TLS_ID`
   constant at the top of `run_websocket_ai.py`.
4. **State-size / action-size drift.** If the checkpoint was trained
   on a different intersection layout, `run_websocket_ai.py` detects
   the mismatch and falls back to actuated. The frontend badge then
   reads `AI: fallback:state_size_mismatch`. Retrain against the same
   TLS rather than patching the loader.
5. **`differential` reward looks bad early.** Episode 1 reward can be
   near zero or negative because the switch penalty dominates before
   the agent has learnt anything. Watch `wait_mean` (it should drop)
   for the actual signal.
6. **Frontend WebSocket URL is hardcoded** to `ws://localhost:8765`.
   If you change `WS_PORT` in `run_websocket_ai.py`, also update
   `connectWebSocket()` in `frontend/index.html` (around line 1238).
7. **Windows console encoding.** Some other scripts in `SUMO/v2/`
   (`run.py`, `run_websocket_sim.py`, etc.) still print emoji and
   will crash on cp1252 Windows shells. The AI runner does not.
   If you ever wrap them, set `PYTHONIOENCODING=utf-8` or strip the
   emoji.

---

## Out of scope

- `SUMO/v2/run.py`, `run_websocket_sim.py`, `run_traci_sim.py`,
  `run_auto_sim.py`, `real_life_simulation.py`, and anything in `v1/`.
- `ai/traffic_base.py` — `dqn_agent.py` depends on its current shape.
