# AI traffic-light control

Double-DQN agent that controls one signalised intersection in the
NE 8th Street corridor SUMO scenario.

## Layout

```
ai/
├── __init__.py
├── traffic_base.py     # plain MLP, used by DQNAgent
├── dqn_agent.py        # DQNAgent + ReplayBuffer (target net, Double-DQN update)
├── sumo_env.py         # TraCI env (yellow transitions, min-green, richer state)
├── train_dqn_sumo.py   # training driver (eps decay, CSV log, best/last ckpt)
├── eval.py             # compare DQN vs fixed vs actuated baselines
├── sanity_check.py     # 60-step end-to-end smoke test
├── make_plots.py       # render PNGs from logs/train_log.csv
├── checkpoints/        # best.pth / last.pth (created on first train run)
└── logs/               # train_log.csv + training_curve.png
```

## Quick start

Run everything from `SUMO/v2` so relative paths (`sim.sumocfg`, `ai/...`) resolve.

```bash
# 1. Sanity-check the pipeline (~30 SUMO seconds, no checkpoint written)
python ai/sanity_check.py

# 2. Train a small model (~10–20 min on CPU for 30 episodes)
python ai/train_dqn_sumo.py --episodes 30 --time-limit 1200

# 3. Plot training curves
python ai/make_plots.py

# 4. Compare against baselines
python ai/eval.py --episodes 3

# 5. Run the live simulation with the agent in the loop
python run_websocket_ai.py
# Open SUMO/v2/frontend/index.html — the top-bar shows AI status.
```

## How the agent decides

* **State** — for each controlled lane: halting count, waiting time, mean
  speed (normalised). Plus a one-hot of the current green slot and time
  spent in that phase.
* **Action** — pick a target *green-phase slot*. The env automatically
  inserts a 4-second yellow transition and enforces a 10-second minimum
  green, so the policy can't flicker the light.
* **Reward** — differential waiting time:
  `r_t = (sum_wait_{t-1} - sum_wait_t) - 0.1 * switch_penalty`.

## Frontend status badge

The top-bar `AI:` badge shows the agent's state each step:

* `active` — agent in control.
* `manual_override` — user clicked a phase button; agent is paused for
  this intersection until a `resumeAI` command is sent.
* `fallback:no_model` / `:tls_missing` / `:state_size_mismatch` —
  no usable checkpoint, so SUMO's actuated controllers stay in charge.
