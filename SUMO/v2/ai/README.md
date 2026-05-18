# AI traffic-light control

Double-DQN agent that controls one signalised intersection (TLS
`3153556582`) in the NE 8th Street corridor SUMO scenario.

**Headline result:** the `max_pressure`-reward model beats SUMO's
built-in actuated controller on **both** objectives — **+7.4% throughput
(5/5 seeds) and +9.4% lower wait/vehicle (3/5 seeds)** over a 5-episode,
paired-seed evaluation. See `logs/eval_ablation.txt`.

## Setup

Requires SUMO installed and on `PATH` (or `SUMO_HOME` set). On Windows
the pip `sumo` package lives at `<python>/Lib/site-packages/sumo` — point
`SUMO_HOME` there if SUMO isn't found.

```bash
pip install -r ../../requirements.txt   # from SUMO/v2: torch, numpy, traci, sumolib, websockets, matplotlib
python -c "import torch, numpy, traci, sumolib, websockets; print('ok')"
```

Run everything from `SUMO/v2` so relative paths (`sim.sumocfg`, `ai/...`)
resolve.

## Layout

```
ai/
├── dqn_agent.py        # Double-DQN: online/target net, replay, save/load
├── sumo_env.py         # TraCI env: regime + reward modes (see below)
├── train_dqn_sumo.py   # training driver (per-episode seed, CSV log, best/last)
├── eval.py             # DQN(s) vs fixed / actuated / native_actuated baselines
├── make_plots.py       # render PNGs from a run's logs/train_log.csv
├── runs/<reward>/       # one trained variant per reward mode
│   ├── checkpoints/best.pth   # committed deliberately — the deployable model
│   └── logs/train_log.csv
└── logs/                # eval result + training console logs
```

## Reproduce the headline result

```bash
cd SUMO/v2

# Re-run the 4-way reward ablation vs all baselines (5 episodes, ~25 min):
python ai/eval.py --episodes 5 --time-limit 1200 --decision-interval 5 \
  --models ai/runs/max_pressure/checkpoints/best.pth \
           ai/runs/combined/checkpoints/best.pth \
           ai/runs/differential/checkpoints/best.pth \
           ai/runs/anti_starve/checkpoints/best.pth \
  --out ai/logs/eval_ablation.txt

# Live demo with the winning model in the loop:
python run_websocket_ai.py
# then open SUMO/v2/frontend/index.html — the top-bar shows AI status.
```

`run_websocket_ai.py` is pre-wired to
`ai/runs/max_pressure/checkpoints/best.pth` with the matching regime
(min_green=5, yellow=5, decision_interval=5). **The regime constants must
match the model's training regime or the live result won't reproduce.**

## Retrain a variant

```bash
# 100 episodes, ~30 min CPU. --reward-mode is the key knob.
python ai/train_dqn_sumo.py --episodes 100 --time-limit 1200 \
  --reward-mode max_pressure --out-dir ai/runs/max_pressure
python ai/make_plots.py   # reads ai/logs/train_log.csv from the last run
```

## How the agent decides

* **State** — per controlled lane: halting count, waiting time, mean
  speed (normalised); plus a one-hot of the current green slot and
  normalised time-in-phase. (29 features for this TLS.)
* **Action** — a target green-phase slot. The env inserts a 5 s yellow,
  enforces a 5 s minimum green, and holds each decision for
  `decision_interval` (5 s) before re-deciding — matching SUMO native
  actuated's decision cadence.
* **Reward** (`--reward-mode`):
  * `max_pressure` *(best)* — `-|queue_in - queue_out| / n`. The
    literature-standard; the normalised signal also keeps DQN loss
    stable (~0.5 vs 28–130 for `differential`).
  * `combined` — `α·arrived − β·Δwait`. Over-weights throughput and
    regressed badly; kept only as a documented negative result.
  * `differential` — `Δwait − switch_penalty` (the original).
  * `anti_starve` — `differential − starve_penalty·max_lane_wait`.
  * `waiting` — `-total_wait`.

## Baselines in `eval.py`

* `fixed` — round-robin cycling.
* `actuated` — greedy max-`waiting-time` slot (corrected link→lane
  mapping; min-green provides hysteresis).
* `native_actuated` — **the bar to beat**: SUMO's own actuated program
  from the `.net.xml`, untouched. The verdict judges each DQN against
  this on both wait and throughput, per-seed.

## Known next levers (not yet done)

The `max_pressure` win is real but its wait-time gain is not yet robust
(3/5 seeds). The reward depends on downstream queue, but the agent's
*state* only sees incoming lanes — adding downstream occupancy to
`sumo_env.get_state()` is the most aligned next improvement, followed by
a structured (cyclic) action space, before scaling to all 12 TLSs.

## Frontend status badge

The top-bar `AI:` badge shows the agent's state each step:

* `active` — agent in control.
* `manual_override` — user clicked a phase button; agent paused until a
  `resumeAI` command.
* `fallback:no_model` / `:tls_missing` / `:state_size_mismatch` — no
  usable checkpoint, so SUMO's actuated controllers stay in charge.
