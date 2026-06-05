# Smart Traffic Lights — Run the AI

One Double-DQN agent per traffic light, all 12 corridor lights driven from a single SUMO process. The trained model lives in `ai/runs/coordinated/`, so you can evaluate or watch the live demo without retraining.

> Run everything from `SUMO/v2/`.

## 1. Setup (one time)

```bash
pip install -r requirements.txt
# SUMO must be on PATH, or set SUMO_HOME.
python -c "import torch, numpy, traci, sumolib, websockets; print('ok')"
```

## 2. Run

```bash
# Live demo — AI drives all 12 lights; then open frontend/index.html.
python run_websocket_ai.py

# Evaluate the trained model on 5 seeds vs SUMO's native actuated baseline:
python ai/eval_network.py --episodes 5 --time-limit 1200

# (Optional) Reproduce the committed model from scratch:
python ai/train_multi_dqn.py --episodes 60 --time-limit 1200
```

## Current result vs SUMO native-actuated

5 seeds × 1200 s, whole 12-light corridor:

| Metric | DQN | Native | Improvement |
|---|---|---|---|
| Throughput (vehicles arrived) | 1699.8 ±86 | 1711.8 ±22 | **−0.7%** (tie — wins 3/5 seeds) |
| Wait per vehicle (s) | 9,670 ±808 | 10,401 ±507 | **+7.0% lower** (wins 4/5 seeds) |

**Net: AI matches native throughput while cutting average network delay ~7%, stably.**

Regression anchor (same recipe on the target intersection alone, single-TLS): **+9.4% lower wait, +7.4% higher throughput** vs native.

## Training provenance

This result comes from **one 60-episode training run** (`ai/train_multi_dqn.py`, defaults). Additional runs were done during development for ablations and to test richer coordination shaping; that machinery is disabled by default because it was found to degrade results (see `ai/train_multi_dqn.py` docstring).

## Estimated ceiling with more training

Under the current design (independent per-light learners + local max-pressure reward), gains taper quickly. A realistic projection from doubling training (~120 episodes) plus multi-seed averaging:

- Throughput: **~parity to +1–2%** vs native (architecturally near the ceiling).
- Wait per vehicle: **~+9–11% lower** vs native.

Throughput is *bottlenecked* at native parity by the local-only credit structure — meaningfully beating native on throughput requires a different per-agent credit-assignment design, not more compute.

---

## V2 (in progress on this branch)

FRAP encoder + CoLight graph attention + MAPPO (centralized critic,
decentralized actors). Closes the architectural ceiling V1 hit:
parameter-shared across all 12 lights, learned attention over
neighbours, joint-value baseline for cross-light credit assignment.
Full plan in `ai/PLAN_V2.md`; calibration provenance in
`ai/sumo_calibration/report.md`.

### Bootstrap order

```bash
# (1) Phase 0: pin V1's headline numbers so any V2 work has a regression net.
python ai/regression_test.py --write-baseline --n-seeds 10
python ai/regression_test.py                        # exits 0 iff V1 still wins

# (2) Phase 1.1: calibrate the sim env from real Bellevue counts.
python -m ai.sumo_calibration.build_calibrated_routes
python -m ai.sumo_calibration.calibrate_carfollow
python "$SUMO_HOME/tools/randomTrips.py" \
    -n NE_8th_St_Corridor.net.xml \
    -r route_pool_calibrated.rou.xml --fringe-factor 100
python "$SUMO_HOME/tools/routeSampler.py" \
    -r route_pool_calibrated.rou.xml \
    --edgedata-files real_world_counts_calibrated.xml \
    -o harvard_simulation_calibrated.rou.xml
python ai/regression_test.py --sumo-cfg sim_calibrated.sumocfg
# Phase 1.1 sanity gate: V1 beats native by >=5pp on wait on the
# calibrated env (GEH histogram reported, not gated -- see report.md R3).

# (3) Phase 1.2: train V2 on the calibrated sim. ~10 GPU-hours / 1500 ep.
python ai/v2/mappo_trainer.py \
    --sumo-cfg sim_calibrated.sumocfg \
    --episodes 1500 --time-limit 1200 \
    --rollout-episodes 6 \
    --out-dir ai/runs/v2_mappo

# (4) Phase 1.3: optional DR training for sim-to-real robustness.
python ai/v2/mappo_trainer.py \
    --sumo-cfg sim_calibrated.sumocfg --randomize \
    --episodes 1500 --out-dir ai/runs/v2_mappo_dr
```

### Comparing V1 and V2

```bash
# V2 spec is appended to the eval automatically when the checkpoint exists.
python ai/eval_network.py \
    --sumo-cfg sim_calibrated.sumocfg \
    --v2-ckpt ai/runs/v2_mappo/checkpoints/best.pth \
    --episodes 10
```

Acceptance gates (per `ai/PLAN_V2.md` §1.2):

- **Honest gate (relative, statistically credible):** paired-seed
  95% CI excludes 0 with mean improvement ≥ 12pp on wait / ≥ 6pp on
  throughput vs V1, and 8/10 seeds individually beat V1.
- **Stretch (literature-edge):** ≥ −25% wait / ≥ +12% throughput vs
  native actuated. Reported with "pending validation against city
  counts" caveat until P0/P1 city data lands.

### V2 in the live demo

No extra step. `run_websocket_ai.py` checks for
`ai/runs/v2_mappo/checkpoints/best.pth` at startup; if present, all 12
lights are driven by the corridor-level V2 policy and the frontend
shows `aiSummary.mode = "v2"`. If absent, the demo runs V1 exactly as
before.

### Verifying the V2 stack without SUMO

```bash
python -m ai.v2.tests.test_smoke   # 7 tests, requires only torch + numpy
```

Knowledge-graph view of the whole codebase (V1 + V2) is in
`graphify-out/` — open `graph.html` in a browser.

---

## V3 (current) — FRAP-DQN: the verifiable win

V2's MAPPO actor never learned (six-hypothesis autopsy in `ai/v2/`). V3
returns FRAP to its native value-based setting on V1's proven Double-DQN
backbone, and **it learns**. Full story: `ai/DECISIONS_V1_V2_V3.md`.

**Result (5-seed, calibrated env, reproducible):** V3 beats conventional
**fixed-time** signal control — the timing real intersections deploy — by
**+3.4% throughput / −20.9% wait**. It does not beat SUMO's *actuated*
baseline (the saturated-corridor ceiling; see `ai/v3/RESULTS.md`).

```bash
# Train (the reward matters: 'combined' avoids max_pressure's gridlock loophole)
python ai/v3/train_frap_dqn.py --episodes 150 --reward-mode combined \
    --tau 0.005 --eval-seeds 1042 1043 1044 1045 1046

# Verifiable eval: V3 vs fixed-time vs native, 5 seeds
python ai/eval_network.py --sumo-cfg sim_calibrated.sumocfg \
    --v3-ckpt ai/runs/v3_frap_dqn_combined/checkpoints/best.pth \
    --episodes 5 --fixed-green-seconds 25

# Unit tests (no SUMO): per-phase Q discrimination + agent
python -m ai.v3.tests.test_frap_q
```

V3 docs: `ai/v3/RESULTS.md` (numbers), `ai/v3/PLAN_BEAT_NATIVE.md` (plan),
`ai/DECISIONS_V1_V2_V3.md` (full V1→V2→V3 decision history).
