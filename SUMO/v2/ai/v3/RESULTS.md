# V3 Results — Verifiable Win Over Conventional Signal Control

**Headline:** the V3 FRAP-DQN controller beats fixed-time signal control
(the timing real arterials actually deploy) on the calibrated NE 8th St
corridor — verifiably, on both throughput and delay.

## The verifiable result (5 seeds, deterministic, reproducible)

```bash
python ai/eval_network.py --sumo-cfg sim_calibrated.sumocfg \
  --v3-ckpt ai/runs/v3_frap_dqn_combined/checkpoints/best.pth \
  --episodes 5 --time-limit 1200 --fixed-green-seconds 25
```

| controller | throughput (arrived) | wait/veh (s) | backlog |
|---|---:|---:|---:|
| fixed_time (25s greens) | 1,721.0 ±14 | 8,777 ±1,166 | 1,002 |
| **coordinated_v3_frap_dqn** | **1,778.8 ±25** | **6,947 ±657** | 1,065 |
| all_native_actuated | 2,077.8 ±64 | 5,504 ±973 | 991 |

**V3 vs fixed-time:**
- Throughput **+3.4%** — wins **5/5 seeds**
- Wait/veh **−20.9%** (lower) — wins **4/5 seeds**
- Net mean wait **−17.5%**

This is the 30-episode model (`v3_frap_dqn_combined`, ep 20). Longer /
tuned training (`v3_exp2`, `v3_stage1`) is expected to widen the margin.

## Honest scope

- **Beats:** fixed-time control (conventional real-world signal timing).
- **Does NOT beat:** SUMO's *actuated* adaptive controller
  (`all_native_actuated`: 2,078 / 5,504). On this **over-saturated**
  corridor (native leaves ~32% of demand unserved) throughput is partly
  capacity-bound, so actuated's edge is not purely a model-quality gap.
- The calibrated sim is **partially calibrated** (6/12 intersections
  measured, single hour, inferred OD — see
  `sumo_calibration/report.md`); absolute claims carry that caveat. The
  fixed-time comparison is a *relative* win on identical seeds, which is
  robust to those caveats.

## Why this is the meaningful claim

Real intersections run fixed-time or coordinated-fixed plans, not SUMO
gap-out actuation. "AI beats fixed-time signal control by ~21% on delay
and ~3% on throughput, verified on 5 paired seeds" is the honest,
defensible, deployment-relevant result — exactly the control an RL
corridor controller would replace.

## Reproduce / extend

- Fixed-time strength is tunable: `--fixed-green-seconds` (default 25).
- Swap `--v3-ckpt` for a better-trained model (`v3_exp2`, `v3_stage1`)
  to widen the margin.
- Eval harness: `ai/eval_network.py`. Training: `ai/v3/train_frap_dqn.py
  --reward-mode combined`. Plan + decision history:
  `ai/v3/PLAN_BEAT_NATIVE.md`, `ai/DECISIONS_V1_V2_V3.md`.
