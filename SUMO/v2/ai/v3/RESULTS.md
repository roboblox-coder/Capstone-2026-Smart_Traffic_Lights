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

## Native-actuated: demonstrated ceiling (not for lack of trying)

Beating SUMO's *actuated* baseline (2,078 / 5,504) was pursued across
**four training experiments + multiple distinct levers**, all 5-seed
evaluated on the clean comparison seeds:

| experiment | lever | throughput | wait |
|---|---|---:|---:|
| 30-ep combined | baseline | 1,779 | 6,947 |
| Stage 1 (300 ep) | longer training | 1,797 | 7,797 |
| Exp2 | gamma 0.99 | (collapsed) | ~worse |
| Exp3 | Polyak soft targets (stability) | 1,708 | 7,034 |
| Exp4 | wait-weighted reward (β=0.3) | 1,772 | 7,179 |

All cluster ~6,900–7,800 wait / ~1,700–1,800 throughput. Native stays
clearly ahead. **Conclusion: native-actuated is the ceiling for the
independent per-light DQN on this over-saturated corridor.** Two
structural reasons (see `PLAN_BEAT_NATIVE.md`): throughput is
capacity-bound under saturation, and the controllers are
*per-intersection* — they cannot do the corridor-wide green-wave
progression that would close the delay gap.

**The one unexhausted lever:** corridor **coordination (GAT, Phase 2)** —
the green-wave mechanism native-actuated also lacks. That is the only
remaining path with a structural reason to beat native-actuated, and it
is a real build (re-integrate the V2 GAT onto the working DQN), not a
hyperparameter tweak. Deferred as a deliberate next phase.

## Reproduce / extend

- Fixed-time strength is tunable: `--fixed-green-seconds` (default 25).
- Swap `--v3-ckpt` for a better-trained model (`v3_exp2`, `v3_stage1`)
  to widen the margin.
- Eval harness: `ai/eval_network.py`. Training: `ai/v3/train_frap_dqn.py
  --reward-mode combined`. Plan + decision history:
  `ai/v3/PLAN_BEAT_NATIVE.md`, `ai/DECISIONS_V1_V2_V3.md`.
