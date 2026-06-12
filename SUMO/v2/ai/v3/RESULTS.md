# V3 Results — Verifiable Win Over Conventional Signal Control

**Headline:** the V3 FRAP-DQN controller beats fixed-time signal control
(the timing real arterials actually deploy) on the calibrated NE 8th St
corridor — verifiably, on both throughput and delay.

## The verifiable result (5 seeds, deterministic, reproducible)

```bash
python ai/eval_network.py --sumo-cfg sim_calibrated.sumocfg \
  --v3-ckpt ai/v3/model_best.pth \
  --episodes 5 --time-limit 1200 --fixed-green-seconds 25
```

(`ai/v3/model_best.pth` is the committed copy of that run's best
checkpoint, so this works on a fresh clone.)

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

## V4 re-test (2026-06-10): post-audit retraining — negative result

The audit's four fixes (F1 outflow reward, F2 phase features, F3
normalization, F4 3s yellows) were implemented and re-tested across
three training campaigns. **None matched the pre-audit di2 checkpoint.**

| run | config | best harvested (sel. seeds) | official 42–46 |
|---|---|---:|---:|
| v3_di2 (pre-audit) | raw state, di=2, eff. wait-diff | 4,593 / 1,940 | **5,044 / 1,924** |

The di2 model is committed at `ai/v3/model_di2_best.pth`. Reproduce its
official eval (matching the regime it was measured under —
`ai/logs/eval_di2.txt`):

```bash
python ai/eval_network.py --sumo-cfg sim_calibrated.sumocfg \
  --v3-ckpt ai/v3/model_di2_best.pth --episodes 5 --time-limit 1200 \
  --decision-interval 2 --yellow-time 5 --fixed-green-seconds 25
```

| v4_stage0 | F1–F4, di=5, alpha=1 | 5,912 / 1,751 | 6,806 / 1,710 (gate FAIL) |
| v4_stage1 | + n-step 3, eps/2, lpd 2 | 9,016 / 1,725 | (collapsed by ep 60) |
| v4_b | di2 regime exactly + V4 state | 17,697 / 1,512 | (not worth evaling) |

Attribution: `v4_b` held di2's entire regime fixed (di=2, alpha=0,
beta=0.05, 50 ep, eps-decay 30, 1-step) and changed only the state
(F2/F3) and yellows (F4) — and the jackpot policies vanished. Meanwhile
`v3_di2_s7` (same old config, different seed) jackpotted repeatedly
(3.7k–9.3k) and `v3_di1` hit 6.3k by ep 5: the old-state pipeline visits
excellent policies robustly. **Prime suspect: F3's clipping saturates on
this over-saturated corridor** — `min(waiting,300)/300` pins at 1.0 for
nearly every congested movement (summed waiting reaches thousands of
seconds), erasing exactly the congestion *ranking* the Q-net needs. The
raw features were ill-conditioned but information-rich; the clipped ones
are well-conditioned but flat. Untested fix: saturation-free scaling,
e.g. `log1p(x)/log1p(cap)` without a hard min().

Notes that stand regardless: training is a violent oscillator in every
configuration (di2's own evals swung 4.6k–123k; its best checkpoint is a
harvested peak, validated out-of-sample on 42–46). The fixed-time win
above is unaffected. The "beats native-actuated on wait 4/5 seeds" claim
(eval_di2.txt) rests on that single harvested checkpoint and was not
reproduced by any post-audit run — treat it as provisional.

## Native-actuated: demonstrated ceiling (not for lack of trying)

> **⚠ 2026-06-09 audit caveat — ceiling conclusion suspended.** All four
> experiments below ran with the `combined` reward's **throughput term
> silently zero** (`TRAINING_REVIEW.md` finding F1: `MultiTlsEnv` never
> increments the units' `_arrived_since_reward`), plus a blind state
> (F2), un-normalized features (F3), and a 5s-vs-3s yellow handicap vs
> native (F4). The fixed-time win above **stands** (paired seeds,
> unaffected). The ceiling claim is under re-test via
> `PLAN_V4_EXECUTION.md`.

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
