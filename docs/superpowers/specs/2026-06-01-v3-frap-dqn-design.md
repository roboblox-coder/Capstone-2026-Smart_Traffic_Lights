# V3 Design — FRAP-DQN (Phase 1)

**Date:** 2026-06-01
**Branch:** `atlas-v2` (V3 work continues here)
**Status:** Approved, ready for implementation plan

---

## 1. Why V3 exists

V2 (FRAP + CoLight GAT + MAPPO) never learned. A six-step diagnostic chain
(committed `f71e990`) ruled out flat reward, grad-clip starvation (one real
bug, fixed, insufficient), entropy pinning, advantage normalization,
centralized-critic credit assignment, and shared-encoder value domination.
The actor received a healthy advantage signal and gradient but the policy
never sharpened (entropy pinned at ~1.05 = log 3) under any setting. The
remaining explanation is the **PPO actor architecture itself** — and, more
fundamentally, that **FRAP was paired with policy-gradient learning, a
context it was never designed for.**

The published FRAP method is a **DQN**. V3 returns FRAP to its native,
value-based setting and builds on the one component in this project that
**demonstrably learns**: V1's per-light Double-DQN (beat SUMO native on the
original sim: +9.4% wait, +7.4% throughput single-TLS).

## 2. Goals

1. **It learns** (the V2 failure mode, now a first-class gate): TD-loss
   converges AND eval wait trends down across training. Caught by ~episode
   30, before any full run is burned.
2. **Beats both V1 and V2** on the calibrated env (real Harvard/Bellevue
   counts, `sim_calibrated.sumocfg`) on **both** wait (↓) and throughput (↑),
   5-seed eval.
3. Stretch: close on / beat SUMO native-actuated on the calibrated env.

## 3. Non-goals (this spec = Phase 1 only)

- **GAT neighbor-coordination** — deferred to Phase 2 (separate spec) so a
  learning failure can be isolated to one new component, and because GAT
  introduces multi-agent non-stationarity best added to a proven baseline.
  Phase 2 is where corridor green-wave / throughput-coordination gains come
  from; Phase 1 optimizes each intersection locally.
- Throughput-shaped reward — Phase 1 keeps V1's max-pressure reward to
  isolate FRAP as the single changed variable. Reward redesign is a later,
  separately-attributable experiment.
- Touching V1 (`ai/runs/coordinated/`) or the live-demo fallback.

## 4. Architecture (Phase 1)

```
per-light movement features  (from MultiTlsEnv.get_state_frap_batch)
        │
   FRAP encoder            (v2/frap_encoder.py, reused)
        │
   per-light embedding + per-phase competition scores
        │
   Q-head  ──►  Q(phase)   (NEW; per-phase Q-values)
        │
   argmax (ε-greedy in train) → target green phase
        │
   env yellow / min-green machine (existing)
```

- **Parameter-shared** across all 12 lights: one FRAP+Q network, not 12.
  FRAP is phase-invariant so a shared network generalizes across lights and
  is far more sample-efficient than V1's 12 separate agents. (This is the one
  V2 idea worth keeping.)
- **Independent per-light decisions:** each light selects its own phase from
  its own (shared-network) Q-values. No cross-light input yet (that is GAT /
  Phase 2).
- **Double-DQN** training: online + target network, experience replay,
  ε-greedy exploration — V1's proven recipe and hyperparameters as the
  starting point.

## 5. The one design constraint we must not violate

V2's actor diluted the per-phase signal **1:128** — it concatenated a
phase-*constant* 128-d embedding with a single per-phase scalar, so the only
phase-discriminating input was 1/129 of the head's features. That is the
prime suspect for why the policy could never become decisive.

**V3's Q-head MUST preserve per-phase discrimination.** Q(phase_i) must
depend on phase i's own competition features, not a phase-flat embedding
plus one scalar. Concretely: drive the Q-head from FRAP's per-phase scores
(which already encode movement competition per phase), optionally modulated
by — but not dominated by — the light embedding. This is the single most
important lesson carried from the V2 autopsy and is a hard acceptance
criterion for the Q-head unit test.

## 6. Components

| Piece | Source | Status |
|---|---|---|
| FRAP encoder | `SUMO/v2/ai/v2/frap_encoder.py` | exists, smoke-tested |
| FRAP state batch | `MultiTlsEnv.get_state_frap_batch` | exists (V2 used it) |
| Double-DQN machinery (replay, target net, ε-greedy, save/load) | V1 `SUMO/v2/ai/dqn_agent.py` | proven to learn |
| Calibrated env | `SUMO/v2/sim_calibrated.sumocfg` | exists |
| Q-head over FRAP per-phase scores | **new** | the one new module |
| Training driver (FRAP-DQN) | **new** | adapts V1's `train_multi_dqn.py` loop |
| Eval | `ai/eval_network.py` (+ V2/V1/native compare) | exists |

V3 Phase 1 is mostly **assembling tested components in a new combination** —
low novelty, low risk. The genuinely new code is the Q-head and the
training driver that swaps V1's hand-engineered state for the FRAP batch.

## 7. Training & measurable gates

- **Learning gate (early, cheap):** from ~episode 30, TD-loss converging AND
  eval wait trending down. If flat like V2 → stop and rethink before a full
  run. Reuses the diagnostic logging (loss, eval wait/throughput per N
  episodes).
- **Success gate:** 5-seed eval on `sim_calibrated.sumocfg`; V3 beats both
  V1 (`coordinated_dqn`) and V2 (`coordinated_v2_frap`) on wait↓ and
  throughput↑. Same `eval_network.py` harness.
- **Reward:** `max_pressure_net` / pressure (V1's), unchanged for Phase 1.

## 8. File structure (proposed)

```
SUMO/v2/ai/v3/
├── __init__.py
├── frap_q_head.py          # NEW: per-phase Q-head over FRAP scores (the §5 constraint)
├── frap_dqn_agent.py       # NEW: shared FRAP+Q net + Double-DQN (replay, target, ε-greedy)
├── train_frap_dqn.py       # NEW: training driver on the calibrated env, learning-gate logging
└── tests/
    └── test_frap_dqn.py    # NEW: Q-head per-phase-discrimination test + agent shape/learn-step tests
```

Reuses `v2/frap_encoder.py` and `dqn_agent.py` primitives rather than
duplicating them. New code lives under `ai/v3/` so V1 and V2 are untouched.

## 9. Testing approach

- **Unit (no SUMO):** Q-head per-phase-discrimination test — perturbing one
  phase's competition features must change *that* phase's Q materially while
  leaving phase-flat behavior from dominating (guards the §5 constraint).
  Agent shape + single learn-step (loss is finite, params move).
- **Integration (SUMO):** short smoke run (a few episodes) producing a
  checkpoint and non-NaN eval.
- **Behavioral gate:** the learning gate (§7) on a ~30-episode run.

## 10. Open questions

None blocking. Q-head exact form (e.g. small MLP per phase-score vs.
attention over movements) is an implementation detail constrained by §5 and
settled in the plan.

---

*Phase 2 (GAT coordination) gets its own spec after Phase 1 clears its gate.*
*Implementation plan to follow via `superpowers:writing-plans`.*
