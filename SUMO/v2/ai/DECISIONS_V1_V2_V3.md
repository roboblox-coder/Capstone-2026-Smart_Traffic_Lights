# ATLAS Decision Log — V1 → V2 → V3

Why the corridor controller has been rebuilt twice, what each rebuild
proved, and where the effort stands against the real goal.

> **The goal (current):** beat SUMO native-actuated control by **≥5% on
> both throughput and wait/veh** on the **calibrated env** (built from the
> real Harvard/Bellevue counts, `sim_calibrated.sumocfg`). Native is
> per-intersection actuated control — a strong baseline.

---

## TL;DR

- **V1** (per-light Double-DQN) beat native on the *original synthetic*
  sim but **loses badly on the calibrated real-data env** (−40% throughput).
- **V2** (FRAP + CoLight-GAT + MAPPO) **never learned** — a six-hypothesis
  diagnostic chain ruled out every training-dynamics cause and one real
  bug; the policy never sharpened under any setting. Abandoned.
- **V3** (FRAP-DQN — FRAP returned to its *native* value-based setting)
  **learns**, and beats both V1 and V2 on the calibrated env.
- **The real wall was the reward, not the architecture.** The
  `max_pressure` reward has a gridlock loophole on heavy demand; switching
  to a throughput+wait reward removed it.
- **We have NOT beaten native on the real data yet.** V3 is the closest
  (−14% throughput / −26% wait, from a 30-episode run). Closing that gap
  to +5%/+5% is the active goal.

---

## V1 — the per-light DQN (shipped, but synthetic-only)

**Architecture:** 12 independent Double-DQN agents, one per light; small
MLP over a hand-engineered per-lane state + a 6-float neighbor summary;
`max_pressure` reward.

**Result:**
- Original synthetic sim: **+9.4% wait / +7.4% throughput** vs native
  (single-TLS); **+7% wait / tie throughput** corridor. A real win.
- Calibrated real-data env: **throughput 1,247 / wait 30,163** vs native
  **2,078 / 5,504** — a heavy loss.

**Lesson:** V1's win was real but *on synthetic demand*. The moment we
moved to calibrated counts, it fell apart. This is what motivated V2 — and,
in hindsight, the calibrated env is where every honest comparison must
happen.

---

## V2 — the coordinated bet that never learned

**Architecture (the literature-standard arterial stack):** FRAP
phase-encoder + CoLight GAT neighbor-attention + MAPPO (centralized critic,
decentralized actors), parameter-shared across all 12 lights.

**The bet:** a *coordinated* controller (learned attention + joint-value
baseline) should beat V1's independent learners and close the gap to native
on the corridor.

**What actually happened:** the actor **never learned**. Across every
configuration, the same signature: a healthy advantage signal, the actor
received gradient, but the policy never sharpened (entropy pinned at
~1.05 = log 3, ratio_dev decaying to ~0.01). Eval was seed-dependent and
worse than native.

### The six-hypothesis autopsy (all committed under `ai/v2/`)

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | Reward too flat (no signal) | raw advantage std | ❌ ruled out — adv_std ≈ 22, healthy |
| 2 | Grad-clip starves the actor | per-head grad-norm split | ✅ **real bug** — one global `clip_grad_norm_` let val_loss (critic_gn ~50) dominate the actor (~0.5), 70–200× imbalance. **Fixed (separate clips). Insufficient.** |
| 3 | Entropy regularizer pins the policy | `entropy_coef = 0` | ❌ ruled out — policy still flat with zero penalty |
| 4 | Pooled advantage-norm washes out per-light signal | per-light normalization | ❌ ruled out — no change |
| 5 | Centralized critic gives bad per-light credit | IndependentCritic (per-light values) | ❌ ruled out — no change |
| 6 | Shared encoder dominated by value gradient | detach critic input | ❌ ruled out — no change |

**Conclusion:** with training-dynamics causes exhausted and one real bug
fixed, the failure localizes to the **V2 actor architecture** itself — the
per-phase action signal was funneled through a single FRAP-prelogit scalar
diluted 1:128 against a phase-constant embedding — and, more fundamentally,
to pairing **FRAP with policy-gradient (PPO)**, a context FRAP was never
designed for.

**Plateau / GAT notes:** the GAT also never trained (unfreeze threshold was
mis-scaled ~10× beyond the achievable gradient-step budget; fixed, but moot
once the actor problem dominated). Two long retrains plateaued early
because the FRAP-only warmup policy degraded before GAT could engage.

---

## The pivot to V3 — FRAP belongs in a DQN

**The insight:** the *published* FRAP method (Zheng et al., CIKM 2019) is a
**DQN**. V2's novelty/mistake was bolting FRAP onto a PPO actor. V3 returns
FRAP to its native value-based setting and builds on the **one component in
this project that demonstrably learns** — V1's Double-DQN loop.

**Architecture (V3 Phase 1, `ai/v3/`):**
- Parameter-shared **FRAP → per-phase Q-head → Double-DQN**, one network
  across all 12 lights, independent per-light decisions.
- **Hard constraint carried from the V2 autopsy:** the Q-head must preserve
  per-phase discrimination — Q(phase) is computed from *that phase's own*
  FRAP embedding (unit-tested), never the diluted phase-constant path that
  killed the V2 actor.
- GAT coordination deferred to a Phase 2 (so any learning failure isolates
  to one new component — the discipline V2 lacked).

**Result:** it **learns** — TD-loss converges, and it produces a sane,
stable policy. First time in the whole effort.

---

## The real wall: the reward, not the architecture

V3's first learning-gate run (max_pressure reward) **gridlocked** — eval
wait climbed 17k → 54k and throughput collapsed to ~430 *as the policy
became greedy*. The agent was learning a policy **worse than random**.

**Root cause (concrete):** `max_pressure = -|incoming_queue −
outgoing_queue|`. Its maximum (zero penalty) is reached when in/out queues
are *equal* — which **includes balanced gridlock** (both huge, difference
0 → reward 0 → best possible). On light synthetic demand V1 never fell in;
on **heavy calibrated demand, balanced gridlock is reachable and
reward-optimal**, so a competent learner correctly learns to gridlock. This
also explains V1's calibrated failure — *it was the reward all along*, not
the learner.

**The fix:** the `combined` reward — `α·throughput + β·wait_reduction` —
directly rewards the deployment metrics and *punishes* gridlock (no
arrivals + rising wait = strongly negative). Re-running the gate:
throughput stabilized at ~1,700 (no collapse), TD-loss converged cleanly.

**Meta-lesson:** three architectures (V1 DQN, V2 PPO, V3 DQN) and many
training-dynamics fixes were tried while a reward loophole was the actual
constraint. **Verify the objective before blaming the learner.**

---

## Where we stand (5-seed eval, calibrated env)

| controller | throughput | wait/veh | vs native |
|---|---:|---:|---|
| **native actuated** | **2,078** | **5,504** | — (still best) |
| V3 FRAP-DQN (30 ep) | 1,779 | 6,947 | −14% thru / −26% wait |
| V2 MAPPO | 1,642 | 7,586 | −21% / −38% |
| V1 DQN | 1,247 | 30,163 | −40% / −5× |

**V3 beats V1 and V2 on both metrics** (and with ~2.7× lower variance —
more reliable). **It does not beat native** — that remains the open goal.

---

## The path to the goal (beat native +5% on both)

Native's structural weakness: it is **per-intersection — zero corridor
coordination.** A coordinated learner can exploit green-wave / platoon
progression that native fundamentally cannot. That is the legitimate route
to beating native on throughput (the harder metric). The risk: heavy
demand means the corridor is near capacity, so throughput gains are partly
physics-bound.

Staged program (cheapest, highest-confidence first):

1. **Stage 1 — train 10–15× longer** (300–450 ep). 30 ep is badly
   undertrained. Biggest cheap lever. *(in progress: `ai/runs/v3_stage1`)*
2. **Stage 2 — tune the reward** (α/β + a queue-clearing term) to target
   the throughput gap specifically.
3. **Stage 3 — coordination layer (GAT, Phase 2)** — the green-wave engine
   native lacks; the real throughput-beating push.
4. Stage 4 — action/decision-interval tuning if margins remain short.

**Decision checkpoint after Stage 1+2 (~1 day):** if V3 reaches roughly
−3% to +2% vs native, native is beatable and Stage 3 likely clears +5%; if
it barely moves off −14%, native is near-ceiling on this saturated corridor
and we have an honest best-achievable answer instead of chasing a wall.

---

*Artifacts: V2 investigation in `ai/v2/` (commits `20d716d`, `f71e990`);
V3 in `ai/v3/`; specs/plans in `docs/superpowers/`. Eval harness:
`ai/eval_network.py` (compares fixed / native / V1 / V2 / V3 on identical
seeds).*
