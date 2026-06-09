# Training Review — Path to Beating Native-Actuated

Independent code-level audit of the V3 FRAP-DQN training stack
(`ai/v3/`, `ai/multi_env.py`, `ai/sumo_env.py`, `ai/eval_network.py`),
done against the stated goal: **beat SUMO native-actuated on the
calibrated corridor (≥5% on both throughput and wait/veh).**

**Headline: the "native-actuated is the ceiling" conclusion is
premature.** Every V3 experiment in `RESULTS.md` was run with a broken
reward (finding F1: the throughput term is silently zero), a state the
agent cannot act optimally from (F2: no current-phase / time-in-phase
input), un-normalized features (F3), and a built-in yellow-time handicap
vs the native baseline (F4). Fix these four and re-run before concluding
anything about ceilings.

---

## F1 (BUG, critical): the `combined` reward's throughput term is dead in multi-light training

`sumo_env.py:548` — `arrived_term = float(self._arrived_since_reward)`.

`_arrived_since_reward` is only incremented in
`SumoTrafficEnv._sim_tick()` (`sumo_env.py:397`). But in multi-light
training, `MultiTlsEnv` owns the clock: it calls its **own** `_tick()`
(`multi_env.py:170`), never the units' `_sim_tick()`, and never calls
`u.reset()`. So every unit's `_arrived_since_reward` stays at its
`__init__` value of `0` for the entire run.

**Consequence:** all V3 training (`train_frap_dqn.py` uses
`MultiTlsEnv`) actually optimized

```
r = beta * (prev_wait - wait)/n_lanes - switch_penalty      # beta=0.05
```

— a pure wait-differential reward. `--reward-alpha` did nothing.
Exp4's "wait-weighted reward (β=0.3)" merely rescaled the wait term
against the fixed 0.1 switch penalty. The agent was **never rewarded
for throughput at all**, which is consistent with V3 clustering at
~1,700–1,800 arrived while native does 2,078.

(Why it still fixed the gridlock loophole: wait-differential alone does
punish gridlock. The diagnosis in `DECISIONS_V1_V2_V3.md` was right
about max_pressure; it just never actually tested the intended fix.)

**Fix options (best first):**
1. **Per-light local outflow** (correct credit assignment): each
   decision, count vehicles that left this TLS's incoming lanes —
   diff the `lane.getLastStepVehicleIDs()` sets across the decision
   interval, or place virtual induction counts at stop lines. Reward
   `alpha * vehicles_served`. This is action-attributable, unlike
   network arrivals split 12 ways.
2. Minimal patch: have `MultiTlsEnv._tick()` push arrivals into every
   unit (`u._arrived_since_reward += n_arr / n_tls`) the way
   `set_shared_arrived` already does for `max_pressure_net`. Works, but
   gives every light the same global signal — noisy credit.

After fixing, re-tune `alpha/beta`: vehicles-served per 5s interval is
O(0–10) per light, wait term is O(±10) — start near alpha=0.5, beta=0.05
and sweep.

## F2 (state deficiency, critical): the agent cannot see the current phase or time-in-phase

`get_state_frap()` (`sumo_env.py:291`) returns per-movement
`(halting, vehicles, waiting)` — **no green-bit, no time_in_phase**.
The dict does carry `current_slot` / `time_in_phase`, and the batch
exposes them, but `train_frap_dqn.py:119` and `eval_network.py:227`
only feed `movement_features / phase_movement_mask / phase_mask` to the
net. `FRAPQNet` consumes nothing else.

The published FRAP state (Zheng et al., CIKM 2019) **includes the
current signal phase per movement**. Without it:
- The Q-function cannot distinguish "keep current green" (free) from
  "switch" (costs 5s yellow + switch_penalty). Both actions look
  identical in the input, so Q can only learn the *average* cost.
- It cannot represent min-green gating or anti-thrash timing.
- V1's flat state had both (one-hot slot + time_in_phase,
  `sumo_env.py:283-286`); the FRAP migration dropped them.

**Fix:** append two features per movement — `is_currently_green` (from
the active phase's state string) and normalized `time_in_phase` —
making `mov_feat_dim=5`. Cheap, no architecture change.

## F3 (conditioning): FRAP inputs are un-normalized raw counts

V1's `get_state()` normalized (`queue/20, waiting/60, speed/15`).
`get_state_frap()` feeds **raw** values: under saturation, summed
per-movement `waiting` reaches hundreds–thousands of seconds while
`halting` is O(10). One input dimension dominates the shared
`movement_mlp` by 2 orders of magnitude; DQN targets also drift with
congestion scale.

**Fix:** normalize at the source, e.g. `halting/20`, `vehicles/40`,
`min(waiting, 300)/300`. Re-train from scratch after (checkpoint
incompatibility is fine — checkpoints are cheap here).

## F4 (unfair baseline comparison): the AI pays 5s yellows; native pays 3s at several lights

`eval_network.py` / training default `yellow_time=5` for every AI
switch. The native programs in `NE_8th_St_Corridor.net.xml` use **3s**
yellows at `53141735`, `53254289`, `7953710843` (and 5s at the bigger
junctions). The AI is handicapped ~2s of dead time per cycle at those
lights relative to the baseline it's judged against.

**Fix:** read each TLS's own yellow duration from its native program
and use it in `MultiTlsEnv.step()` (or at minimum run with
`--yellow-time 3`, matching the corridor majority). Free throughput.

## F5: coordination signal exists but V3 ignores it

`MultiTlsEnv` computes a 6-float upstream/downstream block every
decision (`_neighbor_triplet`: queue, pressure, green-progress) and
pushes it into every unit — V1's flat state consumed it; **V3's FRAP
state does not include it.** So V3 is strictly per-intersection, yet
the docs' own analysis says coordination is the structural lever vs
native-actuated.

**Cheap Phase-2a (before the full GAT):** concatenate the 6-float
neighbor block (plus the F2 phase features) onto each phase embedding
before the Q-head (`FRAPQNet.q_head` input becomes
`embed_dim + 6 + …`). This gives the Q-function green-progress of its
corridor neighbors — the raw ingredient of a learned green wave — for
~10 lines of code. Full GAT (Phase 2 proper) stays the follow-up.

## F6: checkpoint selection is wait-only and 3-seeded

`train_frap_dqn.py:153` saves `best.pth` on **eval wait alone**, over
the default `--eval-seeds 1042 1043 1044` (3 seeds — the docs
themselves concluded 3-seed selection gives false positives, but the
default was never changed). A wait-only criterion can select a
throughput-poor policy, which directly fights the goal metric.

**Fix:** select on a combined score consistent with the goal, e.g.
`score = arrived/native_arrived - wait/native_wait` (or just
`arrived - lambda*wait`), over 5 seeds.

## F7: training-budget and stability levers (why long runs got worse)

- **Gradient budget is tiny.** One `learn()` per decision ⇒ ~120–240
  gradient steps/episode ⇒ ~5k steps for the headline 30-ep run, on a
  net learning all 12 lights. Do 2–4 learn() calls per decision (12
  fresh transitions arrive per decision; replay ratio stays sane).
- **Epsilon schedule is mis-sized for long runs.** Stage 1 (300 ep)
  decayed eps over 20 episodes ⇒ 280 episodes of near-pure
  exploitation filling a 50k buffer (~20 episodes) with on-policy data
  — the classic recipe for the observed mid-run collapse. Scale decay
  to ~half the run; floor 0.02–0.05.
- **Revisit gamma 0.99 only with the stability kit on:** tau=0.005
  Polyak (already built), lr 2.5e-4, and **n-step (3–5) returns** —
  with 5–10s decision steps, gamma 0.95 sees ~100–200s of horizon,
  too short to value letting a platoon through to the next light;
  n-step + 0.99 is the standard way to extend horizon without target
  blow-up. Dueling head and prioritized replay are cheap add-ons after.
- **Seed hygiene:** training uses env seeds `args.seed + ep` = 43–72;
  the headline eval runs seeds 42–46. Four of five eval seeds were
  trained on. Move training seeds to a disjoint range (e.g. 10042+ep).

## F8: evaluation/realism notes (for the "better than current systems" claim)

- **Truncated horizon:** demand file is 3600s; train/eval cut at
  1200s. Run the headline eval at the full hour — ranking can change
  once the post-ramp regime dominates; also report
  backlog-at-end. Paired across controllers either way, so it's a
  robustness check, not a fairness bug.
- **Single demand pattern:** one calibrated hour. Train/eval across
  demand scales (SUMO `--scale 0.7–1.2` via the existing
  `extra_cli_args` hook / DR wrapper) so the policy isn't a
  point-solution to one OD matrix — and so you have an *unsaturated*
  regime where throughput is not capacity-bound and beating actuated
  on throughput is actually possible.
- **Metric note:** `wait_per_vehicle` integrates
  `lane.getWaitingTime()` every second, so a vehicle waiting T seconds
  contributes O(T²). Fine for paired comparisons; for the public claim
  also report SUMO `tripinfo` `timeLoss`/`waitingTime` per vehicle
  (linear, standard in the literature).
- **Statistics:** 5 seeds is thin for a headline. Final claim: 20
  paired seeds + Wilcoxon signed-rank. Episodes are 20 min of sim —
  this is cheap.
- **Add the classical max-pressure controller** (cyclic, fixed
  min-green) as a third baseline. It's the published strong
  non-learned baseline; if V3 can't beat hand-coded max-pressure,
  that localizes the problem faster than comparing to actuated.

---

## Recommended order of attack

| stage | actions | cost | expected effect |
|---|---|---|---|
| 0. Fix the floor | F1 (per-light outflow reward) + F2 (green-bit, time-in-phase) + F3 (normalize) + F4 (3s yellows) , retrain 30 ep, 5-seed eval | ~1 day | This is the first *real* test of the combined reward. Expect throughput to move for the first time; wait gap to native should shrink materially |
| 1. Train properly | F6 (5-seed combined-score selection) + F7 (eps schedule, 2–4 learn/decision, n-step 3, tau 0.005, then gamma 0.99) , 150–300 ep | ~1–2 days compute | Stable long runs that beat the 30-ep model instead of regressing |
| 2a. Cheap coordination | F5 (neighbor block + phase feats into Q-head) | ~½ day | First corridor-aware policy; targets the wait gap (T2) |
| 2b. Full GAT | the planned Phase 2 on the DQN backbone | real build | green-wave progression — the structural edge over actuated |
| 3. Claim hygiene | F8 (full hour, demand scales, tripinfo metrics, 20 seeds, max-pressure baseline) | ~1 day | Defensible public numbers |

**Calibrated expectations.** Beating actuated on **wait** (T2) is a
realistic target once stages 0–2a land: the current −26% wait gap was
produced by a wait-only-by-accident reward with a blind state and a
yellow handicap. Beating it on **throughput** (T3) on *this* demand
stays capacity-bound (~32% unserved even under native); pursue T3 on
the 0.8–0.9 demand scales where the corridor is merely busy rather
than jammed — that is also closer to "current traffic light systems"
reality than a permanently gridlocked corridor.
