# PLAN V4 — Execution Plan (post-audit fixes → beat native-actuated)

**Who this is for:** whoever executes the plan. It was authored after a full
code audit. You do NOT need to re-derive any
finding — every claim is verified in `ai/TRAINING_REVIEW.md` with file:line anchors.
Your job is to implement the tasks below exactly, run the listed verification, and
report numbers at each gate.

**Goal:** beat SUMO native-actuated control (2,078 throughput / 5,504 wait/veh,
5-seed calibrated eval) — at minimum on wait (goal T2), stretch both (T3).
Current best: 1,779 / 6,947 — produced by a training run with the four defects
fixed in Stage 0.

---

## How to execute this plan cheaply (read this first)

1. **Read only these files** (everything you need is in them):
   - `ai/TRAINING_REVIEW.md` — the audit; findings F1–F8 with line anchors
   - this plan
   - the files you edit: `ai/sumo_env.py`, `ai/multi_env.py`,
     `ai/v3/frap_q_net.py`, `ai/v3/frap_dqn_agent.py`,
     `ai/v3/train_frap_dqn.py`, `ai/eval_network.py`
2. **For orientation, use the committed knowledge graph instead of exploring:**
   `graphify-out/GRAPH_REPORT.md`. God-nodes = the core abstractions
   (`MultiTlsEnv` 44 edges, `DQNAgent`, `SumoTrafficEnv`, `FRAPEncoder`); the
   community list maps every subsystem to its files. If you need a relationship
   answered, run `graphify query "..."` against `graphify-out/graph.json` rather
   than grepping the repo.
3. **NEVER read these** (multi-MB, zero signal for this work): `*.rou.xml`,
   `*.net.xml` (use `grep -m1 -A12 'tlLogic id="<TLS>"'` if you need one program
   block), `ai/runs/`, `ai/logs/`, `*.pth`,
   `docs/archive/early-prototypes/Open_Source_Data.ipynb`.
4. **One task = one commit.** Run only the verification listed in the task.
   Don't refactor beyond the task spec. All commands run from `SUMO/v2/`.
5. **Stop at every STAGE GATE** and report the numbers (pass or fail). Do not
   improvise past a failed gate — report and wait for direction.
6. Long runs: training 30 ep ≈ 30–45 min CPU; a 5-seed eval ≈ 15 min. Run them
   in the background and check the JSONL/console logs, don't block on them.

---

## Stage 0 — Fix the floor (4 code tasks, then retrain)

These four defects were live in EVERY V3 experiment to date. Fix all four before
any retraining; partial fixes produce uninterpretable results.

### T0.1 — Per-light outflow reward (fixes F1, the critical bug)

**Problem:** `combined` reward's `arrived_term` reads `_arrived_since_reward`
(`ai/sumo_env.py:548`), which only `SumoTrafficEnv._sim_tick()` increments
(`ai/sumo_env.py:397`) — and `MultiTlsEnv` never calls `_sim_tick`, so the term
is always 0 in multi-light training. `--reward-alpha` has never done anything.

**Fix (action-attributable local throughput, better than the global count):**
1. In `SumoTrafficEnv.__init__` add `self._prev_incoming_ids: set = set()`.
2. Add a method:
   ```python
   def snapshot_served(self) -> None:
       """Update vehicles-served count: vehicles that left this TLS's
       incoming lanes since the last snapshot (≈ vehicles discharged
       through the junction this decision)."""
       conn = self._conn()
       cur = set()
       for l in self._incoming_lanes:
           cur.update(conn.lane.getLastStepVehicleIDs(l))
       self._served_since_reward = len(self._prev_incoming_ids - cur)
       self._prev_incoming_ids = cur
   ```
   Initialize `self._served_since_reward = 0` in `__init__`. Reset both (empty
   set / 0) where `_arrived_since_reward` is reset in `reset()`
   (`ai/sumo_env.py:237`).
3. In the `combined` branch (`ai/sumo_env.py:548`) use it:
   ```python
   arrived_term = float(getattr(self, "_served_since_reward", 0)
                        or self._arrived_since_reward)
   ```
   (keeps the single-TLS path, where `_arrived_since_reward` works, intact).
4. In `MultiTlsEnv._refresh_coordination_signals` (`ai/multi_env.py:215`), first
   line of the loop: `u.snapshot_served()`. Also call once at the end of
   `MultiTlsEnv._start()` to seed the baseline snapshot.

**Caveat to respect:** a vehicle leaving an incoming lane includes lane-changes
to another incoming lane of the same TLS — `cur` is the union over all incoming
lanes, so internal moves cancel out. Vehicles that teleport are negligible
(`--time-to-teleport -1` is set).

**Verify:** new unit test `ai/tests/test_combined_reward.py`: build a
`MultiTlsEnv` on `sim_calibrated.sumocfg`, step ~30 decisions with random
actions, assert `sum(u._served_since_reward over steps) > 0` for at least
half the lights, and that the `combined` reward is not identical to the
beta-only value (i.e. the alpha term is alive). Run:
`python -m pytest ai/tests/test_combined_reward.py -q`

### T0.2 — Green-bit + time-in-phase in the FRAP state (fixes F2)

**Problem:** `get_state_frap()` (`ai/sumo_env.py:291`) emits only
`(halting, vehicles, waiting)` per movement. No current-phase signal, no
time-in-phase → the Q-net cannot distinguish "hold green" (free) from "switch"
(5s yellow + penalty). Published FRAP includes the current signal phase.

**Fix:** in `get_state_frap()` build `feats` with 5 columns:
- col 3: `is_green` — `1.0 if current phase state string[i] in ("G", "g") else 0.0`
  (current phase string =
  `self._phase_states[self._green_phase_indices[self._current_green_slot]]`)
- col 4: `min(self._time_in_phase, 120) / 120.0` (same value for all movements)

Then bump `mov_feat_dim` 3 → 5 at: `FRAPQNet.__init__` default
(`ai/v3/frap_q_net.py:38`), `FRAPDQNAgent.__init__` default
(`ai/v3/frap_dqn_agent.py:75`), and the constructor call in
`ai/v3/train_frap_dqn.py:94` (`mov_feat_dim=3` → `5`). The batch padder in
`ai/multi_env.py:357` uses a hardcoded `3` — change to read the unit's feature
width (e.g. from the first unit's `get_state_frap()["movement_features"].shape[1]`).

**Note:** old `.pth` checkpoints become shape-incompatible. Expected; say so in
the commit message. Do not add back-compat shims.

**Verify:** extend `ai/v3/tests/` smoke test (or add one): after `env.reset()`,
exactly the movements green in the active phase have col-3 == 1.0, and
`get_state_frap_batch()["movement_features"].shape[-1] == 5`. Run pytest on
that file.

### T0.3 — Normalize movement features (fixes F3)

In the same `get_state_frap()` loop, replace the raw triple with:
```python
feats[i, 0] = min(queue, 40) / 40.0
feats[i, 1] = min(vehicles, 60) / 60.0
feats[i, 2] = min(waiting, 300.0) / 300.0
```
(cols 3–4 from T0.2 are already in [0,1]). One commit together with T0.2 is
fine — they touch the same lines and both invalidate checkpoints.

**Verify:** in the T0.2 test, assert `movement_features.min() >= 0` and
`max() <= 1` after a few congested steps.

### T0.4 — Match native yellow times (fixes F4)

**Problem:** the AI pays a fixed 5s yellow per switch; the native baseline's
programs use 3s yellows at `53141735`, `53254289`, `7953710843` (5s at the
bigger junctions). Built-in handicap on exactly the comparison we report.

**Fix (simple version, chosen deliberately):** change the `yellow_time`
defaults from 5 → 3 in `MultiTlsEnv.__init__` (`ai/multi_env.py:51`),
`SumoTrafficEnv.__init__`, `ai/eval_network.py:256` (`--yellow-time`), and add
`--yellow-time` passthrough to `ai/v3/train_frap_dqn.py` (it currently doesn't
expose it — wire it into the `common` dict at line 83). 3s ≥ the real-world
minimum and matches the corridor majority; per-TLS parsing from the net file is
NOT required for V4.

**Verify:** `python -m pytest ai/tests ai/v3/tests -q` (existing suites) still
pass; grep confirms no remaining `yellow_time: int = 5` default.

### STAGE 0 GATE — retrain + 5-seed eval

```bash
# train (background, ~40 min)
python ai/v3/train_frap_dqn.py --sumo-cfg sim_calibrated.sumocfg \
  --reward-mode combined --tau 0.005 --episodes 30 --time-limit 1200 \
  --eval-seeds 1042 1043 1044 1045 1046 \
  --out-dir ai/runs/v4_stage0

# eval (the committed verifiable command)
python ai/eval_network.py --sumo-cfg sim_calibrated.sumocfg \
  --v3-ckpt ai/runs/v4_stage0/checkpoints/best.pth \
  --episodes 5 --time-limit 1200 --fixed-green-seconds 25 \
  --out ai/logs/eval_v4_stage0.txt
```

**Pass:** throughput > 1,820 AND wait/veh < 6,900 (both better than the current
best 1,779 / 6,947 on the same seeds). **Report the full summary table either
way.** On a clear fail (throughput still ~1,700s), the most likely culprits in
order: alpha/beta balance (try `--reward-alpha 0.5`), then the T0.1 snapshot
timing. One retune attempt is in scope; more is not — report instead.

Commit + push after the gate, whatever the outcome.

---

## Stage 1 — Training quality (do after Stage 0 passes)

### T1.1 — Checkpoint selection on the goal metric, 5 seeds
`ai/v3/train_frap_dqn.py:153` selects `best.pth` on wait alone over 3 seeds
(the project's own docs say 3-seed selection gave false positives).
- Default `--eval-seeds` → `[1042, 1043, 1044, 1045, 1046]`.
- Selection score: `score = arr - 0.1 * w` (arrived minus λ·wait, λ=0.1);
  keep `best.pth` on highest score; log both metrics in the JSONL as now.

### T1.2 — Right-size exploration + gradient budget
- Replace `--eps-decay-episodes` default 20 with `max(20, episodes // 2)`
  (computed in `main()` when the flag isn't explicitly set).
- Add `--learn-per-decision` (default 2): call `agent.learn()` that many times
  per env decision in the training loop (`ai/v3/train_frap_dqn.py:138`).

### T1.3 — n-step returns (n=3)
In `FRAPReplayBuffer` (`ai/v3/frap_dqn_agent.py:46`): keep a small per-light
pending deque in the trainer OR implement n-step inside the buffer keyed by
push order per light. Simplest correct approach: the trainer holds, per
`tls_id`, a deque of the last 3 (state, action, reward) and pushes the n-step
transition `(s_t, a_t, sum γ^k r_{t+k}, s_{t+n}, done)` once full (flush on
episode end). Target update in `learn()` then uses `gamma**n`. Add
`--n-step` (default 3) and store `n` on the agent.

### T1.4 — Disjoint training seeds
`ai/v3/train_frap_dqn.py:110`: `env.seed = args.seed + ep` overlaps the
final-eval seeds 42–46. Change to `env.seed = 10000 + args.seed + ep`.

### T1.5 — The long run
```bash
python ai/v3/train_frap_dqn.py --sumo-cfg sim_calibrated.sumocfg \
  --reward-mode combined --tau 0.005 --episodes 150 --time-limit 1200 \
  --out-dir ai/runs/v4_stage1
```
Optionally afterwards, one experiment with `--gamma 0.99` (only valid now that
n-step + Polyak are in): `ai/runs/v4_stage1_g99`.

**STAGE 1 GATE:** `eval_network.py` 5-seed: beats the Stage 0 model on BOTH
metrics. Report table; commit + push.

---

## Stage 2a — Cheap coordination (fixes F5; targets goal T2)

The env already computes a 6-float upstream/downstream block every decision
(`MultiTlsEnv._neighbor_triplet`, `ai/multi_env.py:202`: queue, pressure,
green-progress × 2 neighbors) — V3 just never feeds it to the network.

1. `get_state_frap_batch()` (`ai/multi_env.py:332`): add key
   `"context": float32 [n_tls, 8]` = the unit's 6 neighbor floats
   (`u._neighbor_features`) + `time_in_phase` (normalized) +
   `current_slot / num_green`.
2. `FRAPQNet.forward` takes an extra `context: (B, 8)` tensor; broadcast-concat
   onto each phase embedding: `q_head` input `embed_dim` → `embed_dim + 8`
   (`ai/v3/frap_q_net.py:46`).
3. Thread `context` through `_pad_state`/buffer/`act()`/`learn()` in
   `ai/v3/frap_dqn_agent.py` (it's fixed-size per light — no padding needed)
   and through the two state-dict builders in `ai/v3/train_frap_dqn.py:36,119`
   and `ai/eval_network.py:227`.
4. Retrain with the Stage 1 settings → `ai/runs/v4_stage2a`.

**STAGE 2a GATE (= project goal T2):** 5-seed eval — **wait/veh < 5,504
(native-actuated) on ≥ 3/5 seeds.** Report regardless. If clearly missed
(wait still > 6,000), STOP: Stage 2b (GAT green-wave) needs a separate plan
— do not improvise a GAT.

---

## Stage 3 — Claim hygiene (eval-side, cheap; can run parallel to Stage 2a)

1. **Full-hour eval:** run the headline eval with `--time-limit 3600` too;
   report both horizons.
2. **20 paired seeds + significance:** `--episodes 20`; add a Wilcoxon
   signed-rank (scipy is available; if not, paired sign test) on per-seed
   wait and throughput to the summary block of `ai/eval_network.py`.
3. **tripinfo metric:** add `--tripinfo-output` plumbing (SUMO flag via
   `extra_cli_args`) and report mean `timeLoss` per vehicle alongside
   `wait_per_vehicle` (which integrates quadratically — fine for paired
   comparison, not for public absolute numbers).
4. **Classical max-pressure baseline:** new actions-factory in
   `ai/eval_network.py` next to `fixed_time_actions_factory` (line 116):
   every decision pick the green slot with max `Σ(queue_in − queue_out)` over
   the movements that phase serves, honoring min-green. ~30 lines, uses
   existing `u._incoming_lanes`/`u._outgoing_lanes`/`phase_movement_mask`.
   This is the published strong non-learned baseline — if V4 can't beat it,
   that's the diagnostic to report.
5. **Demand-scale robustness:** eval at `--scale 0.8 / 1.0 / 1.2` via
   `extra_cli_args` (see `DRWrapper` in `ai/v2/domain_randomization.py` for
   the pattern). Report the 3×table. Throughput-vs-native (T3) is expected to
   be winnable at 0.8–0.9, not at 1.0 (over-saturated, capacity-bound).

**Final deliverable of Stage 3:** updated `ai/v3/RESULTS.md` with the V4 table
(all baselines incl. max-pressure, 20 seeds, both horizons, significance),
plots via the existing `ai/v3/make_v3_plots.py` pattern.

---

## Reporting template (use at every gate)

```
STAGE <n> GATE: PASS/FAIL
controller        throughput   wait/veh   (5 seeds, paired)
native_actuated   2,078        5,504      (reference)
fixed_time        ...          ...
v4_stage<n>       ...          ...        (Δ vs native: +x% / −y%)
notes: <one paragraph — anything surprising, what was retuned>
```
