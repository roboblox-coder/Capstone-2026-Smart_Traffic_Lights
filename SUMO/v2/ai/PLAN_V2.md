# ATLAS V2 — From Sim Champion to Live Adaptive Corridor

A planning + research document for the next generation of the NE 8th St
corridor controller. Anchored to what V1 actually shipped and what the
2022-2026 multi-agent TSC literature says about closing the gap to a
live, adaptive, real-world deployment.

> Author: research synthesis, not implementation. Nothing here changes
> code; everything points at it. Decisions left to the team.

---

## 1. Where V1 actually is

**Architecture (current, on this branch):**
- 12 independent Double-DQN agents, one per traffic light.
- Backbone: a small MLP (2×128 ReLU), `BaseTrafficAI`.
- State: per-lane (halting count, waiting time, mean speed) + one-hot
  current green slot + time-in-phase + a hand-engineered 6-float
  upstream/downstream neighbor summary (`adjacency.json` driven).
- Action: pick a target green-phase slot. Yellow + min-green enforced
  inside the env.
- Reward: `max_pressure_net` — local max-pressure penalty + (curriculum-
  blended) shared corridor throughput bonus.
- Decision interval 5 s, min-green 5 s, yellow 5 s.

**Validated headline (5 seeds × 1200 s, whole 12-light corridor, vs SUMO
native actuated):**

| Metric          | DQN          | Native       | Δ                       |
|-----------------|--------------|--------------|-------------------------|
| Throughput      | 1699.8 ± 86  | 1711.8 ± 22  | **−0.7%** (tie, 3/5)    |
| Wait / vehicle  | 9670 ± 808   | 10401 ± 507  | **+7.0% lower** (4/5)   |

Single-TLS regression anchor (target intersection only): **+9.4% lower
wait, +7.4% higher throughput vs native.**

**Empirically rejected by V1 work — do not re-try without architectural
change:**
- Naïve shared corridor-throughput bonus in the reward (`--net-weight`,
  `--coord-penalty`). Monotonically degraded eval performance:
  −0.7% → −4.3% → −33.7% throughput as weight rose. Reward inflation,
  not learnable credit. See `train_multi_dqn.py` docstring.

---

## 2. Why V1 is at its ceiling (the diagnosis)

Three load-bearing limits — addressed in §3:

**L1. Local-only credit assignment.** Each agent's reward is a function
of its own incoming/outgoing lanes. No learned signal flows back to
"this light's decision starved the next-but-one downstream." Throughput
parity vs native is the architectural ceiling, not a training-budget
shortfall.

**L2. Hand-engineered, fixed-width neighbor channel.** Six floats per
neighbor. The agent never learns *which* neighbor matters under *which*
phase — the channel is averaged-across-everything by definition. The
literature consistently shows learned attention over neighbors
(CoLight) beating fixed neighbor summaries by 10-20% on travel time on
arterials.

**L3. State distribution = SUMO ground truth.** Lane occupancy is read
straight from TraCI: exact counts, exact waits, no missed detections.
The instant a real camera replaces TraCI, the state distribution shifts
and Q-values that look great in sim diverge in deployment.

---

## 3. V2 architecture (target)

The literature converges on a recognisable recipe for arterials. We
should not invent a new one.

| Component           | V1                              | V2                                           |
|---------------------|---------------------------------|----------------------------------------------|
| Backbone            | 2×128 MLP per light             | **FRAP** phase-invariant encoder, parameter-shared across all 12 lights |
| Neighbor channel    | Fixed 6-float summary           | **CoLight GAT** — learned attention over upstream/downstream/cross-street neighbors |
| Algorithm           | Independent Double-DQN          | **MAPPO** (centralized critic, decentralized actors) — CTDE |
| Reward              | `max_pressure_net` (blended)    | **Pure per-light pressure**, no shaping. Credit assignment moves into the critic. |
| Action              | Free green-slot pick            | Same, but masked against firmware-allowed transitions at every step |
| State source        | TraCI ground truth              | TraCI in train; **detector-noise-injected** TraCI in domain-randomized train; perception node at deploy |

**Why this combination and not something more exotic:** PressLight +
FRAP + MPLight + CoLight + MAPPO is the boring, replicated, deployment-
grade stack. MetaLight / X-Light / LLM-augmented "CoLLMLight" are
interesting research bets but the field has trouble reproducing their
gains under distribution shift — keep as research spikes, not as the
deployment line.

**Expected gain over V1 in calibrated SUMO** (literature midpoints, not
a promise): throughput +5 to +12% over native actuated (vs V1's tie);
wait/veh +10 to +20% lower (vs V1's +7%).

---

## 4. The deployment surface gap

Audited separately (see commit context). One-line summary: **V1 runs
end-to-end in SUMO. It runs nowhere else.** Real-world deployment needs
five new layers that don't exist:

| Layer                          | V1 status            | What it has to do |
|--------------------------------|----------------------|-------------------|
| Perception                     | None                 | Camera → detector → tracker → lane-occupancy → state vector at ≤200 ms, ≥95% recall daylight / ≥90% night-rain |
| Controller bridge              | None                 | NTCIP-1202 / NEMA TS2 / ATC compatibility; emit phase requests, *not* phase commands |
| Action mask & safety wrapper   | None                 | Enforce min-green / clearance / EVP / ped Walk per MUTCD on top of every RL output |
| Tiered fallback                | Implicit (per-TLS)   | Explicit, monitored, alarmed |
| Ops stack                      | None                 | Model registry, decision logs, monitoring, distribution-shift alarms, retraining trigger |

---

## 5. Phased roadmap (with go/no-go gates)

Every phase has an explicit metric. If a phase fails its gate, the next
phase does not start.

### Phase 0 — Lock V1, build the regression net (1 week)

- Pin the V1 deliverable: `ai/runs/coordinated/checkpoints/<tls>/best.pth`
  + the regime constants + per-seed numbers across 10 seeds (42..51) →
  `ai/baseline_v1.json`.
- **`ai/regression_test.py` (built)** — two-gate check on the first 5
  pinned seeds:
  - (A) current mean inside pinned mean ± 2σ
  - (B) paired-seed delta (current − pinned) 95% CI contains 0
    (catches distribution shifts the mean-only check misses)
  Strict mode: fails loudly if any of the 12 TLS lacks a valid
  checkpoint — refuses to silently substitute round-robin. SHA-pinned
  `sim.sumocfg` change triggers a warning, not a silent pass.
- Snapshot `sim.sumocfg` + `harvard_simulation.rou.xml` as the v1.0
  simulation environment via the `sumocfg_sha256` field in the
  baseline. Any change to demand / routes triggers the SHA warning
  and forces an intentional re-lock.

**Gate:** `python ai/regression_test.py` exits 0 on a clean clone.
Bootstrap once with `python ai/regression_test.py --write-baseline
--n-seeds 10` (slow; needs SUMO + the V1 checkpoints).

### Phase 1 — V2 in calibrated sim (4-6 weeks)

- Calibrate Krauss/IDM car-following + saturation flows + turn ratios
  to the Bellevue field data (`Real_intersection_data/*.xlsx` is already
  in the repo, currently unused). Without this step §5 is fiction —
  see the "Towards Real-World Deployment" finding (arXiv 2103.16223).
- Implement FRAP encoder, CoLight GAT communication, parameter-shared
  policy + MAPPO centralized critic. The training driver pattern in
  `train_multi_dqn.py` mostly transfers; replace per-agent DQN with
  one shared policy + one critic, keep the per-TLS env units.
- Re-run domain-randomized training (demand multiplier ±30%, vehicle
  mix randomization, detector noise injection — false-negative rate
  uniform 0-15%, lateral displacement σ=0.5 m).

**Gate:** beats V1 on **both** headline metrics on the locked sim, on
5/5 seeds. Without 5/5, do not advance — V1 stays the production model.

### Phase 2 — Perception stack (parallel with Phase 1, 6-8 weeks)

- YOLOv8n / YOLO11n + ByteTrack on Jetson Orin Nano, TensorRT INT8.
  Throughput budget: ≤200 ms camera-to-state.
- Surveyed lane polygons → ground-plane homography per camera.
- State publisher (MQTT/DDS) with heartbeat + watermarking; the
  controller drops to fallback if no fresh state in 2 decision ticks.
- All inference in the cabinet. No raw video off the edge. License-plate
  OCR explicitly disabled (zero operational value, large privacy
  liability). This is the "privacy-by-design" claim the ATLAS README
  already makes — make the code match the README.

**Gate:** 24 h field log of the perception stack at one intersection
shows ≥95% / ≥90% (day/adverse) recall against a manual ground-truth
sample of 1000 vehicles. Below this, do not feed the RL agent.

### Phase 3 — Shadow mode at one intersection (4 weeks minimum)

- RL agent receives live perception state, **emits** decisions, but the
  firmware keeps actuated control. Every decision logged with state +
  fallback level + what-the-firmware-actually-did.
- Compute counterfactual: "if we had followed the agent, would the
  observed downstream queue have been lower?" using a digital twin
  re-simulated from logged demand.
- Alarm on: distribution shift (GEH histogram distance from training
  flow > threshold per arXiv 2511.13785), perception degradation,
  decision divergence > N stdev from sim distribution.

**Gate:** 4 weeks of shadow logs show (a) zero unsafe action proposals
under MUTCD constraints, (b) counterfactual delay reduction in line
with sim prediction within 30%. Either failing → return to Phase 1 or
Phase 2 with the root cause.

### Phase 4 — Single-light live pilot (4-8 weeks)

- One intersection — pick the lowest-conflict, highest-instrumented one
  on the corridor.
- RL controls phase **selection only**. Firmware (NEMA TS2 / ATC)
  enforces all timing constraints. EVP and ped calls preempt the RL
  unconditionally.
- Operator-visible dashboard with live tiered-fallback level (see §6).
- Manual kill switch wired to the cabinet, not just to the WebSocket UI.

**Gate:** 30 days live, no safety incidents, no regression vs the
intersection's prior actuated baseline on a paired before/after window.

### Phase 5 — Corridor rollout (2-4 weeks per pair of intersections)

Add lights two at a time; re-run shadow mode for each pair before
flipping it live. Coordination wins compound; the safety story does
not — every new light is another failure surface.

---

## 6. The tiered fallback (the most important section)

The single architectural detail that decides whether this thing is
deployable. Every level monitored, every transition alarmed, every
transition logged immutably.

| Level | When                                       | Who's in charge                              |
|-------|--------------------------------------------|----------------------------------------------|
| 0     | Everything healthy                         | RL agent, full perception + neighbor messages |
| 1     | Neighbor messages stale or detector noisy  | RL agent on local-only state, neighbor block zero-padded |
| 2     | RL agent disagreement / OOD state          | **MaxPressure analytic** on the same detector inputs (cheap, throughput-optimal, no learning) |
| 3     | Perception degraded (recall < threshold)   | Hardware-firmware **actuated** baseline (existing controller) |
| 4     | Detector or cabinet comms down             | **Fixed-time TOD plan** from the .net.xml |

Transitions are not symmetric. Going *down* a level is automatic and
fast. Going *back up* requires N minutes of green-status health to
prevent flapping.

**This pattern is from LemgoRL / arXiv 2206.10122 / arXiv 2506.13836
and is the same pattern WSDOT and most state DOTs already require for
adaptive signal systems.** Implementing it before the live pilot is
non-negotiable, not nice-to-have.

---

## 7. Bottlenecks & failure modes — with proactive mitigations

Catalogued by category. Each has a specific mitigation, not a "we'll
think about it." Order within a category is rough severity.

### A. Modeling / training

| # | Failure                                                                 | Mitigation                                                                                          |
|---|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| A1 | Reward-shaping inflation (already burned us once)                       | Move credit assignment into a centralized critic (MAPPO). Pure per-light pressure reward, no blends. |
| A2 | Local-credit-assignment ceiling                                         | CTDE + GAT communication (Phase 1).                                                                |
| A3 | Sample inefficiency in CTDE PPO                                         | Parameter sharing across all 12 lights via FRAP's phase-symmetry-invariant encoder. ~12x sample reuse. |
| A4 | Stale-state Q-divergence at 5 s decision interval                       | Hard timeout: state older than 1 tick → drop to Level-2 fallback. Encoded in the safety wrapper.   |
| A5 | Distribution shift across seasons / school year / events                | GEH histogram distance monitor against training flow. Alarm at threshold; periodic *offline* retraining on rolling window. Never online. |
| A6 | Best.pth selection cherry-picks brittle snapshot                        | Use the converged Phase-2 selection rule already in `train_multi_dqn.py` (`phase2_arrived_then_-mean_wait`). Keep it. |

### B. Sim-to-real

| # | Failure                                                                 | Mitigation                                                                                          |
|---|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| B1 | Uncalibrated SUMO saturation flow / car-following → overconfident policy | Calibrate Krauss/IDM to field data before training (Phase 1, prerequisite).                        |
| B2 | Detector noise gap (real cameras drop 5-15% of vehicles)                | Inject false-negative + lateral noise during training (domain randomization, arXiv 2307.11357).    |
| B3 | Cold start when policy hits real distribution                           | Shadow mode for 4 weeks (Phase 3) before any actuation. Counterfactual delay check against logged demand. |
| B4 | Vehicle dynamics gap (startup loss, dilemma zone, gap acceptance)       | Grounded Action Transformation (arXiv 2507.15174) as a Phase-3.5 spike if shadow shows residual gap. |

### C. Perception

| # | Failure                                                                 | Mitigation                                                                                          |
|---|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| C1 | Camera dropout in rain / glare / night                                  | Twin redundant cameras per approach where the corridor budget allows. Below recall threshold → Level 3 fallback. |
| C2 | Lane homography drift (camera mount moves over months)                  | Auto-recalibration daily from lane-line detection. Manual recalibration on any reported drift > 1 m. |
| C3 | Edge inference latency overrun (Jetson under load)                      | Hard 200 ms budget. Watchdog kills inference and emits fallback heartbeat if exceeded.             |
| C4 | Detector-count systematic bias vs SUMO ground truth                     | Bias-correction calibration step during shadow mode: regress live detector counts against a periodic manual count, apply offset before state vector. |

### D. Deployment safety

| # | Failure                                                                 | Mitigation                                                                                          |
|---|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| D1 | Emergency vehicle preemption never trained for                          | Hardwired Opticom/GTT preemption at the controller, not in software. RL has no veto path.          |
| D2 | Pedestrian min-greens / ADA compliance                                  | Firmware-enforced (NEMA TS2 / ATC), not the RL's job. Action mask in the safety wrapper rejects any RL proposal that would violate. |
| D3 | RL proposes illegal transition                                          | Action mask at every tick — reject and re-query. If repeated rejection, escalate to Level 2.       |
| D4 | Mid-cycle comms failure                                                 | Watchdog → Level 3 within 1 cycle. Cabinet has the .net.xml fixed-time plan as last-resort.        |
| D5 | Liability / audit trail                                                 | Immutable per-cycle decision log: state, action, fallback level, what played, timestamps. Required for any post-incident review. |
| D6 | Reward gaming — agent starves cross-streets to optimise corridor flow   | Pressure reward is symmetric in lane direction; **additionally**, a hard max-red constraint in the safety wrapper (no lane red > 90 s) overrides RL. Same mechanism handles equity. |
| D7 | Adversarial input (poster of cars taped to a wall in camera view)       | Tracker requires motion + lane assignment. Static "vehicles" don't accumulate as queue. Periodic anomaly detector on detector-output distribution. |
| D8 | Public perception of "the AI made a weird call"                         | Operator dashboard surfaces the *why*: queue snapshot at decision time + the chosen phase + the runner-up. Every override is logged with operator name. |

### E. Operational

| # | Failure                                                                 | Mitigation                                                                                          |
|---|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| E1 | Model drift                                                             | See A5. Periodic retrain on rolling 90-day window of live perception logs replayed in sim.         |
| E2 | No model registry / versioning                                          | Phase 0 deliverable: every checkpoint pinned in git LFS or an artifact store with regime metadata + eval result + git SHA. Live runner only loads a version-stamped model. |
| E3 | No monitoring / alerting                                                | Phase 3 deliverable: Prometheus-style metrics for decision latency, fallback-level histogram, perception recall, GEH distance. PagerDuty on Level≥3 sustained > 5 min. |
| E4 | No regression gate before deploy                                        | Phase 0 deliverable (see §5 Phase 0). No checkpoint reaches the cabinet without it passing.        |
| E5 | Live runner hardcodes a single TLS                                      | `run_websocket_ai.py` is already multi-TLS via adjacency. Generalise the perception adapter + cabinet bridge the same way before Phase 5. |
| E6 | Manual override on the WebSocket UI has no auth                         | Phase 4 prerequisite: token-auth on the operator path. Public viewer stays read-only.              |

---

## 8. What changes in this repo, concretely

A working order, no implementation here:

1. `ai/PLAN_V2.md` — this file (done).
2. `ai/regression_test.py` — Phase 0 (**built**). Two modes:
   `--write-baseline` runs V1 on 10 seeds and pins
   `ai/baseline_v1.json`; default re-runs the first 5 pinned seeds and
   gates on (A) 2σ band + (B) paired-seed 95% CI containing 0.
3. `ai/baseline_v1.json` — Phase 0. Generated by
   `regression_test.py --write-baseline`; needs SUMO + the V1
   checkpoints, so produced locally rather than checked in from CI.
4. `ai/sumo_calibration/` — Phase 1. Notebooks + scripts that take
   `Real_intersection_data/*.xlsx` → calibrated `.sumocfg`.
5. `ai/v2/` — Phase 1. New package: `frap_encoder.py`, `colight_gat.py`,
   `mappo_trainer.py`, `shared_policy.py`. Old `dqn_agent.py` stays
   unchanged for the v1 regression test.
6. `ai/domain_randomization.py` — Phase 1.
7. `perception/` (new top-level dir) — Phase 2. `detector.py`,
   `tracker.py`, `homography.py`, `state_publisher.py`. Hardware-
   agnostic interface so SUMO TraCI can mock it for tests.
8. `safety/` (new top-level dir) — Phase 3-4. `action_mask.py`,
   `fallback_controller.py` (the §6 state machine), `decision_log.py`.
9. `ops/` (new top-level dir) — Phase 3-5. `metrics.py`,
   `drift_monitor.py` (GEH histogram), `model_registry.py`.

Nothing on this list touches anything currently passing in V1. V1 stays
the deliverable until the Phase-1 gate is green.

---

## 9. Hard "don't" list (lessons from V1)

- **Don't** re-enable `--net-weight` / `--coord-penalty` shaping. The
  negative result is in the `train_multi_dqn.py` docstring. Re-doing it
  is a 30-min training run for a confirmed loss.
- **Don't** train online from live data. Drift is real; *online*
  learning from live data is how an adaptive controller becomes an
  unsafe one. Periodic offline retraining only.
- **Don't** put license-plate OCR or PII-capable models in the
  perception path. The privacy-by-design claim is a feature; breaking
  it once kills the project.
- **Don't** let the WebSocket UI's "setPhase" command path into the
  cabinet without operator auth. Today it has neither.
- **Don't** "simplify" the env back to `setPhase` — `setRedYellowGreenState`
  is load-bearing per the V1 HANDOFF.

---

## 10. References (selected — full list in research brief)

- CoLight (arXiv 1905.05717), FRAP (arXiv 1905.04722),
  PressLight (KDD 2019), MPLight (AAAI 2020).
- MAPPO (Yu et al., NeurIPS 2022). QMIX (arXiv 1803.11485),
  QPLEX (arXiv 2008.01062), COMA (arXiv 1705.08926).
- Sim-to-real: arXiv 2103.16223, arXiv 2307.11357, arXiv 2507.15174.
- Safety / fallback: SafeLight (arXiv 2211.10871), LemgoRL
  (arXiv 2206.10122), Robustness under Incidents (arXiv 2506.13836).
- Distribution shift: arXiv 2511.13785, arXiv 2509.15291.
- Perception: aUToLights (arXiv 2305.08673), edge-AI perception
  (arXiv 2601.07845).
- WSDOT TSMO preemption guidance (state-level deployment policy).
