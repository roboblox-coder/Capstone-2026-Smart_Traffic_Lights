# ATLAS — Retrain & Live-Deployment Plan

**Corridor:** NE 8th St, Bellevue · 13 SUMO TLS junctions · 4 prioritized for the demo
**Date:** 2026-05-24
**Owner:** SYNAPTIX / BC Capstone Team 5

This document audits the current AI traffic controller, surveys what other
research groups have shipped, recommends a concrete retrain path, and
pre-empts the failure modes that are most likely to bite us between "works
in simulation" and "live & adaptive on a real corridor."

---

## 1. Where we are right now

### 1.1 Code that exists
| File | Role |
|---|---|
| `ai/sumo_env.py` | Single-TLS Gym-style wrapper over TraCI |
| `ai/traffic_base.py` | 2-layer MLP `[in → 64 → 64 → out]` |
| `ai/train_dqn_sumo.py` | Vanilla DQN trainer |
| `ai/visualize_sumo_ai.py` | Loads model, drives 4 intersections from pygame |
| `run_websocket_ai.py` | Live loop: TraCI + AI inference + WebSocket broadcast |
| `NE_8th_St_Corridor.net.xml` | 13 actuated TLS, real Bellevue geometry |
| `Real_intersection_data/Turn_ratio.xlsx` | Real turn ratios per intersection |
| `detectors.add.xml` | E2 lane-area detectors already wired |

### 1.2 What the model does today
- **Algorithm:** vanilla DQN, no target network, uniform replay
- **State:** per-lane `[halting_count, waiting_time]`, padded/truncated to 16
- **Action:** `argmax → traci.trafficlight.setPhase(tls_id, idx)`
- **Reward:** `-sum(waiting_time)` over controlled lanes
- **Training:** 5 episodes × 300 sim-seconds, single intersection (`3153556582`)

### 1.3 Critical findings (must-fix before any retrain)
1. **No trained weights exist.** There is no `ai/dqn_sumo.pth` anywhere in
   the repo. `visualize_sumo_ai.py` and `run_websocket_ai.py` both fall
   back to **random actions** when the file is missing. The "AI" demo is
   currently fixed-time + random phase jumps.
2. **State-size mismatch across intersections.** The 4 deploy-target
   TLSes have phase-string lengths 19, 21, 19, and 11 (and similarly
   variable lane counts). `STATE_SIZE=16` silently zero-pads or truncates
   in `visualize_sumo_ai.py:44-46` — the model trained on TLS A is being
   asked to control TLS B/C/D with a state vector that means something
   different per intersection. This will never converge correctly.
3. **`setPhase(action)` skips yellow + all-red clearance.** SUMO's
   `tlLogic` for every junction already encodes yellow phases (e.g.
   phase 0 `GGGgg` → phase 1 `yyygg`). Letting the policy jump straight
   from green-A to green-B causes virtual collisions in SUMO and is a
   **safety-critical bug** in any real-world deployment.
4. **No min-green / max-green enforcement.** The DQN can flip phases
   every step (1 s) — pedestrians stranded mid-crossing, drivers slamming
   brakes on yellow-skips.
5. **Trained against the wrong baseline.** Every `tlLogic` in the net is
   already `type="actuated"` (SUMO gap-out). Our real comparison point
   is *gap-actuated*, not fixed-time. We need to beat that bar.
6. **Reward only counts vehicles.** README promises pedestrian, bicycle,
   and emergency-vehicle priority. None of those signals are in the
   state or the reward.
7. **Training budget too small by 3 orders of magnitude.** 5 episodes ≈
   1,500 environment steps. The lightest published baselines train for
   ≥ 1e6 steps; CoLight/PressLight/MPLight all train 1–5e6.
8. **The "Open_Source_Data" notebook is not connected to anything.** It
   generates synthetic Webster-formula wait times. It's not used by the
   trainer or the live loop.

### 1.4 What the architecture gets *right* (keep these)
- SUMO + TraCI is the correct simulator choice for arterial RL.
- E2 lane-area detectors already exist — these are the natural "camera
  count" abstraction; we don't need to rebuild perception to train.
- Real Bellevue net + real turn ratios are checked in — that's gold.
- WebSocket frontend is decoupled — we can swap the controller without
  touching the visualizer.
- PyTorch + edge-deployment story is sound (cheap CPU inference).

---

## 2. Research summary — what the field is doing

| Method | What it brings | Why it matters here |
|---|---|---|
| **PressLight** (KDD '19) | Reward = pressure (incoming queue − downstream queue). Coordinates arterials *without* explicit comms. | NE 8th is a linear arterial → near-perfect fit. |
| **MPLight** (AAAI '20) | FRAP backbone + pressure, parameter-shared across all intersections. | Solves our state-size problem: one shared net, per-intersection observation. |
| **CoLight** (CIKM '19) | Graph attention over neighbor intersections. | Heavier; consider for v2 once MPLight baseline is beaten. |
| **MA-PPO / IPPO** | PPO is more stable than DQN on TSC; parameter sharing + CTDE. | Recommended algorithm — sumo-rl + SB3 supports out of the box. |
| **sumo-rl** (Alegre) | Gymnasium + PettingZoo wrapper, action-masking helpers. | Replaces our hand-rolled `sumo_env.py`. |
| **Domain randomization + meta-RL** (Müller et al. '23) | Closes the sim-to-real gap by training on perturbed demand. | Essential for any chance of beating actuated control in reality. |
| **Action masking / safety shield** (Müller '22) | Mask any action that violates min-green, ped-clear, yellow insertion. | Fixes finding 1.3 #3 and #4. |

**Bottom line:** the consensus 2025 stack for a 4–13-intersection arterial
is *parameter-shared MA-PPO with pressure-based observation, action
masking for phase safety, trained under demand randomization*. That is
what we should aim for, and it is achievable with `sumo-rl` + SB3 inside
a capstone timeline.

---

## 3. Recommended plan (phased, capstone-sized)

### Phase 0 — Stop the bleeding (~1 week)
Goal: a *correct* baseline that we can defend.

- [ ] Delete `setPhase(action)` everywhere. Replace with a `SafePhaseController`
      that exposes 2 actions per TLS: `{keep_current_green, advance_to_next_green}`.
      The controller inserts the SUMO-defined yellow + all-red between
      greens and enforces `min_green=8s`, `max_green=50s` (matches the
      `minDur`/`maxDur` already in the `.net.xml`).
- [ ] Add a **benchmark harness** (`ai/benchmark.py`) that runs three
      controllers on the same routes and dumps a CSV:
      1. Fixed-time (Webster cycle from real Bellevue data)
      2. SUMO actuated (the current `type="actuated"` baseline)
      3. AI controller (whatever we have)
      Metrics: avg travel time, avg wait, max queue, throughput,
      phase-switches/hour, fairness (95th-pct wait per approach),
      pedestrian wait. **No model gets accepted that doesn't beat
      actuated on at least 4 of these.**
- [ ] Replace the synthetic demand with real Bellevue counts driven by
      `Turn_ratio.xlsx` (already in `Real_intersection_data/`). Pin the
      AM-peak and PM-peak demand profiles as our two canonical scenarios.

### Phase 1 — Rebuild the env on `sumo-rl` (~1 week)
- [ ] `pip install sumo-rl stable-baselines3[extra]` — add to
      `requirements.txt`.
- [ ] Replace `sumo_env.py` with a thin adapter around
      `sumo_rl.parallel_env(...)` so we get PettingZoo MARL for free.
- [ ] Observation per intersection (pressure-style):
      `[phase_one_hot, time_in_phase, density_per_lane, queue_per_lane,
       pressure_per_movement, ped_call_flag, emergency_flag]`.
      All entries normalized to [0,1]; pad with **explicit mask** (not
      silent zeros) when lane counts differ.
- [ ] Reward (multi-objective, scalar):
      `r = -α·Σwait - β·Σqueue + γ·throughput - δ·phase_switch_cost
           - ε·ped_wait + emergency_bonus`.
      Default weights: `α=1.0, β=0.25, γ=2.0, δ=0.1, ε=2.0, EV=50`.
      Tune by sweeping on the AM-peak scenario.

### Phase 2 — Train the new model (~2 weeks of wall-clock, mostly unattended)
- [ ] Algorithm: **PPO with parameter sharing across the 4 TLSes**
      (one policy, per-TLS observation+action). FRAP-style backbone is
      ideal but a 3-layer MLP + 1-head attention is enough for v1.
- [ ] Budget: ≥ 2e6 environment steps, 8 parallel SUMO workers.
- [ ] Curriculum: 30 % light traffic → 40 % real-peak → 30 % surge
      (peak × 1.4) and 5 % emergency-vehicle injection.
- [ ] Domain randomization each episode:
      - arrival rate × U(0.7, 1.3)
      - turn ratios ± 15 % (Dirichlet noise)
      - vehicle mix (car/truck/bus) ± 20 %
      - 10 % sensor miss rate, 5 % false-positive on detectors
        (this is the sim-to-real bridge — see §4.3)
- [ ] Eval every 25k steps against the Phase-0 benchmark harness. Save
      `best.zip` whenever it beats the current `best.zip` on weighted
      score.

### Phase 3 — Sim-validation gate (~1 week)
- [ ] Test scenarios the model has never seen: rainy-day demand
      profile, school-letout surge, NE-8th-closed detour.
- [ ] Sensor-degradation test: drop 25 % of E2 detectors at random and
      confirm graceful degradation (should still beat fixed-time, may
      lose to actuated).
- [ ] Emergency-vehicle test: inject an EV every 5 minutes; verify
      preemption + recovery within 2 cycles.
- [ ] Pedestrian fairness test: confirm no crossing waits > 90 s.
- [ ] If any test fails → loop back to Phase 2 with revised reward.

### Phase 4 — Shadow + tiered live deploy (~ongoing; needs city partner)
- [ ] **Shadow mode (no actuation).** Wire the trained model to live
      camera-derived counts (or city loop counts if we can get a data
      feed) and log what it *would have done* for 2-4 weeks. Compare to
      what the city signal actually did.
- [ ] **One intersection, off-peak only.** Lowest-volume TLS, 22:00 –
      05:00, with a human-engineer kill-switch. Run for 1 week.
- [ ] **Expand the window**, then add intersections one at a time,
      always with a fixed-time fallback program that activates on TraCI
      / heartbeat loss (see §4.6).

---

## 4. Failure modes & pre-emptive mitigations

Each is a real risk; each has a concrete countermeasure to put in
the code *before* the failure costs us a week.

### 4.1 State-size drift between training & deployment
**Risk:** intersection geometry changes (lane added, detector breaks),
state vector size shifts, model silently produces garbage.
**Mitigation:** observation is a dict `{lanes: [...], mask: [...]}`,
mask is zero for non-existent lanes. Trainer + inference both refuse to
run if `mask.sum() != expected_lane_count_for_tls`. Hard-fail loud, not
silent-pad quiet.

### 4.2 Reward hacking → starved minor approaches
**Risk:** model learns it can minimize total wait by giving the busy
direction a permanent green, starving the side street.
**Mitigation:** add `ε·Σmax(0, lane_wait − 60s)²` penalty + cap any
green at 50 s (already in `.net.xml`, enforce in shield). Eval-time
fairness metric (95th-pct wait per approach) is a hard go/no-go gate.

### 4.3 Sim-to-real gap
**Risk:** model is brilliant in SUMO, mediocre or worse on the real
corridor because SUMO doesn't model camera noise, weather, blocked
lanes, or driver heterogeneity.
**Mitigation:**
- Domain randomization in training (§3 Phase 2).
- **Sensor abstraction layer** (`TrafficSensor.read_counts(lane)`) that
  is the *only* way the policy sees the world. Backed by SUMO E2
  detectors in sim, by camera-YOLO in real. Same noise model in both.
- 2–4 week shadow-mode log before any actuation (§3 Phase 4).

### 4.4 Camera-pipeline failure / occlusion / night
**Risk:** A truck parks in front of the camera, snow blocks the lens,
night-time miss rate spikes.
**Mitigation:** `TrafficSensor` returns `(count, confidence)`. Below
0.4 confidence → policy treats that lane as "unknown" (mask = 0) and
shield falls back to actuated control for that approach.

### 4.5 Catastrophic forgetting on rare events (EV, surge, crash)
**Risk:** rare events get drowned out by 99 % normal-day samples and
the model un-learns them.
**Mitigation:** reservoir buffer that *always* keeps a few thousand
samples from each rare-event class (EV preemption, oversaturation,
ped-heavy). Sample replay batches 70 % uniform / 30 % from reservoirs.

### 4.6 Loss of TraCI / network / model server
**Risk:** controller crashes mid-rush-hour and intersections freeze.
**Mitigation:** every TLS keeps the original `type="actuated"` program
loaded as program ID `0`; AI control is program ID `1`. On heartbeat
loss (no command in 3 s), the gateway calls
`traci.trafficlight.setProgram(tls, "0")` and signals revert to
gap-actuated. Test in chaos-monkey mode every CI run.

### 4.7 Phase-skip safety violation
**Risk:** policy outputs a phase index that doesn't have a yellow
transition from the current phase → simulated collision, real-world
catastrophe.
**Mitigation:** `SafePhaseController` is the *only* code that calls
`traci.trafficlight.setPhase`. Policy never sees raw phase indices —
only `{keep, advance}` actions. Controller looks up the SUMO-defined
yellow-then-green sequence and plays it through.

### 4.8 Trained on stale data
**Risk:** Bellevue grows, turn ratios drift, model from 2026 mis-prices
2028 traffic.
**Mitigation:** monthly retraining cron on rolling 30-day camera-count
data, automatic A/B vs. the currently-deployed checkpoint. Promote new
checkpoint only on stat-sig improvement.

### 4.9 We can't actually touch the real signals
**Risk:** City of Bellevue rightfully won't let a capstone team
actuate live signals on day one. Project optics suffer.
**Mitigation:** the **shadow-mode deliverable** (§3 Phase 4) is the
real public demo. "We watched the city's signals for 3 weeks and would
have saved X vehicle-hours" is a publishable result and a credible
ask for a city pilot conversation. Frame this as the headline outcome
*from the start*, not a fallback.

### 4.10 Capstone-team bandwidth
**Risk:** Phases 2-4 take longer than 1 semester.
**Mitigation:** Phases 0 + 1 (3 weeks) alone produce a defensible
result: "actuated baseline beaten by safe-phase MARL in simulation on
real Bellevue geometry and real turn ratios." That's the minimum viable
capstone. Everything past it is a stretch goal — plan accordingly.

---

## 5. Concrete next actions (in order)

1. Stand up the **benchmark harness** (§3 Phase 0) — without this we
   can't tell whether any retrain "worked." This is the single highest-
   leverage week of work in the whole plan.
2. Fix the **safety shield** + delete raw `setPhase(action)` calls.
3. Swap to **`sumo-rl` + SB3 PPO** with the pressure observation.
4. Train v1 against AM-peak only, get past actuated baseline,
   *then* expand.
5. Build the **`TrafficSensor`** abstraction so the same policy code
   runs on SUMO E2 detectors *and* on the camera/YOLO pipeline. Don't
   defer this; it's the seam between research and deployment.

---

## 6. Success metrics (define "done")

A model is shippable when, averaged over 50 episodes of AM-peak +
PM-peak with domain randomization enabled, it beats SUMO `actuated`
on:
- avg travel time ≥ 8 % lower
- avg vehicle wait ≥ 12 % lower
- 95th-pct per-approach wait no worse than +5 %
- pedestrian wait no worse than baseline
- zero phase-safety violations in the shield audit log

And in shadow mode against real Bellevue counts for 2 weeks, the model
must show a statistically significant predicted improvement (paired
t-test, p < 0.05) before *any* live actuation is proposed.

---

## Sources

- [PressLight (KDD '19)](https://www.researchgate.net/publication/334715234_PressLight_Learning_Max_Pressure_Control_to_Coordinate_Traffic_Signals_in_Arterial_Network)
- [MPLight / FRAP](https://proceedings.mlr.press/v162/zhang22ah/zhang22ah.pdf)
- [CoLight benchmarks](https://github.com/traffic-signal-control/RL_signals)
- [LibSignal open library](https://arxiv.org/pdf/2211.10649)
- [sumo-rl + PettingZoo + SB3](https://github.com/LucasAlegre/sumo-rl)
- [Safe TSC with action masking (Müller '22)](https://arxiv.org/pdf/2206.10122)
- [Sim-to-real via domain randomization (Müller '23)](https://arxiv.org/pdf/2307.11357)
- [Real-world deployment survey](https://arxiv.org/pdf/2103.16223)
- [MARL TSC review (MDPI '25)](https://www.mdpi.com/2076-3417/15/15/8605)
- [PyTSC unified platform](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902778/)
