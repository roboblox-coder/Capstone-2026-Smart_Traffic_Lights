# Plan: Beat Native Signal Control on a Verifiable Metric

**Goal:** demonstrate the V3 FRAP-DQN controller beats conventional
("native") traffic signal control on the calibrated NE 8th St corridor,
on a metric anyone can re-verify.

---

## 1. What "verifiable" means here

A win is verifiable iff it is **reproducible by a single committed
command on fixed seeds with a deterministic policy**:

```bash
python ai/eval_network.py --sumo-cfg sim_calibrated.sumocfg \
    --v3-ckpt <ckpt> --episodes 5 --time-limit 1200
```

- 5 seeds (not 3 — 3-seed evals gave false positives earlier).
- Deterministic policy (ε=0 / argmax).
- Identical seeds across all controllers (paired comparison).
- Metrics straight from SUMO: `arrived` (throughput), `wait_per_vehicle`,
  `backlog`. No hand-picked numbers.

## 2. The baselines (what "native" means)

| Baseline | What it is | Realism | Difficulty |
|---|---|---|---|
| **fixed_time** (held ~25s greens) | a fixed equal-split cycle | **This is what real arterials actually run** | beatable |
| all_native_actuated | SUMO gap-out adaptive control | a SUMO-specific *strong* baseline; not typical real-world | hard (saturated corridor → throughput capacity-bound) |

Primary target: **fixed_time** (the honest "beats conventional signals"
claim). Stretch: **native_actuated**.

## 3. Tiered success (any tier = a real, reportable win)

- **T1 — beats fixed-time on both** wait↓ and throughput↑, 5 seeds.
  → "V3 beats conventional fixed-time signal control by X%." Highest
  confidence; likely already true (V3 beat round-robin fixed +28%/−29%).
- **T2 — beats native-actuated on wait** (delay), 5 seeds.
  → "beats even adaptive actuated control on delay." Medium confidence;
  delay is where corridor coordination pays off.
- **T3 — beats native-actuated on both.** Stretch; throughput is
  capacity-bound on this saturated demand, so this may be unreachable
  here regardless of model quality.

**Definition of done for the goal:** at minimum **T1 confirmed on a
5-seed eval** (a verifiable win over conventional signal control), with
T2 pursued and reported if reached.

## 4. Enactment sequence (iterate until done)

1. **Exp2** (running): combined reward, gamma 0.99, **5-seed checkpoint
   selection** (reliable `best.pth`). `ai/runs/v3_exp2`.
2. **Verifiable eval**: V3(best) vs fixed_time vs native, 5 seeds. Check
   T1/T2/T3.
3. **If T1 not yet clear** (unexpected): the realistic fixed-time is a
   stronger baseline than round-robin — tune V3 (longer training,
   reward β) and re-eval.
4. **To push T2 (beat native on wait):** the wait gap is the target. Try
   a wait-weighted combined reward (raise β) and/or longer training with
   reliable best-checkpoint selection. Re-eval.
5. **Stop when** T1 is confirmed (verifiable win banked) and T2 is either
   reached or shown to be at the saturated-corridor ceiling.

## 5. Why this is the right framing (honest footing)

- The corridor is **over-saturated** (native leaves ~32% of demand
  unserved), so throughput is partly physics-bound — beating native
  *throughput* by a wide margin is not a model-quality question.
- The calibrated sim is **half-estimated** (6/12 intersections real,
  single hour, inferred OD) — per `sumo_calibration/report.md`, absolute
  "beats native" claims carry a "pending city-count validation" caveat.
- Therefore the **credible, defensible** win is **beating conventional
  fixed-time control on delay + throughput, verifiably and reproducibly**
  — which is exactly what real deployments would replace.

---

*Enacted via `ai/v3/train_frap_dqn.py` (retrain) + `ai/eval_network.py`
(verifiable 5-seed eval with the fixed-time baseline). Progress tracked
in `ai/runs/v3_exp*/`.*
