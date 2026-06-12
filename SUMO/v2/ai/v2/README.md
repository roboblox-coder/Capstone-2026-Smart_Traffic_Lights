# V2 MAPPO Corridor Controller - Training Session Findings & Diagnostics

This README documents the results, findings, and technical diagnoses from the latest **V2 MAPPO** training session (`v2_mappo_retrain4`) on the Bellevue corridor. It is the starting point for implementing the next training iteration.

---

## 1. Executive Summary

* **Objective:** Unfreeze the GAT (Graph Attention) attention weights, stabilize PPO training, and evaluate the trained model against the **SUMO Native Actuated** controller on Bellevue's real-world calibrated RouteSampler corridor network (`sim_calibrated.sumocfg`).
* **Current Checkpoint:** `SUMO/v2/ai/runs/v2_mappo_retrain4/checkpoints/best.pth` (Episode 480 snapshot, early-stopped at episode 630).
* **Verdict:** **FAIL.** The policy fails to clear the honest gate (+6% throughput AND +12% lower wait) on both standard base seeds (42–46) and training evaluation seeds (1042–1044).
  * *Base seeds throughput:* **-27.2%** vs native actuated.
  * *Base seeds wait time:* **-38.8%** higher delay vs native actuated.
* **Diagnosis:** The V2 MAPPO model was trained using `pressure_only` reward mode, which lacks global throughput incentives, coordination penalties, and switch penalties. While the centralized critic stabilized value estimation, local pressure minimization alone cannot compete with Bellevue's coordinated green-wave offsets and suffers from yellow-light transition losses due to excessive phase switching.

---

## 2. Training Run & Evaluation Details (`v2_mappo_retrain4`)

### Training Parameters
* **Total planned episodes:** 1200
* **Early stop plateau threshold:** 150 episodes of no eval improvement (with GAT unfrozen)
* **GAT freeze schedule:** GAT uniform-frozen for first 500 gradient steps, linearly ramped to step 1000, then fully learning.
* **Actor LR schedule:** Cosine decay from `3e-4` to `5e-5`.
* **GAT LR schedule:** Ramped and decayed in sync with `actor_lr` to prevent attention entropy collapse.

### Training Progress
* The GAT successfully unfroze, and attention entropy moved dynamically in the `0.30`–`0.97` range.
* The plateau detector triggered an early stop at **Episode 630** because evaluation wait times plateaued.
* **Peak Training Performance (Episode 480):** Mean wait of **`6478.67s`** and throughput of **`1556.33`** across 3 evaluation seeds (1042, 1043, 1044).

---

## 3. Head-to-Head Evaluation Results

Evaluations were run for 5 episodes (seeds 42–46) and 3 episodes (seeds 1042–1044) for 1200 simulation seconds.

### 5-Seed Standard Eval (Seeds 42–46)
| Policy | Arrived (Throughput) | Backlog | Wait per Vehicle | Net Mean Wait |
|---|---|---|---|---|
| **SUMO Native Actuated** | **2077.8** $\pm$ 63.5 | **991.2** $\pm$ 28.2 | **5504.04** $\pm$ 973.41s | **14040.96** $\pm$ 2287.33s |
| **V2 MAPPO (`best.pth`)** | **1513.4** $\pm$ 112.6 | **1229.8** $\pm$ 68.9 | **7638.09** $\pm$ 2769.79s | **17355.74** $\pm$ 6009.24s |

* **Throughput difference:** -27.2%
* **Wait/Veh difference:** -38.8% (worse wait time)

### 3-Seed Training Eval Verification (Seeds 1042–1044)
| Policy | Arrived (Throughput) | Backlog | Wait per Vehicle | Net Mean Wait |
|---|---|---|---|---|
| **SUMO Native Actuated** | **2121.0** $\pm$ 48.8 | **957.0** $\pm$ 25.9 | **4881.05** $\pm$ 817.13s | **12502.52** $\pm$ 2012.71s |
| **V2 MAPPO (`best.pth`)** | **1556.3** $\pm$ 72.8 | **1225.3** $\pm$ 44.1 | **6478.67** $\pm$ 1076.30s | **14980.29** $\pm$ 2279.01s |

* **Throughput difference:** -26.6%
* **Wait/Veh difference:** -32.7% (worse wait time)
* *Note: The V2 metrics match the training log at Episode 480 exactly, validating the correctness of the inference adapter.*

---

## 4. Root Cause Analysis

### 4.1 Bellevue Green-Wave Coordination
Bellevue routes are highly calibrated and directional (morning/evening flows). SUMO's native actuated traffic lights operate with coordinated offsets, allowing large blocks of vehicles to traverse the 12-intersection corridor in green waves. Beating this baseline requires the RL policy to discover and maintain green-wave offsets.

### 4.2 Reward Mode Limitation (`pressure_only`)
The V2 MAPPO trainer is configured to use `reward_mode="pressure_only"`, which only rewards local pressure minimization at each signal:
$$r_i = -\frac{|q_{in} - q_{out}|}{n}$$
It completely lacks the components of the DQN baseline's coordinated `max_pressure_net` reward:
1. **Corridor Throughput Term:** No global arrival reward is shared among agents. Thus, they have no incentive to coordinate green-wave offsets to maximize total corridor exits.
2. **Downstream Saturation Penalty:** No penalty for dumping vehicles into downstream lanes that are already jammed. This causes queue spillbacks and gridlock cascades.
3. **Switch Penalty:** No cost for changing phases. PPO's entropy regularization does not penalize switching. In SUMO, every phase switch triggers a **5-second yellow light transition** during which no vehicles can cross. Without a switch penalty, the V2 policy switches phases too frequently, losing massive throughput to yellow time.

---

## 5. Next Steps

To close the gap and successfully clear the honest gate, the reward structure must be updated to align with the coordinated corridor objective:

### 1. Update Trainer to Coordinated Rewards
* Modify `SUMO/v2/ai/v2/mappo_trainer.py` to instantiate both training and evaluation environments using `reward_mode="max_pressure_net"`.
* Configure the trainer to use curriculum scaling:
  * **Phase 1 (stabilization):** Set `local_w = 1.0`, `net_w = 0.0`, `coord_w = 0.0`.
  * **Phase 2 (coordination fine-tuning):** Linearly ramp `net_w` and `coord_w` to their final values (e.g., `net_w = 0.05`, `coord_w = 0.1`) over the episodes.
* Ensure a `switch_penalty` (e.g., `0.1` or `0.2`) is active in `max_pressure_net` to discourage excessive yellow-light transitions.

### 2. Relaunch Training
* Run a new training session with the coordinated reward settings for 1200 episodes.
* Verify that V2 evaluation wait times trend below native actuated as the coordination weights ramp up.
