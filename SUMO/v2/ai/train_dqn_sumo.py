"""Train a Double-DQN traffic-light controller against SUMO via TraCI.

Run from the ``SUMO/v2`` directory:

    python ai/train_dqn_sumo.py --episodes 30 --tls-id 3153556582

Outputs (under ``ai/``):
    checkpoints/best.pth      # checkpoint with the highest mean reward
    checkpoints/last.pth      # final checkpoint
    logs/train_log.csv        # per-episode metrics
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow running this file directly from ``SUMO/v2`` or ``SUMO/v2/ai``.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sumo_env import SumoTrafficEnv  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402


def linear_epsilon(step: int, total_steps: int,
                   start: float = 1.0, end: float = 0.05) -> float:
    if total_steps <= 0:
        return end
    frac = min(1.0, step / float(total_steps))
    return start + (end - start) * frac


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sumo-cfg", default="sim.sumocfg")
    p.add_argument("--tls-id", default="3153556582")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--time-limit", type=int, default=1200,
                   help="SUMO seconds per episode")
    p.add_argument("--min-green", type=int, default=5)
    p.add_argument("--yellow-time", type=int, default=5)
    p.add_argument("--decision-interval", type=int, default=5,
                   help="Sim-seconds per env.step(). 5s = boundary cadence.")
    p.add_argument("--reward-mode",
                   choices=["waiting", "differential", "anti_starve",
                            "max_pressure", "combined"],
                   default="differential")
    p.add_argument("--starve-penalty", type=float, default=0.01,
                   help="Weight on max-lane-wait in anti_starve reward.")
    p.add_argument("--reward-alpha", type=float, default=1.0,
                   help="Weight on throughput term in combined reward.")
    p.add_argument("--reward-beta", type=float, default=0.05,
                   help="Weight on wait term in combined reward.")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--out-dir", default="ai")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps-warmup-frac", type=float, default=0.6,
                   help="Fraction of total steps over which epsilon decays.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import random
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_root = Path(args.out_dir)
    ckpt_dir = out_root / "checkpoints"
    log_dir = out_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Probing SUMO network ({args.sumo_cfg}, TLS={args.tls_id}) …")
    env = SumoTrafficEnv(
        sumo_cfg_file=args.sumo_cfg,
        tls_id=args.tls_id,
        time_limit=args.time_limit,
        use_gui=args.gui,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        reward_mode=args.reward_mode,
        decision_interval=args.decision_interval,
        starve_penalty=args.starve_penalty,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        seed=args.seed,
    )
    print(f"  controlled lanes: {len(env.controlled_lanes)}")
    print(f"  green phases:     {env.green_phase_indices}")
    print(f"  state_size={env.state_size}  action_size={env.action_size}")

    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    total_decisions = max(1, args.episodes * (args.time_limit // max(1, args.min_green)))
    eps_total = max(1, int(total_decisions * args.eps_warmup_frac))

    log_path = log_dir / "train_log.csv"
    log_file = log_path.open("w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "episode", "decisions", "total_reward", "mean_reward_per_decision",
        "mean_waiting", "switches", "loss", "epsilon", "wall_time_s",
    ])

    best_mean = -float("inf")
    decision_counter = 0
    eps = 1.0

    for ep in range(1, args.episodes + 1):
        ep_start = time.time()
        # Vary the SUMO seed per episode so the agent sees different demand
        # samples — important for generalisation and to avoid memorising one
        # arrival pattern.
        env.seed = args.seed + ep - 1
        state = env.reset()
        ep_reward = 0.0
        ep_switches = 0
        ep_decisions = 0
        ep_losses = []
        ep_waits = []
        done = False

        while not done:
            eps = linear_epsilon(decision_counter, eps_total)
            action = agent.act(state, epsilon=eps)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            ep_reward += reward
            ep_switches += int(info.get("switched", False))
            ep_waits.append(env._total_waiting_time())
            decision_counter += 1
            ep_decisions += 1

        wall = time.time() - ep_start
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        mean_wait = float(np.mean(ep_waits)) if ep_waits else 0.0
        mean_per_dec = ep_reward / max(1, ep_decisions)

        writer.writerow([
            ep, ep_decisions, f"{ep_reward:.2f}", f"{mean_per_dec:.4f}",
            f"{mean_wait:.2f}", ep_switches, f"{mean_loss:.4f}",
            f"{eps:.3f}", f"{wall:.1f}",
        ])
        log_file.flush()

        print(
            f"ep {ep:>3d}/{args.episodes}  reward={ep_reward:>9.1f}  "
            f"wait_mean={mean_wait:>6.1f}  switches={ep_switches:>3d}  "
            f"loss={mean_loss:>7.4f}  eps={eps:.2f}  ({wall:.1f}s)"
        )

        agent.save(str(ckpt_dir / "last.pth"),
                   meta={"episode": ep, "tls_id": args.tls_id,
                         "green_phase_indices": env.green_phase_indices,
                         "controlled_lanes": env.controlled_lanes,
                         "reward_mode": args.reward_mode})
        if mean_per_dec > best_mean:
            best_mean = mean_per_dec
            agent.save(str(ckpt_dir / "best.pth"),
                       meta={"episode": ep, "tls_id": args.tls_id,
                             "green_phase_indices": env.green_phase_indices,
                             "controlled_lanes": env.controlled_lanes,
                             "reward_mode": args.reward_mode,
                             "best_mean_reward_per_decision": best_mean})

    log_file.close()
    env.stop_simulation()
    print(f"\nDone. Best mean reward / decision = {best_mean:.4f}")
    print(f"Checkpoints in: {ckpt_dir}")
    print(f"Log:            {log_path}")


if __name__ == "__main__":
    main()
