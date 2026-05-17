"""Train one Double-DQN per traffic light, all coordinated in ONE SUMO sim.

Decentralised independent learners with neighbour-aware observations and the
``max_pressure_net`` reward (validated max_pressure shaping + a small shared
network-throughput bonus). Every light is controlled simultaneously so each
agent learns against the others' live behaviour.

Run from ``SUMO/v2``::

    # smoke
    python ai/train_multi_dqn.py --episodes 2 --time-limit 120
    # full
    python ai/train_multi_dqn.py --episodes 30 --time-limit 1200

Outputs under ``ai/runs/coordinated/``::
    checkpoints/<tls_id>/best.pth   coherent network-best snapshot
    checkpoints/<tls_id>/last.pth   latest
    logs/train_log.csv              per-TLS + per-episode network rows
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from multi_env import MultiTlsEnv, load_adjacency  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402


def linear_epsilon(step: int, total_steps: int,
                   start: float = 1.0, end: float = 0.05) -> float:
    if total_steps <= 0:
        return end
    frac = min(1.0, step / float(total_steps))
    return start + (end - start) * frac


def curriculum_net_weight(step: int, total_steps: int,
                          phase1_steps: int, final_w: float) -> float:
    """Two-phase weight on the shared corridor-throughput reward term.

    Phase 1 (step < phase1_steps): 0.0 — every agent learns the *validated*
    local max-pressure policy in a near-stationary setting, so joint
    training starts from individually-competent policies instead of from
    scratch (the root cause of the earlier from-scratch instability).

    Phase 2 (remaining steps): linearly ramp 0 -> final_w so the agents
    fine-tune toward corridor-wide flow without a destabilising jump.
    """
    if step < phase1_steps:
        return 0.0
    span = max(1, total_steps - phase1_steps)
    frac = min(1.0, (step - phase1_steps) / float(span))
    return final_w * frac


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sumo-cfg", default="sim.sumocfg")
    p.add_argument("--adjacency", default="ai/adjacency.json")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--time-limit", type=int, default=1200)
    p.add_argument("--min-green", type=int, default=5)
    p.add_argument("--yellow-time", type=int, default=5)
    p.add_argument("--decision-interval", type=int, default=5)
    p.add_argument("--reward-mode", default="max_pressure_net",
                   choices=["max_pressure_net", "max_pressure",
                            "differential", "anti_starve", "combined",
                            "waiting"])
    p.add_argument("--reward-gamma", type=float, default=0.05,
                   help="Legacy fixed weight on the shared bonus; only used "
                        "if --net-weight is not set / curriculum disabled.")
    p.add_argument("--out-dir", default="ai/runs/coordinated")
    p.add_argument("--seed", type=int, default=42)
    # The reward curriculum (Phase 1 pure local, Phase 2 ramped coordination)
    # is what keeps independent learners stable, so the old long epsilon
    # warmup is no longer needed — a shorter one leaves real exploitation
    # time in Phase 2 for the coordination fine-tune.
    p.add_argument("--eps-warmup-frac", type=float, default=0.35)
    p.add_argument("--curriculum-frac", type=float, default=0.6,
                   help="Fraction of total decisions spent in Phase 1 "
                        "(pure validated local max-pressure, net_w=0).")
    p.add_argument("--net-weight", type=float, default=4.0,
                   help="Final Phase-2 weight on the shared corridor-"
                        "throughput term (≈ local-term magnitude so it "
                        "actually influences behaviour; the old 0.05 was "
                        "~100x too small to matter).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import random
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_root = Path(args.out_dir)
    ckpt_root = out_root / "checkpoints"
    log_dir = out_root / "logs"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    adjacency = load_adjacency(args.adjacency)
    print(f"Probing {len(adjacency)} TLS from {args.sumo_cfg} …")
    env = MultiTlsEnv(
        sumo_cfg_file=args.sumo_cfg,
        adjacency=adjacency,
        time_limit=args.time_limit,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        decision_interval=args.decision_interval,
        reward_mode=args.reward_mode,
        reward_gamma=args.reward_gamma,
        seed=args.seed,
    )
    ss, as_ = env.state_sizes, env.action_sizes
    agents = {
        tid: DQNAgent(state_size=ss[tid], action_size=as_[tid])
        for tid in env.tls_ids
    }
    for tid in env.tls_ids:
        print(f"  {tid:<55s} state={ss[tid]:>3d} action={as_[tid]}")

    total_decisions = max(
        1, args.episodes * (args.time_limit // max(1, args.decision_interval))
    )
    eps_total = max(1, int(total_decisions * args.eps_warmup_frac))
    curric_phase1 = int(total_decisions * args.curriculum_frac)
    print(
        f"Curriculum: Phase 1 = decisions [0, {curric_phase1}) net_w=0; "
        f"Phase 2 ramp 0 -> {args.net_weight} over the rest "
        f"(total {total_decisions} decisions)."
    )

    log_path = log_dir / "train_log.csv"
    log_file = log_path.open("w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "episode", "tls_id", "decisions", "total_reward",
        "mean_reward_per_decision", "mean_loss", "epsilon",
        "net_arrived", "net_mean_wait", "wall_time_s",
    ])

    # best.pth is selected by the eval-aligned objective (network throughput,
    # tie-broken by lower mean wait), restricted to Phase-2 episodes. Raw
    # reward/decision is NOT usable here: the curriculum deliberately shifts
    # reward scale between phases, so a reward-"best" cherry-picks a brittle
    # snapshot (confirmed — it eval'd at -18% throughput while the converged
    # policy reached parity + 7% lower wait).
    best_score = (-1.0, -float("inf"))  # (arrived, -mean_wait)
    best_written = False
    decision_counter = 0

    for ep in range(1, args.episodes + 1):
        ep_start = time.time()
        env.seed = args.seed + ep - 1
        states = env.reset()

        ep_reward = {t: 0.0 for t in env.tls_ids}
        ep_losses = {t: [] for t in env.tls_ids}
        ep_decisions = 0
        done = False
        net_w = 0.0

        while not done:
            eps = linear_epsilon(decision_counter, eps_total)
            net_w = curriculum_net_weight(
                decision_counter, total_decisions,
                curric_phase1, args.net_weight,
            )
            env.set_reward_weights(1.0, net_w)
            actions = {
                t: agents[t].act(states[t], epsilon=eps)
                for t in env.tls_ids
            }
            next_states, rewards, done, _info = env.step(actions)

            for t in env.tls_ids:
                agents[t].remember(states[t], actions[t], rewards[t],
                                   next_states[t], done)
                loss = agents[t].learn()
                if loss is not None:
                    ep_losses[t].append(loss)
                ep_reward[t] += rewards[t]

            states = next_states
            decision_counter += 1
            ep_decisions += 1

        wall = time.time() - ep_start
        m = env.metrics_summary()
        net_reward = float(sum(ep_reward.values()))
        net_mean = net_reward / max(1, ep_decisions)

        for t in env.tls_ids:
            ml = float(np.mean(ep_losses[t])) if ep_losses[t] else 0.0
            writer.writerow([
                ep, t, ep_decisions, f"{ep_reward[t]:.2f}",
                f"{ep_reward[t] / max(1, ep_decisions):.4f}",
                f"{ml:.4f}", f"{eps:.3f}", "", "", "",
            ])
        writer.writerow([
            ep, "__network__", ep_decisions, f"{net_reward:.2f}",
            f"{net_mean:.4f}", "", f"{eps:.3f}",
            m["arrived"], f"{m['mean_wait']:.2f}", f"{wall:.1f}",
        ])
        log_file.flush()

        phase = 1 if decision_counter <= curric_phase1 else 2
        print(
            f"ep {ep:>3d}/{args.episodes}  net_reward={net_reward:>10.1f}  "
            f"arrived={m['arrived']:>5d}  net_wait={m['mean_wait']:>7.2f}  "
            f"eps={eps:.2f}  P{phase} net_w={net_w:.2f}  ({wall:.1f}s)"
        )

        def _meta(tid):
            u = env.units[tid]
            return {
                "episode": ep,
                "tls_id": tid,
                "green_phase_indices": u.green_phase_indices,
                "controlled_lanes": u.controlled_lanes,
                "reward_mode": args.reward_mode,
                "neighbor_ids": u.neighbor_ids,
                "state_layout": {"neighbor_aware": True, "neighbor_block": 6},
            }

        for tid in env.tls_ids:
            agents[tid].save(
                str(ckpt_root / tid / "last.pth"), meta=_meta(tid)
            )
        ep_score = (float(m["arrived"]), -float(m["mean_wait"]))
        if phase == 2 and ep_score > best_score:
            best_score = ep_score
            best_written = True
            for tid in env.tls_ids:
                meta = _meta(tid)
                meta["best_select"] = {
                    "by": "phase2_arrived_then_-mean_wait",
                    "episode": ep,
                    "arrived": int(m["arrived"]),
                    "mean_wait": float(m["mean_wait"]),
                }
                agents[tid].save(
                    str(ckpt_root / tid / "best.pth"), meta=meta
                )

    log_file.close()

    # Guarantee best.pth exists even if no Phase-2 episode ran (e.g. a very
    # short run or curriculum_frac≈1): fall back to the final converged
    # policy, which the diagnostic showed is the robust one anyway.
    if not best_written:
        print("No Phase-2 episode selected a best — falling back to the "
              "final (converged) policy as best.pth.")
        for tid in env.tls_ids:
            meta = _meta(tid)
            meta["best_select"] = {"by": "fallback_final_policy",
                                   "episode": ep}
            agents[tid].save(str(ckpt_root / tid / "best.pth"), meta=meta)

    env.stop()
    print(
        f"\nDone. best.pth = Phase-2 episode with arrived={int(best_score[0])} "
        f"mean_wait={-best_score[1]:.2f}"
        if best_written else "\nDone. best.pth = final converged policy."
    )
    print(f"Checkpoints: {ckpt_root}/<tls_id>/best.pth (and last.pth)")
    print(f"Log:         {log_path}")


if __name__ == "__main__":
    main()
