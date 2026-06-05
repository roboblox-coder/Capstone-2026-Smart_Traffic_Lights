"""Plot MAPPO V2 training curves from ``ai/runs/v2_mappo/train_log.jsonl``.

Run from ``SUMO/v2``:
    python ai/v2/make_v2_plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    log_path = Path("ai/runs/v2_mappo/train_log.jsonl")
    if not log_path.exists():
        sys.exit(f"no log at {log_path}; train first with MAPPO trainer")

    updates, episodes_done, rewards, pol_losses, val_losses, entropies = [], [], [], [], [], []
    eval_updates, eval_waits, eval_throughputs = [], [], []

    with log_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("kind")
            if kind == "update":
                updates.append(row["update"])
                episodes_done.append(row["episodes_done"])
                rewards.append(row["reward_per_episode"])
                pol_losses.append(row.get("pol_loss", 0.0))
                val_losses.append(row.get("val_loss", 0.0))
                entropies.append(row.get("entropy", 0.0))
            elif kind == "eval":
                eval_updates.append(row["update"])
                eval_waits.append(row["wait_per_vehicle_mean"])
                eval_throughputs.append(row["throughput_mean"])

    out_dir = Path("ai/runs/v2_mappo")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Rewards
    if rewards:
        axes[0, 0].plot(episodes_done, rewards, label="Reward/Ep", color="tab:blue")
        axes[0, 0].set_title("Reward per Episode")
        axes[0, 0].set_xlabel("Episodes Done")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # 2. Eval waiting times
    if eval_waits:
        axes[0, 1].plot(eval_updates, eval_waits, label="Mean Wait/Veh", color="tab:orange")
        axes[0, 1].set_title("Eval Mean Wait per Vehicle")
        axes[0, 1].set_xlabel("Updates")
        axes[0, 1].set_ylabel("Wait Time (seconds)")
        axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # 3. Eval throughput
    if eval_throughputs:
        axes[1, 0].plot(eval_updates, eval_throughputs, label="Throughput", color="tab:green")
        axes[1, 0].set_title("Eval Mean Throughput")
        axes[1, 0].set_xlabel("Updates")
        axes[1, 0].set_ylabel("Vehicles Arrived")
        axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # 4. Losses
    if pol_losses and val_losses:
        ax4 = axes[1, 1]
        ax4.plot(updates, pol_losses, label="Policy Loss", color="tab:red", alpha=0.8)
        ax4.set_title("Policy & Value Loss")
        ax4.set_xlabel("Updates")
        ax4.set_ylabel("Policy Loss", color="tab:red")
        ax4.tick_params(axis="y", labelcolor="tab:red")
        
        ax4_twin = ax4.twinx()
        ax4_twin.plot(updates, val_losses, label="Value Loss", color="tab:purple", alpha=0.8)
        ax4_twin.set_ylabel("Value Loss", color="tab:purple")
        ax4_twin.tick_params(axis="y", labelcolor="tab:purple")
        ax4.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
