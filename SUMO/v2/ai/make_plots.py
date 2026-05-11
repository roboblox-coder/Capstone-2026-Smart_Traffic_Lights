"""Plot training curves from ``ai/logs/train_log.csv``.

Run from ``SUMO/v2``:
    python ai/make_plots.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    log_path = Path("ai/logs/train_log.csv")
    if not log_path.exists():
        sys.exit(f"no log at {log_path}; train first with ai/train_dqn_sumo.py")

    episodes, rewards, waits, losses, epsilons = [], [], [], [], []
    with log_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
            waits.append(float(row["mean_waiting"]))
            losses.append(float(row["loss"]))
            epsilons.append(float(row["epsilon"]))

    out_dir = Path("ai/logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(episodes, rewards)
    axes[0, 0].set_title("Total reward / episode")
    axes[0, 0].set_xlabel("episode")

    axes[0, 1].plot(episodes, waits)
    axes[0, 1].set_title("Mean waiting time / episode")
    axes[0, 1].set_xlabel("episode")

    axes[1, 0].plot(episodes, losses)
    axes[1, 0].set_title("Mean loss / episode")
    axes[1, 0].set_xlabel("episode")

    axes[1, 1].plot(episodes, epsilons)
    axes[1, 1].set_title("Epsilon")
    axes[1, 1].set_xlabel("episode")

    fig.tight_layout()
    out_path = out_dir / "training_curve.png"
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
