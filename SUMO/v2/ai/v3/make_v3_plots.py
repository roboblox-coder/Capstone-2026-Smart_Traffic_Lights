"""Turn the V3 AI's logs into visual proof it works.

Produces two PNGs (committed under ai/v3/plots/ so they render on GitHub):

  1. learning_curve.png  -- the agent's eval wait + throughput across
     training episodes (read from a run's train_log.jsonl). Shows it
     LEARNING (and, honestly, the instability spikes).
  2. comparison.png      -- grouped bars: V3 vs fixed-time vs native on
     throughput and wait (read from an eval_network.py summary). Shows
     the verifiable win over conventional fixed-time signal control.

Run from SUMO/v2:
    python ai/v3/make_v3_plots.py
    python ai/v3/make_v3_plots.py --run-dir ai/runs/v3_exp4 \
        --eval-file ai/logs/eval_v3_vs_fixedtime.txt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# name -> (mean_arrived, mean_backlog, mean_wait, mean_net) from a summary row
_ROW = re.compile(
    r"^(\S+)\s+([\d.]+)\s*\+/-\s*[\d.]+\s+([\d.]+)\s*\+/-\s*[\d.]+\s+"
    r"([\d.]+)\s*\+/-\s*[\d.]+\s+([\d.]+)")

_PRETTY = {
    "fixed_time": "Fixed-time\n(conventional)",
    "all_native_actuated": "SUMO actuated\n(strong baseline)",
    "coordinated_v3_frap_dqn": "V3 FRAP-DQN\n(this AI)",
    "coordinated_v2_frap": "V2 (failed)",
    "coordinated_dqn": "V1 DQN",
}
_COLOR = {
    "fixed_time": "#9e9e9e",
    "all_native_actuated": "#42a5f5",
    "coordinated_v3_frap_dqn": "#2e7d32",
    "coordinated_v2_frap": "#bdbdbd",
    "coordinated_dqn": "#cfcfcf",
}


def plot_learning_curve(run_dir: Path, out: Path) -> None:
    lines = [json.loads(x) for x in
             (run_dir / "train_log.jsonl").open(encoding="utf-8")]
    ev = [r for r in lines if r.get("kind") == "eval"]
    if not ev:
        print(f"  (no eval entries in {run_dir}, skipping learning curve)")
        return
    eps = [e["episodes_done"] if "episodes_done" in e else e["episode"]
           for e in ev]
    wait = [e["wait_per_vehicle"] for e in ev]
    thru = [e["throughput"] for e in ev]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(eps, wait, "o-", color="#2e7d32", label="V3 wait/veh")
    ax1.axhline(5504, ls="--", color="#42a5f5", lw=1,
                label="native-actuated (5504)")
    ax1.axhline(8777, ls="--", color="#9e9e9e", lw=1,
                label="fixed-time (8777)")
    ax1.set_xlabel("training episode")
    ax1.set_ylabel("wait / vehicle (s)  — lower is better", color="#2e7d32")
    ax1.set_yscale("log")  # collapses span 10^3..10^5
    ax1.tick_params(axis="y", labelcolor="#2e7d32")

    ax2 = ax1.twinx()
    ax2.plot(eps, thru, "s-", color="#ef6c00", alpha=0.6,
             label="V3 throughput")
    ax2.set_ylabel("throughput (vehicles arrived)", color="#ef6c00")
    ax2.tick_params(axis="y", labelcolor="#ef6c00")

    ax1.set_title(f"V3 FRAP-DQN learning curve ({run_dir.name})\n"
                  "eval every N episodes on held-out seeds")
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def parse_eval_summary(eval_file: Path) -> dict:
    rows = {}
    in_summary = False
    for line in eval_file.open(encoding="utf-8"):
        if "=== summary" in line:
            in_summary = True
            continue
        if in_summary:
            if line.startswith("===") and "summary" not in line:
                break
            m = _ROW.match(line.strip())
            if m:
                name, arr, _bk, wait, _net = m.groups()
                rows[name] = {"throughput": float(arr),
                              "wait": float(wait)}
    return rows


def plot_comparison(eval_file: Path, out: Path) -> None:
    rows = parse_eval_summary(eval_file)
    order = [n for n in ("fixed_time", "coordinated_v3_frap_dqn",
                         "all_native_actuated") if n in rows]
    if not order:
        print(f"  (no recognizable policies in {eval_file}, skipping)")
        return
    labels = [_PRETTY.get(n, n) for n in order]
    colors = [_COLOR.get(n, "#777") for n in order]

    fig, (axt, axw) = plt.subplots(1, 2, figsize=(10, 4.5))
    thru = [rows[n]["throughput"] for n in order]
    wait = [rows[n]["wait"] for n in order]

    bars1 = axt.bar(labels, thru, color=colors)
    axt.set_title("Throughput (vehicles arrived)\nhigher is better")
    axt.bar_label(bars1, fmt="%.0f", fontsize=9)
    axt.set_ylim(0, max(thru) * 1.15)

    bars2 = axw.bar(labels, wait, color=colors)
    axw.set_title("Wait per vehicle (s)\nlower is better")
    axw.bar_label(bars2, fmt="%.0f", fontsize=9)
    axw.set_ylim(0, max(wait) * 1.15)

    fig.suptitle("V3 FRAP-DQN vs conventional signal control "
                 "(calibrated corridor, 5-seed paired eval)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="ai/runs/v3_exp4",
                    help="Training run dir with train_log.jsonl.")
    ap.add_argument("--eval-file",
                    default="ai/v3/eval_v3_vs_fixedtime.txt",
                    help="eval_network.py summary output to chart "
                         "(committed copy works on a fresh clone).")
    ap.add_argument("--out-dir", default="ai/v3/plots")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("V3 plots:")
    rd = Path(args.run_dir)
    if (rd / "train_log.jsonl").exists():
        plot_learning_curve(rd, out / "learning_curve.png")
    else:
        print(f"  (run dir {rd} not found; skip learning curve)")
    ef = Path(args.eval_file)
    if ef.exists():
        plot_comparison(ef, out / "comparison.png")
    else:
        print(f"  (eval file {ef} not found; skip comparison)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
