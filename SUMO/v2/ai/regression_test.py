"""Phase 0 regression net: does V1 still win?

A single command answering "did anything we changed silently break the V1
win?" Pins V1's headline numbers to ``baseline_v1.json`` and re-runs V1
under identical regime on a fixed seed set, gating on two checks:

  (A) Mean of each metric stays inside pinned mean +/- 2*sigma.
  (B) Paired-seed delta (current - pinned) per metric has its 95% CI
      containing 0 (no detectable distribution shift). A mean-only check
      misses shifts where variance ate the move.

Both gates must pass; exit code is non-zero on failure so the script
slots into pre-commit / CI hooks.

Usage (run from ``SUMO/v2``):

    # First-time setup, populate the pinned baseline (slow):
    python ai/regression_test.py --write-baseline --n-seeds 10

    # Every subsequent commit touching ai/, runs in CI:
    python ai/regression_test.py            # exits 0 iff V1 still wins
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from multi_env import MultiTlsEnv, load_adjacency  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402
from eval_network import run_controlled  # noqa: E402

# Two-tailed t critical values at alpha=0.05. Inlined to avoid a scipy
# dependency for a 5-row table. Indexed by degrees of freedom (n-1).
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    14: 2.145, 19: 2.093, 29: 2.045,
}


def t_crit_95(n: int) -> float:
    """Two-tailed 95% t critical for sample size n. Falls back to 1.96
    (normal limit) for n > 30 where the table isn't needed."""
    df = n - 1
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    if df > 30:
        return 1.96
    # Linear interp between bracketing entries — coarse but adequate.
    keys = sorted(_T_CRIT_95)
    for i in range(len(keys) - 1):
        if keys[i] < df < keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            f = (df - lo) / (hi - lo)
            return _T_CRIT_95[lo] + f * (_T_CRIT_95[hi] - _T_CRIT_95[lo])
    raise ValueError(f"n={n} (df={df}) outside the inlined t-table")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def run_v1_on_seeds(
    seeds: list,
    sumo_cfg: str,
    adjacency_path: str,
    ckpt_dir: str,
    ckpt_name: str,
    time_limit: int,
    min_green: int,
    yellow_time: int,
    decision_interval: int,
) -> list:
    """Run the V1 coordinated DQN on each seed. Returns one metrics dict
    per seed in input order.

    Strict-mode contract: if ANY of the 12 TLS lacks a valid checkpoint
    (missing file OR shape mismatch vs env), this raises rather than
    silently substituting round-robin. The regression baseline pins the
    full-agent V1; partial-V1 measurements would lie about the gate.
    """
    adjacency = load_adjacency(adjacency_path)
    env = MultiTlsEnv(
        sumo_cfg_file=sumo_cfg, adjacency=adjacency,
        time_limit=time_limit, min_green=min_green,
        yellow_time=yellow_time, decision_interval=decision_interval,
        reward_mode="max_pressure_net", control_tls=True, seed=seeds[0],
    )

    agents: dict = {}
    missing: list = []
    for tid in env.tls_ids:
        path = os.path.join(ckpt_dir, tid, ckpt_name)
        if not os.path.exists(path):
            missing.append((tid, "no_checkpoint", path))
            continue
        a = DQNAgent.load_for_inference(path)
        if (a.state_size != env.state_sizes[tid]
                or a.action_size != env.action_sizes[tid]):
            missing.append((
                tid, "shape_mismatch",
                f"{a.state_size}x{a.action_size} vs "
                f"{env.state_sizes[tid]}x{env.action_sizes[tid]}",
            ))
            continue
        agents[tid] = a

    if missing:
        env.stop()
        lines = "\n".join(f"  {tid}: {reason} ({detail})"
                          for tid, reason, detail in missing)
        raise RuntimeError(
            f"Regression test requires all {len(env.tls_ids)} TLS to "
            f"have valid V1 checkpoints. Missing/invalid:\n{lines}"
        )

    def dqn_actions(states, e):
        return {t: agents[t].act(states[t], epsilon=0.0) for t in e.tls_ids}

    results = []
    for ep, ep_seed in enumerate(seeds, 1):
        env.seed = int(ep_seed)
        out = run_controlled(env, dqn_actions)
        # Keep only the fields the regression cares about; metrics_summary
        # returns more but we don't want schema drift to break baseline
        # comparisons.
        results.append({
            "seed": int(ep_seed),
            "wait_per_vehicle": float(out["wait_per_vehicle"]),
            "throughput": int(out["arrived"]),
            "backlog": int(out["backlog"]),
            "mean_wait": float(out["mean_wait"]),
        })
        print(f"  seed={ep_seed:>4d}  "
              f"wait/veh={out['wait_per_vehicle']:>8.2f}  "
              f"throughput={out['arrived']:>5d}  "
              f"backlog={out['backlog']:>5d}")
    env.stop()
    return results


def write_baseline(path: Path, results: list, args: argparse.Namespace) -> None:
    wpv = np.array([r["wait_per_vehicle"] for r in results], dtype=float)
    arr = np.array([r["throughput"] for r in results], dtype=float)
    sumo_cfg_abs = str(Path(args.sumo_cfg).resolve())

    payload = {
        "schema_version": 1,
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "sumocfg_path": args.sumo_cfg,
        "sumocfg_sha256": _sha256_file(sumo_cfg_abs),
        "regime": {
            "min_green": args.min_green,
            "yellow_time": args.yellow_time,
            "decision_interval": args.decision_interval,
            "time_limit": args.time_limit,
            "reward_mode": "max_pressure_net",
            "adjacency": args.adjacency,
            "ckpt_dir": args.ckpt_dir,
            "ckpt_name": args.ckpt_name,
        },
        "seeds": [int(r["seed"]) for r in results],
        "per_seed": {str(r["seed"]): r for r in results},
        "wait_per_vehicle": {
            "mean": float(wpv.mean()),
            "sigma": float(wpv.std(ddof=1)) if len(wpv) > 1 else 0.0,
            "n": int(len(wpv)),
        },
        "throughput": {
            "mean": float(arr.mean()),
            "sigma": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    print(f"  wait/veh:    {payload['wait_per_vehicle']['mean']:.2f} "
          f"+/- {payload['wait_per_vehicle']['sigma']:.2f}  "
          f"(n={payload['wait_per_vehicle']['n']})")
    print(f"  throughput:  {payload['throughput']['mean']:.2f} "
          f"+/- {payload['throughput']['sigma']:.2f}  "
          f"(n={payload['throughput']['n']})")


def _gate_metric(
    name: str,
    pinned_mean: float,
    pinned_sigma: float,
    current_values: list,
    pinned_values: list,
    higher_is_better: bool,
) -> tuple:
    """Returns (passed: bool, lines: list[str]) for one metric.

    Two checks:
      (A) current_mean inside pinned_mean +/- 2*pinned_sigma
      (B) 95% CI of paired-seed deltas (current - pinned) contains 0
    """
    cur = np.array(current_values, dtype=float)
    pin = np.array(pinned_values, dtype=float)
    cur_mean = float(cur.mean())

    # (A) 2-sigma band
    lo = pinned_mean - 2.0 * pinned_sigma
    hi = pinned_mean + 2.0 * pinned_sigma
    a_pass = (lo <= cur_mean <= hi)

    # (B) Paired-seed CI
    deltas = cur - pin
    n = len(deltas)
    d_mean = float(deltas.mean())
    d_sigma = float(deltas.std(ddof=1)) if n > 1 else 0.0
    se = d_sigma / math.sqrt(n) if n > 0 else 0.0
    tcrit = t_crit_95(n)
    ci_lo = d_mean - tcrit * se
    ci_hi = d_mean + tcrit * se
    b_pass = (ci_lo <= 0.0 <= ci_hi) or (d_sigma == 0.0 and d_mean == 0.0)

    arrow = "v" if higher_is_better else "^"  # which direction is "worse"
    direction = ("more" if higher_is_better
                 else "less") + " is better"

    lines = [
        f"  {name}  ({direction})",
        f"    pinned    : {pinned_mean:>10.3f} +/- {pinned_sigma:.3f}  "
        f"(2-sigma band [{lo:.3f}, {hi:.3f}])",
        f"    current   : {cur_mean:>10.3f}  (n={n})",
        f"    (A) within 2-sigma band       : "
        f"{'PASS' if a_pass else 'FAIL'}",
        f"    paired delta: mean={d_mean:>+10.3f}  "
        f"se={se:.3f}  t_crit={tcrit:.3f}  "
        f"95%CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]",
        f"    (B) paired-CI contains 0      : "
        f"{'PASS' if b_pass else 'FAIL'}",
    ]
    if not a_pass:
        side = "above" if cur_mean > hi else "below"
        lines.append(f"    -> current mean is {side} the 2-sigma band; "
                     f"investigate ({arrow}).")
    if not b_pass:
        lines.append(f"    -> paired-CI excludes 0; statistically "
                     f"distinguishable shift from V1 baseline ({arrow}).")
    return (a_pass and b_pass, lines)


def gate_against_baseline(baseline: dict, results: list) -> int:
    """Returns 0 if both metrics pass both gates, 1 otherwise. Prints
    a human-readable report."""
    # Pair on seed order; baseline's seeds[:len(results)] should equal
    # results' seeds because we drove the runner from those same seeds.
    pinned_seeds = baseline["seeds"][:len(results)]
    current_seeds = [r["seed"] for r in results]
    if pinned_seeds != current_seeds:
        print(f"ERROR: seed order mismatch.\n"
              f"  pinned : {pinned_seeds}\n"
              f"  current: {current_seeds}\n"
              f"Regression test requires the first N pinned seeds to "
              f"match the seeds the runner was driven with.")
        return 1

    pinned_per_seed = baseline["per_seed"]

    print("\n=== regression gates ===")

    wpv_pass, wpv_lines = _gate_metric(
        "wait_per_vehicle",
        pinned_mean=baseline["wait_per_vehicle"]["mean"],
        pinned_sigma=baseline["wait_per_vehicle"]["sigma"],
        current_values=[r["wait_per_vehicle"] for r in results],
        pinned_values=[pinned_per_seed[str(s)]["wait_per_vehicle"]
                       for s in current_seeds],
        higher_is_better=False,
    )
    for line in wpv_lines:
        print(line)

    arr_pass, arr_lines = _gate_metric(
        "throughput",
        pinned_mean=baseline["throughput"]["mean"],
        pinned_sigma=baseline["throughput"]["sigma"],
        current_values=[r["throughput"] for r in results],
        pinned_values=[pinned_per_seed[str(s)]["throughput"]
                       for s in current_seeds],
        higher_is_better=True,
    )
    for line in arr_lines:
        print(line)

    overall = wpv_pass and arr_pass
    print("\n=== VERDICT ===")
    if overall:
        print("PASS: V1 still wins. No detectable regression on either "
              "metric on the regression seed set.")
        return 0
    print("FAIL: V1 regression detected. Either revert the offending "
          "change OR re-baseline (`--write-baseline`) if the new numbers "
          "are an intentional, validated improvement.")
    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sumo-cfg", default="sim.sumocfg")
    p.add_argument("--adjacency", default="ai/adjacency.json")
    p.add_argument("--ckpt-dir", default="ai/runs/coordinated/checkpoints")
    p.add_argument("--ckpt-name", default="best.pth")
    p.add_argument("--baseline-path", default="ai/baseline_v1.json")
    p.add_argument("--time-limit", type=int, default=1200)
    p.add_argument("--min-green", type=int, default=5)
    p.add_argument("--yellow-time", type=int, default=5)
    p.add_argument("--decision-interval", type=int, default=5)
    p.add_argument("--write-baseline", action="store_true",
                   help="Run V1 and write baseline_v1.json. Slow.")
    p.add_argument("--n-seeds", type=int, default=10,
                   help="Seeds 42..42+N-1. Used with --write-baseline "
                        "(default 10). In gate mode, the first 5 pinned "
                        "seeds are re-run.")
    p.add_argument("--regression-n", type=int, default=5,
                   help="In gate mode, re-run this many of the pinned "
                        "seeds (default 5).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    baseline_path = Path(args.baseline_path)

    if args.write_baseline:
        seeds = list(range(42, 42 + args.n_seeds))
        print(f"# write_baseline: V1 on seeds {seeds} "
              f"(time_limit={args.time_limit}s, "
              f"regime min_green={args.min_green} "
              f"yellow={args.yellow_time} "
              f"decision_interval={args.decision_interval})")
        results = run_v1_on_seeds(
            seeds=seeds,
            sumo_cfg=args.sumo_cfg, adjacency_path=args.adjacency,
            ckpt_dir=args.ckpt_dir, ckpt_name=args.ckpt_name,
            time_limit=args.time_limit, min_green=args.min_green,
            yellow_time=args.yellow_time,
            decision_interval=args.decision_interval,
        )
        write_baseline(baseline_path, results, args)
        return 0

    # Gate mode.
    if not baseline_path.exists():
        print(f"ERROR: no pinned baseline at {baseline_path}.\n"
              f"Generate it first (slow, ~5-10 min on 10 seeds):\n"
              f"  python ai/regression_test.py --write-baseline "
              f"--n-seeds 10")
        return 2

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    n = min(args.regression_n, len(baseline["seeds"]))
    seeds = list(baseline["seeds"][:n])

    # Sanity: warn (don't fail) if sumocfg has changed since baseline
    # was pinned. A change can be intentional (calibration landed); the
    # test still runs, but the user should know.
    cur_sha = _sha256_file(str(Path(args.sumo_cfg).resolve()))
    if cur_sha != baseline.get("sumocfg_sha256"):
        print(f"WARNING: {args.sumo_cfg} sha256 has changed since "
              f"baseline was pinned. Gate results may not be meaningful "
              f"-- consider re-baselining.")
        print(f"  pinned : {baseline.get('sumocfg_sha256')}")
        print(f"  current: {cur_sha}")

    print(f"# regression: V1 on seeds {seeds} (re-running pinned set)")
    results = run_v1_on_seeds(
        seeds=seeds,
        sumo_cfg=args.sumo_cfg, adjacency_path=args.adjacency,
        ckpt_dir=args.ckpt_dir, ckpt_name=args.ckpt_name,
        time_limit=args.time_limit, min_green=args.min_green,
        yellow_time=args.yellow_time,
        decision_interval=args.decision_interval,
    )
    return gate_against_baseline(baseline, results)


if __name__ == "__main__":
    sys.exit(main())
