"""Network-level evaluation: coordinated DQN vs fair baselines.

Compares three policies on identical SUMO seeds, scoring the WHOLE corridor
(all 12 TLS) rather than one intersection:

    all_fixed            every TLS round-robins its green slots
    all_native_actuated  SUMO's own .net.xml programs, untouched
    coordinated_dqn      per-TLS best.pth from ai/runs/coordinated/

Headline metrics are network throughput (``arrived``) and network mean wait,
with a ``% improvement vs all_native_actuated`` verdict — the multi-TLS
analogue of ai/eval.py's single-intersection verdict.

Run from ``SUMO/v2``::

    python ai/eval_network.py --episodes 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from multi_env import MultiTlsEnv, load_adjacency  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402
from eval import _fmt_mean_std  # noqa: E402  (reuse formatter)


def run_controlled(env: MultiTlsEnv, choose_actions) -> dict:
    states = env.reset()
    done = False
    while not done:
        actions = choose_actions(states, env)
        states, _r, done, _i = env.step(actions)
    return env.metrics_summary()


def run_native(env: MultiTlsEnv) -> dict:
    env.reset()
    done = False
    while not done:
        _s, done = env.passive_step()
    return env.metrics_summary()


def fixed_actions_factory():
    """Per-TLS round-robin: each light advances one green slot per decision."""
    ctr: dict = {}

    def choose(_states, env):
        out = {}
        for t in env.tls_ids:
            n = env.units[t]._num_green
            i = ctr.get(t, 0)
            out[t] = i % n
            ctr[t] = i + 1
        return out
    return choose


def load_coordinated_agents(ckpt_dir: str, env: MultiTlsEnv,
                            ckpt_name: str = "best.pth") -> dict:
    """One agent per TLS. A per-TLS shape mismatch falls back to fixed for
    that light only (mirrors the live demo's per-TLS fallback)."""
    agents = {}
    for tid in env.tls_ids:
        path = os.path.join(ckpt_dir, tid, ckpt_name)
        if not os.path.exists(path):
            print(f"  [fallback] {tid}: no checkpoint at {path}")
            agents[tid] = None
            continue
        a = DQNAgent.load_for_inference(path)
        if (a.state_size != env.state_sizes[tid]
                or a.action_size != env.action_sizes[tid]):
            print(f"  [fallback] {tid}: shape "
                  f"{a.state_size}x{a.action_size} vs env "
                  f"{env.state_sizes[tid]}x{env.action_sizes[tid]}")
            agents[tid] = None
        else:
            agents[tid] = a
    return agents


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sumo-cfg", default="sim.sumocfg")
    p.add_argument("--adjacency", default="ai/adjacency.json")
    p.add_argument("--ckpt-dir", default="ai/runs/coordinated/checkpoints")
    p.add_argument("--ckpt-name", default="best.pth",
                   help="Checkpoint file per TLS (best.pth or last.pth).")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--time-limit", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-green", type=int, default=5)
    p.add_argument("--yellow-time", type=int, default=5)
    p.add_argument("--decision-interval", type=int, default=5)
    p.add_argument("--out", default="ai/logs/eval_network.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adjacency = load_adjacency(args.adjacency)

    common = dict(
        sumo_cfg_file=args.sumo_cfg, adjacency=adjacency,
        time_limit=args.time_limit, min_green=args.min_green,
        yellow_time=args.yellow_time, decision_interval=args.decision_interval,
        reward_mode="max_pressure_net", seed=args.seed,
    )
    controlled_env = MultiTlsEnv(control_tls=True, **common)
    native_env = MultiTlsEnv(control_tls=False, **common)

    agents = load_coordinated_agents(args.ckpt_dir, controlled_env,
                                     args.ckpt_name)
    rr = fixed_actions_factory()

    def dqn_actions(states, env):
        out = {}
        for t in env.tls_ids:
            a = agents.get(t)
            if a is None:  # per-TLS fallback to round-robin
                n = env.units[t]._num_green
                out[t] = (env.units[t]._current_green_slot + 1) % n
            else:
                out[t] = a.act(states[t], epsilon=0.0)
        return out

    specs = [
        ("all_fixed", controlled_env,
         lambda: lambda e: run_controlled(e, fixed_actions_factory())),
        ("all_native_actuated", native_env, lambda: run_native),
        ("coordinated_dqn", controlled_env,
         lambda: lambda e: run_controlled(e, dqn_actions)),
    ]

    results = {n: [] for n, _, _ in specs}
    out_lines = []

    def emit(s):
        print(s)
        out_lines.append(s)

    emit(f"# network eval: episodes={args.episodes} "
         f"time_limit={args.time_limit}s base_seed={args.seed} "
         f"tls={len(controlled_env.tls_ids)}")
    emit(f"# regime: min_green={args.min_green} "
         f"yellow_time={args.yellow_time} "
         f"decision_interval={args.decision_interval}  "
         f"reward=max_pressure_net")

    for ep in range(1, args.episodes + 1):
        ep_seed = args.seed + ep - 1
        controlled_env.seed = ep_seed
        native_env.seed = ep_seed
        emit(f"\n--- episode {ep}  seed={ep_seed} ---")
        for name, env, make in specs:
            out = make()(env)
            results[name].append(out)
            emit(f"  {name:<22} arrived={out['arrived']:>5d}  "
                 f"backlog={out['backlog']:>5d}  "
                 f"wait/veh={out['wait_per_vehicle']:>8.2f}  "
                 f"net_mean_wait={out['mean_wait']:>9.2f}")

    controlled_env.stop()
    native_env.stop()

    emit("\n=== summary (network totals, mean +/- std across episodes) ===")
    header = (f"{'policy':<22}  {'arrived':>14}  {'backlog':>14}  "
              f"{'wait/veh':>14}  {'net_mean_wait':>14}")
    emit(header)
    emit("-" * len(header))
    for name, runs in results.items():
        emit(f"{name:<22}  "
             f"{_fmt_mean_std([r['arrived'] for r in runs], 10, 1)}  "
             f"{_fmt_mean_std([r['backlog'] for r in runs], 10, 1)}  "
             f"{_fmt_mean_std([r['wait_per_vehicle'] for r in runs], 10, 2)}  "
             f"{_fmt_mean_std([r['mean_wait'] for r in runs], 10, 2)}")

    base = "all_native_actuated"
    if results[base] and results["coordinated_dqn"]:
        def _mean(n, k):
            return float(np.mean([r[k] for r in results[n]]))

        nat_arr = _mean(base, "arrived")
        nat_wpv = _mean(base, "wait_per_vehicle")
        dqn_arr = _mean("coordinated_dqn", "arrived")
        dqn_wpv = _mean("coordinated_dqn", "wait_per_vehicle")
        arr_imp = (dqn_arr - nat_arr) / max(1e-9, nat_arr) * 100.0
        wait_imp = (nat_wpv - dqn_wpv) / max(1e-9, nat_wpv) * 100.0

        ns = len(results["coordinated_dqn"])
        arr_wins = sum(
            1 for i in range(ns)
            if results["coordinated_dqn"][i]["arrived"]
            > results[base][i]["arrived"]
        )
        wait_wins = sum(
            1 for i in range(ns)
            if results["coordinated_dqn"][i]["wait_per_vehicle"]
            < results[base][i]["wait_per_vehicle"]
        )

        emit("\n=== coordinated_dqn vs all_native_actuated (network) ===")
        emit(f"  throughput : {dqn_arr:8.1f} vs {nat_arr:8.1f}  "
             f"({arr_imp:+.1f}%, beats native on {arr_wins}/{ns} seeds)")
        emit(f"  wait/veh   : {dqn_wpv:8.2f} vs {nat_wpv:8.2f}  "
             f"({wait_imp:+.1f}%, beats native on {wait_wins}/{ns} seeds)")
        emit("")
        if arr_imp > 0 and wait_imp >= 0:
            emit("Verdict: coordinated DQN raises NETWORK throughput without "
                 "regressing network wait. Coordination objective met.")
        elif arr_imp > 0:
            emit(f"Verdict: coordinated DQN raises network throughput "
                 f"(+{arr_imp:.1f}%) but network wait regressed "
                 f"({wait_imp:+.1f}%). Tune reward_gamma.")
        else:
            emit("Verdict: coordinated DQN does NOT beat native on network "
                 "throughput. Retrain (more episodes / reward_gamma) before "
                 "wiring into the demo.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
