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


def compute_vs_native_verdict(policy_runs: list, native_runs: list) -> dict:
    """Pure helper: compute mean throughput / wait, per-seed wins, and
    pct improvement of ``policy_runs`` vs ``native_runs``. Caller passes
    parallel lists of per-episode result dicts with keys ``arrived`` and
    ``wait_per_vehicle``. Returns:

        {
          "throughput_pct": float,   # positive = policy beats native
          "wait_pct":       float,   # positive = policy has lower wait
          "throughput_wins": int,    # seeds where policy > native arrived
          "wait_wins":      int,     # seeds where policy < native wait
          "n_seeds":        int,
          "policy_arr":     float,   # mean arrived
          "policy_wpv":     float,   # mean wait per vehicle
          "native_arr":     float,
          "native_wpv":     float,
        }
    """
    n = min(len(policy_runs), len(native_runs))
    if n == 0:
        return {"throughput_pct": 0.0, "wait_pct": 0.0,
                "throughput_wins": 0, "wait_wins": 0, "n_seeds": 0,
                "policy_arr": 0.0, "policy_wpv": 0.0,
                "native_arr": 0.0, "native_wpv": 0.0}
    p_arr = sum(r["arrived"] for r in policy_runs[:n]) / n
    p_wpv = sum(r["wait_per_vehicle"] for r in policy_runs[:n]) / n
    n_arr = sum(r["arrived"] for r in native_runs[:n]) / n
    n_wpv = sum(r["wait_per_vehicle"] for r in native_runs[:n]) / n
    arr_pct = (p_arr - n_arr) / max(1e-9, n_arr) * 100.0
    wait_pct = (n_wpv - p_wpv) / max(1e-9, n_wpv) * 100.0
    arr_wins = sum(1 for i in range(n)
                   if policy_runs[i]["arrived"] > native_runs[i]["arrived"])
    wait_wins = sum(1 for i in range(n)
                    if policy_runs[i]["wait_per_vehicle"]
                    < native_runs[i]["wait_per_vehicle"])
    return {
        "throughput_pct": arr_pct,
        "wait_pct": wait_pct,
        "throughput_wins": arr_wins,
        "wait_wins": wait_wins,
        "n_seeds": n,
        "policy_arr": p_arr,
        "policy_wpv": p_wpv,
        "native_arr": n_arr,
        "native_wpv": n_wpv,
    }


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


def load_coordinated_agents_v2(ckpt_path: str, env: MultiTlsEnv):
    """Load a V2 corridor policy from a MAPPO checkpoint.

    Returns ``(policy, choose_actions_callable)`` where the callable
    matches the (states, env) -> dict[tls_id, action] shape expected by
    ``run_controlled``. The ``states`` argument is ignored -- V2 reads
    its own FRAP-form batch from the env each tick.

    On missing checkpoint or shape mismatch, returns ``(None, None)``
    so callers can fall back the same way V1's missing-checkpoint path
    does.
    """
    if not os.path.exists(ckpt_path):
        print(f"  [v2 fallback] no checkpoint at {ckpt_path}")
        return None, None
    try:
        from v2.inference_adapter import V2CorridorPolicy
    except ImportError as exc:
        print(f"  [v2 fallback] inference adapter import failed: {exc}")
        return None, None
    try:
        policy = V2CorridorPolicy.load_for_inference(ckpt_path)
    except Exception as exc:
        print(f"  [v2 fallback] load failed: {exc}")
        return None, None
    # Refuse silent shape drift: env's tls_ids order MUST match the
    # checkpoint's. The policy enforces it, but failing here gives a
    # clearer message before the first action is dispatched.
    if list(env.tls_ids) != policy.tls_ids:
        print(f"  [v2 fallback] tls_ids mismatch: env "
              f"{list(env.tls_ids)} vs ckpt {policy.tls_ids}")
        return None, None
    adjacency = env.frap_adjacency_tensor()

    def choose(_states, e):
        batch = e.get_state_frap_batch()
        return policy.act(batch, adjacency, deterministic=True)

    return policy, choose


def load_v3_frap_dqn(ckpt_path: str, env):
    """Load the shared FRAP-DQN and return a (states, env)->actions dict
    callable, matching run_controlled's contract. None if missing."""
    import os
    if not os.path.exists(ckpt_path):
        print(f"  [v3 fallback] no checkpoint at {ckpt_path}")
        return None, None
    try:
        from v3.frap_dqn_agent import FRAPDQNAgent
        agent = FRAPDQNAgent.load_for_inference(ckpt_path)
    except Exception as exc:
        print(f"  [v3 fallback] load failed: {exc}")
        return None, None

    def choose(_states, e):
        batch = e.get_state_frap_batch()
        out = {}
        for i, tid in enumerate(batch["tls_ids"]):
            st = {"movement_features": batch["movement_features"][i],
                  "phase_movement_mask": batch["phase_movement_mask"][i],
                  "phase_mask": batch["phase_mask"][i]}
            out[tid] = agent.act(st, epsilon=0.0)
        return out
    return agent, choose


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sumo-cfg", default="sim.sumocfg")
    p.add_argument("--adjacency", default="ai/adjacency.json")
    p.add_argument("--ckpt-dir", default="ai/runs/coordinated/checkpoints")
    p.add_argument("--ckpt-name", default="best.pth",
                   help="Checkpoint file per TLS (best.pth or last.pth).")
    p.add_argument("--v2-ckpt",
                   default="ai/runs/v2_mappo/checkpoints/best.pth",
                   help="Corridor-level V2 (FRAP/GAT/MAPPO) checkpoint. "
                        "Compared as 'coordinated_v2_frap' alongside V1 "
                        "and native when the file exists.")
    p.add_argument("--v3-ckpt",
                   default="ai/runs/v3_frap_dqn/checkpoints/best.pth",
                   help="V3 FRAP-DQN checkpoint; added as "
                        "'coordinated_v3_frap_dqn' when present.")
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
        ("all_native_actuated", native_env, lambda: run_native),
    ]

    # V2 spec only joins the comparison when a real checkpoint is on
    # disk. Otherwise we'd be racing fixed/native against an
    # unitialized policy.
    _v2_policy, v2_choose = load_coordinated_agents_v2(
        args.v2_ckpt, controlled_env)
    if v2_choose is not None:
        specs.append(
            ("coordinated_v2_frap", controlled_env,
             lambda: lambda e: run_controlled(e, v2_choose))
        )

    _v3_agent, v3_choose = load_v3_frap_dqn(args.v3_ckpt, controlled_env)
    if v3_choose is not None:
        specs.append(
            ("coordinated_v3_frap_dqn", controlled_env,
             lambda: lambda e: run_controlled(e, v3_choose))
        )

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
    if results[base] and results.get("coordinated_dqn"):
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

    # V2-vs-native verdict (parallel to the V1 block above). Only emitted
    # when the V2 spec ran in this eval (i.e. v2_choose was not None).
    if "coordinated_v2_frap" in results and results["coordinated_v2_frap"]:
        v2 = compute_vs_native_verdict(
            results["coordinated_v2_frap"], results[base])
        emit("\n=== coordinated_v2_frap vs all_native_actuated (network) ===")
        emit(f"  throughput : {v2['policy_arr']:8.1f} vs "
             f"{v2['native_arr']:8.1f}  "
             f"({v2['throughput_pct']:+.1f}%, beats native on "
             f"{v2['throughput_wins']}/{v2['n_seeds']} seeds)")
        emit(f"  wait/veh   : {v2['policy_wpv']:8.2f} vs "
             f"{v2['native_wpv']:8.2f}  "
             f"({v2['wait_pct']:+.1f}%, beats native on "
             f"{v2['wait_wins']}/{v2['n_seeds']} seeds)")
        emit("")
        # Honest gate from RUN_AI.md: +6pp throughput AND -12pp wait.
        passes_throughput = v2["throughput_pct"] >= 6.0
        passes_wait = v2["wait_pct"] >= 12.0
        if passes_throughput and passes_wait:
            emit("Verdict: V2 clears the honest gate vs native "
                 f"(throughput +{v2['throughput_pct']:.1f}% >= +6%, "
                 f"wait +{v2['wait_pct']:.1f}% lower >= +12%). "
                 "Eligible for promotion to live demo.")
        else:
            reasons = []
            if not passes_throughput:
                reasons.append(f"throughput {v2['throughput_pct']:+.1f}% < +6%")
            if not passes_wait:
                reasons.append(f"wait {v2['wait_pct']:+.1f}% < +12% lower")
            emit("Verdict: V2 does NOT clear the honest gate vs native "
                 f"({'; '.join(reasons)}). Do not promote.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
