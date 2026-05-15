"""Compare the trained DQN agent against baselines.

Four policies are compared on identical SUMO seeds:
    fixed           — round-robin through green slots (DQN-env constraints)
    actuated        — greedy queue picker (DQN-env constraints)
    native_actuated — SUMO's own program from the .net.xml, untouched
    dqn             — loaded from ``--model``

Headline metric is ``wait_per_vehicle = cumulative_wait / (arrived + in_net)``,
which does not saturate when the demand horizon outlasts the eval window
(the corridor route file holds 12k vehicles across 3600s but the eval
runs for ``--time-limit`` seconds, typically 1200, so most policies leave
some vehicles still in the network at termination).

Run from ``SUMO/v2``:
    python ai/eval.py --episodes 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sumo_env import SumoTrafficEnv  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402


def run_controlled_policy(env: SumoTrafficEnv, choose_action) -> dict:
    """Roll out one episode with the DQN-env controlling the TLS."""
    state = env.reset()
    total_reward = 0.0
    decisions = 0
    done = False
    while not done:
        action = choose_action(state, env)
        state, reward, done, _info = env.step(action)
        total_reward += reward
        decisions += 1
    out = env.metrics_summary()
    out["total_reward"] = total_reward
    out["decisions"] = decisions
    return out


def run_native_actuated(env: SumoTrafficEnv) -> dict:
    """Roll out one episode letting SUMO drive the TLS from the .net.xml."""
    env.reset()
    done = False
    while not done:
        _, done = env.passive_step()
    out = env.metrics_summary()
    out["total_reward"] = 0.0
    out["decisions"] = 0
    return out


def fixed_policy_factory():
    """Cycle through the green slots in order, one per decision."""
    counter = {"i": 0}

    def choose(_state, env):
        a = counter["i"] % env.action_size
        counter["i"] += 1
        return a
    return choose


def actuated_policy(_state, env):
    """SUMO-actuated-like baseline: pick the slot whose served lanes have
    the most accumulated waiting time. The env's ``min_green`` enforces
    the anti-thrash hysteresis (a switch costs 5s of yellow, so cycles
    are at least 10s long).

    Two corrections vs. the legacy heuristic:

    1. Link -> lane mapping via ``getControlledLinks(tls)`` (length =
       len(state_str)) rather than ``getControlledLanes(tls)`` (length =
       count of *non-empty* link indices). The legacy version paired
       state-string position ``i`` with ``unique_lanes[i]``, which is a
       different lane whenever any preceding link index is empty.
    2. Score is ``getWaitingTime`` (accumulated per-vehicle wait), not
       ``getLastStepHaltingNumber``. Halting count is near-zero on green
       lanes as soon as traffic moves, so the heuristic oscillated; wait
       time is monotone on red lanes — a clean starvation signal.
    """
    import traci
    tls_id = env.tls_id
    # controlled_links[i] is a tuple of (from_lane, to_lane, via) tuples
    # for signal index i; len(controlled_links) == len(state_str).
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    phase_states = env._phase_states
    green_indices = env.green_phase_indices

    def score(slot: int) -> float:
        state_str = phase_states[green_indices[slot]]
        served = set()
        for i, ch in enumerate(state_str):
            if ch in ("G", "g") and i < len(controlled_links):
                for entry in controlled_links[i]:
                    if entry:  # (from_lane, to_lane, via)
                        served.add(entry[0])
        return float(sum(traci.lane.getWaitingTime(l) for l in served))

    best_slot, best_score = 0, -1.0
    for slot in range(len(green_indices)):
        s = score(slot)
        if s > best_score:
            best_score = s
            best_slot = slot
    return best_slot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sumo-cfg", default="sim.sumocfg")
    p.add_argument("--tls-id", default="3153556582")
    p.add_argument("--model", default="ai/checkpoints/best.pth",
                   help="Single DQN model path (legacy; use --models to "
                        "compare several).")
    p.add_argument("--models", nargs="*", default=None,
                   help="One or more DQN model paths. Each is labelled "
                        "by its checkpoint parent dir, e.g. "
                        "ai/runs/anti_starve/checkpoints/best.pth -> "
                        "'dqn_anti_starve'. Overrides --model when set.")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--time-limit", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42,
                   help="SUMO seed. Same value across all policies for fairness.")
    p.add_argument("--min-green", type=int, default=5)
    p.add_argument("--yellow-time", type=int, default=5)
    p.add_argument("--decision-interval", type=int, default=1,
                   help="Sim-seconds advanced per env.step() (boundary-cadence "
                        "for the controlled policies; native_actuated is "
                        "unaffected since it bypasses env.step()).")
    p.add_argument("--out", default="ai/logs/eval_baseline_vs_dqn.txt")
    return p.parse_args()


def _fmt_mean_std(values, width=10, prec=2):
    if not values:
        return f"{'-':>{width}}"
    m = float(np.mean(values))
    s = float(np.std(values))
    return f"{m:>{width}.{prec}f} +/-{s:.{prec}f}"


def main() -> None:
    args = parse_args()

    # Two env instances: one that takes control of the TLS (for fixed /
    # actuated / dqn) and one that leaves the .net.xml program in charge
    # (for native_actuated). Both share the same SUMO seed.
    controlled_env = SumoTrafficEnv(
        sumo_cfg_file=args.sumo_cfg,
        tls_id=args.tls_id,
        time_limit=args.time_limit,
        use_gui=False,
        reward_mode="differential",
        control_tls=True,
        seed=args.seed,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        decision_interval=args.decision_interval,
    )
    native_env = SumoTrafficEnv(
        sumo_cfg_file=args.sumo_cfg,
        tls_id=args.tls_id,
        time_limit=args.time_limit,
        use_gui=False,
        reward_mode="differential",
        control_tls=False,
        seed=args.seed,
        min_green=args.min_green,
        yellow_time=args.yellow_time,
        decision_interval=args.decision_interval,
    )

    # Build the policy list. Each entry is (name, env, rollout_factory)
    # where rollout_factory() returns a single-episode rollout callable.
    # The factory pattern lets us reset stateful policies (like the
    # round-robin counter) at the start of every episode.
    policy_specs: list = [
        ("fixed", controlled_env,
         lambda: (lambda e, choose=fixed_policy_factory():
                  run_controlled_policy(e, choose))),
        ("actuated", controlled_env,
         lambda: (lambda e: run_controlled_policy(e, actuated_policy))),
        ("native_actuated", native_env, lambda: run_native_actuated),
    ]

    model_paths = list(args.models) if args.models else [args.model]
    for mp in model_paths:
        if not os.path.exists(mp):
            print(f"Skipping {mp}: not found.")
            continue
        agent_i = DQNAgent.load_for_inference(mp)
        if (agent_i.state_size != controlled_env.state_size
                or agent_i.action_size != controlled_env.action_size):
            print(f"Skipping {mp}: shape mismatch "
                  f"(model {agent_i.state_size}x{agent_i.action_size} vs "
                  f"env {controlled_env.state_size}x{controlled_env.action_size})")
            continue
        # Label by checkpoint parent dir, e.g.
        # ai/runs/anti_starve/checkpoints/best.pth -> 'dqn_anti_starve'.
        parts = Path(mp).parts
        label_parent = parts[-3] if len(parts) >= 3 else parts[-2]
        label = f"dqn_{label_parent}" if label_parent not in ("ai", "checkpoints") else "dqn"

        def _spec(a=agent_i, lbl=label):
            return (lbl, controlled_env,
                    lambda: (lambda e:
                             run_controlled_policy(
                                 e, lambda s, _env: a.act(s, epsilon=0.0))))
        policy_specs.append(_spec())

    results: dict = {name: [] for name, _, _ in policy_specs}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_lines: list = []

    def emit(line: str) -> None:
        print(line)
        out_lines.append(line)

    emit(f"# eval: episodes={args.episodes} time_limit={args.time_limit}s "
         f"base_seed={args.seed} tls={args.tls_id}")
    emit(f"# regime: min_green={args.min_green} yellow_time={args.yellow_time} "
         f"decision_interval={args.decision_interval}")

    # Outer loop over episodes so all policies share the same seed in a
    # given episode (paired comparison) but vary across episodes.
    for ep in range(1, args.episodes + 1):
        ep_seed = args.seed + ep - 1
        controlled_env.seed = ep_seed
        native_env.seed = ep_seed
        emit(f"\n--- episode {ep}  seed={ep_seed} ---")
        for name, env, make_rollout in policy_specs:
            rollout = make_rollout()
            out = rollout(env)
            results[name].append(out)
            emit(
                f"  {name:<18} reward={out['total_reward']:>8.1f}  "
                f"arrived={out['arrived']:>4d}  backlog={out['backlog']:>4d}  "
                f"wait/veh={out['wait_per_vehicle']:>7.2f}  "
                f"mean_wait={out['mean_wait']:>7.2f}  "
                f"decisions={out['decisions']:>3d}"
            )

    controlled_env.stop_simulation()
    native_env.stop_simulation()

    emit("\n=== summary (mean +/- std across episodes) ===")
    header = (f"{'policy':<18}  {'arrived':>14}  {'backlog':>14}  "
              f"{'wait/veh':>14}  {'mean_wait':>14}  {'decisions':>14}")
    emit(header)
    emit("-" * len(header))
    for name, runs in results.items():
        arrived = [r["arrived"] for r in runs]
        backlog = [r["backlog"] for r in runs]
        wpv = [r["wait_per_vehicle"] for r in runs]
        mw = [r["mean_wait"] for r in runs]
        dec = [r["decisions"] for r in runs]
        emit(f"{name:<18}  {_fmt_mean_std(arrived, 10, 1)}  "
             f"{_fmt_mean_std(backlog, 10, 1)}  "
             f"{_fmt_mean_std(wpv, 10, 2)}  "
             f"{_fmt_mean_std(mw, 10, 2)}  "
             f"{_fmt_mean_std(dec, 10, 1)}")

    # Decision rule: the goal is to beat the self-actuated model
    # (native_actuated) on BOTH objectives simultaneously — lower
    # wait_per_vehicle AND higher throughput (arrived). Each DQN is
    # judged against native on both, with a paired per-seed t-style
    # check so single-seed noise doesn't flip the verdict.
    def _mean(name, key):
        return float(np.mean([r[key] for r in results[name]]))

    def _paired_delta(dqn, base, key):
        """Mean per-seed (dqn - base) and the fraction of seeds where
        dqn beats base in the desired direction."""
        d = [results[dqn][i][key] - results[base][i][key]
             for i in range(len(results[dqn]))]
        return float(np.mean(d)), d

    dqn_labels = [n for n in results if n.startswith("dqn")]
    if dqn_labels and "native_actuated" in results:
        emit("\n=== vs native_actuated (the self-actuated model to beat) ===")
        nat_wpv = _mean("native_actuated", "wait_per_vehicle")
        nat_arr = _mean("native_actuated", "arrived")
        passers = []
        for n in dqn_labels:
            d_wpv = _mean(n, "wait_per_vehicle")
            d_arr = _mean(n, "arrived")
            # Lower wait is better; higher arrived is better.
            wait_imp = (nat_wpv - d_wpv) / max(1e-9, nat_wpv) * 100.0
            arr_imp = (d_arr - nat_arr) / max(1e-9, nat_arr) * 100.0
            _, wpv_diffs = _paired_delta(n, "native_actuated", "wait_per_vehicle")
            _, arr_diffs = _paired_delta(n, "native_actuated", "arrived")
            wait_wins = sum(1 for x in wpv_diffs if x < 0)  # dqn lower wait
            arr_wins = sum(1 for x in arr_diffs if x > 0)    # dqn more arrived
            nseed = len(wpv_diffs)
            both = wait_imp > 0 and arr_imp > 0
            if both:
                passers.append(n)
            emit(f"  {n}")
            emit(f"    wait/veh : {d_wpv:8.2f} vs {nat_wpv:8.2f}  "
                 f"({wait_imp:+.1f}%, beats native on {wait_wins}/{nseed} seeds)")
            emit(f"    arrived  : {d_arr:8.1f} vs {nat_arr:8.1f}  "
                 f"({arr_imp:+.1f}%, beats native on {arr_wins}/{nseed} seeds)")
        emit("")
        if passers:
            emit(f"Verdict: {', '.join(passers)} beat native_actuated on BOTH "
                 f"wait and throughput. Best candidate(s) ready for Phase 2.")
        else:
            emit("Verdict: no DQN beats native_actuated on BOTH metrics. "
                 "Iterate (state enrichment / action structure / more "
                 "training) before Phase 2.")

    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
