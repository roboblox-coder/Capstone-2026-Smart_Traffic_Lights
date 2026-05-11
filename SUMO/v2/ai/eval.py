"""Compare the trained DQN agent against ``fixed`` and ``actuated`` baselines.

Each policy is run headlessly through SUMO for ``--episodes`` rollouts and we
report mean cumulative reward and mean total waiting-time per episode.

Run from ``SUMO/v2``:
    python ai/eval.py --episodes 3
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


def run_policy(env: SumoTrafficEnv, choose_action):
    """Roll out one episode using ``choose_action(state, env) -> int``."""
    state = env.reset()
    total_reward = 0.0
    total_wait = 0.0
    steps = 0
    done = False
    while not done:
        action = choose_action(state, env)
        state, reward, done, info = env.step(action)
        total_reward += reward
        total_wait += env._total_waiting_time()
        steps += 1
    return {
        "total_reward": total_reward,
        "mean_wait": total_wait / max(1, steps),
        "decisions": steps,
    }


def fixed_policy_factory():
    """Cycle through the green slots in order, one per decision."""
    counter = {"i": 0}

    def choose(_state, env):
        a = counter["i"] % env.action_size
        counter["i"] += 1
        return a
    return choose


def actuated_policy(_state, env):
    """Pick the green slot whose lanes currently have the most queueing."""
    import traci
    lanes = env.controlled_lanes
    phase_states = env._phase_states
    green_indices = env.green_phase_indices
    # For each green phase, score by total queue length on lanes it serves.
    best_slot, best_score = 0, -1.0
    for slot, phase_idx in enumerate(green_indices):
        state_str = phase_states[phase_idx]
        score = 0.0
        for ch, lane in zip(state_str[:len(lanes)], lanes):
            if ch in ("G", "g"):
                score += traci.lane.getLastStepHaltingNumber(lane)
        if score > best_score:
            best_score = score
            best_slot = slot
    return best_slot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sumo-cfg", default="sim.sumocfg")
    p.add_argument("--tls-id", default="3153556582")
    p.add_argument("--model", default="ai/checkpoints/best.pth")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--time-limit", type=int, default=1200)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = SumoTrafficEnv(
        sumo_cfg_file=args.sumo_cfg,
        tls_id=args.tls_id,
        time_limit=args.time_limit,
        use_gui=False,
        reward_mode="differential",
    )

    policies: dict = {"fixed": fixed_policy_factory(),
                      "actuated": actuated_policy}

    if os.path.exists(args.model):
        agent = DQNAgent.load_for_inference(args.model)
        if (agent.state_size == env.state_size
                and agent.action_size == env.action_size):
            policies["dqn"] = lambda s, _env: agent.act(s, epsilon=0.0)
        else:
            print(f"Skipping DQN: shape mismatch "
                  f"(model {agent.state_size}x{agent.action_size} vs "
                  f"env {env.state_size}x{env.action_size})")
    else:
        print(f"No model at {args.model} — comparing baselines only.")

    results: dict = {name: [] for name in policies}
    for name, choose in policies.items():
        print(f"\n=== {name} ===")
        for ep in range(1, args.episodes + 1):
            out = run_policy(env, choose)
            results[name].append(out)
            print(f"  ep {ep}  reward={out['total_reward']:.1f}  "
                  f"mean_wait={out['mean_wait']:.2f}  "
                  f"decisions={out['decisions']}")

    env.stop_simulation()

    print("\n=== summary ===")
    print(f"{'policy':<10}  {'reward (mean)':>14}  {'wait (mean)':>12}")
    for name, runs in results.items():
        rmean = float(np.mean([r["total_reward"] for r in runs]))
        wmean = float(np.mean([r["mean_wait"] for r in runs]))
        print(f"{name:<10}  {rmean:>14.1f}  {wmean:>12.2f}")


if __name__ == "__main__":
    main()
