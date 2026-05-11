"""One-minute headless smoke test: does the DQN pipeline run end-to-end?

Verifies:
 1. ``SumoTrafficEnv`` opens and probes the network correctly.
 2. ``DQNAgent`` constructs with the right shapes.
 3. A handful of training transitions can be collected and learned from.

Run from ``SUMO/v2``:
    python ai/sanity_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from sumo_env import SumoTrafficEnv  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402


SUMO_CFG = "sim.sumocfg"
TLS_ID = "3153556582"
STEPS = 60


def main() -> None:
    start = time.time()
    env = SumoTrafficEnv(SUMO_CFG, TLS_ID, time_limit=STEPS, use_gui=False,
                         min_green=5, yellow_time=3,
                         reward_mode="differential")
    print(f"[ok] env probed: state_size={env.state_size} action_size={env.action_size}")

    agent = DQNAgent(env.state_size, env.action_size,
                     hidden_sizes=(32, 32), batch_size=8, target_sync_steps=10)
    print("[ok] agent built")

    state = env.reset()
    transitions = 0
    losses = []
    rewards = []
    done = False
    while not done:
        a = agent.act(state, epsilon=0.5)
        s2, r, done, info = env.step(a)
        agent.remember(state, a, r, s2, done)
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        rewards.append(r)
        state = s2
        transitions += 1

    env.stop_simulation()
    elapsed = time.time() - start
    print(f"[ok] {transitions} transitions, "
          f"mean_reward={float(np.mean(rewards)):.3f}, "
          f"learn_calls={len(losses)}, "
          f"last_loss={losses[-1] if losses else 0:.4f}, "
          f"elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
