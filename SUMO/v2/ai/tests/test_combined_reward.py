"""Regression test for F1: the ``combined`` reward's throughput term must
be alive in multi-light training.

Before the fix, ``arrived_term`` read ``_arrived_since_reward``, which only
SumoTrafficEnv._sim_tick() increments -- but MultiTlsEnv owns the clock and
never calls it, so the throughput term was silently 0 in every V3 run and
``--reward-alpha`` did nothing. The fix counts per-light local outflow via
snapshot_served(). This test builds a real MultiTlsEnv on the calibrated
corridor and asserts (a) vehicles are actually served at most lights and
(b) that served count flows through the ``combined`` reward exactly.
"""
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))   # SUMO/v2/ai

from multi_env import MultiTlsEnv, load_adjacency  # noqa: E402

SUMO_CFG = "sim_calibrated.sumocfg"
ADJ = "ai/adjacency.json"
SWITCH_PENALTY = 0.1   # SumoTrafficEnv default; MultiTlsEnv units inherit it
N_DECISIONS = 60


def test_combined_reward_throughput_term_alive():
    # beta=0 isolates the throughput (alpha) term so the reward equals
    # exactly  alpha*served - switch_penalty*switched  -- a clean, exact
    # check that the previously-dead term is wired in. With beta=0 the
    # wait term contributes nothing, and arrived_term == served because
    # _arrived_since_reward is 0 in multi-light mode.
    alpha = 1.0
    env = MultiTlsEnv(
        sumo_cfg_file=SUMO_CFG,
        adjacency=load_adjacency(ADJ),
        time_limit=1200,
        reward_mode="combined",
        reward_alpha=alpha,
        reward_beta=0.0,
        seed=7,
    )
    served_total = {t: 0 for t in env.tls_ids}
    any_flow = False
    try:
        env.reset()
        rng = np.random.default_rng(0)
        for _ in range(N_DECISIONS):
            actions = {t: int(rng.integers(0, env.units[t]._num_green))
                       for t in env.tls_ids}
            _s, rewards, done, infos = env.step(actions)
            for t in env.tls_ids:
                served = env.units[t]._served_since_reward
                served_total[t] += served
                if served > 0:
                    any_flow = True
                expected = alpha * served - (
                    SWITCH_PENALTY if infos[t]["switched"] else 0.0)
                assert abs(rewards[t] - expected) < 1e-5, (
                    f"{t}: combined reward {rewards[t]} != "
                    f"alpha*served-penalty {expected} "
                    f"(served={served}, switched={infos[t]['switched']})")
            if done:
                break
    finally:
        env.stop()

    # F1 core: per-light outflow is non-zero across the corridor, i.e. the
    # signal the term reads is no longer dead.
    lights_with_flow = sum(1 for t in env.tls_ids if served_total[t] > 0)
    assert lights_with_flow >= len(env.tls_ids) / 2, (
        f"only {lights_with_flow}/{len(env.tls_ids)} lights served any "
        f"vehicles over {N_DECISIONS} decisions -- outflow term looks dead")
    # And non-zero outflow actually reached the reward (alpha is alive).
    assert any_flow, ("no light ever served a vehicle; cannot prove the "
                      "throughput term is alive")


if __name__ == "__main__":
    test_combined_reward_throughput_term_alive()
    print("test_combined_reward OK")
