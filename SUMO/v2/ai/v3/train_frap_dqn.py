"""Train the shared FRAP Double-DQN on the calibrated corridor.

One shared FRAPDQNAgent drives all 12 lights; every light contributes
its own transitions to the shared replay each decision. Logs TD-loss and
periodic eval (wait/veh, throughput) as JSONL so the LEARNING GATE is
visible early: TD-loss should fall AND eval wait should trend down by
~episode 30. If it is flat (the V2 failure), stop before a long run.

Run from SUMO/v2:
    python ai/v3/train_frap_dqn.py --episodes 30 --time-limit 1200
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))     # SUMO/v2/ai

from multi_env import MultiTlsEnv, load_adjacency  # noqa: E402
from v3.frap_dqn_agent import FRAPDQNAgent          # noqa: E402


def _eval(agent, env, seeds, time_limit):
    waits, arrived = [], []
    for s in seeds:
        env.seed = s
        env.reset()
        done = False
        while not done:
            batch = env.get_state_frap_batch()
            actions = {}
            for i, tid in enumerate(batch["tls_ids"]):
                st = {"movement_features": batch["movement_features"][i],
                      "phase_movement_mask":
                          batch["phase_movement_mask"][i],
                      "phase_mask": batch["phase_mask"][i]}
                actions[tid] = agent.act(st, epsilon=0.0)
            _s, _r, done, _i = env.step(actions)
        m = env.metrics_summary()
        waits.append(m["wait_per_vehicle"])
        arrived.append(m["arrived"])
    return float(np.mean(waits)), float(np.mean(arrived))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumo-cfg", default="sim_calibrated.sumocfg")
    ap.add_argument("--adjacency", default="ai/adjacency.json")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--time-limit", type=int, default=1200)
    ap.add_argument("--yellow-time", type=int, default=3,
                    help="Yellow (s) per switch. 3 matches the native "
                         "corridor majority; 5 was a ~2s/cycle handicap "
                         "vs the baseline at several lights.")
    ap.add_argument("--decision-interval", type=int, default=5,
                    help="Seconds each decision holds the green. The di2 "
                         "run (best model to date) used 2 — min_green "
                         "still prevents thrash; finer decisions let the "
                         "policy react faster. Was previously only "
                         "settable by editing MultiTlsEnv's default.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-episodes", type=int, default=None,
                    help="Episodes to decay epsilon over. Default "
                         "max(20, episodes//2) so long runs don't spend "
                         "most of training in near-pure exploitation.")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, nargs="+",
                    default=[1042, 1043, 1044, 1045, 1046])
    ap.add_argument("--reward-mode", default="max_pressure_net",
                    help="Env reward. 'combined' directly rewards "
                         "throughput + wait (no gridlock loophole that "
                         "max_pressure has on heavy demand).")
    ap.add_argument("--gamma", type=float, default=0.95,
                    help="DQN discount. 0.99 for longer-horizon credit.")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--target-sync", type=int, default=500)
    ap.add_argument("--tau", type=float, default=0.0,
                    help="Polyak soft target-update rate (e.g. 0.005). "
                         "0 = hard sync. Fixes mid-training collapse.")
    ap.add_argument("--n-step", type=int, default=3,
                    help="n-step return horizon. With 5-10s decisions, "
                         "1-step (gamma 0.95) sees too short a horizon to "
                         "value releasing a platoon to the next light; "
                         "n-step extends it without target blow-up.")
    ap.add_argument("--learn-per-decision", type=int, default=2,
                    help="agent.learn() calls per env decision. 12 fresh "
                         "transitions arrive per decision; 2-4 uses the "
                         "gradient budget while keeping replay ratio sane.")
    ap.add_argument("--reward-alpha", type=float, default=1.0,
                    help="combined reward: throughput (arrived) weight.")
    ap.add_argument("--reward-beta", type=float, default=0.05,
                    help="combined reward: wait-reduction weight. Raise "
                         "to optimize delay harder (toward native's wait).")
    ap.add_argument("--out-dir", default="ai/runs/v3_frap_dqn")
    args = ap.parse_args()
    if args.eps_decay_episodes is None:
        # Decay over ~half the run (floor 20) so a long run isn't mostly
        # near-pure exploitation filling the buffer with on-policy data
        # (the recipe for the observed mid-run collapse, finding F7).
        args.eps_decay_episodes = max(20, args.episodes // 2)

    adjacency = load_adjacency(args.adjacency)
    common = dict(sumo_cfg_file=args.sumo_cfg, adjacency=adjacency,
                  time_limit=args.time_limit, yellow_time=args.yellow_time,
                  decision_interval=args.decision_interval,
                  reward_mode=args.reward_mode,
                  reward_alpha=args.reward_alpha, reward_beta=args.reward_beta,
                  control_tls=True, seed=args.seed)
    env = MultiTlsEnv(**common)
    eval_env = MultiTlsEnv(**common)

    env.reset()
    b0 = env.get_state_frap_batch()
    p_max = b0["phase_mask"].shape[1]
    m_max = b0["movement_features"].shape[1]
    agent = FRAPDQNAgent(mov_feat_dim=5, p_max=p_max, m_max=m_max,
                         gamma=args.gamma, lr=args.lr,
                         target_sync_steps=args.target_sync, tau=args.tau,
                         n_step=args.n_step)

    out = Path(args.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.jsonl").open("w", encoding="utf-8")

    def emit(d):
        log.write(json.dumps(d) + "\n")
        log.flush()

    best_score = float("-inf")
    for ep in range(1, args.episodes + 1):
        frac = min(1.0, (ep - 1) / max(1, args.eps_decay_episodes))
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)
        env.seed = 10000 + args.seed + ep   # disjoint from eval seeds 42-46
        env.reset()
        done = False
        losses = []
        # Per-light pending deques for n-step returns: hold recent
        # (state, action, reward) tuples until n accumulate, then push the
        # n-step transition (s_t, a_t, sum gamma^k r_{t+k}, s_{t+n}, done).
        pending = {tid: deque() for tid in env.tls_ids}
        while not done:
            batch = env.get_state_frap_batch()
            states = {}
            actions = {}
            for i, tid in enumerate(batch["tls_ids"]):
                st = {"movement_features": batch["movement_features"][i],
                      "phase_movement_mask":
                          batch["phase_movement_mask"][i],
                      "phase_mask": batch["phase_mask"][i]}
                states[tid] = st
                actions[tid] = agent.act(st, epsilon=eps)
            _ns, rewards, done, _info = env.step(actions)
            # On the terminal step the sim has ended; querying the FRAP
            # batch again can fail. The next-state is masked by `done` in
            # the DQN target anyway, so reuse the pre-step batch as a
            # harmless placeholder.
            nbatch = batch if done else env.get_state_frap_batch()
            for i, tid in enumerate(batch["tls_ids"]):
                nst = {"movement_features": nbatch["movement_features"][i],
                       "phase_movement_mask":
                           nbatch["phase_movement_mask"][i],
                       "phase_mask": nbatch["phase_mask"][i]}
                pend = pending[tid]
                pend.append((states[tid], actions[tid], float(rewards[tid])))
                if done:
                    # Flush every pending transition as a truncated n-step
                    # ending at this terminal state. done=True masks the
                    # bootstrap, so the gamma**n in learn() is harmless here.
                    while pend:
                        s0, a0, _ = pend[0]
                        ret = sum((args.gamma ** k) * pend[k][2]
                                  for k in range(len(pend)))
                        agent.remember(s0, a0, ret, nst, True)
                        pend.popleft()
                elif len(pend) >= args.n_step:
                    s0, a0, _ = pend[0]
                    ret = sum((args.gamma ** k) * pend[k][2]
                              for k in range(args.n_step))
                    agent.remember(s0, a0, ret, nst, False)
                    pend.popleft()
            loss = None
            for _ in range(max(1, args.learn_per_decision)):
                loss = agent.learn()
            if loss is not None:
                losses.append(loss)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[ep {ep:3d}/{args.episodes}] eps={eps:.3f} "
              f"td_loss={mean_loss:.4f}")
        emit({"kind": "episode", "episode": ep, "epsilon": eps,
              "td_loss": mean_loss})

        if ep % args.eval_every == 0:
            w, arr = _eval(agent, eval_env, args.eval_seeds,
                           args.time_limit)
            # Select best.pth on a goal-aligned score (throughput up, wait
            # down) over 5 seeds, not wait alone over 3 -- wait-only
            # selection can pick a throughput-poor policy that fights the
            # goal metric (finding F6).
            score = arr - 0.1 * w
            print(f"  eval: wait/veh={w:.1f} throughput={arr:.0f} "
                  f"score={score:.1f}")
            emit({"kind": "eval", "episode": ep, "wait_per_vehicle": w,
                  "throughput": arr, "score": score})
            if score > best_score:
                best_score = score
                agent.save(str(out / "checkpoints" / "best.pth"),
                           meta={"episode": ep, "wait": w,
                                 "throughput": arr, "score": score})
        agent.save(str(out / "checkpoints" / "last.pth"),
                   meta={"episode": ep})

    log.close()
    env.stop()
    eval_env.stop()
    print(f"Done. best eval score={best_score:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
