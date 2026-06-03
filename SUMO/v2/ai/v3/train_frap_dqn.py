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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-episodes", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, nargs="+",
                    default=[1042, 1043, 1044])
    ap.add_argument("--reward-mode", default="max_pressure_net",
                    help="Env reward. 'combined' directly rewards "
                         "throughput + wait (no gridlock loophole that "
                         "max_pressure has on heavy demand).")
    ap.add_argument("--gamma", type=float, default=0.95,
                    help="DQN discount. 0.99 for longer-horizon credit.")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--target-sync", type=int, default=500)
    ap.add_argument("--out-dir", default="ai/runs/v3_frap_dqn")
    args = ap.parse_args()

    adjacency = load_adjacency(args.adjacency)
    common = dict(sumo_cfg_file=args.sumo_cfg, adjacency=adjacency,
                  time_limit=args.time_limit, reward_mode=args.reward_mode,
                  control_tls=True, seed=args.seed)
    env = MultiTlsEnv(**common)
    eval_env = MultiTlsEnv(**common)

    env.reset()
    b0 = env.get_state_frap_batch()
    p_max = b0["phase_mask"].shape[1]
    m_max = b0["movement_features"].shape[1]
    agent = FRAPDQNAgent(mov_feat_dim=3, p_max=p_max, m_max=m_max,
                         gamma=args.gamma, lr=args.lr,
                         target_sync_steps=args.target_sync)

    out = Path(args.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.jsonl").open("w", encoding="utf-8")

    def emit(d):
        log.write(json.dumps(d) + "\n")
        log.flush()

    best_wait = float("inf")
    for ep in range(1, args.episodes + 1):
        frac = min(1.0, (ep - 1) / max(1, args.eps_decay_episodes))
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)
        env.seed = args.seed + ep
        env.reset()
        done = False
        losses = []
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
                agent.remember(states[tid], actions[tid],
                               float(rewards[tid]), nst, done)
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
            print(f"  eval: wait/veh={w:.1f} throughput={arr:.0f}")
            emit({"kind": "eval", "episode": ep,
                  "wait_per_vehicle": w, "throughput": arr})
            if w < best_wait:
                best_wait = w
                agent.save(str(out / "checkpoints" / "best.pth"),
                           meta={"episode": ep, "wait": w})
        agent.save(str(out / "checkpoints" / "last.pth"),
                   meta={"episode": ep})

    log.close()
    env.stop()
    eval_env.stop()
    print(f"Done. best eval wait/veh={best_wait:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
