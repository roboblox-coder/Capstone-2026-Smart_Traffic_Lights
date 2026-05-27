"""MAPPO trainer: rollout + PPO update + GAT unfreeze schedule + checkpoints.

The orchestration that turns the four neural modules in ``ai/v2/`` into
a corridor policy. CTDE: actors run per-light during execution, critic
sees joint state during training, both update from PPO-style clipped
gradients on shared rollout buffers.

Two key contracts:

  Rollout buffer is on-policy. Every rollout is generated with the
  current policy and consumed in K PPO epochs of minibatch SGD before
  the buffer is discarded.

  Advantage normalization is per-MINIBATCH, not per-rollout. The plan
  (PLAN_V2.md §1.2) pre-commits this -- normalizing per-rollout averages
  the per-light advantage signal toward zero and the policy stops
  learning when one light dominates the variance.

Run from ``SUMO/v2``:

    python ai/v2/mappo_trainer.py \\
        --sumo-cfg sim_calibrated.sumocfg \\
        --episodes 1500 --time-limit 1200 \\
        --rollout-episodes 6 \\
        --out-dir ai/runs/v2_mappo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from multi_env import MultiTlsEnv, load_adjacency  # noqa: E402
from v2.frap_encoder import FRAPEncoder  # noqa: E402
from v2.colight_gat import CoLightGAT  # noqa: E402
from v2.shared_policy import SharedActor  # noqa: E402
from v2.centralized_critic import CentralCritic  # noqa: E402


# ---------- hyperparameters ----------

@dataclass
class MAPPOConfig:
    # PPO / GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.5
    entropy_coef: float = 0.01
    entropy_coef_final: float = 0.001
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Optimization
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    ppo_epochs: int = 4
    minibatch_size: int = 2048

    # Architecture
    embed_dim: int = 128
    gat_heads: int = 4
    gat_head_dim: int = 32
    critic_hidden: int = 512

    # GAT unfreeze schedule (steps refer to gradient updates).
    gat_freeze_until_step: int = 50_000
    gat_ramp_end_step: int = 60_000

    # Rollout / training
    rollout_episodes: int = 6
    total_episodes: int = 1500
    seed: int = 42

    # Eval / checkpoint
    eval_every_updates: int = 5
    eval_seeds: tuple = (1042, 1043, 1044)
    plateau_episodes: int = 100  # early-stop if no eval improvement
    save_dir: str = "ai/runs/v2_mappo"


# ---------- rollout buffer ----------

class RolloutBuffer:
    """Stores (state, action, logprob, reward, value, done) per
    (step, light). Tensors are appended in numpy / int / float and
    stacked at end-of-rollout."""

    def __init__(self):
        self.batches: list = []

    def append(self, step_record: dict) -> None:
        self.batches.append(step_record)

    def __len__(self) -> int:
        return len(self.batches)

    def clear(self) -> None:
        self.batches.clear()

    def stack(self) -> dict:
        keys = self.batches[0].keys()
        out = {}
        for k in keys:
            vals = [b[k] for b in self.batches]
            if isinstance(vals[0], np.ndarray):
                out[k] = np.stack(vals)
            elif isinstance(vals[0], (int, float, bool, np.floating,
                                      np.integer)):
                out[k] = np.array(vals)
            else:
                out[k] = vals  # list of dicts or strings
        return out


# ---------- trainer ----------

class MAPPOTrainer:

    def __init__(self, env: MultiTlsEnv, cfg: MAPPOConfig,
                 device: Optional[str] = None,
                 eval_env: Optional[MultiTlsEnv] = None):
        self.env = env
        # eval_env defaults to env (V1-style training: no DR, no need
        # for a separate eval env). When training with the DR wrapper
        # the trainer's CLI passes a clean (non-DR) env here so eval
        # metrics aren't muddied by the per-episode noise.
        self.eval_env = eval_env if eval_env is not None else env
        self.cfg = cfg
        self.device = torch.device(device or
                                   ("cuda" if torch.cuda.is_available()
                                    else "cpu"))
        # Cached env shapes (require the env to be open once).
        self.p_max = env.frap_p_max
        self.max_movements = env.frap_max_movements
        self.n_tls = len(env.tls_ids)
        adj = env.frap_adjacency_tensor()
        self.adjacency = torch.from_numpy(adj).to(self.device)

        # Modules.
        self.encoder = FRAPEncoder(mov_feat_dim=3,
                                   embed_dim=cfg.embed_dim).to(self.device)
        self.gat = CoLightGAT(embed_dim=cfg.embed_dim,
                              num_heads=cfg.gat_heads,
                              head_dim=cfg.gat_head_dim).to(self.device)
        self.actor = SharedActor(embed_dim=cfg.embed_dim).to(self.device)
        self.critic = CentralCritic(embed_dim=cfg.embed_dim,
                                    n_tls=self.n_tls,
                                    hidden_dim=cfg.critic_hidden
                                    ).to(self.device)

        # Two optimizers so actor LR != critic LR (MAPPO recipe). The
        # GAT goes in with the actor since its gradients flow through
        # actor + critic, and the GAT-LR scaling is implemented as a
        # multiplier on the actor LR via param_groups.
        self.actor_opt = optim.Adam([
            {"params": self.encoder.parameters(), "lr": cfg.actor_lr},
            {"params": self.gat.parameters(), "lr": 0.0},  # ramped below
            {"params": self.actor.parameters(), "lr": cfg.actor_lr},
        ])
        self.critic_opt = optim.Adam(self.critic.parameters(),
                                     lr=cfg.critic_lr)

        # GAT unfreeze: start frozen-uniform.
        self.gat.set_frozen_uniform(True)
        self._gradient_steps = 0
        self._episodes_done = 0

        # Eval / checkpoint state.
        self.best_eval_wpv = float("inf")
        self.episodes_since_improvement = 0
        # Episode count at which best_eval_wpv was last beaten. Used
        # by the plateau detector so resumed runs continue counting
        # from where they left off.
        self._last_improvement_ep = 0

        self.save_dir = Path(cfg.save_dir)
        (self.save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        # Save the config so the resume / inference path can reconstruct
        # the same shapes without command-line argument drift.
        (self.save_dir / "config.json").write_text(
            json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")

    # ---------- GAT unfreeze schedule ----------

    def _update_gat_schedule(self) -> None:
        """Called once per gradient step. Implements the plan's frozen-
        uniform (0..50k) -> linear ramp (50k..60k) -> full-rate (60k+)
        schedule from PLAN_V2.md §1.2 R2."""
        s = self._gradient_steps
        cfg = self.cfg
        if s < cfg.gat_freeze_until_step:
            self.gat.set_frozen_uniform(True)
            new_lr = 0.0
        elif s < cfg.gat_ramp_end_step:
            self.gat.set_frozen_uniform(False)
            f = ((s - cfg.gat_freeze_until_step) /
                 max(1, cfg.gat_ramp_end_step - cfg.gat_freeze_until_step))
            new_lr = f * cfg.actor_lr
        else:
            self.gat.set_frozen_uniform(False)
            new_lr = cfg.actor_lr
        # param_groups[1] is the GAT block (see __init__).
        self.actor_opt.param_groups[1]["lr"] = new_lr

    def _current_entropy_coef(self) -> float:
        """Linear decay across total episodes."""
        progress = min(1.0, self._episodes_done /
                       max(1, self.cfg.total_episodes))
        return (self.cfg.entropy_coef * (1.0 - progress)
                + self.cfg.entropy_coef_final * progress)

    # ---------- rollout ----------

    def _encode_step(self, mov_feats_t: torch.Tensor,
                     pm_mask_t: torch.Tensor,
                     phase_mask_t: torch.Tensor) -> tuple:
        """Vectorized FRAP + GAT forward.

        Accepts either (N_tls, ...) for a single rollout step or
        (B, N_tls, ...) for a PPO minibatch. The leading shape is
        preserved on the way out. Returns
        ``(light_embeds_ctx, phase_prelogits, light_embeds_raw)``;
        ``light_embeds_raw`` is the pre-GAT representation that feeds
        the critic (keeps the critic free of the GAT frozen-uniform vs
        learned weights phase shift).

        Args:
            mov_feats_t: (..., M_max, 3) movement features.
            pm_mask_t: (..., P_max, M_max) bool phase-movement mask.
            phase_mask_t: (..., P_max) bool phase mask.
        """
        squeeze_b = (mov_feats_t.dim() == 3)  # rollout / eval shape
        if squeeze_b:
            mov_feats_t = mov_feats_t.unsqueeze(0)
            pm_mask_t = pm_mask_t.unsqueeze(0)
            phase_mask_t = phase_mask_t.unsqueeze(0)

        B, n_tls = mov_feats_t.shape[:2]
        # Flatten batch + TLS for FRAP (FRAP is per-light, no
        # cross-light interactions). One forward, no Python loop over
        # the corridor.
        mov_flat = mov_feats_t.reshape(B * n_tls,
                                       *mov_feats_t.shape[2:])
        pm_flat = pm_mask_t.reshape(B * n_tls, *pm_mask_t.shape[2:])
        phase_flat = phase_mask_t.reshape(B * n_tls, -1)

        le_flat, ppl_flat = self.encoder.forward_batched(
            mov_flat, pm_flat, phase_flat)
        # Reshape back to (B, N_tls, ...).
        le_raw = le_flat.reshape(B, n_tls, -1)
        ppl = ppl_flat.reshape(B, n_tls, -1)
        # GAT operates per-step: (B, N_tls, D) -> (B, N_tls, D), with
        # the same adjacency broadcast across the batch.
        le_ctx = self.gat.forward_batched(le_raw, self.adjacency)

        if squeeze_b:
            return le_ctx.squeeze(0), ppl.squeeze(0), le_raw.squeeze(0)
        return le_ctx, ppl, le_raw

    def _collect_rollout(self) -> dict:
        """Run ``cfg.rollout_episodes`` episodes; return stacked buffer."""
        buf = RolloutBuffer()
        ep_rewards = []
        for ep in range(self.cfg.rollout_episodes):
            self.env.seed = self.cfg.seed + self._episodes_done
            self.env.reset()
            done = False
            ep_reward_sum = 0.0
            while not done:
                batch = self.env.get_state_frap_batch()
                with torch.no_grad():
                    mov_t = torch.from_numpy(
                        batch["movement_features"]).to(self.device)
                    pm_t = torch.from_numpy(
                        batch["phase_movement_mask"]).to(self.device)
                    phase_t = torch.from_numpy(
                        batch["phase_mask"]).to(self.device)
                    le_ctx, ppl, le_raw = self._encode_step(
                        mov_t, pm_t, phase_t)
                    logits = self.actor(le_ctx, ppl, phase_t)
                    actions, logprobs, _entropy = (
                        SharedActor.sample_actions(logits))
                    values = self.critic(le_raw)
                # Submit per-TLS actions.
                actions_np = actions.cpu().numpy()
                actions_dict = {tid: int(actions_np[i])
                                for i, tid in enumerate(batch["tls_ids"])}
                _next_states, rewards, done, _infos = self.env.step(
                    actions_dict)
                rewards_arr = np.array([rewards[t]
                                        for t in batch["tls_ids"]],
                                       dtype=np.float32)
                ep_reward_sum += float(rewards_arr.sum())
                buf.append({
                    "movement_features": batch["movement_features"],
                    "movement_mask": batch["movement_mask"],
                    "phase_movement_mask": batch["phase_movement_mask"],
                    "phase_mask": batch["phase_mask"],
                    "actions": actions_np,
                    "logprobs": logprobs.cpu().numpy(),
                    "values": values.cpu().numpy(),
                    "rewards": rewards_arr,
                    "dones": bool(done),
                })
            ep_rewards.append(ep_reward_sum)
            self._episodes_done += 1

        stacked = buf.stack()
        stacked["episode_reward_mean"] = float(np.mean(ep_rewards))
        return stacked

    # ---------- advantage computation ----------

    def _compute_advantages(self, traj: dict) -> tuple:
        """GAE-lambda per (step, light). Returns (advantages, returns).

        Rewards are per-light, values are scalar (joint); we broadcast
        the joint value to all lights for the advantage compute. This
        is the MAPPO-CTDE recipe: per-light advantages computed against
        a single joint value baseline.
        """
        rewards = traj["rewards"]   # (T, n_tls)
        values = traj["values"]     # (T,) scalar joint value
        dones = traj["dones"]       # (T,) bool
        T, n = rewards.shape

        # Broadcast joint value to per-light.
        v = np.broadcast_to(values.reshape(T, 1), (T, n))
        # Bootstrap last value from a "virtual" V(s_T) of zero on done.
        # Cheap and consistent: episodes end at fixed horizon, so V(s_T)
        # is mostly noise. Refine here if it shows up as bias later.
        next_v = np.concatenate([v[1:], np.zeros((1, n), dtype=v.dtype)],
                                axis=0)
        not_done = (~dones).astype(np.float32).reshape(T, 1)

        deltas = rewards + self.cfg.gamma * next_v * not_done - v
        adv = np.zeros_like(deltas)
        lastgae = np.zeros(n, dtype=np.float32)
        for t in reversed(range(T)):
            lastgae = (deltas[t]
                       + self.cfg.gamma * self.cfg.gae_lambda
                       * not_done[t, 0] * lastgae)
            adv[t] = lastgae
        returns = adv + v
        return adv, returns

    # ---------- PPO update ----------

    def _ppo_update(self, traj: dict) -> dict:
        """K epochs over the rollout; per-minibatch advantage norm."""
        adv, returns = self._compute_advantages(traj)
        # Flatten over (time, light) for shuffled minibatching.
        T, n = traj["rewards"].shape
        flat_idx = np.arange(T * n).reshape(T, n)
        # Per-light tensors stay shape (T, n_tls, ...). For PPO we need
        # to slice by (time_idx, light_idx) -- easiest is to keep step
        # tensors whole and minibatch by random (t, light) pairs. With
        # T*n = ~2880 (240*12) per episode * 6 episodes = ~17k samples,
        # minibatch 2048 is realistic.

        # Convert step-level tensors to device once.
        device = self.device
        mov_feats_all = torch.from_numpy(traj["movement_features"]).to(device)
        pm_mask_all = torch.from_numpy(traj["phase_movement_mask"]).to(device)
        phase_mask_all = torch.from_numpy(traj["phase_mask"]).to(device)
        actions_all = torch.from_numpy(traj["actions"]).long().to(device)
        old_logprobs_all = torch.from_numpy(traj["logprobs"]).to(device)
        old_values_all = torch.from_numpy(traj["values"]).to(device)
        adv_all = torch.from_numpy(adv).to(device)
        ret_all = torch.from_numpy(returns).to(device)

        logs = {"pol_loss": [], "val_loss": [], "entropy": [],
                "approx_kl": []}

        steps = T
        for _epoch in range(self.cfg.ppo_epochs):
            perm = np.random.permutation(steps)
            n_minibatches = max(1, (steps * n) // self.cfg.minibatch_size)
            mb_size = max(1, steps // n_minibatches)
            for start in range(0, steps, mb_size):
                idx = perm[start:start + mb_size]
                if len(idx) == 0:
                    continue
                idx_t = torch.from_numpy(idx).long().to(device)

                # Per-minibatch advantage normalization (PLAN_V2.md §1.2).
                mb_adv = adv_all[idx_t]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Forward through the V2 stack for the whole minibatch
                # in one batched pass. No Python loop over (B, N_tls),
                # no cpu round-trip -- tensors stay on device.
                mov_feats_b = mov_feats_all[idx_t]    # (B, N, M, 3)
                pm_mask_b = pm_mask_all[idx_t]        # (B, N, P, M)
                phase_mask_b = phase_mask_all[idx_t]  # (B, N, P)
                le_ctx_b, ppl_b, le_raw_b = self._encode_step(
                    mov_feats_b, pm_mask_b, phase_mask_b)
                logits_b = self.actor.forward_batched(
                    le_ctx_b, ppl_b, phase_mask_b)   # (B, N, P)
                values_b = self.critic(le_raw_b)     # (B,)

                new_logprobs, entropy = SharedActor.evaluate_actions(
                    logits_b, actions_all[idx_t])  # (B, n)
                old_logprobs = old_logprobs_all[idx_t]
                ratio = torch.exp(new_logprobs - old_logprobs)

                # PPO clipped policy loss.
                pol_loss_unclipped = ratio * mb_adv
                pol_loss_clipped = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_eps,
                    1.0 + self.cfg.clip_eps) * mb_adv
                pol_loss = -torch.min(pol_loss_unclipped,
                                      pol_loss_clipped).mean()

                # Value loss with clipping (MAPPO recipe).
                old_values = old_values_all[idx_t]
                returns_mb = ret_all[idx_t].mean(dim=-1)  # joint return
                v_clipped = old_values + torch.clamp(
                    values_b - old_values,
                    -self.cfg.value_clip_eps, self.cfg.value_clip_eps)
                val_loss = 0.5 * torch.max(
                    (values_b - returns_mb) ** 2,
                    (v_clipped - returns_mb) ** 2).mean()

                ent_coef = self._current_entropy_coef()
                ent_loss = -ent_coef * entropy.mean()

                loss = (pol_loss
                        + self.cfg.value_loss_coef * val_loss
                        + ent_loss)

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters())
                    + list(self.gat.parameters())
                    + list(self.actor.parameters())
                    + list(self.critic.parameters()),
                    self.cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()
                self._gradient_steps += 1
                self._update_gat_schedule()

                approx_kl = (old_logprobs - new_logprobs).mean().item()
                logs["pol_loss"].append(float(pol_loss.detach()))
                logs["val_loss"].append(float(val_loss.detach()))
                logs["entropy"].append(float(entropy.mean().detach()))
                logs["approx_kl"].append(approx_kl)

        return {k: float(np.mean(v)) for k, v in logs.items()
                if len(v) > 0}

    # ---------- checkpoint ----------

    def save(self, tag: str, meta: Optional[dict] = None) -> None:
        path = self.save_dir / "checkpoints" / f"{tag}.pth"
        torch.save({
            "agent_type": "frap_gat_mappo",
            "config": asdict(self.cfg),
            "n_tls": self.n_tls,
            "p_max": self.p_max,
            "max_movements": self.max_movements,
            "tls_ids": list(self.env.tls_ids),
            "encoder": self.encoder.state_dict(),
            "gat": self.gat.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            # Optimizer state so --resume picks up exactly where it
            # left off (LR schedule, Adam moments). Without these,
            # resuming would restart momentum from zero and hurt
            # mid-training stability.
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "episodes_done": self._episodes_done,
            "gradient_steps": self._gradient_steps,
            "best_eval_wpv": self.best_eval_wpv,
            "episodes_since_improvement":
                self.episodes_since_improvement,
            "last_improvement_ep": self._last_improvement_ep,
            "meta": meta or {},
        }, path)

    def load(self, path: str) -> dict:
        """Restore everything save() persisted; returns the meta dict.

        Validates that the on-disk shape contract matches the trainer
        the user is resuming into (n_tls, p_max, max_movements). A
        mismatch means the env or sumocfg was changed underneath the
        run -- refuse rather than silently misrouting.
        """
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("agent_type") != "frap_gat_mappo":
            raise ValueError(
                f"checkpoint at {path} is "
                f"agent_type={ckpt.get('agent_type')!r}; expected "
                f"'frap_gat_mappo'.")
        for field, want in (("n_tls", self.n_tls),
                            ("p_max", self.p_max),
                            ("max_movements", self.max_movements)):
            got = ckpt.get(field)
            if got != want:
                raise ValueError(
                    f"checkpoint {field}={got} vs current env "
                    f"{field}={want}; refusing to resume.")
        if list(ckpt.get("tls_ids", [])) != list(self.env.tls_ids):
            raise ValueError(
                "tls_ids ordering mismatch between checkpoint and "
                "current env; refusing to resume.")
        self.encoder.load_state_dict(ckpt["encoder"])
        self.gat.load_state_dict(ckpt["gat"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self._episodes_done = int(ckpt.get("episodes_done", 0))
        self._gradient_steps = int(ckpt.get("gradient_steps", 0))
        self.best_eval_wpv = float(
            ckpt.get("best_eval_wpv", float("inf")))
        self.episodes_since_improvement = int(
            ckpt.get("episodes_since_improvement", 0))
        self._last_improvement_ep = int(
            ckpt.get("last_improvement_ep", self._episodes_done))
        # Snap the GAT schedule + LR onto the current gradient step
        # count so resuming mid-ramp doesn't reset the GAT to frozen.
        self._update_gat_schedule()
        print(f"[resume] loaded {path} | "
              f"episodes_done={self._episodes_done} | "
              f"gradient_steps={self._gradient_steps} | "
              f"best_eval_wpv={self.best_eval_wpv:.3f}")
        return ckpt.get("meta", {})

    # ---------- eval ----------

    def evaluate(self, seeds: tuple) -> dict:
        """Deterministic rollouts on fixed seeds. Returns aggregate
        metrics suitable for the plateau-detection auto-stop.

        Always runs against ``self.eval_env`` -- which is a separate
        non-randomized env when DR is on, so eval signal isn't
        contaminated by the per-episode demand/noise sampling.
        """
        wpvs, arrs = [], []
        was_training = self.encoder.training
        self.encoder.eval(); self.gat.eval()
        self.actor.eval(); self.critic.eval()
        with torch.no_grad():
            for s in seeds:
                self.eval_env.seed = int(s)
                self.eval_env.reset()
                done = False
                while not done:
                    batch = self.eval_env.get_state_frap_batch()
                    mov_t = torch.from_numpy(
                        batch["movement_features"]).to(self.device)
                    pm_t = torch.from_numpy(
                        batch["phase_movement_mask"]).to(self.device)
                    phase_t = torch.from_numpy(
                        batch["phase_mask"]).to(self.device)
                    le_ctx, ppl, _le_raw = self._encode_step(
                        mov_t, pm_t, phase_t)
                    logits = self.actor(le_ctx, ppl, phase_t)
                    actions, _, _ = SharedActor.sample_actions(
                        logits, deterministic=True)
                    a_np = actions.cpu().numpy()
                    actions_dict = {tid: int(a_np[i])
                                    for i, tid in enumerate(
                                        batch["tls_ids"])}
                    _, _, done, _ = self.eval_env.step(actions_dict)
                m = self.eval_env.metrics_summary()
                wpvs.append(m["wait_per_vehicle"])
                arrs.append(m["arrived"])
        if was_training:
            self.encoder.train(); self.gat.train()
            self.actor.train(); self.critic.train()
        return {"wait_per_vehicle_mean": float(np.mean(wpvs)),
                "throughput_mean": float(np.mean(arrs)),
                "n_seeds": len(seeds)}

    # ---------- train loop ----------

    def _log_jsonl(self, payload: dict) -> None:
        """Append one JSON object to ``<save_dir>/train_log.jsonl`` and
        flush. Durable per-update metrics for offline analysis -- stdout
        is for humans, this is for plots."""
        path = self.save_dir / "train_log.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _format_elapsed(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"

    def train(self) -> None:
        import time as _time
        n_updates = 0
        if self.eval_env is not self.env:
            print("[train] DR active: training env is randomized; eval "
                  "env is clean (separate SUMO process).")
        # Anchor episode counter at whatever resume left us at; ETA
        # math uses ep delta from this start so partial resumes don't
        # confuse the projection.
        t_start = _time.perf_counter()
        ep_at_start = self._episodes_done
        while self._episodes_done < self.cfg.total_episodes:
            t_update_start = _time.perf_counter()
            traj = self._collect_rollout()
            logs = self._ppo_update(traj)
            n_updates += 1

            update_seconds = _time.perf_counter() - t_update_start
            elapsed = _time.perf_counter() - t_start
            # ETA on remaining episodes vs episodes-per-second so far.
            ep_done_this_run = self._episodes_done - ep_at_start
            ep_per_sec = ep_done_this_run / max(elapsed, 1e-6)
            ep_remaining = self.cfg.total_episodes - self._episodes_done
            eta_seconds = (ep_remaining / ep_per_sec
                           if ep_per_sec > 0 else float("inf"))

            print(f"[update {n_updates:4d}] "
                  f"ep={self._episodes_done:>4d}/"
                  f"{self.cfg.total_episodes}  "
                  f"upd={self._format_elapsed(update_seconds)}  "
                  f"eta={self._format_elapsed(eta_seconds)}  "
                  f"reward/ep={traj['episode_reward_mean']:>10.2f}  "
                  f"pol={logs.get('pol_loss', float('nan')):+.4f}  "
                  f"val={logs.get('val_loss', float('nan')):+.4f}  "
                  f"ent={logs.get('entropy', float('nan')):+.3f}  "
                  f"kl={logs.get('approx_kl', float('nan')):+.4f}  "
                  f"gat_lr={self.actor_opt.param_groups[1]['lr']:.2e}")

            self._log_jsonl({
                "kind": "update",
                "update": n_updates,
                "episodes_done": self._episodes_done,
                "gradient_steps": self._gradient_steps,
                "update_seconds": update_seconds,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta_seconds,
                "reward_per_episode": traj["episode_reward_mean"],
                "gat_lr": self.actor_opt.param_groups[1]["lr"],
                "entropy_coef": self._current_entropy_coef(),
                **logs,
            })

            if n_updates % self.cfg.eval_every_updates == 0:
                ev = self.evaluate(self.cfg.eval_seeds)
                print(f"  eval: wait/veh={ev['wait_per_vehicle_mean']:.2f}  "
                      f"throughput={ev['throughput_mean']:.1f}  "
                      f"(n={ev['n_seeds']})")
                self._log_jsonl({
                    "kind": "eval",
                    "update": n_updates,
                    "episodes_done": self._episodes_done,
                    "wait_per_vehicle_mean": ev["wait_per_vehicle_mean"],
                    "throughput_mean": ev["throughput_mean"],
                    "n_seeds": ev["n_seeds"],
                })
                if ev["wait_per_vehicle_mean"] < self.best_eval_wpv:
                    self.best_eval_wpv = ev["wait_per_vehicle_mean"]
                    self._last_improvement_ep = self._episodes_done
                    self.episodes_since_improvement = 0
                    self.save("best", meta={"eval": ev})
                    print(f"  new best: wait/veh={self.best_eval_wpv:.2f}")
                else:
                    # Count ACTUAL elapsed episodes since the last
                    # improvement (previous revision counted by eval
                    # cadence * rollout, which over-counted on the
                    # first miss after improvement).
                    self.episodes_since_improvement = (
                        self._episodes_done - self._last_improvement_ep)
                if (self.episodes_since_improvement
                        >= self.cfg.plateau_episodes):
                    print(f"  plateau: no eval improvement for "
                          f"{self.episodes_since_improvement} episodes; "
                          f"stopping.")
                    self._log_jsonl({
                        "kind": "plateau_stop",
                        "update": n_updates,
                        "episodes_done": self._episodes_done,
                    })
                    break

        self.save("last")
        self.env.stop()
        if self.eval_env is not self.env:
            self.eval_env.stop()


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sumo-cfg", default="sim_calibrated.sumocfg")
    p.add_argument("--adjacency", default="ai/adjacency.json")
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--time-limit", type=int, default=1200)
    p.add_argument("--rollout-episodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-green", type=int, default=5)
    p.add_argument("--yellow-time", type=int, default=5)
    p.add_argument("--decision-interval", type=int, default=5)
    p.add_argument("--out-dir", default="ai/runs/v2_mappo")
    p.add_argument("--device", default=None)
    p.add_argument("--randomize", action="store_true",
                   help="Wrap training env in DRWrapper (per-episode "
                        "demand + detector noise per PLAN_V2.md §1.3). "
                        "Eval env stays clean.")
    p.add_argument("--dr-seed", type=int, default=4242)
    p.add_argument("--resume", default=None,
                   help="Resume from a checkpoint .pth. Restores nets, "
                        "optimizers, episode + gradient counters, and "
                        "plateau tracker. Refuses on shape mismatch.")
    p.add_argument("--eval-every", type=int, default=5,
                   help="Run eval every N PPO updates (default 5).")
    p.add_argument("--eval-seeds", type=int, nargs="+",
                   default=[1042, 1043, 1044],
                   help="Seeds for deterministic eval rollouts. "
                        "Distinct from training seeds so eval isn't "
                        "evaluated against trained-on demand draws.")
    p.add_argument("--plateau-episodes", type=int, default=100,
                   help="Stop if no eval improvement for N episodes "
                        "(0 disables).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    adjacency = load_adjacency(args.adjacency)
    env_kwargs = dict(
        sumo_cfg_file=args.sumo_cfg, adjacency=adjacency,
        time_limit=args.time_limit, min_green=args.min_green,
        yellow_time=args.yellow_time,
        decision_interval=args.decision_interval,
        reward_mode="pressure_only", control_tls=True, seed=args.seed,
    )
    if args.randomize:
        from v2.domain_randomization import DRWrapper
        env = DRWrapper(**env_kwargs, dr_seed=args.dr_seed)
        eval_env = MultiTlsEnv(**env_kwargs)
    else:
        env = MultiTlsEnv(**env_kwargs)
        eval_env = None  # trainer falls back to training env

    cfg = MAPPOConfig(
        total_episodes=args.episodes,
        rollout_episodes=args.rollout_episodes,
        seed=args.seed,
        save_dir=args.out_dir,
        eval_every_updates=args.eval_every,
        eval_seeds=tuple(args.eval_seeds),
        plateau_episodes=args.plateau_episodes,
    )
    trainer = MAPPOTrainer(env, cfg, device=args.device,
                           eval_env=eval_env)
    if args.resume:
        trainer.load(args.resume)
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
