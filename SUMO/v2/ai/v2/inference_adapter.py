"""V2 inference: corridor-level policy reloaded from a MAPPO checkpoint.

V1's ``DQNAgent.load_for_inference`` returned a per-TLS object. V2 is
parameter-shared across the 12 lights and reads neighbor embeddings via
GAT, so the natural inference surface is corridor-level: one
``V2CorridorPolicy`` instance owns the encoder + GAT + actor and
dispatches actions for every light per decision tick.

The two consumers:

  eval_network.py
    Loads via ``V2CorridorPolicy.load_for_inference(ckpt_path)`` and
    calls ``act(env.get_state_frap_batch(), env.frap_adjacency_tensor())``
    once per decision.

  run_websocket_ai.py
    Wraps the same call in a ``V2InferenceLoop`` class so the existing
    per-TLS yellow / min-green bookkeeping stays in the runner's
    decision loop.

Critic + meta-fields are also loaded so the same file can resume
training -- callers that just want inference can ignore them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from v2.frap_encoder import FRAPEncoder  # noqa: E402
from v2.colight_gat import CoLightGAT  # noqa: E402
from v2.shared_policy import SharedActor  # noqa: E402
from v2.centralized_critic import CentralCritic  # noqa: E402


@dataclass
class _Modules:
    encoder: FRAPEncoder
    gat: CoLightGAT
    actor: SharedActor
    critic: Optional[CentralCritic] = None


class V2CorridorPolicy:
    """Corridor-level inference surface for FRAP / GAT / SharedActor.

    Instantiated from a checkpoint produced by ``MAPPOTrainer.save``.
    Tracks the n_tls / P_max / max_movements counts the checkpoint was
    trained at so a shape mismatch against the env fails loudly rather
    than silently misrouting actions.
    """

    agent_type = "frap_gat_mappo"

    def __init__(self, modules: _Modules, meta: dict,
                 device: torch.device):
        self.modules = modules
        self.meta = meta
        self.device = device
        # Cached shapes from training.
        self.n_tls = int(meta["n_tls"])
        self.p_max = int(meta["p_max"])
        self.max_movements = int(meta["max_movements"])
        self.tls_ids = list(meta["tls_ids"])

    # ---------- loaders ----------

    @classmethod
    def load_for_inference(cls, ckpt_path: str,
                           device: Optional[str] = None,
                           include_critic: bool = False
                           ) -> "V2CorridorPolicy":
        dev = torch.device(device or
                           ("cuda" if torch.cuda.is_available()
                            else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=dev)
        if ckpt.get("agent_type") != cls.agent_type:
            raise ValueError(
                f"checkpoint at {ckpt_path} is "
                f"agent_type={ckpt.get('agent_type')!r}; "
                f"expected {cls.agent_type!r}.")
        cfg = ckpt["config"]
        encoder = FRAPEncoder(
            mov_feat_dim=3, embed_dim=int(cfg["embed_dim"])).to(dev)
        gat = CoLightGAT(
            embed_dim=int(cfg["embed_dim"]),
            num_heads=int(cfg["gat_heads"]),
            head_dim=int(cfg["gat_head_dim"])).to(dev)
        actor = SharedActor(embed_dim=int(cfg["embed_dim"])).to(dev)
        encoder.load_state_dict(ckpt["encoder"])
        gat.load_state_dict(ckpt["gat"])
        actor.load_state_dict(ckpt["actor"])
        encoder.eval(); gat.eval(); actor.eval()
        # Inference doesn't need the frozen-uniform warmup; whatever the
        # training run was up to is locked in by load_state_dict.
        gat.set_frozen_uniform(False)

        critic = None
        if include_critic:
            critic = CentralCritic(
                embed_dim=int(cfg["embed_dim"]),
                n_tls=int(ckpt["n_tls"]),
                hidden_dim=int(cfg["critic_hidden"])).to(dev)
            critic.load_state_dict(ckpt["critic"])
            critic.eval()

        modules = _Modules(encoder=encoder, gat=gat, actor=actor,
                           critic=critic)
        return cls(modules=modules, meta=ckpt, device=dev)

    # ---------- inference ----------

    def _encode_batch(self, batch: dict,
                      adjacency: torch.Tensor) -> tuple:
        # One batched FRAP forward across the whole corridor; no
        # Python per-TLS loop and no GPU<->CPU round-trip.
        mov_feats = torch.from_numpy(batch["movement_features"]).to(
            self.device)
        pm_mask = torch.from_numpy(batch["phase_movement_mask"]).to(
            self.device)
        phase_mask = torch.from_numpy(batch["phase_mask"]).to(self.device)
        light_embeds, phase_prelogits = self.modules.encoder.forward_batched(
            mov_feats, pm_mask, phase_mask)
        light_embeds_ctx = self.modules.gat(light_embeds, adjacency)
        return light_embeds_ctx, phase_prelogits, phase_mask

    @torch.no_grad()
    def act(self, batch: dict, adjacency,
            deterministic: bool = True) -> dict:
        """One corridor-level decision.

        Args:
            batch: dict from ``MultiTlsEnv.get_state_frap_batch`` (so the
                ``tls_ids`` ordering inside the batch IS the canonical
                action-output ordering).
            adjacency: bool numpy array (n_tls, n_tls) or torch tensor.
            deterministic: argmax over the masked categorical when True
                (eval / live runner default), else sample.

        Returns:
            dict mapping each TLS id to the chosen green-slot index.

        Raises if the env's tls_ids / n_tls don't match training time.
        """
        env_ids = batch["tls_ids"]
        if list(env_ids) != self.tls_ids:
            raise ValueError(
                f"V2 checkpoint was trained on tls_ids={self.tls_ids}; "
                f"the env presents tls_ids={list(env_ids)}. Refusing "
                f"to dispatch actions on a re-ordered corridor.")
        if isinstance(adjacency, np.ndarray):
            adjacency = torch.from_numpy(adjacency).to(self.device)
        elif adjacency.device != self.device:
            adjacency = adjacency.to(self.device)

        le_ctx, ppl, pm = self._encode_batch(batch, adjacency)
        logits = self.modules.actor(le_ctx, ppl, pm)
        actions, _, _ = SharedActor.sample_actions(
            logits, deterministic=deterministic)
        a_np = actions.cpu().numpy()
        return {tid: int(a_np[i]) for i, tid in enumerate(env_ids)}
