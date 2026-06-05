"""Centralized critic for MAPPO (CTDE).

The critic sees the joint state during training (concatenated per-light
FRAP embeddings, optionally augmented with cross-light context from
GAT) and predicts a scalar value -- the same value for all 12 lights in
the rollout, since they share the global reward signal during training.

At execution the critic is never queried: each light's actor consumes
only its own (GAT-context-enriched) embedding, satisfying the
decentralized-execution constraint.

Why concatenated FRAP embeddings and not raw lane counts: the
embeddings are already a learned compression of the per-light state and
have a fixed dim per light. Feeding raw lane state -- variable per TLS
-- would either need padding to a per-TLS max or its own encoder. The
critic re-using the actor's embeddings is parameter-efficient and
matches the MAPPO paper's "share representation across actor + critic"
recommendation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CentralCritic(nn.Module):
    """V_{phi}(s_joint) for MAPPO.

    Args:
        embed_dim: per-light embedding dim (matches FRAPEncoder).
        n_tls: number of lights in the corridor (corridor-fixed; not a
            multi-task model).
        hidden_dim: width of the value head MLP.
    """

    def __init__(self, embed_dim: int = 128, n_tls: int = 12,
                 hidden_dim: int = 512):
        super().__init__()
        joint_dim = embed_dim * n_tls
        # Two-layer MLP with LayerNorm + residual on the first hidden
        # block. This is the recipe in the MAPPO paper's appendix; the
        # LayerNorm matters more than the residual but both help with
        # the 1.5k-dim concatenated input.
        self.fc1 = nn.Linear(joint_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.embed_dim = embed_dim
        self.n_tls = n_tls

    def forward(self, light_embeddings: torch.Tensor) -> torch.Tensor:
        """Args:
            light_embeddings: (B, n_tls, embed_dim) or
                              (n_tls, embed_dim) -- batched per-step or
                              single rollout-step.
        Returns:
            (B,) or scalar value estimate.
        """
        if light_embeddings.dim() == 2:
            flat = light_embeddings.reshape(-1)
            squeeze = True
        else:
            B = light_embeddings.size(0)
            flat = light_embeddings.reshape(B, -1)
            squeeze = False

        h = torch.relu(self.ln1(self.fc1(flat)))
        # Residual on the second block: keeps the value head from
        # collapsing to a near-constant during early training.
        h2 = torch.relu(self.ln2(self.fc2(h)) + h)
        v = self.head(h2).squeeze(-1)
        if squeeze:
            v = v.squeeze()
        return v


class IndependentCritic(nn.Module):
    """Per-light value V_i(s_i) for IPPO-style training (CTDE dropped).

    Each light's value is computed from ITS OWN embedding by a
    parameter-shared MLP (shared across all lights, exactly like
    ``SharedActor`` shares its body). The output is a per-light value
    VECTOR, so each light's GAE advantage is taken against its own
    baseline.

    Why this exists: the ``CentralCritic`` predicts a single joint value
    broadcast to all 12 lights. With a shared baseline, a light's
    advantage = (its local return) - (the same joint V) does not isolate
    whether *that light's own action* was good -- it mostly reflects
    corridor-wide noise. Diagnostics showed the resulting policy gradient
    never sharpened the actor (entropy pinned, ratio_dev ~0.01) under any
    clip / entropy / normalization setting. An independent per-light
    baseline restores per-agent credit assignment, matching the
    independent-learner structure that V1 (per-TLS DQN) already won with.

    Shape contract mirrors ``CentralCritic`` on the batch dim so the
    trainer can swap them by config, but the OUTPUT gains a per-light
    axis: (B, n_tls) instead of (B,).
    """

    def __init__(self, embed_dim: int = 128, n_tls: int = 12,
                 hidden_dim: int = 512):
        super().__init__()
        # Body operates on a single light's embed_dim; applied
        # broadcast over the n_tls axis so lights stay independent and
        # the parameter count is light-count-agnostic.
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.embed_dim = embed_dim
        self.n_tls = n_tls

    def forward(self, light_embeddings: torch.Tensor) -> torch.Tensor:
        """Args:
            light_embeddings: (B, n_tls, embed_dim) or (n_tls, embed_dim).
        Returns:
            (B, n_tls) or (n_tls,) -- one value per light. The MLP is
            applied over the last (embed) dim only; the n_tls axis is
            preserved, so each light's value depends only on its own
            embedding.
        """
        h = torch.relu(self.ln1(self.fc1(light_embeddings)))
        h2 = torch.relu(self.ln2(self.fc2(h)) + h)
        v = self.head(h2).squeeze(-1)  # (..., n_tls)
        return v
