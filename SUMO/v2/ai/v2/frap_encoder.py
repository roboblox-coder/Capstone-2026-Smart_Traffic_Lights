"""FRAP: phase-symmetry-invariant encoder for a single intersection.

Reference: Zheng et al., "Learning Phase Competition for Traffic Signal
Control" (CIKM 2019, arXiv 1905.04722).

Why FRAP instead of a flat MLP per light: a 4-way intersection has 8
canonical movements (NS-left, NS-through, NS-right, EW-left, EW-through,
EW-right, and so on). A flat MLP assigns each lane-index in the input
vector its own weights, so the encoder has to relearn the
"NS-straight and EW-straight are symmetric phases" relationship in
every TLS. FRAP encodes a movement-pair "competition" score that is
permutation-invariant in the movement ordering and shares the same
parameters across all 12 TLS in the corridor -- roughly a 12x
sample-efficiency multiplier vs. per-light MLPs.

Two outputs per forward:

  light_embedding   - (embed_dim,) summary used by CoLight GAT and the
                      MAPPO critic.
  phase_prelogits   - (num_green,) score per available phase, fed into
                      the SharedActor after GAT context is mixed in.

The encoder is invoked once per TLS per decision; movement counts vary
per TLS (variable signal-state-string length) but ``embed_dim`` and
``num_green`` are both per-TLS so downstream code can pad to the
corridor maximum.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FRAPEncoder(nn.Module):
    """Parameter-shared per-light encoder.

    Args:
        mov_feat_dim: feature dim of each movement input row. Default 3
            for the (halting, vehicles, waiting) triple emitted by
            ``SumoTrafficEnv.get_state_frap``.
        embed_dim: dim of both the per-light embedding and the
            per-movement / per-phase intermediate embeddings.
        hidden_dim: MLP hidden width for the movement and competition
            sub-networks.
    """

    def __init__(self, mov_feat_dim: int = 3, embed_dim: int = 128,
                 hidden_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.movement_mlp = nn.Sequential(
            nn.Linear(mov_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Pairwise phase competition: takes concat(phase_i, phase_j) and
        # produces a scalar "how much does i beat j right now" score.
        # Parameter-shared across all (i, j) pairs.
        self.phase_competition = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mov_feats: torch.Tensor,
                phase_movement_mask: torch.Tensor) -> tuple:
        """One light, one forward pass.

        Args:
            mov_feats: (N_mov, mov_feat_dim) tensor of per-movement
                features. Order matches the SUMO signal-state-string
                index ordering.
            phase_movement_mask: (num_green, N_mov) bool tensor.
                ``mask[s, m]`` is True iff movement ``m`` is green in
                phase slot ``s``.

        Returns:
            (light_embedding, phase_prelogits) where
              light_embedding   : (embed_dim,)
              phase_prelogits   : (num_green,)
        """
        # mov_embeds: (N_mov, embed_dim)
        mov_embeds = self.movement_mlp(mov_feats)

        # Phase embedding = mean of movement embeddings for movements
        # green in that phase. Masked mean avoids division by zero on
        # the (degenerate but possible) all-red phase. Equivalent to
        # FRAP's "phase representation" sum-pool with normalization.
        mask_f = phase_movement_mask.to(mov_embeds.dtype)
        per_phase_count = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        weights = mask_f / per_phase_count
        phase_embeds = weights @ mov_embeds  # (num_green, embed_dim)

        num_green = phase_embeds.size(0)
        if num_green > 1:
            # Build the (num_green x num_green) pair grid and feed
            # concat(pi, pj) through the competition MLP. Mask out the
            # diagonal (a phase doesn't compete with itself) before
            # summing.
            pi = phase_embeds.unsqueeze(1).expand(-1, num_green, -1)
            pj = phase_embeds.unsqueeze(0).expand(num_green, -1, -1)
            pairs = torch.cat([pi, pj], dim=-1)
            scores = self.phase_competition(pairs).squeeze(-1)
            eye = torch.eye(num_green, dtype=torch.bool,
                            device=scores.device)
            scores = scores.masked_fill(eye, 0.0)
            phase_prelogits = scores.sum(dim=-1)
        else:
            phase_prelogits = torch.zeros(num_green,
                                          device=phase_embeds.device,
                                          dtype=phase_embeds.dtype)

        # Light embedding: mean across phases. Stays fixed-dim regardless
        # of num_green so the GAT and critic can batch it across TLS.
        light_embedding = phase_embeds.mean(dim=0)
        return light_embedding, phase_prelogits
