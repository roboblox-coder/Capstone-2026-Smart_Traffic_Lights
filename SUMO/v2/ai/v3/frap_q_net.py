"""FRAP -> per-phase Q network for V3 (value-based, the published FRAP
setting). Parameter-shared across all lights.

Q(phase_i) is computed from phase i's OWN FRAP embedding through a
shared per-phase head. This is the §5 design constraint from the V3
spec: the V2 actor diluted per-phase signal 1:128 by mixing a
phase-constant light embedding with a single per-phase scalar; here each
phase's value comes from its own embedding, so the head can actually
discriminate phases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))   # SUMO/v2/ai

from v2.frap_encoder import FRAPEncoder  # noqa: E402

_NEG_INF = -1e9


class FRAPQNet(nn.Module):
    """movement features -> per-phase embedding (FRAP) -> per-phase Q.

    forward args (batched over lights):
        mov_feats:           (B, M_max, mov_feat_dim)
        phase_movement_mask: (B, P_max, M_max) bool
        phase_mask:          (B, P_max) bool
    returns:
        (B, P_max) Q-values, with padded phase slots at -1e9.
    """

    def __init__(self, mov_feat_dim: int = 3, embed_dim: int = 128,
                 head_hidden: int = 128):
        super().__init__()
        self.encoder = FRAPEncoder(mov_feat_dim=mov_feat_dim,
                                   embed_dim=embed_dim)
        # Shared per-phase Q-head: applied to each phase embedding
        # independently (broadcast over the P_max axis), so phases stay
        # discriminable.
        self.q_head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, mov_feats: torch.Tensor,
                phase_movement_mask: torch.Tensor,
                phase_mask: torch.Tensor) -> torch.Tensor:
        phase_embeds = self.encoder.phase_embeddings_batched(
            mov_feats, phase_movement_mask)        # (B, P_max, D)
        q = self.q_head(phase_embeds).squeeze(-1)  # (B, P_max)
        return q.masked_fill(~phase_mask, _NEG_INF)
