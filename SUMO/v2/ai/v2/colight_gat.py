"""CoLight: graph attention over per-light embeddings.

Reference: Wei et al., "CoLight: Learning Network-level Cooperation for
Traffic Signal Control" (KDD 2019, arXiv 1905.05717).

Sits between the per-light FRAP encoder and the per-light actor head:
each light reads its neighbors' embeddings, weighted by learned
attention over upstream/downstream/cross-street adjacency. The output
embedding for light i is its own embedding plus a learned, attended
combination of its neighbors' embeddings -- so the per-light actor can
condition on "what is the next-but-one downstream light doing right now"
without it being a hand-engineered feature (V1's load-bearing limit).

Two implementation choices worth noting:

  1. Multi-head attention with per-head linear projections, residual
     connection, and post-attention LayerNorm. The pattern is identical
     to a Transformer encoder block over a graph: the GAT is what
     restricts which (i, j) entries the attention can fire on.
  2. ``frozen_uniform`` mode replaces the learned scores with uniform
     1/n_neighbors weights, used during the 50k-step warmup the plan
     pre-commits (see PLAN_V2.md §1.2 R2 mitigation). Toggled via
     ``set_frozen_uniform(True/False)`` so the trainer can ramp it off
     without rebuilding the optimizer state.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoLightGAT(nn.Module):
    """Multi-head graph attention over corridor adjacency.

    Args:
        embed_dim: input/output dim (matches FRAPEncoder.embed_dim).
        num_heads: attention heads. 4 heads x 32-dim per head is the
            FRAP-CoLight baseline.
        head_dim: per-head Q/K/V dim.
        residual: True to add the residual + LayerNorm wrap on output.
            Disable for ablation studies.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4,
                 head_dim: int = 32, residual: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim)
        self.residual = residual
        self.layer_norm = nn.LayerNorm(embed_dim)
        # Frozen-uniform pretraining hook (PLAN_V2.md §1.2 R2). Toggled
        # by the trainer's GAT-unfreeze schedule.
        self._frozen_uniform = False
        # Last forward's attention weights, kept for entropy logging.
        # Shape (N_tls, num_heads, N_tls). None until first forward.
        self.last_attention: torch.Tensor = None  # type: ignore[assignment]

    def set_frozen_uniform(self, frozen: bool) -> None:
        self._frozen_uniform = bool(frozen)

    def forward(self, light_embeddings: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """Args:
            light_embeddings: (N_tls, embed_dim) per-light embeddings
                from FRAPEncoder.
            adjacency: (N_tls, N_tls) bool tensor. ``adj[i, j]`` is True
                if message can flow from j to i. The diagonal must be
                True (self-loop) so a light can attend to its own
                embedding when neighbors are uninformative.
        Returns:
            (N_tls, embed_dim) context-enriched embeddings.
        """
        n_tls = light_embeddings.size(0)
        h = self.num_heads
        d = self.head_dim

        q = self.q_proj(light_embeddings).view(n_tls, h, d)
        k = self.k_proj(light_embeddings).view(n_tls, h, d)
        v = self.v_proj(light_embeddings).view(n_tls, h, d)

        if self._frozen_uniform:
            # Uniform weight across neighbors (incl. self via the
            # adjacency diagonal). Equivalent to a Boolean
            # "max-pressure-with-mask" warmup.
            mask = adjacency.to(v.dtype).unsqueeze(1).expand(n_tls, h,
                                                             n_tls)
            neighbor_count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            attn = mask / neighbor_count
        else:
            # scores[i, h, j] = (q[i, h] . k[j, h]) / sqrt(d)
            scores = torch.einsum('ihd,jhd->ihj', q, k) / (d ** 0.5)
            mask = adjacency.unsqueeze(1).expand(n_tls, h, n_tls)
            scores = scores.masked_fill(~mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)

        # Stash for entropy diagnostics (cheap; no grad path).
        self.last_attention = attn.detach()

        # context[i, h] = sum_j attn[i, h, j] * v[j, h]
        ctx = torch.einsum('ihj,jhd->ihd', attn, v)
        ctx = ctx.reshape(n_tls, h * d)
        out = self.out_proj(ctx)

        if self.residual:
            out = self.layer_norm(out + light_embeddings)
        return out

    def attention_entropy(self) -> torch.Tensor:
        """Per-head, per-light attention entropy from the last forward.

        Returns shape ``(num_heads,)`` -- mean entropy across lights per
        head, in nats. The plan's diagnostic alert fires if entropy
        stays at log(n_neighbors) for >100 episodes post-unfreeze.
        """
        if self.last_attention is None:
            raise RuntimeError("Call forward() before attention_entropy().")
        a = self.last_attention.clamp_min(1e-12)
        # Per (light, head) entropy over neighbors
        per_lh = -(a * a.log()).sum(dim=-1)  # (N_tls, num_heads)
        return per_lh.mean(dim=0)  # (num_heads,)
