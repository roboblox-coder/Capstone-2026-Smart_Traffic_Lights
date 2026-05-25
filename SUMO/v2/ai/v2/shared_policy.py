"""Parameter-shared actor head with masked-categorical action output.

Sits at the top of the V2 stack:
  FRAPEncoder (per light)  ->  CoLightGAT (cross-light)  ->  SharedActor

The actor takes:
  - the GAT-enriched per-light embedding (size embed_dim)
  - the per-light FRAP phase_prelogits (size num_green)
  - the per-light phase_mask telling which slots are real vs padding

It produces a categorical distribution over phase slots, with masked
positions clamped to -inf before softmax so they receive zero
probability AND zero gradient through the cross-entropy. PPO logprobs
and entropy use the same masked distribution; that consistency matters
when the policy is updated.

The "shared" part: a single SharedActor is reused across all 12 TLS in
the corridor. The (very small) per-light state -- num_green, current
slot, phase_mask -- is passed in as a tensor at forward time, so no
TLS-specific weights are needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedActor(nn.Module):
    """Param-shared phase-policy head.

    Args:
        embed_dim: matches FRAPEncoder / CoLightGAT.
        head_hidden: hidden dim for the per-light combining MLP.
    """

    def __init__(self, embed_dim: int = 128, head_hidden: int = 64):
        super().__init__()
        # Combines the GAT-context embedding with the FRAP phase prelogits
        # into a per-phase score. The phase prelogit is a single scalar
        # per phase; concatenate with the (broadcast) light embedding so
        # the head's effective input dim is embed_dim + 1 per phase.
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 1, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def logits(self, light_embedding: torch.Tensor,
               phase_prelogits: torch.Tensor,
               phase_mask: torch.Tensor) -> torch.Tensor:
        """Per-phase masked logits for a single TLS.

        Args:
            light_embedding: (embed_dim,) GAT-enriched per-light vector.
            phase_prelogits: (P_max,) FRAP per-phase scores, padded with
                zeros beyond ``num_green``.
            phase_mask: (P_max,) bool, True for valid phase slots.

        Returns:
            (P_max,) tensor of logits with invalid slots at -inf.
        """
        p_max = phase_prelogits.size(0)
        # Broadcast the light embedding to all P_max phases and concat
        # with the per-phase prelogit (treated as a 1-dim feature).
        emb = light_embedding.unsqueeze(0).expand(p_max, -1)
        feats = torch.cat([emb, phase_prelogits.unsqueeze(-1)], dim=-1)
        scores = self.head(feats).squeeze(-1)  # (P_max,)
        # Masking: invalid slots -> -inf so softmax assigns 0 probability.
        # Using -1e9 instead of float('-inf') keeps gradient stable when
        # the mask happens to coincide with a row of -inf (rare edge).
        return scores.masked_fill(~phase_mask, -1e9)

    def forward(self, light_embeddings: torch.Tensor,
                phase_prelogits: torch.Tensor,
                phase_masks: torch.Tensor) -> torch.Tensor:
        """Batched over the corridor.

        Args:
            light_embeddings: (N_tls, embed_dim)
            phase_prelogits: (N_tls, P_max) -- FRAP outputs already padded.
            phase_masks: (N_tls, P_max) bool.
        Returns:
            (N_tls, P_max) logits with invalid slots masked.
        """
        n_tls = light_embeddings.size(0)
        out = torch.empty_like(phase_prelogits)
        for i in range(n_tls):
            out[i] = self.logits(light_embeddings[i],
                                 phase_prelogits[i],
                                 phase_masks[i])
        return out

    @staticmethod
    def sample_actions(logits: torch.Tensor,
                       deterministic: bool = False) -> tuple:
        """Sample actions + log-probs from masked logits.

        Args:
            logits: (N_tls, P_max) with masked positions at -1e9.
            deterministic: if True, argmax; else categorical sample.
        Returns:
            actions: (N_tls,) long
            logprobs: (N_tls,) float, log p(a_i | state_i)
            entropy: (N_tls,) float, distribution entropy (nats)
        """
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, logprobs, entropy

    @staticmethod
    def evaluate_actions(logits: torch.Tensor,
                         actions: torch.Tensor) -> tuple:
        """Compute log-prob + entropy for replayed actions (PPO update).

        Args:
            logits: (B, N_tls, P_max) batched per-step.
            actions: (B, N_tls) long.
        Returns:
            (logprobs, entropy), each shape (B, N_tls).
        """
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()
