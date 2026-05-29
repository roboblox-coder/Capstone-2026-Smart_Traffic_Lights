"""End-to-end smoke test for the V2 stack (FRAP -> GAT -> actor -> critic).

Synthesizes the dict that ``MultiTlsEnv.get_state_frap_batch`` produces,
runs forward + backward through every module, and verifies the inference
adapter round-trips a checkpoint. No SUMO required.

This is the test the dev env CAN run -- the eventual end-to-end with a
real env still has to happen locally.

Run from ``SUMO/v2``::

    python -m ai.v2.tests.test_smoke
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent.parent))  # SUMO/v2/ai

from v2.frap_encoder import FRAPEncoder  # noqa: E402
from v2.colight_gat import CoLightGAT  # noqa: E402
from v2.shared_policy import SharedActor  # noqa: E402
from v2.centralized_critic import CentralCritic  # noqa: E402
from v2.inference_adapter import V2CorridorPolicy  # noqa: E402

# Note: ``MAPPOConfig`` lives in ``v2.mappo_trainer`` which imports
# ``multi_env`` -> ``sumo_env`` -> ``sumolib``. Importing it here would
# require SUMO on the dev machine just to syntax-check the V2 stack.
# Instead, the inference-adapter roundtrip below builds the config dict
# inline -- matches the schema the trainer writes to disk.


N_TLS = 12
P_MAX = 6           # corridor varies 2..6 green slots per TLS
M_MAX = 16          # max controlled-link count
EMBED_DIM = 128
MOV_FEAT_DIM = 3


def synth_batch(seed: int = 0) -> dict:
    """Mock get_state_frap_batch() output. Per-TLS num_green / n_mov
    varies; per-TLS positions beyond the real count are masked / zeroed
    so the test exercises the padding paths."""
    rng = np.random.default_rng(seed)
    # Varying real per-TLS counts (clamped to {P_MAX, M_MAX}).
    n_green_per_tls = rng.integers(2, P_MAX + 1, size=N_TLS)
    n_mov_per_tls = rng.integers(4, M_MAX + 1, size=N_TLS)

    mov_feats = np.zeros((N_TLS, M_MAX, MOV_FEAT_DIM), dtype=np.float32)
    mov_mask = np.zeros((N_TLS, M_MAX), dtype=bool)
    pm_mask = np.zeros((N_TLS, P_MAX, M_MAX), dtype=bool)
    phase_mask = np.zeros((N_TLS, P_MAX), dtype=bool)
    cur_slot = np.zeros((N_TLS,), dtype=np.int64)
    t_in_phase = np.zeros((N_TLS,), dtype=np.float32)

    for i in range(N_TLS):
        nm = int(n_mov_per_tls[i])
        ng = int(n_green_per_tls[i])
        mov_feats[i, :nm] = rng.uniform(0, 10,
                                        size=(nm, MOV_FEAT_DIM))
        mov_mask[i, :nm] = True
        # Each green phase serves a random non-empty subset of movements.
        for s in range(ng):
            srv = rng.random(nm) > 0.5
            if not srv.any():
                srv[0] = True
            pm_mask[i, s, :nm] = srv
        phase_mask[i, :ng] = True
        cur_slot[i] = int(rng.integers(0, ng))
        t_in_phase[i] = float(rng.uniform(0, 1))

    return {
        "movement_features": mov_feats,
        "movement_mask": mov_mask,
        "phase_movement_mask": pm_mask,
        "phase_mask": phase_mask,
        "current_slot": cur_slot,
        "time_in_phase": t_in_phase,
        "tls_ids": [f"tls_{i:02d}" for i in range(N_TLS)],
    }


def synth_adjacency() -> np.ndarray:
    """A toy corridor adjacency: linear chain with self-loops."""
    adj = np.eye(N_TLS, dtype=bool)
    for i in range(N_TLS - 1):
        adj[i, i + 1] = True
        adj[i + 1, i] = True
    return adj


def test_frap_only() -> None:
    print("  test_frap_only ... ", end="")
    enc = FRAPEncoder(mov_feat_dim=MOV_FEAT_DIM, embed_dim=EMBED_DIM)
    batch = synth_batch(seed=1)
    for i in range(N_TLS):
        n_grn = int(batch["phase_mask"][i].sum())
        feats = torch.from_numpy(batch["movement_features"][i])
        mask = torch.from_numpy(batch["phase_movement_mask"][i, :n_grn])
        le, ppl = enc(feats, mask)
        assert le.shape == (EMBED_DIM,), \
            f"light_embedding {le.shape} != ({EMBED_DIM},)"
        assert ppl.shape == (n_grn,), \
            f"phase_prelogits {ppl.shape} != ({n_grn},)"
    print("OK")


def test_full_stack_forward() -> None:
    print("  test_full_stack_forward ... ", end="")
    enc = FRAPEncoder(mov_feat_dim=MOV_FEAT_DIM, embed_dim=EMBED_DIM)
    gat = CoLightGAT(embed_dim=EMBED_DIM)
    actor = SharedActor(embed_dim=EMBED_DIM)
    critic = CentralCritic(embed_dim=EMBED_DIM, n_tls=N_TLS)

    batch = synth_batch(seed=2)
    adj = torch.from_numpy(synth_adjacency())

    light_embeds = []
    phase_prelogits = []
    for i in range(N_TLS):
        n_grn = int(batch["phase_mask"][i].sum())
        feats = torch.from_numpy(batch["movement_features"][i])
        mask = torch.from_numpy(batch["phase_movement_mask"][i, :n_grn])
        le, ppl = enc(feats, mask)
        padded = torch.zeros(P_MAX, dtype=ppl.dtype)
        padded[:ppl.size(0)] = ppl
        light_embeds.append(le)
        phase_prelogits.append(padded)
    light_embeds = torch.stack(light_embeds, dim=0)
    phase_prelogits = torch.stack(phase_prelogits, dim=0)
    phase_mask = torch.from_numpy(batch["phase_mask"])

    ctx = gat(light_embeds, adj)
    assert ctx.shape == (N_TLS, EMBED_DIM), f"gat out {ctx.shape}"

    logits = actor(ctx, phase_prelogits, phase_mask)
    assert logits.shape == (N_TLS, P_MAX), f"actor out {logits.shape}"

    # Masked positions should receive ~0 probability after softmax.
    probs = torch.softmax(logits, dim=-1)
    masked_mass = (probs * (~phase_mask).float()).sum().item()
    assert masked_mass < 1e-3, \
        f"masked positions hold {masked_mass:.6f} probability mass"

    actions, logprobs, entropy = SharedActor.sample_actions(
        logits, deterministic=False)
    assert actions.shape == (N_TLS,)
    assert logprobs.shape == (N_TLS,)
    assert entropy.shape == (N_TLS,)
    # No sampled action should fall on a masked slot.
    for i in range(N_TLS):
        assert phase_mask[i, actions[i]].item(), \
            f"sampled masked slot at tls {i}"

    value = critic(light_embeds)
    assert value.dim() == 0, f"critic out {value.shape} not scalar"

    # Entropy in [0, log(P_MAX)] per-tls.
    upper = float(np.log(P_MAX) + 1e-3)
    for i, e in enumerate(entropy):
        assert 0.0 <= e.item() <= upper, \
            f"entropy[{i}]={e.item()} out of [0, log(P_MAX)]"
    print("OK")


def test_full_stack_backward() -> None:
    print("  test_full_stack_backward ... ", end="")
    enc = FRAPEncoder(mov_feat_dim=MOV_FEAT_DIM, embed_dim=EMBED_DIM)
    gat = CoLightGAT(embed_dim=EMBED_DIM)
    actor = SharedActor(embed_dim=EMBED_DIM)
    critic = CentralCritic(embed_dim=EMBED_DIM, n_tls=N_TLS)

    batch = synth_batch(seed=3)
    adj = torch.from_numpy(synth_adjacency())

    light_embeds = []
    phase_prelogits = []
    for i in range(N_TLS):
        n_grn = int(batch["phase_mask"][i].sum())
        feats = torch.from_numpy(batch["movement_features"][i])
        mask = torch.from_numpy(batch["phase_movement_mask"][i, :n_grn])
        le, ppl = enc(feats, mask)
        padded = torch.zeros(P_MAX, dtype=ppl.dtype)
        padded[:ppl.size(0)] = ppl
        light_embeds.append(le)
        phase_prelogits.append(padded)
    light_embeds = torch.stack(light_embeds, dim=0)
    phase_prelogits = torch.stack(phase_prelogits, dim=0)
    phase_mask = torch.from_numpy(batch["phase_mask"])

    ctx = gat(light_embeds, adj)
    logits = actor(ctx, phase_prelogits, phase_mask)
    value = critic(light_embeds)

    loss = logits.sum() + value
    loss.backward()

    grad_found = False
    for p in (list(enc.parameters()) + list(gat.parameters())
              + list(actor.parameters()) + list(critic.parameters())):
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            grad_found = True
            break
    assert grad_found, "no parameter received gradient"
    print("OK")


def test_gat_attention_modes() -> None:
    print("  test_gat_attention_modes ... ", end="")
    gat = CoLightGAT(embed_dim=EMBED_DIM)
    adj = torch.from_numpy(synth_adjacency())
    le = torch.randn(N_TLS, EMBED_DIM)

    gat.set_frozen_uniform(True)
    _out_frozen = gat(le, adj)
    ent_frozen = gat.attention_entropy()
    assert ent_frozen.shape == (gat.num_heads,)
    # Frozen-uniform on a chain with self-loops: each row has 2..3
    # neighbours. Entropy should be > 0 (uniform over >=2 neighbours).
    assert ent_frozen.min().item() > 0.0, \
        f"frozen-uniform entropy collapsed to 0: {ent_frozen}"

    gat.set_frozen_uniform(False)
    _out_learned = gat(le, adj)
    ent_learned = gat.attention_entropy()
    assert ent_learned.shape == (gat.num_heads,)
    # Learned attention with random weights -> roughly uniform too at
    # init; just sanity-check it's a real number.
    assert torch.isfinite(ent_learned).all()
    print("OK")


def test_inference_adapter_roundtrip() -> None:
    print("  test_inference_adapter_roundtrip ... ", end="")
    # Inline config dict mirrors MAPPOConfig fields the inference
    # adapter actually reads (embed_dim, gat_heads, gat_head_dim,
    # critic_hidden). Keeping this matched to MAPPOConfig is a manual
    # contract; mappo_trainer's own integration tests cover drift.
    cfg = {
        "embed_dim": EMBED_DIM,
        "gat_heads": 4,
        "gat_head_dim": 32,
        "critic_hidden": 128,
    }
    enc = FRAPEncoder(mov_feat_dim=MOV_FEAT_DIM, embed_dim=EMBED_DIM)
    gat = CoLightGAT(embed_dim=EMBED_DIM)
    actor = SharedActor(embed_dim=EMBED_DIM)
    critic = CentralCritic(embed_dim=EMBED_DIM, n_tls=N_TLS,
                           hidden_dim=128)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "best.pth"
        torch.save({
            "agent_type": "frap_gat_mappo",
            "config": cfg,
            "n_tls": N_TLS,
            "p_max": P_MAX,
            "max_movements": M_MAX,
            "tls_ids": [f"tls_{i:02d}" for i in range(N_TLS)],
            "encoder": enc.state_dict(),
            "gat": gat.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "episodes_done": 0,
            "gradient_steps": 0,
            "best_eval_wpv": float("inf"),
            "meta": {},
        }, ckpt_path)

        policy = V2CorridorPolicy.load_for_inference(str(ckpt_path))
        assert policy.n_tls == N_TLS
        assert policy.p_max == P_MAX
        assert policy.max_movements == M_MAX

        batch = synth_batch(seed=4)
        adj = synth_adjacency()
        actions = policy.act(batch, adj, deterministic=True)
        assert isinstance(actions, dict)
        assert len(actions) == N_TLS
        # Returned slots must be within the per-TLS num_green band.
        for i, tid in enumerate(batch["tls_ids"]):
            a = actions[tid]
            n_grn = int(batch["phase_mask"][i].sum())
            assert 0 <= a < n_grn, \
                f"action {a} out of [0, {n_grn}) for {tid}"
    print("OK")


def test_frap_batched_matches_per_tls() -> None:
    """The batched FRAP forward must agree with the per-light forward
    on every (b, t) entry within fp32 tolerance. This is the
    correctness contract for the trainer's PPO inner-loop refactor."""
    print("  test_frap_batched_matches_per_tls ... ", end="")
    torch.manual_seed(0)
    enc = FRAPEncoder(mov_feat_dim=MOV_FEAT_DIM, embed_dim=EMBED_DIM)
    enc.eval()
    batch = synth_batch(seed=7)

    # Per-light path: call the original forward on each TLS.
    per_le, per_ppl = [], []
    for i in range(N_TLS):
        n_grn = int(batch["phase_mask"][i].sum())
        feats = torch.from_numpy(batch["movement_features"][i])
        mask = torch.from_numpy(batch["phase_movement_mask"][i, :n_grn])
        with torch.no_grad():
            le, ppl = enc(feats, mask)
        padded = torch.zeros(P_MAX, dtype=ppl.dtype)
        padded[:ppl.size(0)] = ppl
        per_le.append(le)
        per_ppl.append(padded)
    per_le = torch.stack(per_le, dim=0)        # (N_TLS, D)
    per_ppl = torch.stack(per_ppl, dim=0)      # (N_TLS, P_MAX)

    # Batched path.
    mov_feats = torch.from_numpy(batch["movement_features"])
    pm_mask = torch.from_numpy(batch["phase_movement_mask"])
    phase_mask = torch.from_numpy(batch["phase_mask"])
    with torch.no_grad():
        bat_le, bat_ppl = enc.forward_batched(
            mov_feats, pm_mask, phase_mask)

    le_err = (per_le - bat_le).abs().max().item()
    ppl_err = (per_ppl - bat_ppl).abs().max().item()
    assert le_err < 1e-5, f"light_embed max diff {le_err} too high"
    assert ppl_err < 1e-5, f"phase_prelogit max diff {ppl_err} too high"
    print(f"OK (le diff={le_err:.2e}, ppl diff={ppl_err:.2e})")


def test_batched_minibatch_shapes() -> None:
    """End-to-end batched pass mirroring the trainer's PPO inner loop:
    (B, N_tls, ...) -> FRAP -> GAT -> actor / critic, with batched
    actor.forward_batched."""
    print("  test_batched_minibatch_shapes ... ", end="")
    torch.manual_seed(0)
    enc = FRAPEncoder(mov_feat_dim=MOV_FEAT_DIM, embed_dim=EMBED_DIM)
    gat = CoLightGAT(embed_dim=EMBED_DIM)
    actor = SharedActor(embed_dim=EMBED_DIM)
    critic = CentralCritic(embed_dim=EMBED_DIM, n_tls=N_TLS)

    B = 5
    batches = [synth_batch(seed=10 + i) for i in range(B)]
    mov = torch.from_numpy(np.stack(
        [b["movement_features"] for b in batches]))
    pm = torch.from_numpy(np.stack(
        [b["phase_movement_mask"] for b in batches]))
    phase = torch.from_numpy(np.stack(
        [b["phase_mask"] for b in batches]))
    adj = torch.from_numpy(synth_adjacency())

    # Flatten (B, N_tls) for FRAP.
    mov_flat = mov.reshape(B * N_TLS, *mov.shape[2:])
    pm_flat = pm.reshape(B * N_TLS, *pm.shape[2:])
    phase_flat = phase.reshape(B * N_TLS, -1)
    le_flat, ppl_flat = enc.forward_batched(mov_flat, pm_flat, phase_flat)
    le_raw = le_flat.reshape(B, N_TLS, -1)
    ppl = ppl_flat.reshape(B, N_TLS, -1)
    assert le_raw.shape == (B, N_TLS, EMBED_DIM)
    assert ppl.shape == (B, N_TLS, P_MAX)

    le_ctx = gat.forward_batched(le_raw, adj)
    assert le_ctx.shape == (B, N_TLS, EMBED_DIM)

    logits = actor.forward_batched(le_ctx, ppl, phase)
    assert logits.shape == (B, N_TLS, P_MAX)
    # Masked positions still zero-mass under softmax.
    probs = torch.softmax(logits, dim=-1)
    masked_mass = (probs * (~phase).float()).sum().item()
    assert masked_mass < 1e-3, f"masked mass {masked_mass}"

    values = critic(le_raw)
    assert values.shape == (B,), f"critic batched shape {values.shape}"

    # evaluate_actions over the batch.
    actions = torch.zeros((B, N_TLS), dtype=torch.long)
    for b in range(B):
        for t in range(N_TLS):
            n_grn = int(phase[b, t].sum())
            actions[b, t] = int(np.random.randint(0, n_grn))
    logprobs, entropy = SharedActor.evaluate_actions(logits, actions)
    assert logprobs.shape == (B, N_TLS)
    assert entropy.shape == (B, N_TLS)
    print("OK")


def test_cosine_lr_schedule() -> None:
    """cosine_lr returns lr_start at progress=0, lr_final at progress=1,
    and the midpoint average between them. Must clamp progress to [0, 1]."""
    print("  test_cosine_lr_schedule ... ", end="")
    from v2.mappo_trainer import cosine_lr

    lr0 = 3e-4
    lrN = 5e-5
    # Endpoints
    assert abs(cosine_lr(0.0, lr0, lrN) - lr0) < 1e-12, \
        f"progress=0 should give lr_start, got {cosine_lr(0.0, lr0, lrN)}"
    assert abs(cosine_lr(1.0, lr0, lrN) - lrN) < 1e-12, \
        f"progress=1 should give lr_final, got {cosine_lr(1.0, lr0, lrN)}"
    # Midpoint of cosine (1+cos(pi/2))/2 = 0.5, so result is the average
    mid_expected = (lr0 + lrN) / 2
    assert abs(cosine_lr(0.5, lr0, lrN) - mid_expected) < 1e-12, \
        f"progress=0.5 should give mean, got {cosine_lr(0.5, lr0, lrN)}"
    # Clamping below 0 / above 1
    assert cosine_lr(-0.5, lr0, lrN) == cosine_lr(0.0, lr0, lrN)
    assert cosine_lr(2.0, lr0, lrN) == cosine_lr(1.0, lr0, lrN)
    print("OK")


def main() -> int:
    print("V2 smoke tests:")
    test_frap_only()
    test_frap_batched_matches_per_tls()
    test_full_stack_forward()
    test_full_stack_backward()
    test_gat_attention_modes()
    test_batched_minibatch_shapes()
    test_inference_adapter_roundtrip()
    test_cosine_lr_schedule()
    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
