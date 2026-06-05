"""V3 FRAP-DQN unit tests (no SUMO; torch + numpy only)."""
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent.parent))   # SUMO/v2/ai

from v2.frap_encoder import FRAPEncoder  # noqa: E402

EMBED = 128
P_MAX = 6
M_MAX = 16


def _synth(seed=0):
    rng = np.random.default_rng(seed)
    B = 4
    mov = rng.uniform(0, 10, size=(B, M_MAX, 3)).astype(np.float32)
    pm = np.zeros((B, P_MAX, M_MAX), dtype=bool)
    phase = np.zeros((B, P_MAX), dtype=bool)
    for b in range(B):
        ng = int(rng.integers(2, P_MAX + 1))
        phase[b, :ng] = True
        for s in range(ng):
            srv = rng.random(M_MAX) > 0.5
            if not srv.any():
                srv[0] = True
            pm[b, s] = srv
    return (torch.from_numpy(mov), torch.from_numpy(pm),
            torch.from_numpy(phase))


def test_phase_embeddings_batched_shape():
    enc = FRAPEncoder(mov_feat_dim=3, embed_dim=EMBED)
    mov, pm, phase = _synth(1)
    pe = enc.phase_embeddings_batched(mov, pm)
    assert pe.shape == (mov.shape[0], P_MAX, EMBED), \
        f"phase embeds {pe.shape} != (B, {P_MAX}, {EMBED})"


def test_q_per_phase_discrimination():
    """Perturbing ONE phase's served movements must change THAT phase's
    Q materially, while leaving other (real) phases' Q nearly unchanged.
    This is the §5 no-dilution guarantee: Q(phase_i) depends on phase i's
    own features, not a phase-constant embedding."""
    from v3.frap_q_net import FRAPQNet
    net = FRAPQNet(mov_feat_dim=3, embed_dim=EMBED)
    net.eval()

    # One light: 3 real phases, disjoint served movements so a change to
    # phase 0's movements cannot leak through shared movements.
    M = 9
    mov = torch.zeros(1, M, 3)
    mov[0, :, 0] = 1.0  # uniform baseline halting
    pm = torch.zeros(1, P_MAX, M, dtype=torch.bool)
    pm[0, 0, 0:3] = True
    pm[0, 1, 3:6] = True
    pm[0, 2, 6:9] = True
    phase = torch.zeros(1, P_MAX, dtype=torch.bool)
    phase[0, :3] = True

    with torch.no_grad():
        q_base = net(mov, pm, phase)[0].clone()   # (P_max,)
        mov2 = mov.clone()
        mov2[0, 0:3, 0] += 25.0                    # spike phase 0's lanes
        q_after = net(mov2, pm, phase)[0]

    d0 = (q_after[0] - q_base[0]).abs().item()
    d_others = (q_after[1:3] - q_base[1:3]).abs().max().item()
    assert d0 > 1e-3, f"phase 0 Q did not respond to its own change ({d0})"
    assert d_others < d0 * 0.25, \
        f"change leaked into other phases (d0={d0}, others={d_others})"


def test_q_masks_padded_phases():
    """Q at padded (non-real) phase slots must be -inf so argmax never
    selects them."""
    from v3.frap_q_net import FRAPQNet
    net = FRAPQNet(mov_feat_dim=3, embed_dim=EMBED)
    mov, pm, phase = _synth(2)
    q = net(mov, pm, phase)             # (B, P_max)
    masked = q[~phase]
    # Masked with -1e9 (not -inf: keeps the DQN target NaN-safe and
    # matches the codebase convention). Just needs to be far below any
    # real Q so argmax never selects a padded slot.
    assert (masked < -1e8).all(), \
        "padded phase slots must be strongly negative (never argmax-able)"


def _agent_synth_state(seed=0):
    """A single-light FRAP state dict like the env emits per light."""
    rng = np.random.default_rng(seed)
    M, P = M_MAX, P_MAX
    ng = 3
    mov = rng.uniform(0, 10, size=(M, 3)).astype(np.float32)
    pm = np.zeros((P, M), dtype=bool)
    for s in range(ng):
        pm[s, (s * 3):(s * 3 + 3)] = True
    phase = np.zeros((P,), dtype=bool)
    phase[:ng] = True
    return {"movement_features": mov, "phase_movement_mask": pm,
            "phase_mask": phase}


def test_agent_act_in_range():
    from v3.frap_dqn_agent import FRAPDQNAgent
    ag = FRAPDQNAgent(mov_feat_dim=3, p_max=P_MAX, m_max=M_MAX,
                      embed_dim=EMBED, batch_size=8)
    st = _agent_synth_state(1)
    a = ag.act(st, epsilon=0.0)
    ng = int(st["phase_mask"].sum())
    assert 0 <= a < ng, f"action {a} outside real phases [0,{ng})"
    # epsilon=1.0 must also stay within real phases (never picks padded).
    for _ in range(20):
        a = ag.act(st, epsilon=1.0)
        assert 0 <= a < ng


def test_agent_learn_step_runs():
    """A learn() call on a filled buffer returns a finite loss and steps
    the params."""
    from v3.frap_dqn_agent import FRAPDQNAgent
    ag = FRAPDQNAgent(mov_feat_dim=3, p_max=P_MAX, m_max=M_MAX,
                      embed_dim=EMBED, batch_size=8)
    for i in range(32):
        s = _agent_synth_state(i)
        ns = _agent_synth_state(i + 100)
        ag.remember(s, action=i % 3, reward=float(i % 5),
                    next_state=ns, done=(i % 7 == 0))
    before = [p.clone() for p in ag.online.parameters()]
    loss = ag.learn()
    assert loss is not None and np.isfinite(loss), f"bad loss {loss}"
    after = list(ag.online.parameters())
    moved = any((a - b).abs().sum().item() > 0
                for a, b in zip(after, before))
    assert moved, "online params did not update"


if __name__ == "__main__":
    test_phase_embeddings_batched_shape()
    test_q_per_phase_discrimination()
    test_q_masks_padded_phases()
    test_agent_act_in_range()
    test_agent_learn_step_runs()
    print("task3 OK")
