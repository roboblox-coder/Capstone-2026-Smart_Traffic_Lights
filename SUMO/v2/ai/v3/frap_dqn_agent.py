"""Parameter-shared FRAP Double-DQN agent.

One shared FRAPQNet drives all 12 lights. Each light's per-decision
transition (its own FRAP state, chosen phase, reward, next FRAP state)
goes into one shared replay buffer; one network learns from all of them.
Mirrors V1's ai/dqn_agent.py (Double-DQN, target net, eps-greedy,
SmoothL1) but over structured FRAP states padded to (P_max, M_max).
"""
from __future__ import annotations

import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))   # SUMO/v2/ai

from v3.frap_q_net import FRAPQNet  # noqa: E402


def _pad_state(state: dict, p_max: int, m_max: int):
    """Right-pad one light's FRAP state to (p_max, m_max). Returns
    numpy arrays (mov_feats, phase_movement_mask, phase_mask)."""
    mov = state["movement_features"]              # (M, F)
    pm = state["phase_movement_mask"]             # (P, M)
    phase = state["phase_mask"]                   # (P,)
    m, f = mov.shape
    p = phase.shape[0]
    mov_p = np.zeros((m_max, f), dtype=np.float32)
    mov_p[:m] = mov
    pm_p = np.zeros((p_max, m_max), dtype=bool)
    pm_p[:p, :m] = pm
    ph_p = np.zeros((p_max,), dtype=bool)
    ph_p[:p] = phase
    return mov_p, pm_p, ph_p


class FRAPReplayBuffer:
    def __init__(self, p_max: int, m_max: int, capacity: int = 50_000):
        self._buf: deque = deque(maxlen=capacity)
        self.p_max = p_max
        self.m_max = m_max

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, state, action, reward, next_state, done) -> None:
        s = _pad_state(state, self.p_max, self.m_max)
        ns = _pad_state(next_state, self.p_max, self.m_max)
        self._buf.append((s, int(action), float(reward), ns, bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        mov = np.stack([b[0][0] for b in batch])
        pm = np.stack([b[0][1] for b in batch])
        ph = np.stack([b[0][2] for b in batch])
        a = np.asarray([b[1] for b in batch], dtype=np.int64)
        r = np.asarray([b[2] for b in batch], dtype=np.float32)
        nmov = np.stack([b[3][0] for b in batch])
        npm = np.stack([b[3][1] for b in batch])
        nph = np.stack([b[3][2] for b in batch])
        d = np.asarray([b[4] for b in batch], dtype=np.float32)
        return (mov, pm, ph, a, r, nmov, npm, nph, d)


class FRAPDQNAgent:
    def __init__(self, mov_feat_dim: int = 3, p_max: int = 6,
                 m_max: int = 16, embed_dim: int = 128,
                 lr: float = 5e-4, gamma: float = 0.95,
                 buffer_capacity: int = 50_000, batch_size: int = 64,
                 target_sync_steps: int = 500,
                 device: Optional[str] = None):
        self.p_max = p_max
        self.m_max = m_max
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_steps = target_sync_steps
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online = FRAPQNet(mov_feat_dim, embed_dim).to(self.device)
        self.target = FRAPQNet(mov_feat_dim, embed_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = FRAPReplayBuffer(p_max, m_max, buffer_capacity)
        self._learn_steps = 0

    def _state_tensors(self, state: dict):
        mov, pm, ph = _pad_state(state, self.p_max, self.m_max)
        t = lambda x, dt: torch.as_tensor(x, dtype=dt,
                                          device=self.device).unsqueeze(0)
        return (t(mov, torch.float32), t(pm, torch.bool),
                t(ph, torch.bool))

    def act(self, state: dict, epsilon: float = 0.0) -> int:
        ng = int(state["phase_mask"].sum())
        if random.random() < epsilon:
            return random.randint(0, ng - 1)
        with torch.no_grad():
            mov, pm, ph = self._state_tensors(state)
            q = self.online(mov, pm, ph)[0]   # (P_max,)
            return int(torch.argmax(q).item())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None
        (mov, pm, ph, a, r, nmov, npm, nph, d) = self.buffer.sample(
            self.batch_size)
        dev = self.device
        mov = torch.as_tensor(mov, device=dev)
        pm = torch.as_tensor(pm, device=dev)
        ph = torch.as_tensor(ph, device=dev)
        a = torch.as_tensor(a, device=dev).unsqueeze(1)
        r = torch.as_tensor(r, device=dev)
        nmov = torch.as_tensor(nmov, device=dev)
        npm = torch.as_tensor(npm, device=dev)
        nph = torch.as_tensor(nph, device=dev)
        d = torch.as_tensor(d, device=dev)

        current_q = self.online(mov, pm, ph).gather(1, a).squeeze(1)
        with torch.no_grad():
            next_online = self.online(nmov, npm, nph)
            next_actions = torch.argmax(next_online, dim=1, keepdim=True)
            next_q = self.target(nmov, npm, nph).gather(
                1, next_actions).squeeze(1)
            target_q = r + self.gamma * next_q * (1.0 - d)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str, meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.online.state_dict(),
            "p_max": self.p_max, "m_max": self.m_max,
            "meta": dict(meta or {}),
        }, path)

    @classmethod
    def load_for_inference(cls, path: str,
                           device: Optional[str] = None) -> "FRAPDQNAgent":
        dev = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        ag = cls(p_max=int(ckpt["p_max"]), m_max=int(ckpt["m_max"]),
                 device=str(dev))
        ag.online.load_state_dict(ckpt["state_dict"])
        ag.online.eval()
        ag.target.load_state_dict(ag.online.state_dict())
        return ag
