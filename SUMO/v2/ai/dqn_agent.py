"""Double-DQN agent with target network and uniform replay buffer."""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from traffic_base import BaseTrafficAI


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self._buf: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, state, action, reward, next_state, done) -> None:
        self._buf.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states = np.stack([b[0] for b in batch])
        actions = np.asarray([b[1] for b in batch], dtype=np.int64)
        rewards = np.asarray([b[2] for b in batch], dtype=np.float32)
        next_states = np.stack([b[3] for b in batch])
        dones = np.asarray([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


class DQNAgent:
    """Double-DQN over a small MLP defined in ``traffic_base.BaseTrafficAI``."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes=(128, 128),
        lr: float = 5e-4,
        gamma: float = 0.95,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_sync_steps: int = 500,
        device: Optional[str] = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_steps = target_sync_steps
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.online = BaseTrafficAI(state_size, list(hidden_sizes), action_size).to(self.device)
        self.target = BaseTrafficAI(state_size, list(hidden_sizes), action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_capacity)

        self._learn_steps = 0

    def act(self, state, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s)
            return int(torch.argmax(q, dim=1).item())

    def remember(self, *transition) -> None:
        self.buffer.push(*transition)

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)

        current_q = self.online(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = torch.argmax(self.online(next_states), dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    # ── persistence ──────────────────────────────────────────

    def save(self, path: str, meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.online.state_dict(),
            "state_size": self.state_size,
            "action_size": self.action_size,
            "meta": dict(meta or {}),
        }, path)

    @classmethod
    def load_for_inference(cls, path: str, hidden_sizes=(128, 128),
                           device: Optional[str] = None) -> "DQNAgent":
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location=dev)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_size = int(ckpt["state_size"])
            action_size = int(ckpt["action_size"])
            sd = ckpt["state_dict"]
        else:
            # legacy: a raw state_dict
            sd = ckpt
            first_w = next(iter(sd.values()))
            last_w = list(sd.values())[-2]  # last Linear weight
            state_size = first_w.shape[1]
            action_size = last_w.shape[0]

        agent = cls(state_size=state_size, action_size=action_size,
                    hidden_sizes=hidden_sizes, device=str(dev))
        agent.online.load_state_dict(sd)
        agent.online.eval()
        agent.target.load_state_dict(agent.online.state_dict())
        return agent
