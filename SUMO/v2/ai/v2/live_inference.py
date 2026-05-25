"""V2 inference loop for the live WebSocket runner.

run_websocket_ai.py keeps its per-TLS yellow / min-green state machine
(that bookkeeping is correct and well-tested for V1) and replaces only
the decision step. For V1 the decision is

    actions = {tid: agents[tid].act(build_state(tid))  for tid in ...}

For V2 the decision is

    actions = v2_loop.decide(slot, tip)

where ``v2_loop`` reads each TLS's current TraCI state, builds the
FRAP-form padded batch, and calls the corridor-level
``V2CorridorPolicy.act()`` once. The 40-line refactor noted in
PLAN_V2.md §1.2 lives here, not in the runner.

Architectural note: V1 used a per-TLS state vector mirroring
``SumoTrafficEnv.get_state``. V2 needs per-movement features (and the
phase-movement mask) which the env's ``get_state_frap`` produces from a
live TraCI connection. The runner does NOT instantiate
``MultiTlsEnv`` -- it uses raw TraCI -- so we replicate the per-TLS
piece of ``get_state_frap`` here, against the same TraCI session the
runner already holds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from v2.inference_adapter import V2CorridorPolicy  # noqa: E402


def _build_frap_state_for_tls(traci_mod, tls_id: str,
                              phase_states: list, green_indices: list,
                              num_green: int,
                              current_slot: int,
                              time_in_phase: int) -> dict:
    """Mirror of ``SumoTrafficEnv.get_state_frap`` for the live runner.

    Pulled out as a free function so any future drift between the env's
    get_state_frap and the live build path will show up as a code
    diff, not a behaviour diff at deploy time.
    """
    links = traci_mod.trafficlight.getControlledLinks(tls_id)
    n_mov = len(links)
    feats = np.zeros((n_mov, 3), dtype=np.float32)
    for i, link_group in enumerate(links):
        lanes = set()
        for entry in link_group:
            if entry:
                lanes.add(entry[0])
        if not lanes:
            continue
        q = sum(traci_mod.lane.getLastStepHaltingNumber(l) for l in lanes)
        v = sum(traci_mod.lane.getLastStepVehicleNumber(l) for l in lanes)
        w = sum(traci_mod.lane.getWaitingTime(l) for l in lanes)
        feats[i] = (q, v, w)

    mask = np.zeros((num_green, n_mov), dtype=bool)
    for slot_idx, phase_idx in enumerate(green_indices):
        s = phase_states[phase_idx]
        for j in range(min(n_mov, len(s))):
            mask[slot_idx, j] = s[j] in ("G", "g")

    return {
        "movement_features": feats,
        "phase_movement_mask": mask,
        "num_green": int(num_green),
        "current_slot": int(current_slot),
        "time_in_phase": float(min(time_in_phase, 120) / 120.0),
    }


class V2InferenceLoop:
    """Corridor-level decision driver for the live runner.

    Usage from run_websocket_ai.py::

        v2 = V2InferenceLoop.try_load(traci, ckpt_path, tls_ids, struct,
                                      adjacency)
        if v2 is not None:
            # All 12 lights will use v2; per-TLS DQN load is skipped.
            ...
            actions = v2.decide(slot, tip)
            for tid, action in actions.items():
                # use existing yellow / min-green machine

    The factory ``try_load`` returns None when the V2 checkpoint is
    missing or fails shape checks -- the runner falls back to its
    existing V1 per-TLS path in that case.
    """

    def __init__(self, traci_mod, policy: V2CorridorPolicy,
                 tls_ids: list, struct: dict, adjacency_tensor):
        self._traci = traci_mod
        self.policy = policy
        self.tls_ids = list(tls_ids)
        self.struct = struct  # {tid: {"phase_states", "green", ...}}
        # Pre-build a torch tensor for the adjacency on the policy's
        # device; the dispatch path won't pay for the H2D copy per
        # decision.
        if isinstance(adjacency_tensor, np.ndarray):
            self.adjacency = torch.from_numpy(adjacency_tensor).to(
                policy.device)
        else:
            self.adjacency = adjacency_tensor.to(policy.device)

        # Pre-compute padding shapes the policy expects.
        self.n_tls = len(tls_ids)
        self.p_max = policy.p_max
        self.m_max = policy.max_movements

    @classmethod
    def try_load(cls, traci_mod, ckpt_path: str, tls_ids: list,
                 struct: dict, adjacency: dict,
                 device: Optional[str] = None
                 ) -> Optional["V2InferenceLoop"]:
        """Return an instance if the V2 checkpoint loads cleanly and
        the env shape matches; ``None`` otherwise (runner stays on V1)."""
        if not Path(ckpt_path).exists():
            return None
        try:
            policy = V2CorridorPolicy.load_for_inference(
                ckpt_path, device=device)
        except Exception as exc:
            print(f"V2 checkpoint failed to load ({ckpt_path}): {exc!r}")
            return None
        if list(tls_ids) != policy.tls_ids:
            print(f"V2 checkpoint tls_ids ordering mismatch -- "
                  f"falling back to V1.")
            return None

        # Build adjacency tensor from the runner's adjacency dict (same
        # convention as MultiTlsEnv.frap_adjacency_tensor: undirected,
        # self-loops on diagonal).
        idx = {t: i for i, t in enumerate(tls_ids)}
        n = len(tls_ids)
        adj = np.zeros((n, n), dtype=bool)
        for i, t in enumerate(tls_ids):
            adj[i, i] = True
            nbrs = adjacency.get(t, {})
            for direction in ("upstream", "downstream"):
                nbr = nbrs.get(direction)
                if nbr in idx:
                    j = idx[nbr]
                    adj[i, j] = True
                    adj[j, i] = True
        return cls(traci_mod, policy, tls_ids, struct, adj)

    def build_frap_batch(self, slot: dict, tip: dict) -> dict:
        """Pull per-TLS FRAP state from TraCI and pad to the policy's
        (n_tls, P_max, M_max) shape contract."""
        mov_feats = np.zeros((self.n_tls, self.m_max, 3), dtype=np.float32)
        mov_mask = np.zeros((self.n_tls, self.m_max), dtype=bool)
        pm_mask = np.zeros((self.n_tls, self.p_max, self.m_max),
                           dtype=bool)
        phase_mask = np.zeros((self.n_tls, self.p_max), dtype=bool)
        cur_slot = np.zeros((self.n_tls,), dtype=np.int64)
        t_in_phase = np.zeros((self.n_tls,), dtype=np.float32)

        for i, tid in enumerate(self.tls_ids):
            st = self.struct[tid]
            num_green = len(st["green"])
            s = _build_frap_state_for_tls(
                traci_mod=self._traci, tls_id=tid,
                phase_states=st["phase_states"],
                green_indices=st["green"],
                num_green=num_green,
                current_slot=slot[tid],
                time_in_phase=tip[tid],
            )
            n_mov = s["movement_features"].shape[0]
            mov_feats[i, :n_mov] = s["movement_features"]
            mov_mask[i, :n_mov] = True
            pm_mask[i, :num_green, :n_mov] = s["phase_movement_mask"]
            phase_mask[i, :num_green] = True
            cur_slot[i] = s["current_slot"]
            t_in_phase[i] = s["time_in_phase"]

        return {
            "movement_features": mov_feats,
            "movement_mask": mov_mask,
            "phase_movement_mask": pm_mask,
            "phase_mask": phase_mask,
            "current_slot": cur_slot,
            "time_in_phase": t_in_phase,
            "tls_ids": list(self.tls_ids),
        }

    def decide(self, slot: dict, tip: dict) -> dict:
        """One corridor-level decision. Returns {tid: green_slot_index}."""
        batch = self.build_frap_batch(slot, tip)
        return self.policy.act(batch, self.adjacency, deterministic=True)
