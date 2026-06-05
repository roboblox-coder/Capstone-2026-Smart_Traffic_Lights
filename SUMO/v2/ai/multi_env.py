"""Multi-intersection environment: one SUMO process, N coordinated agents.

``MultiTlsEnv`` drives every traffic light in the corridor from a single
labelled TraCI connection. Each light is represented by a normal
``SumoTrafficEnv`` *unit* reused only for its probe / state / reward logic —
the units never advance SUMO themselves (that would desync the lights).
``MultiTlsEnv`` owns the clock:

    set all yellows -> tick yellow_time -> set all greens -> tick
    decision_interval -> push shared-arrival + neighbour blocks ->
    read each unit's reward & next state

Coordination signal: before every reward/state read each unit gets

  * ``set_shared_arrived`` — network arrivals this interval (max_pressure_net)
  * ``set_neighbor_features`` — a fixed 6-float block summarising its
    upstream + downstream TLS (queue / pressure / green-progress each),
    built from ``adjacency.json``.

Run order matters; see ``step()``.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from sumo_env import (  # reuse validated helpers / per-TLS logic
    SumoTrafficEnv,
    _next_label,
    _yellow_between,
    checkBinary,
    traci,
)

_NEI_NORM = 40.0  # queue normaliser for the neighbour block


class MultiTlsEnv:
    def __init__(
        self,
        sumo_cfg_file: str,
        adjacency: dict,
        tls_ids: Optional[list] = None,
        time_limit: int = 1200,
        min_green: int = 5,
        yellow_time: int = 5,
        decision_interval: int = 5,
        reward_mode: str = "max_pressure_net",
        reward_gamma: float = 0.05,
        reward_alpha: float = 1.0,
        reward_beta: float = 0.05,
        control_tls: bool = True,
        seed: int = 42,
        use_gui: bool = False,
    ):
        self.sumo_cfg = sumo_cfg_file
        self.adjacency = dict(adjacency)
        self.tls_ids = list(tls_ids) if tls_ids else sorted(self.adjacency)
        self.time_limit = int(time_limit)
        self.min_green = int(min_green)
        self.yellow_time = int(yellow_time)
        self.decision_interval = max(1, int(decision_interval))
        self.reward_mode = reward_mode
        self.control_tls = bool(control_tls)
        self.seed = int(seed)
        self.use_gui = use_gui

        self._label: Optional[str] = None
        self._step_time = 0
        self._arrived_total = 0
        self._departed_total = 0
        self._cumulative_wait = 0.0

        # Build one SumoTrafficEnv per TLS. Its __init__ opens a transient
        # probe connection, reads structure, closes it — safe to do 12x
        # sequentially. We only ever reuse get_state / _reward / probe data.
        self.units: dict = {}
        for tid in self.tls_ids:
            self.units[tid] = SumoTrafficEnv(
                sumo_cfg_file=sumo_cfg_file,
                tls_id=tid,
                time_limit=time_limit,
                min_green=min_green,
                yellow_time=yellow_time,
                decision_interval=decision_interval,
                reward_mode=reward_mode,
                reward_gamma=reward_gamma,
                reward_alpha=reward_alpha,
                reward_beta=reward_beta,
                control_tls=control_tls,
                neighbor_aware=True,
                neighbor_ids=[
                    self.adjacency[tid].get("upstream"),
                    self.adjacency[tid].get("downstream"),
                ],
                seed=seed,
            )
        self._n_tls = len(self.units)

    # ── shape helpers ─────────────────────────────────────────
    @property
    def state_sizes(self) -> dict:
        return {t: u.state_size for t, u in self.units.items()}

    @property
    def action_sizes(self) -> dict:
        return {t: u.action_size for t, u in self.units.items()}

    # ── lifecycle ─────────────────────────────────────────────
    def _conn(self):
        return traci.getConnection(self._label)

    def _start(self) -> None:
        binary = checkBinary("sumo-gui" if self.use_gui else "sumo")
        self._label = _next_label()
        cmd = [
            binary,
            "-c", self.sumo_cfg,
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--seed", str(self.seed),
            "--start",
            "--quit-on-end",
        ]
        # Subclasses (DRWrapper) can append --scale / other CLI args per
        # episode via this attribute; default of [] keeps V1 byte-identical.
        cmd.extend(list(getattr(self, "extra_cli_args", []) or []))
        traci.start(cmd, label=self._label)
        self._step_time = 0
        self._arrived_total = 0
        self._departed_total = 0
        self._cumulative_wait = 0.0

        conn = self._conn()
        for u in self.units.values():
            # Hand the shared connection to every unit so its get_state /
            # _reward read from the one live sim.
            u._label = self._label
            u._time_in_phase = 0
            u._current_green_slot = 0
            u._prev_total_wait = 0.0
            if self.control_tls:
                conn.trafficlight.setRedYellowGreenState(
                    u.tls_id,
                    u._phase_states[u._green_phase_indices[0]],
                )

    def stop(self) -> None:
        if self._label is None:
            return
        try:
            self._conn().close()
        except Exception:
            pass
        self._label = None

    def reset(self) -> dict:
        self.stop()
        for u in self.units.values():
            u.seed = self.seed
        self._start()
        self._refresh_coordination_signals(net_arrived=0)
        return {t: u.get_state() for t, u in self.units.items()}

    # ── ticking ───────────────────────────────────────────────
    def _tick(self) -> int:
        """Advance the shared sim one second; update network accumulators.
        Returns arrivals this second."""
        conn = self._conn()
        conn.simulationStep()
        self._step_time += 1
        n_arr = 0
        try:
            n_arr = int(conn.simulation.getArrivedNumber())
            self._arrived_total += n_arr
            self._departed_total += int(conn.simulation.getDepartedNumber())
        except Exception:
            pass
        # Network waiting time = sum over every controlled lane in the corridor.
        self._cumulative_wait += sum(
            u._total_waiting_time() for u in self.units.values()
        )
        return n_arr

    def _terminal(self) -> bool:
        if self._step_time >= self.time_limit:
            return True
        try:
            return self._conn().simulation.getMinExpectedNumber() <= 0
        except Exception:
            return True

    # ── coordination signals ──────────────────────────────────
    def _lane_queue(self, lanes) -> float:
        conn = self._conn()
        return float(sum(conn.lane.getLastStepHaltingNumber(l) for l in lanes))

    def _neighbor_triplet(self, nbr_id) -> list:
        """[incoming_queue_norm, pressure_norm, green_progress] for a
        neighbour TLS, or zeros if it is absent / not in scope."""
        u = self.units.get(nbr_id)
        if u is None:
            return [0.0, 0.0, 0.0]
        q_in = self._lane_queue(u._incoming_lanes)
        q_out = self._lane_queue(u._outgoing_lanes)
        inq = min(1.0, q_in / _NEI_NORM)
        press = float(np.clip((q_in - q_out) / _NEI_NORM, -1.0, 1.0))
        prog = min(u._time_in_phase, 120) / 120.0
        return [inq, press, prog]

    def _refresh_coordination_signals(self, net_arrived: int) -> None:
        for tid, u in self.units.items():
            u.set_shared_arrived(net_arrived, self._n_tls)
            adj = self.adjacency.get(tid, {})
            block = (self._neighbor_triplet(adj.get("upstream"))
                     + self._neighbor_triplet(adj.get("downstream")))
            u.set_neighbor_features(block)

    def set_reward_weights(self, local_w: float, net_w: float,
                           coord_w: float = 0.0) -> None:
        """Fan the curriculum reward weights out to every unit. Called by
        the trainer once per decision to anneal Phase 1 -> Phase 2.
        ``coord_w`` scales the per-agent downstream-saturation penalty."""
        for u in self.units.values():
            u.set_reward_weights(local_w, net_w, coord_w)

    # ── stepping ──────────────────────────────────────────────
    def step(self, actions: dict):
        """One synchronous decision cycle for every controlled TLS.

        ``actions`` maps tls_id -> green-slot index. Returns
        (states, rewards, done, infos) keyed by tls_id (done is shared).
        """
        conn = self._conn()
        switching = {}

        # 1. Decide switches and fire all yellows together.
        for tid, u in self.units.items():
            a = int(actions[tid]) % u._num_green
            target = u._green_phase_indices[a]
            current = u._green_phase_indices[u._current_green_slot]
            if target != current and u._time_in_phase >= self.min_green:
                ys = _yellow_between(
                    u._phase_states[current], u._phase_states[target]
                )
                conn.trafficlight.setRedYellowGreenState(u.tls_id, ys)
                switching[tid] = a

        net_arrived = 0

        # 2. If anyone switched, run the shared yellow window.
        if switching:
            for _ in range(self.yellow_time):
                net_arrived += self._tick()
                if self._terminal():
                    break

        # 3. Apply every switching unit's new green together.
        for tid, a in switching.items():
            u = self.units[tid]
            conn.trafficlight.setRedYellowGreenState(
                u.tls_id, u._phase_states[u._green_phase_indices[a]]
            )
            u._current_green_slot = a
            u._time_in_phase = 0

        # 4. Hold the (possibly new) greens for the decision interval.
        for _ in range(self.decision_interval):
            net_arrived += self._tick()
            if self._terminal():
                break
        for u in self.units.values():
            u._time_in_phase += self.decision_interval

        # 5. Publish coordination signals, then read rewards / next states.
        self._refresh_coordination_signals(net_arrived)
        done = self._terminal()
        states, rewards, infos = {}, {}, {}
        for tid, u in self.units.items():
            states[tid] = u.get_state()
            rewards[tid] = u._reward(tid in switching)
            infos[tid] = {
                "switched": tid in switching,
                "green_slot": u._current_green_slot,
            }
        return states, rewards, done, infos

    # ── FRAP-form state batch (V2) ────────────────────────────────
    @property
    def frap_p_max(self) -> int:
        """Maximum num_green across all 12 TLS. Used to right-pad the
        per-light phase logits / masks so the shared actor can operate
        on a fixed-shape (n_tls, P_max) tensor."""
        return max(u._num_green for u in self.units.values())

    @property
    def frap_max_movements(self) -> int:
        """Maximum number of controlled-link signal indices across all
        TLS. Used to right-pad the per-light movement features tensor."""
        return max(len(s) for u in self.units.values()
                   for s in u._phase_states)

    def frap_adjacency_tensor(self):
        """(n_tls, n_tls) bool numpy array for the GAT.

        Includes the self-loop on every diagonal entry so a light can
        always attend to its own embedding -- the GAT softmax can hit a
        degenerate case otherwise when no neighbor data is informative.
        Treats adjacency.json's upstream/downstream pairings as
        undirected: if j is a neighbour of i in either direction, both
        ``adj[i, j]`` and ``adj[j, i]`` are True.
        """
        ids = list(self.tls_ids)
        idx = {t: i for i, t in enumerate(ids)}
        n = len(ids)
        adj = np.zeros((n, n), dtype=bool)
        for i, t in enumerate(ids):
            adj[i, i] = True  # self-loop
            nbrs = self.adjacency.get(t, {})
            for direction in ("upstream", "downstream"):
                nbr = nbrs.get(direction)
                if nbr in idx:
                    j = idx[nbr]
                    adj[i, j] = True
                    adj[j, i] = True  # symmetrize
        return adj

    def get_state_frap_batch(self) -> dict:
        """Padded FRAP-form state across all 12 TLS, ready for the V2 net.

        Returns:
            {
              "movement_features"   : float32 [n_tls, max_movements, 3]
              "movement_mask"       : bool    [n_tls, max_movements]
                  (True for the real signal indices on this TLS;
                  False at right-pad positions.)
              "phase_movement_mask" : bool    [n_tls, P_max, max_movements]
              "phase_mask"          : bool    [n_tls, P_max]
              "current_slot"        : int64   [n_tls]
              "time_in_phase"       : float32 [n_tls]
              "tls_ids"             : list[str], ordering of n_tls dim
            }

        Pads with zeros for movement features and False for masks. The
        actor's masked-categorical head reads ``phase_mask`` to clamp
        invalid slots to -inf.
        """
        ids = list(self.tls_ids)
        n = len(ids)
        p_max = self.frap_p_max
        m_max = self.frap_max_movements

        mov_feats = np.zeros((n, m_max, 3), dtype=np.float32)
        mov_mask = np.zeros((n, m_max), dtype=bool)
        pm_mask = np.zeros((n, p_max, m_max), dtype=bool)
        phase_mask = np.zeros((n, p_max), dtype=bool)
        cur_slot = np.zeros((n,), dtype=np.int64)
        t_in_phase = np.zeros((n,), dtype=np.float32)

        for i, tid in enumerate(ids):
            s = self.units[tid].get_state_frap()
            f = s["movement_features"]
            n_mov = f.shape[0]
            n_grn = s["num_green"]
            mov_feats[i, :n_mov] = f
            mov_mask[i, :n_mov] = True
            pm_mask[i, :n_grn, :n_mov] = s["phase_movement_mask"]
            phase_mask[i, :n_grn] = True
            cur_slot[i] = s["current_slot"]
            t_in_phase[i] = s["time_in_phase"]

        return {
            "movement_features": mov_feats,
            "movement_mask": mov_mask,
            "phase_movement_mask": pm_mask,
            "phase_mask": phase_mask,
            "current_slot": cur_slot,
            "time_in_phase": t_in_phase,
            "tls_ids": ids,
        }

    def passive_step(self):
        """Native baseline: advance one second, SUMO's own programs in
        charge (no setRedYellowGreenState calls). Returns (states, done)."""
        net_arrived = self._tick()
        self._refresh_coordination_signals(net_arrived)
        done = self._terminal()
        return {t: u.get_state() for t, u in self.units.items()}, done

    # ── metrics ───────────────────────────────────────────────
    def vehicles_in_network(self) -> int:
        try:
            return int(self._conn().vehicle.getIDCount())
        except Exception:
            return 0

    def metrics_summary(self) -> dict:
        """Network-wide episode metrics (analogous to
        SumoTrafficEnv.metrics_summary but summed over all 12 TLS)."""
        in_net = self.vehicles_in_network()
        arrived = self._arrived_total
        denom = max(1, arrived + in_net)
        return {
            "arrived": arrived,
            "departed": self._departed_total,
            "backlog": int(self._departed_total - arrived),
            "in_network": in_net,
            "cumulative_wait": float(self._cumulative_wait),
            "mean_wait": float(self._cumulative_wait / max(1, self._step_time)),
            "wait_per_vehicle": float(self._cumulative_wait / denom),
            "sim_seconds": int(self._step_time),
        }


def load_adjacency(path: str = "ai/adjacency.json") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python ai/network_topology.py` first."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
