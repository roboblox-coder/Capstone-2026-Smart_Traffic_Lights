"""Single-intersection RL environment for SUMO via TraCI.

Action semantics
----------------
The agent picks a target green-phase index (one of ``green_phase_indices``).
The env automatically inserts a yellow transition and enforces a minimum
green time, so the agent cannot flicker the light.

State
-----
For each controlled lane:   [halting_count, waiting_time, mean_speed]
Plus:                       one-hot of current green-phase slot,
                            time_in_current_phase (normalised)

Reward
------
Default:        r_t = -sum(waiting_time over controlled lanes)
``differential``: r_t = (sum_wait_{t-1} - sum_wait_t) - alpha * switch_penalty
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    try:
        import sumolib as _sl
        sys.path.append(
            os.path.join(os.path.dirname(os.path.dirname(_sl.__file__)), "tools")
        )
    except ImportError as e:
        raise EnvironmentError("Set SUMO_HOME or install sumolib") from e

import traci  # noqa: E402
from sumolib import checkBinary  # noqa: E402


_TRACI_LABEL_COUNTER = 0


def _next_label() -> str:
    global _TRACI_LABEL_COUNTER
    _TRACI_LABEL_COUNTER += 1
    return f"sumo_env_{os.getpid()}_{_TRACI_LABEL_COUNTER}"


def _is_green(state_str: str) -> bool:
    s = state_str.lower()
    return ("g" in s) and ("y" not in s)


def _yellow_between(from_state: str, to_state: str) -> str:
    """Build a yellow transition string: G/g -> r becomes y."""
    out = []
    for a, b in zip(from_state, to_state):
        if a in ("G", "g") and b in ("r", "s"):
            out.append("y")
        else:
            out.append(a)
    return "".join(out)


class SumoTrafficEnv:
    """Single-intersection environment for a DQN agent."""

    def __init__(
        self,
        sumo_cfg_file: str,
        tls_id: str,
        time_limit: int = 3600,
        use_gui: bool = False,
        min_green: int = 5,
        yellow_time: int = 5,
        reward_mode: str = "waiting",
        switch_penalty: float = 0.1,
        extra_sumo_args: Optional[list] = None,
        control_tls: bool = True,
        seed: int = 42,
        decision_interval: int = 1,
        starve_penalty: float = 0.01,
        reward_alpha: float = 1.0,
        reward_beta: float = 0.05,
        reward_gamma: float = 0.05,
        neighbor_aware: bool = False,
        neighbor_ids: Optional[list] = None,
    ):
        self.sumo_cfg = sumo_cfg_file
        self.tls_id = tls_id
        self.time_limit = int(time_limit)
        self.use_gui = use_gui
        self.min_green = int(min_green)
        self.yellow_time = int(yellow_time)
        self.reward_mode = reward_mode
        self.switch_penalty = float(switch_penalty)
        self.extra_sumo_args = list(extra_sumo_args or [])
        self.control_tls = bool(control_tls)
        self.seed = int(seed)
        self.decision_interval = max(1, int(decision_interval))
        self.starve_penalty = float(starve_penalty)
        self.reward_alpha = float(reward_alpha)
        self.reward_beta = float(reward_beta)
        self.reward_gamma = float(reward_gamma)
        # Coordination: when neighbor_aware, get_state() appends a fixed
        # 6-float block describing the upstream + downstream TLS. The block
        # is computed by the multi-TLS wrapper (which can see other lights)
        # and pushed in via set_neighbor_features(); a lone single-TLS env
        # leaves it at zeros, so default behaviour is byte-for-byte
        # unchanged and old checkpoints still load.
        self.neighbor_aware = bool(neighbor_aware)
        self.neighbor_ids = list(neighbor_ids or [])
        self._neighbor_block_size = 6
        self._neighbor_features = np.zeros(
            self._neighbor_block_size, dtype=np.float32
        )
        # Network-wide arrivals this decision interval + the TLS count,
        # set by the multi-TLS wrapper before _reward() for max_pressure_net.
        self._shared_arrived: int = 0
        self._n_tls: int = 1
        # Curriculum weights for the ``max_pressure_net`` reward. The trainer
        # anneals these (Phase 1: pure validated local max-pressure with
        # net_w=0; Phase 2: ramp net_w up so the corridor-throughput term
        # actually influences behaviour). Defaults are chosen so the reward
        # is *byte-identical* to the previous fixed formula
        #   r = -|q_in-q_out|/n + reward_gamma * (shared/n_tls)
        # whenever set_reward_weights() is never called — preserving the
        # single-TLS regression guarantee and old-checkpoint reproducibility.
        self._reward_local_w: float = 1.0
        self._reward_net_w: float = float(reward_gamma)
        # Per-agent downstream-saturation penalty weight. Default 0.0 keeps
        # the reward byte-identical (single-TLS regression + Phase-A
        # reproducibility); the multi-TLS trainer ramps it in during Phase 2.
        self._reward_coord_w: float = 0.0

        self._label: Optional[str] = None
        self._step_time: int = 0
        self._time_in_phase: int = 0
        self._current_green_slot: int = 0
        self._prev_total_wait: float = 0.0

        # Metric accumulators — reset on every start_simulation()
        self._arrived_total: int = 0
        self._departed_total: int = 0
        self._cumulative_wait: float = 0.0
        # Vehicles that arrived since the last _reward() call (for the
        # throughput term in the ``combined`` reward).
        self._arrived_since_reward: int = 0

        self._controlled_lanes: list = []
        # Incoming/outgoing lane sets for the ``max_pressure`` reward.
        self._incoming_lanes: list = []
        self._outgoing_lanes: list = []
        self._green_phase_indices: list = []
        self._phase_states: list = []
        self._num_green: int = 0

        self._probe_network()

    # ── lifecycle ─────────────────────────────────────────────

    def _probe_network(self) -> None:
        """Open SUMO once headlessly to read TLS structure, then close."""
        label = _next_label()
        cmd = [
            checkBinary("sumo"),
            "-c", self.sumo_cfg,
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--start",
            "--quit-on-end",
        ]
        traci.start(cmd, label=label)
        conn = traci.getConnection(label)
        try:
            if self.tls_id not in conn.trafficlight.getIDList():
                raise ValueError(
                    f"TLS id {self.tls_id!r} not found. Available: "
                    f"{list(conn.trafficlight.getIDList())[:5]} ..."
                )

            lanes = conn.trafficlight.getControlledLanes(self.tls_id)
            self._controlled_lanes = list(dict.fromkeys(lanes))

            # Per-link (from_lane, to_lane, via) — used by max_pressure.
            # getControlledLinks has one entry per signal index (empty
            # tuples for unused indices); de-dup into incoming/outgoing.
            links = conn.trafficlight.getControlledLinks(self.tls_id)
            inc, out = [], []
            for entry in links:
                for ct in entry:
                    if ct:
                        if ct[0]:
                            inc.append(ct[0])
                        if ct[1]:
                            out.append(ct[1])
            self._incoming_lanes = list(dict.fromkeys(inc))
            self._outgoing_lanes = list(dict.fromkeys(out))

            logic = conn.trafficlight.getAllProgramLogics(self.tls_id)[0]
            self._phase_states = [p.state for p in logic.getPhases()]
            self._green_phase_indices = [
                i for i, s in enumerate(self._phase_states) if _is_green(s)
            ]
            if not self._green_phase_indices:
                raise ValueError(
                    f"TLS {self.tls_id} has no detectable green phases."
                )
            self._num_green = len(self._green_phase_indices)
        finally:
            conn.close()

    def start_simulation(self) -> None:
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
        ] + self.extra_sumo_args
        traci.start(cmd, label=self._label)
        self._step_time = 0
        self._time_in_phase = 0
        self._current_green_slot = 0
        self._prev_total_wait = 0.0
        self._arrived_total = 0
        self._departed_total = 0
        self._cumulative_wait = 0.0
        self._arrived_since_reward = 0

        if self.control_tls:
            # Lock the TLS into the agent-controlled regime. Drive via raw
            # state strings rather than setPhase because mixing setPhase with
            # setRedYellowGreenState (used for yellows below) lands us on a
            # one-phase override program where setPhase indices stop matching.
            conn = self._conn()
            conn.trafficlight.setRedYellowGreenState(
                self.tls_id, self._phase_states[self._green_phase_indices[0]]
            )

    def stop_simulation(self) -> None:
        if self._label is None:
            return
        try:
            self._conn().close()
        except Exception:
            pass
        self._label = None

    def reset(self) -> np.ndarray:
        self.stop_simulation()
        self.start_simulation()
        self._prev_total_wait = self._total_waiting_time()
        return self.get_state()

    # ── observations ──────────────────────────────────────────

    def _conn(self):
        return traci.getConnection(self._label)

    def _total_waiting_time(self) -> float:
        conn = self._conn()
        return float(sum(conn.lane.getWaitingTime(l) for l in self._controlled_lanes))

    def get_state(self) -> np.ndarray:
        conn = self._conn()
        feats: list = []
        for lane in self._controlled_lanes:
            queue = conn.lane.getLastStepHaltingNumber(lane)
            waiting = conn.lane.getWaitingTime(lane)
            speed = conn.lane.getLastStepMeanSpeed(lane)
            # cheap normalisation
            feats.extend([queue / 20.0, waiting / 60.0, speed / 15.0])

        one_hot = [0.0] * self._num_green
        one_hot[self._current_green_slot] = 1.0
        feats.extend(one_hot)
        feats.append(min(self._time_in_phase, 120) / 120.0)
        if self.neighbor_aware:
            feats.extend(float(x) for x in self._neighbor_features)
        return np.asarray(feats, dtype=np.float32)

    def get_state_frap(self) -> dict:
        """Per-movement features + phase masks for the FRAP encoder (V2).

        Returns a dict, not a vector. V2's parameter-shared encoder needs
        variable-length movement input; flattening to a fixed vector here
        would erase the phase symmetries FRAP exploits.

        Keys:
          movement_features    : np.ndarray[num_movements, 3] of
                                 (halting, vehicles, waiting_time)
          phase_movement_mask  : np.ndarray[num_green, num_movements] bool;
                                 mask[slot, m] is True iff movement m has
                                 a green signal ('G' or 'g') in that phase.
          num_green            : int, this TLS's actual green-slot count
                                 (the multi-TLS wrapper pads to P_max).
          current_slot         : int in [0, num_green), this TLS's
                                 currently active green slot.
          time_in_phase        : float in [0,1], min(t, 120) / 120.

        Movement definition: one entry per signal index in the phase state
        string -- i.e. one per element of ``getControlledLinks(tls)``. This
        is the granularity SUMO controls the light at, and the granularity
        FRAP's pairwise competition matrix is defined over.
        """
        conn = self._conn()
        controlled_links = conn.trafficlight.getControlledLinks(self.tls_id)
        n_mov = len(controlled_links)

        feats = np.zeros((n_mov, 3), dtype=np.float32)
        for i, link_group in enumerate(controlled_links):
            lanes = set()
            for entry in link_group:
                if entry:
                    lanes.add(entry[0])  # from-lane
            if not lanes:
                continue
            queue = sum(conn.lane.getLastStepHaltingNumber(l) for l in lanes)
            vehicles = sum(conn.lane.getLastStepVehicleNumber(l)
                           for l in lanes)
            waiting = sum(conn.lane.getWaitingTime(l) for l in lanes)
            feats[i] = (queue, vehicles, waiting)

        mask = np.zeros((self._num_green, n_mov), dtype=bool)
        for slot_idx, phase_idx in enumerate(self._green_phase_indices):
            state_str = self._phase_states[phase_idx]
            for j in range(min(n_mov, len(state_str))):
                mask[slot_idx, j] = state_str[j] in ("G", "g")

        return {
            "movement_features": feats,
            "phase_movement_mask": mask,
            "num_green": int(self._num_green),
            "current_slot": int(self._current_green_slot),
            "time_in_phase": float(min(self._time_in_phase, 120) / 120.0),
        }

    def set_neighbor_features(self, arr) -> None:
        """Push the upstream/downstream summary (len == 6). Called by the
        multi-TLS wrapper before get_state(); no-op effect unless
        neighbor_aware is on."""
        a = np.asarray(arr, dtype=np.float32).ravel()
        if a.size != self._neighbor_block_size:
            raise ValueError(
                f"neighbor block must be {self._neighbor_block_size} floats, "
                f"got {a.size}"
            )
        self._neighbor_features = a

    def set_shared_arrived(self, n_arrived: int, n_tls: int) -> None:
        """Network-wide arrivals this interval + TLS count, consumed by the
        ``max_pressure_net`` reward."""
        self._shared_arrived = int(n_arrived)
        self._n_tls = max(1, int(n_tls))

    def set_reward_weights(self, local_w: float, net_w: float,
                           coord_w: float = 0.0) -> None:
        """Set the curriculum weights on the ``max_pressure_net`` reward.

        ``local_w`` scales the validated local max-pressure penalty;
        ``net_w`` scales the shared corridor-throughput bonus;
        ``coord_w`` scales the per-agent downstream-saturation penalty
        (action-attributable: punishes feeding an already-jammed downstream
        neighbour). The trainer calls this every decision to ramp net_w /
        coord_w from 0 (Phase 1: stable local learning) up to their final
        values (Phase 2: coordination). ``coord_w`` defaults to 0.0 so
        omitting it leaves the reward byte-identical. No effect unless
        reward_mode == "max_pressure_net"."""
        self._reward_local_w = float(local_w)
        self._reward_net_w = float(net_w)
        self._reward_coord_w = float(coord_w)

    # ── dynamics ──────────────────────────────────────────────

    def _sim_tick(self) -> None:
        """Advance SUMO by one second and update metric accumulators.

        Every place that calls ``simulationStep`` should go through this so
        arrivals during yellow phases are counted and the wait-time series
        is sampled at 1Hz consistently across policies.
        """
        conn = self._conn()
        conn.simulationStep()
        self._step_time += 1
        try:
            n_arrived = int(conn.simulation.getArrivedNumber())
            self._arrived_total += n_arrived
            self._arrived_since_reward += n_arrived
            self._departed_total += int(conn.simulation.getDepartedNumber())
        except Exception:
            pass
        self._cumulative_wait += self._total_waiting_time()

    def passive_step(self):
        """Advance one sim-second without changing the TLS.

        Used by the ``native_actuated`` eval baseline: SUMO's program from
        the .net.xml stays in charge. Returns ``(state, done)``.
        """
        self._sim_tick()
        return self.get_state(), self._terminal()

    def step(self, action: int):
        """Advance the simulation by one decision cycle.

        ``action`` indexes ``green_phase_indices`` (0..num_green-1).
        """
        action = int(action) % self._num_green
        target_phase = self._green_phase_indices[action]
        current_phase = self._green_phase_indices[self._current_green_slot]

        conn = self._conn()
        switched = False

        if target_phase != current_phase and self._time_in_phase >= self.min_green:
            yellow_state = _yellow_between(
                self._phase_states[current_phase],
                self._phase_states[target_phase],
            )
            conn.trafficlight.setRedYellowGreenState(self.tls_id, yellow_state)
            for _ in range(self.yellow_time):
                self._sim_tick()
                if self._terminal():
                    break
            conn.trafficlight.setRedYellowGreenState(
                self.tls_id, self._phase_states[target_phase]
            )
            self._current_green_slot = action
            self._time_in_phase = 0
            switched = True

        # Hold the (possibly new) green for ``decision_interval`` seconds.
        # SUMO-native actuated controllers decide on phase boundaries, not
        # every tick; emulating that cadence is the main lever in Option A.
        for _ in range(self.decision_interval):
            self._sim_tick()
            self._time_in_phase += 1
            if self._terminal():
                break

        next_state = self.get_state()
        reward = self._reward(switched)
        done = self._terminal()
        info = {
            "switched": switched,
            "step_time": self._step_time,
            "green_slot": self._current_green_slot,
            "phase_index": self._green_phase_indices[self._current_green_slot],
        }
        return next_state, reward, done, info

    def _reward(self, switched: bool) -> float:
        total_wait = self._total_waiting_time()
        if self.reward_mode == "differential":
            r = (self._prev_total_wait - total_wait)
            if switched:
                r -= self.switch_penalty
        elif self.reward_mode == "anti_starve":
            # Differential + penalty for the worst-starved lane. The plain
            # differential reward lets the agent camp on the busiest slot
            # because no-switch -> no wait increase on the flowing lanes;
            # the max-lane-wait term punishes leaving any single lane
            # starved indefinitely.
            conn = self._conn()
            max_lane_wait = max(
                (conn.lane.getWaitingTime(l) for l in self._controlled_lanes),
                default=0.0,
            )
            r = (self._prev_total_wait - total_wait) - self.starve_penalty * max_lane_wait
            if switched:
                r -= self.switch_penalty
        elif self.reward_mode == "pressure_only":
            # V2/MAPPO reward: pure per-light pressure, no switch penalty,
            # no shaping. Credit assignment for cross-light coordination
            # lives in the centralized critic, not in this per-light
            # reward. Differs from "max_pressure" above in dropping the
            # switch_penalty subtraction -- PPO's entropy regularisation
            # already discourages thrash without needing a hand-tuned cost.
            conn = self._conn()
            q_in = sum(conn.lane.getLastStepHaltingNumber(l)
                       for l in self._incoming_lanes)
            q_out = sum(conn.lane.getLastStepHaltingNumber(l)
                        for l in self._outgoing_lanes)
            n = max(1, len(self._incoming_lanes))
            r = -abs(q_in - q_out) / n
        elif self.reward_mode == "max_pressure":
            # Max-pressure control (PressLight/MPLight): pressure =
            # (queue on incoming lanes) - (queue on outgoing lanes).
            # Minimising |pressure| provably stabilises queues and
            # maximises throughput, and the downstream term stops the
            # agent dumping vehicles into an already-full link.
            conn = self._conn()
            q_in = sum(conn.lane.getLastStepHaltingNumber(l)
                       for l in self._incoming_lanes)
            q_out = sum(conn.lane.getLastStepHaltingNumber(l)
                        for l in self._outgoing_lanes)
            n = max(1, len(self._incoming_lanes))
            r = -abs(q_in - q_out) / n
            if switched:
                r -= self.switch_penalty
        elif self.reward_mode == "max_pressure_net":
            # Validated single-TLS max_pressure shaping (the local term is
            # kept identical to the ``max_pressure`` branch above so its
            # result stays reproducible) plus a *shared* corridor-throughput
            # bonus. The two terms are blended with curriculum weights set by
            # the trainer via set_reward_weights():
            #   Phase 1  local_w=1, net_w=0    -> pure validated local policy
            #   Phase 2  local_w=1, net_w↑     -> coordination fine-tune
            # With the default weights (1.0, reward_gamma) this is byte-for-
            # byte the previous formula, so old runs stay reproducible.
            conn = self._conn()
            q_in = sum(conn.lane.getLastStepHaltingNumber(l)
                       for l in self._incoming_lanes)
            q_out = sum(conn.lane.getLastStepHaltingNumber(l)
                        for l in self._outgoing_lanes)
            n = max(1, len(self._incoming_lanes))
            local = -abs(q_in - q_out) / n
            net = self._shared_arrived / self._n_tls
            # Per-agent downstream-saturation penalty. _neighbor_features[3]
            # is the downstream neighbour's normalised incoming queue (∈[0,1],
            # 0 for a lone single-TLS env). Scaling it by THIS light's own
            # outgoing queue makes it action-attributable: an agent holding a
            # green that dumps flow into an already-jammed downstream is
            # penalised; one that isn't, is not — directly targeting the
            # cascade gridlock seen on the worst seeds.
            ds_sat = float(self._neighbor_features[3]) if self.neighbor_aware else 0.0
            coord_pen = ds_sat * (q_out / n)
            r = (self._reward_local_w * local
                 + self._reward_net_w * net
                 - self._reward_coord_w * coord_pen)
            if switched:
                r -= self.switch_penalty
        elif self.reward_mode == "combined":
            # Directly encodes the two stated goals: reward throughput
            # (vehicles that completed their trip this decision interval)
            # and penalise growth in total waiting time. Wait term is
            # normalised by lane count to keep DQN targets O(1).
            n = max(1, len(self._controlled_lanes))
            arrived_term = float(self._arrived_since_reward)
            wait_term = (self._prev_total_wait - total_wait) / n
            r = self.reward_alpha * arrived_term + self.reward_beta * wait_term
            if switched:
                r -= self.switch_penalty
        else:  # "waiting"
            r = -total_wait
        self._prev_total_wait = total_wait
        self._arrived_since_reward = 0
        return float(r)

    def _terminal(self) -> bool:
        if self._step_time >= self.time_limit:
            return True
        try:
            return self._conn().simulation.getMinExpectedNumber() <= 0
        except Exception:
            return True

    # ── shape helpers for the agent ───────────────────────────

    @property
    def state_size(self) -> int:
        base = len(self._controlled_lanes) * 3 + self._num_green + 1
        return base + (self._neighbor_block_size if self.neighbor_aware else 0)

    @property
    def action_size(self) -> int:
        return self._num_green

    @property
    def green_phase_indices(self):
        return list(self._green_phase_indices)

    @property
    def controlled_lanes(self):
        return list(self._controlled_lanes)

    # ── metrics ───────────────────────────────────────────────

    @property
    def arrived_total(self) -> int:
        return int(self._arrived_total)

    @property
    def departed_total(self) -> int:
        return int(self._departed_total)

    @property
    def cumulative_wait(self) -> float:
        return float(self._cumulative_wait)

    @property
    def vehicles_in_network(self) -> int:
        try:
            return int(self._conn().vehicle.getIDCount())
        except Exception:
            return 0

    def metrics_summary(self) -> dict:
        """Snapshot of episode-level metrics. Call before stop_simulation()."""
        in_net = self.vehicles_in_network
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
