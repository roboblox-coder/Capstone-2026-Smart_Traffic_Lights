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
        return np.asarray(feats, dtype=np.float32)

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
        return len(self._controlled_lanes) * 3 + self._num_green + 1

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
