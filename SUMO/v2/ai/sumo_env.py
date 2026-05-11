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
        min_green: int = 10,
        yellow_time: int = 4,
        reward_mode: str = "waiting",
        switch_penalty: float = 0.1,
        extra_sumo_args: Optional[list] = None,
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

        self._label: Optional[str] = None
        self._step_time: int = 0
        self._time_in_phase: int = 0
        self._current_green_slot: int = 0
        self._prev_total_wait: float = 0.0

        self._controlled_lanes: list = []
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
            "--start",
            "--quit-on-end",
        ] + self.extra_sumo_args
        traci.start(cmd, label=self._label)
        self._step_time = 0
        self._time_in_phase = 0
        self._current_green_slot = 0
        self._prev_total_wait = 0.0

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
                conn.simulationStep()
                self._step_time += 1
                if self._terminal():
                    break
            conn.trafficlight.setRedYellowGreenState(
                self.tls_id, self._phase_states[target_phase]
            )
            self._current_green_slot = action
            self._time_in_phase = 0
            switched = True

        # Hold the (possibly new) green for one second.
        conn.simulationStep()
        self._step_time += 1
        self._time_in_phase += 1

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
        else:
            r = -total_wait
        self._prev_total_wait = total_wait
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
