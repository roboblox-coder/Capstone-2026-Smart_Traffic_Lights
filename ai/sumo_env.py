"""
SUMO environment for traffic light control using TraCI.
No pre-parsing; uses a short simulation to initialize sizes.
"""

import os
import sys
import traci
import numpy as np

# Add SUMO tools to path if not already
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set SUMO_HOME environment variable.")

class SumoTrafficEnv:
    """
    A reinforcement learning environment for a single traffic light in SUMO.
    """

    def __init__(self, sumo_cfg_file, tls_id, time_limit=3600, use_gui=False):
        """
        Args:
            sumo_cfg_file: path to the .sumocfg configuration file.
            tls_id: the ID of the traffic light to control.
            time_limit: simulation time in seconds (stop after this).
            use_gui: if True, launch sumo-gui, else sumo (no GUI).
        """
        self.sumo_cfg = sumo_cfg_file
        self.tls_id = tls_id
        self.time_limit = time_limit
        self.use_gui = use_gui
        self.step_time = 0

        # Initialize by running a short simulation to get lane and phase info
        self._init_from_traci()

    def _init_from_traci(self):
        """Start a temporary simulation to fetch lane list and phase count."""
        sumo_binary = "sumo"  # use headless for initialization (faster)
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg, "--start", "--quit-on-end", "--no-step-log"]
        traci.start(sumo_cmd)

        # Get controlled lanes for this traffic light
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        self._controlled_lanes = list(dict.fromkeys(lanes))  # remove duplicates

        # Get number of phases
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        self._num_phases = len(logic.getPhases())

        traci.close()

    def start_simulation(self):
        """Launch SUMO with TraCI (full simulation)."""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg, "--start", "--quit-on-end"]
        traci.start(sumo_cmd)
        self.step_time = 0

    def stop_simulation(self):
        """Close TraCI connection safely."""
        try:
            traci.close()
        except:
            pass

    def get_state(self):
        """
        Extract state from the intersection.
        Returns a numpy array of features.
        Features: queue length and waiting time for each controlled lane.
        """
        state = []
        for lane in self._controlled_lanes:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            waiting = traci.lane.getWaitingTime(lane)
            state.extend([queue, waiting])
        return np.array(state, dtype=np.float32)

    def get_reward(self):
        """
        Compute reward from current simulation state.
        Here we use negative total waiting time (we want to minimise waiting).
        """
        total_wait = 0.0
        for lane in self._controlled_lanes:
            total_wait += traci.lane.getWaitingTime(lane)
        return -total_wait

    def step(self, action):
        """
        Apply action (set traffic light phase) and advance simulation one second.
        Returns: (next_state, reward, done, info)
        """
        traci.trafficlight.setPhase(self.tls_id, action)
        traci.simulationStep()
        self.step_time += 1

        next_state = self.get_state()
        reward = self.get_reward()
        done = self.step_time >= self.time_limit
        return next_state, reward, done, {}

    def reset(self):
        """Restart simulation."""
        self.stop_simulation()
        self.start_simulation()
        return self.get_state()

    @property
    def state_size(self):
        """Number of features in the state vector."""
        return len(self._controlled_lanes) * 2

    @property
    def action_size(self):
        """Number of possible actions (phases)."""
        return self._num_phases
