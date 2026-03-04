"""
SUMO environment for traffic light control using TraCI.
Assumes SUMO is installed and SUMO_HOME is set.
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
            tls_id: the ID of the traffic light to control (e.g., "3153556582").
            time_limit: simulation time in seconds (stop after this).
            use_gui: if True, launch sumo-gui, else sumo (no GUI).
        """
        self.sumo_cfg = sumo_cfg_file
        self.tls_id = tls_id
        self.time_limit = time_limit
        self.use_gui = use_gui
        self.step_time = 0

        # Will be filled after start
        self.controlled_lanes = None
        self.num_phases = None

    def start_simulation(self):
        """Launch SUMO with TraCI."""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg, "--start", "--quit-on-end"]
        traci.start(sumo_cmd)

        # Get controlled lanes for this traffic light
        self.controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        # Remove duplicates (multiple lanes per approach may be listed)
        self.controlled_lanes = list(dict.fromkeys(self.controlled_lanes))

        # Get number of phases (to define action space)
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        self.num_phases = len(logic.getPhases())
        self.step_time = 0

    def stop_simulation(self):
        """Close TraCI connection."""
        traci.close()

    def get_state(self):
        """
        Extract state from the intersection.
        Returns a numpy array of features.
        Features: queue length and waiting time for each controlled lane.
        """
        state = []
        for lane in self.controlled_lanes:
            # Number of stopped vehicles on this lane
            queue = traci.lane.getLastStepHaltingNumber(lane)
            # Total waiting time (seconds) on this lane
            waiting = traci.lane.getWaitingTime(lane)
            state.extend([queue, waiting])

        # Pad if something went wrong (shouldn't happen)
        return np.array(state, dtype=np.float32)

    def get_reward(self):
        """
        Compute reward from current simulation state.
        Here we use negative total waiting time (we want to minimise waiting).
        """
        total_wait = 0.0
        for lane in self.controlled_lanes:
            total_wait += traci.lane.getWaitingTime(lane)
        return -total_wait

    def step(self, action):
        """
        Apply action (set traffic light phase) and advance simulation one second.
        Returns: (next_state, reward, done, info)
        """
        # Set the traffic light phase
        traci.trafficlight.setPhase(self.tls_id, action)

        # Advance simulation by one second
        traci.simulationStep()
        self.step_time += 1

        # Get new state and reward
        next_state = self.get_state()
        reward = self.get_reward()

        # Check if simulation ended
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
        return len(self.controlled_lanes) * 2

    @property
    def action_size(self):
        """Number of possible actions (phases)."""
        return self.num_phases
