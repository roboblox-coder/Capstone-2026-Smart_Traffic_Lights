import os
import sys
import traci
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Set SUMO_HOME environment variable.")

class SumoTrafficEnv:
    def __init__(self, sumo_cfg_file, tls_id, time_limit=3600, use_gui=False):
        self.sumo_cfg = sumo_cfg_file
        self.tls_id = tls_id
        self.time_limit = time_limit
        self.use_gui = use_gui
        self.step_time = 0
        self._init_from_traci()

    def _init_from_traci(self):
        sumo_cmd = ["sumo", "-c", self.sumo_cfg, "--start", "--quit-on-end", "--no-step-log"]
        traci.start(sumo_cmd)
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        self._controlled_lanes = list(dict.fromkeys(lanes))
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        self._num_phases = len(logic.getPhases())
        traci.close()

    def start_simulation(self):
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([sumo_binary, "-c", self.sumo_cfg, "--start", "--quit-on-end"])
        self.step_time = 0

    def stop_simulation(self):
        try:
            traci.close()
        except:
            pass

    def get_state(self):
        state = []
        for lane in self._controlled_lanes:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            waiting = traci.lane.getWaitingTime(lane)
            state.extend([queue, waiting])
        return np.array(state, dtype=np.float32)

    def get_reward(self):
        total_wait = 0.0
        for lane in self._controlled_lanes:
            total_wait += traci.lane.getWaitingTime(lane)
        return -total_wait

    def step(self, action):
        traci.trafficlight.setPhase(self.tls_id, action)
        traci.simulationStep()
        self.step_time += 1
        return self.get_state(), self.get_reward(), self.step_time >= self.time_limit, {}

    def reset(self):
        self.stop_simulation()
        self.start_simulation()
        return self.get_state()

    @property
    def state_size(self):
        return len(self._controlled_lanes) * 2

    @property
    def action_size(self):
        return self._num_phases
