"""WebSocket runner with a trained DQN agent in the loop.

Starts SUMO via TraCI, runs a trained Double-DQN agent on one intersection,
and broadcasts vehicle / traffic-light state to every connected frontend.

If the agent checkpoint is missing or fails to load, the simulation still
runs (SUMO's built-in actuated controllers stay in charge) and the frontend
shows ``aiMode = "fallback"``.

Run from ``SUMO/v2``:
    python run_websocket_ai.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# ── SUMO setup ──────────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    try:
        import sumolib as _sl
        sys.path.append(
            os.path.join(os.path.dirname(os.path.dirname(_sl.__file__)), "tools")
        )
    except ImportError:
        sys.exit("Set SUMO_HOME or install sumolib.")

import traci  # noqa: E402
from sumolib import checkBinary  # noqa: E402

# AI imports — ``ai`` is on sys.path because the package has an __init__.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "ai"))
from dqn_agent import DQNAgent  # noqa: E402

from websocket_server import SimulationWebSocketServer  # noqa: E402


# ── Configuration ────────────────────────────────────────────────────
SUMO_CONFIG = "sim.sumocfg"
TLS_ID = "3153556582"
MODEL_PATH = "ai/checkpoints/best.pth"
MIN_GREEN = 10
YELLOW_TIME = 4
MAX_STEPS = 3600
USE_GUI = False
WS_HOST = "localhost"
WS_PORT = 8765
# =====================================================================


def _is_green(state_str: str) -> bool:
    s = state_str.lower()
    return ("g" in s) and ("y" not in s)


def _yellow_between(from_state: str, to_state: str) -> str:
    out = []
    for a, b in zip(from_state, to_state):
        if a in ("G", "g") and b in ("r", "s"):
            out.append("y")
        else:
            out.append(a)
    return "".join(out)


def load_agent(path: str):
    """Try to load a trained agent. Return (agent, meta) or (None, error_str)."""
    if not os.path.exists(path):
        return None, f"model not found at {path}"
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        agent = DQNAgent.load_for_inference(path)
        return agent, meta
    except Exception as exc:
        return None, f"failed to load: {exc!r}"


def main() -> None:
    print("Starting WebSocket server …")
    ws = SimulationWebSocketServer(host=WS_HOST, port=WS_PORT)
    ws.start()
    time.sleep(1)

    agent, meta_or_err = load_agent(MODEL_PATH)
    if agent is None:
        print(f"AI disabled ({meta_or_err}). SUMO's actuated controllers will run.")
        agent_meta = {}
    else:
        agent_meta = meta_or_err or {}
        print(f"Loaded agent  state_size={agent.state_size} "
              f"action_size={agent.action_size}")

    print("Starting SUMO …")
    sumo_binary = checkBinary("sumo-gui" if USE_GUI else "sumo")
    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CONFIG,
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--start",
        "--quit-on-end",
    ]

    try:
        traci.start(sumo_cmd)
    except Exception as exc:
        print(f"Failed to start TraCI: {exc}")
        ws.stop()
        return

    # ── Resolve the TLS we'll control ────────────────────────────────
    ai_status = "active"
    controlled_lanes: list = []
    green_phase_indices: list = []
    phase_states: list = []
    current_green_slot = 0
    time_in_phase = 0

    if TLS_ID not in traci.trafficlight.getIDList():
        print(f"TLS {TLS_ID} not found in network. AI control disabled.")
        agent = None
        ai_status = "fallback:tls_missing"

    if agent is not None:
        controlled_lanes = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(TLS_ID)
        ))
        logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
        phase_states = [p.state for p in logic.getPhases()]
        green_phase_indices = [i for i, s in enumerate(phase_states) if _is_green(s)]

        # If the checkpoint was trained on a different intersection layout, abort.
        expected_state_size = len(controlled_lanes) * 3 + len(green_phase_indices) + 1
        if expected_state_size != agent.state_size:
            print(
                f"State-size mismatch: env={expected_state_size} "
                f"vs checkpoint={agent.state_size}. Falling back to actuated."
            )
            agent = None
            ai_status = "fallback:state_size_mismatch"
        elif len(green_phase_indices) != agent.action_size:
            print(
                f"Action-size mismatch: env={len(green_phase_indices)} "
                f"vs checkpoint={agent.action_size}. Falling back to actuated."
            )
            agent = None
            ai_status = "fallback:action_size_mismatch"
        else:
            # Hand the intersection over to the agent. Use raw state strings
            # (setRedYellowGreenState) because once we apply a yellow override
            # SUMO switches to a one-phase override program where setPhase
            # indices stop matching the static logic.
            traci.trafficlight.setRedYellowGreenState(
                TLS_ID, phase_states[green_phase_indices[0]]
            )

    if agent is None and ai_status == "active":
        ai_status = "fallback:no_model"

    # ── Simulation loop ──────────────────────────────────────────────
    step = 0
    decisions = 0
    in_yellow = False
    yellow_steps_left = 0
    pending_target_slot = current_green_slot
    cumulative_wait = 0.0
    no_vehicle_counter = 0

    def total_waiting():
        return float(sum(traci.lane.getWaitingTime(l) for l in controlled_lanes))

    def build_state():
        feats = []
        for lane in controlled_lanes:
            feats.extend([
                traci.lane.getLastStepHaltingNumber(lane) / 20.0,
                traci.lane.getWaitingTime(lane) / 60.0,
                traci.lane.getLastStepMeanSpeed(lane) / 15.0,
            ])
        one_hot = [0.0] * len(green_phase_indices)
        one_hot[current_green_slot] = 1.0
        feats.extend(one_hot)
        feats.append(min(time_in_phase, 120) / 120.0)
        return feats

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
            # ── Agent decision (only on green frames, respecting min-green) ──
            if agent is not None and not in_yellow:
                if time_in_phase >= MIN_GREEN:
                    state = build_state()
                    action = agent.act(state, epsilon=0.0)
                    target_phase_idx = green_phase_indices[action]
                    current_phase_idx = green_phase_indices[current_green_slot]
                    if target_phase_idx != current_phase_idx:
                        yellow_state = _yellow_between(
                            phase_states[current_phase_idx],
                            phase_states[target_phase_idx],
                        )
                        traci.trafficlight.setRedYellowGreenState(TLS_ID, yellow_state)
                        in_yellow = True
                        yellow_steps_left = YELLOW_TIME
                        pending_target_slot = action
                        decisions += 1

            traci.simulationStep()
            step += 1
            cumulative_wait += total_waiting() if controlled_lanes else 0.0

            if in_yellow:
                yellow_steps_left -= 1
                if yellow_steps_left <= 0:
                    target_phase_idx = green_phase_indices[pending_target_slot]
                    traci.trafficlight.setRedYellowGreenState(
                        TLS_ID, phase_states[target_phase_idx]
                    )
                    current_green_slot = pending_target_slot
                    time_in_phase = 0
                    in_yellow = False
            else:
                time_in_phase += 1

            # ── Gather frame for the frontend ────────────────────────
            vehicles = []
            for vid in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vid)
                vehicles.append({
                    "id": vid,
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "speed": round(traci.vehicle.getSpeed(vid), 2),
                    "angle": round(traci.vehicle.getAngle(vid), 2),
                    "edge": traci.vehicle.getRoadID(vid),
                })

            traffic_lights = []
            for tl_id in traci.trafficlight.getIDList():
                traffic_lights.append({
                    "id": tl_id,
                    "phase": traci.trafficlight.getPhase(tl_id),
                    "state": traci.trafficlight.getRedYellowGreenState(tl_id),
                })

            ws.broadcast({
                "type": "step",
                "step": step,
                "vehicles": vehicles,
                "trafficLights": traffic_lights,
                "ai": {
                    "status": ai_status,
                    "tlsId": TLS_ID if agent is not None else None,
                    "decisions": decisions,
                    "currentGreenSlot": current_green_slot,
                    "inYellow": in_yellow,
                    "minGreen": MIN_GREEN,
                    "yellowTime": YELLOW_TIME,
                    "modelPath": MODEL_PATH if agent is not None else None,
                },
            })

            # ── Honour incoming commands (manual overrides) ──────────
            for cmd in ws.get_pending_commands():
                action_name = cmd.get("action")
                try:
                    if action_name == "setPhase":
                        traci.trafficlight.setPhase(cmd["tlsId"], int(cmd["phase"]))
                        if cmd["tlsId"] == TLS_ID and agent is not None:
                            # User took manual control — pause the agent.
                            ai_status = "manual_override"
                    elif action_name == "resumeAI":
                        if agent is not None:
                            traci.trafficlight.setRedYellowGreenState(
                                TLS_ID,
                                phase_states[green_phase_indices[current_green_slot]],
                            )
                            time_in_phase = 0
                            ai_status = "active"
                    elif action_name == "setSpeed":
                        traci.vehicle.setSpeed(cmd["vehId"], float(cmd["speed"]))
                except Exception as exc:
                    print(f"   command error: {exc}")
                    ws.broadcast({"type": "error", "message": str(exc)})

            if step % 100 == 0:
                active = traci.vehicle.getIDCount()
                avg_wait = cumulative_wait / max(1, step)
                print(f"step {step:>4d}  vehicles={active:>3d}  "
                      f"avg_wait={avg_wait:6.2f}  decisions={decisions:>3d}  "
                      f"ai={ai_status}")

            active = traci.vehicle.getIDCount()
            if active == 0:
                no_vehicle_counter += 1
                if no_vehicle_counter > 100:
                    print("No vehicles for 100 steps — ending.")
                    break
            else:
                no_vehicle_counter = 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        traceback.print_exc()
    finally:
        try:
            traci.close()
        except Exception:
            pass
        ws.stop()
        print(f"Done. steps={step} decisions={decisions} "
              f"avg_wait={cumulative_wait / max(1, step):.2f}")


if __name__ == "__main__":
    main()
