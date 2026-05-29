"""WebSocket runner: the coordinated DQN drives EVERY corridor light.

Starts SUMO via TraCI, loads one trained Double-DQN per traffic light from
``ai/runs/coordinated/checkpoints/<tls_id>/best.pth``, and broadcasts vehicle
/ traffic-light state plus a per-intersection AI status block to every
connected frontend.

Each light is independent: a missing / mismatched checkpoint falls that one
light back to SUMO's actuated program without disturbing the others. Agents
observe the same neighbour-aware state they were trained on (own lanes +
phase + a 6-float upstream/downstream summary built from ai/adjacency.json).

Run from ``SUMO/v2``::

    python run_websocket_ai.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

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

sys.path.insert(0, str(Path(__file__).resolve().parent / "ai"))
from dqn_agent import DQNAgent  # noqa: E402

# V2 (FRAP/GAT/MAPPO) inference is optional: torch is the dependency
# gate. If the import fails, V2_INFERENCE_AVAILABLE is False and the
# runner stays on the V1 per-TLS path -- the existing demo is
# unaffected by V2 work being incomplete on a given checkout.
try:
    from v2.live_inference import V2InferenceLoop  # noqa: E402
    V2_INFERENCE_AVAILABLE = True
except Exception as _v2_exc:
    V2InferenceLoop = None  # type: ignore[assignment]
    V2_INFERENCE_AVAILABLE = False
    _V2_IMPORT_ERR = _v2_exc

from websocket_server import SimulationWebSocketServer  # noqa: E402


# ── Configuration ────────────────────────────────────────────────────
SUMO_CONFIG = "sim.sumocfg"
ADJACENCY_PATH = "ai/adjacency.json"
CKPT_DIR = "ai/runs/coordinated/checkpoints"
V2_CKPT_PATH = os.path.join(
    os.path.dirname(__file__), "ai", "runs", "v2_mappo", "checkpoints", "best_ep200.pth"
)# Regime MUST match training (multi_env defaults).
MIN_GREEN = 5
YELLOW_TIME = 5
DECISION_INTERVAL = 5
MAX_STEPS = 3600
USE_GUI = False
WS_HOST = "localhost"
WS_PORT = 8765
_NEI_NORM = 40.0  # queue normaliser; identical to multi_env._NEI_NORM
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


def _probe_tls(tls_id: str) -> dict:
    """Live structural probe (mirrors sumo_env._probe_network)."""
    lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
    links = traci.trafficlight.getControlledLinks(tls_id)
    inc, out = [], []
    for entry in links:
        for ct in entry:
            if ct:
                if ct[0]:
                    inc.append(ct[0])
                if ct[1]:
                    out.append(ct[1])
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phase_states = [p.state for p in logic.getPhases()]
    green = [i for i, s in enumerate(phase_states) if _is_green(s)]
    return {
        "lanes": lanes,
        "incoming": list(dict.fromkeys(inc)),
        "outgoing": list(dict.fromkeys(out)),
        "phase_states": phase_states,
        "green": green,
    }


def load_agent(path: str):
    if not os.path.exists(path):
        return None, f"model not found at {path}"
    try:
        DQNAgent  # noqa: B018
        agent = DQNAgent.load_for_inference(path)
        return agent, ""
    except Exception as exc:
        return None, f"failed to load: {exc!r}"


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sumo-cfg", default=None)
    args = p.parse_args()

    sumo_cfg = args.sumo_cfg
    if sumo_cfg is None:
        if os.path.exists(V2_CKPT_PATH) and os.path.exists("sim_calibrated.sumocfg"):
            sumo_cfg = "sim_calibrated.sumocfg"
            print(f"Auto-selected calibrated config: {sumo_cfg} for V2 inference.")
        else:
            sumo_cfg = SUMO_CONFIG

    print("Starting WebSocket server …")
    ws = SimulationWebSocketServer(host=WS_HOST, port=WS_PORT)
    ws.start()
    time.sleep(1)

    with open(ADJACENCY_PATH, encoding="utf-8") as fh:
        adjacency = json.load(fh)

    print("Starting SUMO …")
    sumo_binary = checkBinary("sumo-gui" if USE_GUI else "sumo")
    sumo_cmd = [
        sumo_binary, "-c", sumo_cfg,
        "--no-step-log", "true", "--time-to-teleport", "-1",
        "--start", "--quit-on-end",
    ]
    try:
        traci.start(sumo_cmd)
    except Exception as exc:
        print(f"Failed to start TraCI: {exc}")
        ws.stop()
        return

    tls_ids = [t for t in sorted(adjacency)
               if t in traci.trafficlight.getIDList()]

    # ── Per-TLS state ────────────────────────────────────────────────
    struct, agents, status = {}, {}, {}
    slot, tip, in_yellow = {}, {}, {}
    yellow_left, pending, since_decision, decisions = {}, {}, {}, {}
    prev_obs_state = {}

    for tid in tls_ids:
        st = _probe_tls(tid)
        struct[tid] = st
        slot[tid] = 0
        tip[tid] = 0
        in_yellow[tid] = False
        yellow_left[tid] = 0
        pending[tid] = 0
        since_decision[tid] = DECISION_INTERVAL
        decisions[tid] = 0
        prev_obs_state[tid] = traci.trafficlight.getRedYellowGreenState(tid)

    # ── V2 corridor-level policy (preferred when checkpoint exists) ──
    v2_loop = None
    if V2_INFERENCE_AVAILABLE:
        v2_loop = V2InferenceLoop.try_load(
            traci_mod=traci, ckpt_path=V2_CKPT_PATH,
            tls_ids=tls_ids, struct=struct, adjacency=adjacency,
        )
        if v2_loop is not None:
            print(f"V2 inference: loaded {V2_CKPT_PATH} "
                  f"(corridor-level FRAP/GAT/MAPPO).")
            # All 12 lights are driven by the single V2 policy; flag the
            # per-TLS load loop below to skip and mark status as v2.
            for tid in tls_ids:
                agents[tid] = None
                status[tid] = "active:v2"
                # The light needs to be in a known green at simulation
                # start; pick slot 0 like the V1 path does.
                traci.trafficlight.setRedYellowGreenState(
                    tid, struct[tid]["phase_states"][
                        struct[tid]["green"][0]]
                )

    for tid in tls_ids:
        if v2_loop is not None:
            # V2 has already populated agents[tid] = None / status = active:v2.
            continue
        path = os.path.join(CKPT_DIR, tid, "best.pth")
        agent, err = load_agent(path)
        if agent is None:
            agents[tid] = None
            status[tid] = "fallback:no_model"
            continue
        # Agents were trained neighbour-aware: +6 floats on the state.
        expected = len(st["lanes"]) * 3 + len(st["green"]) + 1 + 6
        if expected != agent.state_size:
            agents[tid] = None
            status[tid] = "fallback:state_size_mismatch"
        elif len(st["green"]) != agent.action_size:
            agents[tid] = None
            status[tid] = "fallback:action_size_mismatch"
        else:
            agents[tid] = agent
            status[tid] = "active"
            traci.trafficlight.setRedYellowGreenState(
                tid, st["phase_states"][st["green"][0]]
            )

    n_active = sum(1 for s in status.values()
                   if s in ("active", "active:v2"))
    print(f"Loaded {n_active}/{len(tls_ids)} agents "
          f"(rest fall back to actuated).")

    def _queue(lanes):
        return float(sum(traci.lane.getLastStepHaltingNumber(l)
                         for l in lanes))

    def neighbor_triplet(nbr):
        st = struct.get(nbr)
        if nbr is None or st is None:
            return [0.0, 0.0, 0.0]
        q_in = _queue(st["incoming"])
        q_out = _queue(st["outgoing"])
        return [
            min(1.0, q_in / _NEI_NORM),
            float(np.clip((q_in - q_out) / _NEI_NORM, -1.0, 1.0)),
            min(tip.get(nbr, 0), 120) / 120.0,
        ]

    def build_state(tid):
        st = struct[tid]
        feats = []
        for lane in st["lanes"]:
            feats.extend([
                traci.lane.getLastStepHaltingNumber(lane) / 20.0,
                traci.lane.getWaitingTime(lane) / 60.0,
                traci.lane.getLastStepMeanSpeed(lane) / 15.0,
            ])
        one_hot = [0.0] * len(st["green"])
        one_hot[slot[tid]] = 1.0
        feats.extend(one_hot)
        feats.append(min(tip[tid], 120) / 120.0)
        adj = adjacency.get(tid, {})
        feats.extend(neighbor_triplet(adj.get("upstream")))
        feats.extend(neighbor_triplet(adj.get("downstream")))
        return feats

    # ── Simulation loop ──────────────────────────────────────────────
    step = 0
    no_vehicle_counter = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
            # Decisions (pre-step, mirrors single-TLS cadence) per light.
            if v2_loop is not None:
                # Corridor-level decision: fire when at least one light
                # has hit the cadence and no light is mid-yellow. Keeps
                # all 12 phase machines in lock-step the way V2 was
                # trained.
                any_due = any(since_decision[t] >= DECISION_INTERVAL
                              for t in tls_ids)
                any_yellow = any(in_yellow[t] for t in tls_ids)
                if any_due and not any_yellow:
                    v2_actions = v2_loop.decide(slot, tip)
                    for tid in tls_ids:
                        action = int(v2_actions[tid])
                        st = struct[tid]
                        tgt = st["green"][action]
                        cur = st["green"][slot[tid]]
                        if tgt != cur and tip[tid] >= MIN_GREEN:
                            traci.trafficlight.setRedYellowGreenState(
                                tid, _yellow_between(
                                    st["phase_states"][cur],
                                    st["phase_states"][tgt])
                            )
                            in_yellow[tid] = True
                            yellow_left[tid] = YELLOW_TIME
                            pending[tid] = action
                        decisions[tid] += 1
                        since_decision[tid] = 0
            else:
                for tid in tls_ids:
                    if (agents[tid] is not None and status[tid] == "active"
                            and not in_yellow[tid]
                            and since_decision[tid] >= DECISION_INTERVAL):
                        st = struct[tid]
                        action = agents[tid].act(build_state(tid),
                                                 epsilon=0.0)
                        tgt = st["green"][action]
                        cur = st["green"][slot[tid]]
                        if tgt != cur and tip[tid] >= MIN_GREEN:
                            traci.trafficlight.setRedYellowGreenState(
                                tid, _yellow_between(
                                    st["phase_states"][cur],
                                    st["phase_states"][tgt])
                            )
                            in_yellow[tid] = True
                            yellow_left[tid] = YELLOW_TIME
                            pending[tid] = action
                        decisions[tid] += 1
                        since_decision[tid] = 0

            traci.simulationStep()
            step += 1

            # Per-light phase machine + time-in-phase bookkeeping.
            # "Driven" means the runner (V1 per-TLS DQN or V2 corridor
            # policy) owns the phase transitions for this light; fallback /
            # manual-override lights stay in the observed-transition path.
            for tid in tls_ids:
                driven = (status[tid] == "active:v2"
                          or (status[tid] == "active"
                              and agents[tid] is not None))
                if driven:
                    if in_yellow[tid]:
                        yellow_left[tid] -= 1
                        if yellow_left[tid] <= 0:
                            st = struct[tid]
                            traci.trafficlight.setRedYellowGreenState(
                                tid,
                                st["phase_states"][st["green"][pending[tid]]]
                            )
                            slot[tid] = pending[tid]
                            tip[tid] = 0
                            in_yellow[tid] = False
                    else:
                        tip[tid] += 1
                        since_decision[tid] += 1
                else:
                    # Fallback / manual: track time-in-phase from observed
                    # state transitions so neighbour signals stay meaningful.
                    cur = traci.trafficlight.getRedYellowGreenState(tid)
                    if cur != prev_obs_state[tid]:
                        tip[tid] = 0
                        prev_obs_state[tid] = cur
                    else:
                        tip[tid] += 1

            # ── Frontend frame ───────────────────────────────────────
            vehicles = []
            for vid in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vid)
                vehicles.append({
                    "id": vid, "x": round(x, 2), "y": round(y, 2),
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

            ai_block = {
                tid: {
                    "status": status[tid],
                    "decisions": decisions[tid],
                    "currentGreenSlot": slot[tid],
                    "inYellow": in_yellow[tid],
                    "minGreen": MIN_GREEN,
                    "yellowTime": YELLOW_TIME,
                    "modelPath": (V2_CKPT_PATH
                                  if status[tid] == "active:v2"
                                  else (os.path.join(CKPT_DIR, tid,
                                                     "best.pth")
                                        if agents[tid] is not None
                                        else None)),
                }
                for tid in tls_ids
            }

            ws.broadcast({
                "type": "step",
                "step": step,
                "vehicles": vehicles,
                "trafficLights": traffic_lights,
                "ai": ai_block,
                "aiSummary": {
                    "active": sum(1 for s in status.values()
                                  if s in ("active", "active:v2")),
                    "total": len(tls_ids),
                    "mode": "v2" if v2_loop is not None else "v1",
                },
            })

            # ── Commands (manual overrides), keyed by tlsId ───────────
            for cmd in ws.get_pending_commands():
                action_name = cmd.get("action")
                try:
                    if action_name == "setPhase":
                        cid = cmd["tlsId"]
                        traci.trafficlight.setPhase(cid, int(cmd["phase"]))
                        if cid in status and agents.get(cid) is not None:
                            status[cid] = "manual_override"
                    elif action_name == "resumeAI":
                        cid = cmd.get("tlsId")
                        v2_drives_this = (v2_loop is not None
                                          and cid in struct)
                        v1_drives_this = (cid in struct
                                          and agents.get(cid) is not None)
                        if v2_drives_this or v1_drives_this:
                            st = struct[cid]
                            traci.trafficlight.setRedYellowGreenState(
                                cid,
                                st["phase_states"][st["green"][slot[cid]]]
                            )
                            tip[cid] = 0
                            in_yellow[cid] = False
                            status[cid] = ("active:v2" if v2_drives_this
                                           else "active")
                    elif action_name == "setSpeed":
                        traci.vehicle.setSpeed(
                            cmd["vehId"], float(cmd["speed"]))
                except Exception as exc:
                    print(f"   command error: {exc}")
                    ws.broadcast({"type": "error", "message": str(exc)})

            if step % 100 == 0:
                active = traci.vehicle.getIDCount()
                tot_dec = sum(decisions.values())
                print(f"step {step:>4d}  vehicles={active:>3d}  "
                      f"decisions={tot_dec:>4d}  "
                      f"ai_active={n_active}/{len(tls_ids)}")

            if traci.vehicle.getIDCount() == 0:
                no_vehicle_counter += 1
                if no_vehicle_counter > 100:
                    print("No vehicles for 100 steps — ending.")
                    break
            else:
                no_vehicle_counter = 0

            # Pace the simulation if clients are connected to prevent GIL starvation and socket flooding
            if ws.client_count > 0:
                time.sleep(0.05)

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
        print(f"Done. steps={step} total_decisions={sum(decisions.values())}")


if __name__ == "__main__":
    main()
