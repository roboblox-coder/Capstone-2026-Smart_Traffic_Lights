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
    from v2.live_inference import (V2InferenceLoop,  # noqa: E402
                                   _build_frap_state_for_tls)
    V2_INFERENCE_AVAILABLE = True
except Exception as _v2_exc:
    V2InferenceLoop = None  # type: ignore[assignment]
    _build_frap_state_for_tls = None  # type: ignore[assignment]
    V2_INFERENCE_AVAILABLE = False
    _V2_IMPORT_ERR = _v2_exc

# V3 (FRAP-DQN) inference is also optional.
try:
    from v3.frap_dqn_agent import FRAPDQNAgent  # noqa: E402
    V3_INFERENCE_AVAILABLE = True
except Exception as _v3_exc:
    FRAPDQNAgent = None  # type: ignore[assignment]
    V3_INFERENCE_AVAILABLE = False
    _V3_IMPORT_ERR = _v3_exc

from websocket_server import SimulationWebSocketServer  # noqa: E402


# ── Configuration ────────────────────────────────────────────────────
SUMO_CONFIG = "sim.sumocfg"
ADJACENCY_PATH = "ai/adjacency.json"
CKPT_DIR = "ai/runs/coordinated/checkpoints"
V2_CKPT_PATH = os.path.join(
    os.path.dirname(__file__), "ai", "runs", "v2_mappo", "checkpoints", "best.pth"
)
V3_CKPT_PATH = next(
    (p for p in [
        os.path.join(os.path.dirname(__file__), "ai", "v3", "model_best.pth"),
        os.path.join(os.path.dirname(__file__), "ai", "runs",
                     "v3_frap_dqn_combined", "checkpoints", "best.pth"),
    ] if os.path.exists(p)),
    os.path.join(os.path.dirname(__file__), "ai", "v3", "model_best.pth"),
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
        # V2/V3 were trained on the calibrated corridor; prefer it when
        # either checkpoint is present.
        if ((os.path.exists(V2_CKPT_PATH) or os.path.exists(V3_CKPT_PATH))
                and os.path.exists("sim_calibrated.sumocfg")):
            sumo_cfg = "sim_calibrated.sumocfg"
            print(f"Auto-selected calibrated config: {sumo_cfg}")
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
    struct = {}
    slot, tip, in_yellow = {}, {}, {}
    yellow_left, pending, since_decision, decisions = {}, {}, {}, {}
    prev_obs_state, manual_override = {}, {}

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
        manual_override[tid] = False
        prev_obs_state[tid] = traci.trafficlight.getRedYellowGreenState(tid)

    # ── Load every available model (V1 / V2 / V3) so the frontend can
    #    hot-swap between them live. Each is wrapped (below, after the
    #    state helpers) as decide(slot, tip) -> {tid: target_green_slot}.
    # V1: one Double-DQN per light from ai/runs/coordinated/.
    v1_agents = {}
    v1_ok = 0
    for tid in tls_ids:
        agent, _err = load_agent(os.path.join(CKPT_DIR, tid, "best.pth"))
        if agent is not None:
            exp = len(struct[tid]["lanes"]) * 3 + len(struct[tid]["green"]) + 1 + 6
            if exp == agent.state_size and len(struct[tid]["green"]) == agent.action_size:
                v1_agents[tid] = agent
                v1_ok += 1
    v1_available = v1_ok == len(tls_ids)

    # V2: corridor-level FRAP/GAT/MAPPO.
    v2_loop = None
    if V2_INFERENCE_AVAILABLE:
        v2_loop = V2InferenceLoop.try_load(
            traci_mod=traci, ckpt_path=V2_CKPT_PATH,
            tls_ids=tls_ids, struct=struct, adjacency=adjacency,
        )

    # V3: parameter-shared FRAP Double-DQN.
    v3_agent = None
    if V3_INFERENCE_AVAILABLE and os.path.exists(V3_CKPT_PATH):
        try:
            v3_agent = FRAPDQNAgent.load_for_inference(V3_CKPT_PATH)
        except Exception as exc:
            print(f"V3 load failed: {exc}")
            v3_agent = None

    print(f"Models loaded -> V1:{v1_available} V2:{v2_loop is not None} "
          f"V3:{v3_agent is not None}")

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

    # ── Unified per-model decision functions ────────────────────────
    # Each returns {tid: target_green_slot}; the phase machine applies it.
    from collections import deque as _deque

    def decide_v1(slot, tip):
        out = {}
        for tid in tls_ids:
            a = v1_agents.get(tid)
            out[tid] = (a.act(build_state(tid), epsilon=0.0)
                        if a is not None else slot[tid])
        return out

    def decide_v2(slot, tip):
        acts = v2_loop.decide(slot, tip)
        return {tid: int(acts[tid]) for tid in tls_ids}

    def decide_v3(slot, tip):
        out = {}
        for tid in tls_ids:
            st = struct[tid]
            ng = len(st["green"])
            s = _build_frap_state_for_tls(
                traci, tid, st["phase_states"], st["green"], ng,
                slot[tid], tip[tid])
            state = {
                "movement_features": s["movement_features"],
                "phase_movement_mask": s["phase_movement_mask"],
                "phase_mask": np.ones(ng, dtype=bool),
            }
            out[tid] = int(v3_agent.act(state, epsilon=0.0))
        return out

    model_decide = {}
    if v1_available:
        model_decide["v1"] = decide_v1
    if v2_loop is not None:
        model_decide["v2"] = decide_v2
    if v3_agent is not None:
        model_decide["v3"] = decide_v3
    available_models = list(model_decide.keys())
    # Default active driver: prefer V3 > V2 > V1.
    active = {"name": ("v3" if "v3" in model_decide else
                       "v2" if "v2" in model_decide else
                       "v1" if "v1" in model_decide else None)}
    decisions_log = _deque(maxlen=60)
    print(f"Available models: {available_models} | active: {active['name']}")

    # Put every light into a known green so the phase machine has a
    # defined current slot for whichever model drives.
    if active["name"] is not None:
        for tid in tls_ids:
            traci.trafficlight.setRedYellowGreenState(
                tid, struct[tid]["phase_states"][struct[tid]["green"][0]])

    # ── Simulation loop ──────────────────────────────────────────────
    step = 0
    no_vehicle_counter = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
            # Decisions: corridor-level cadence; dispatch to the ACTIVE
            # model (hot-swappable from the frontend). Fire when at least
            # one light is due and none is mid-yellow.
            if active["name"] is not None:
                any_due = any(since_decision[t] >= DECISION_INTERVAL
                              for t in tls_ids)
                any_yellow = any(in_yellow[t] for t in tls_ids)
                if any_due and not any_yellow:
                    actions = model_decide[active["name"]](slot, tip)
                    for tid in tls_ids:
                        action = int(actions[tid])
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
                            # Concise decision record for the frontend log.
                            decisions_log.append({
                                "step": step, "light": tid,
                                "slot": action, "model": active["name"]})
                        decisions[tid] += 1
                        since_decision[tid] = 0

            traci.simulationStep()
            step += 1

            # Per-light phase machine + time-in-phase bookkeeping. When a
            # model is active it owns every light's phase transitions;
            # with no model, lights follow their observed (actuated) state.
            for tid in tls_ids:
                driven = active["name"] is not None and not manual_override.get(tid, False)
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

            cur_model = active["name"]
            ai_block = {
                tid: {
                    "status": ("manual_override" if manual_override[tid]
                               else (f"active:{cur_model}" if cur_model
                                     else "idle")),
                    "decisions": decisions[tid],
                    "currentGreenSlot": slot[tid],
                    "inYellow": in_yellow[tid],
                    "minGreen": MIN_GREEN,
                    "yellowTime": YELLOW_TIME,
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
                    "active": (0 if cur_model is None
                               else sum(1 for tid in tls_ids
                                        if not manual_override[tid])),
                    "total": len(tls_ids),
                    "mode": cur_model or "none",
                    "availableModels": available_models,
                    "activeModel": cur_model,
                },
                "decisionsLog": list(decisions_log)[-25:],
            })

            # ── Commands (manual overrides), keyed by tlsId ───────────
            for cmd in ws.get_pending_commands():
                action_name = cmd.get("action")
                try:
                    if action_name == "selectModel":
                        m = cmd.get("model")
                        if m in model_decide:
                            active["name"] = m
                            decisions_log.append({
                                "step": step, "light": "—",
                                "slot": -1, "model": m,
                                "event": f"switched to {m.upper()}"})
                            print(f"[model] switched active -> {m}")
                    elif action_name == "setPhase":
                        cid = cmd["tlsId"]
                        traci.trafficlight.setPhase(cid, int(cmd["phase"]))
                        if cid in struct:
                            manual_override[cid] = True
                    elif action_name == "resumeAI":
                        cid = cmd.get("tlsId")
                        if cid in struct:
                            st = struct[cid]
                            traci.trafficlight.setRedYellowGreenState(
                                cid,
                                st["phase_states"][st["green"][slot[cid]]]
                            )
                            tip[cid] = 0
                            in_yellow[cid] = False
                            manual_override[cid] = False
                    elif action_name == "setSpeed":
                        traci.vehicle.setSpeed(
                            cmd["vehId"], float(cmd["speed"]))
                except Exception as exc:
                    print(f"   command error: {exc}")
                    ws.broadcast({"type": "error", "message": str(exc)})

            if step % 100 == 0:
                nveh = traci.vehicle.getIDCount()
                tot_dec = sum(decisions.values())
                print(f"step {step:>4d}  vehicles={nveh:>3d}  "
                      f"decisions={tot_dec:>4d}  model={active['name']}")

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
