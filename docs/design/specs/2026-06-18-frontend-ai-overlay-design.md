# Design — AI overlay for the updated frontends

**Date:** 2026-06-18
**Goal:** Make the team's two newer visual frontends
(`frontend/index_CyberPunk.html`, `frontend/index_performance_optimized.html`)
display the live AI controller, matching the AI integration that
`frontend/index.html` already has. Showcase frontends, so the scope is
**read-only AI visuals + a live model switcher** — no manual phase
override / resume-AI controls.

## Background

`run_websocket_ai.py` drives the V3 FRAP-DQN corridor controller across
all 12 lights, auto-uses `sim_calibrated.sumocfg`, and broadcasts each
step:

```
{ type:"step", step, vehicles[], trafficLights:[{id,phase,state}],
  ai:{ <tid>:{status,decisions,currentGreenSlot,inYellow,minGreen,yellowTime} },
  aiSummary:{ active,total,mode,availableModels,activeModel },
  decisionsLog:[ {step,light,slot,model,event} ] }
```

`index.html` consumes all of this; the two updated frontends connect to
the same WebSocket and render `vehicles` + `trafficLights` but ignore
`ai` / `aiSummary` / `decisionsLog` entirely.

## What gets added to each updated frontend

Ported from `index.html`, themed to match each file (CyberPunk neon /
perf minimal), inside a delimited `// === AI overlay (keep in sync with
index.html) ===` block so the three stay reconcilable:

1. **AI badge** in the header — `AI: <active>/<total> active`,
   color-coded via the shared `aiColors(status)` helper. Fed by
   `updateAiBadge(msg.aiSummary)`.
2. **Model switcher** `<select>` in the header — options from
   `aiSummary.availableModels` (labelled V1·DQN / V2·MAPPO / V3·FRAP-DQN);
   `change` → `ws.send({action:'selectModel', model})`. Reflects
   `activeModel` from the backend. `updateModelSelect(msg.aiSummary)`.
   Swapping to a *different* model reloads the sim from step 0 runner-side
   (`traci.load` + per-light state reset) so each controller is shown on a
   clean run rather than inheriting the previous model's traffic.
3. **AI decisions panel** in the sidebar/HUD — newest-first stream of the
   last ~25 `decisionsLog` rows. `renderDecisions(msg.decisionsLog,
   activeModel)`.
4. **Per-light AI status** — each frontend's `updateTLPanel(trafficLights)`
   becomes `updateTLPanel(trafficLights, msg.ai)`, adding a per-light
   `AI: <status> slot=N [Y] Δ=k` chip.

Each frontend's `handleStep(msg)` calls the four functions. No backend
changes for the overlay (the payload already carries everything).

## Model shown by the demo

The live runner builds its **own 3-feature** FRAP state
(`ai/v2/live_inference.py`) and drives V3 from `ai/v3/model_best.pth`
(3-feature); it already defaults to V3 when V3 loads. The runner's
checkpoint path is **unchanged** — `model_di2_best.pth` was trained on a
10-feature state, so it is not compatible with the live runner and stays
an eval-only artifact (it remains the best *eval* model).

**Bug fixed en route:** the V4 T0.2 change bumped `FRAPDQNAgent`'s default
`mov_feat_dim` 3 -> 5, which silently broke
`FRAPDQNAgent.load_for_inference` for the 3-feature `model_best.pth` (net
built at 5, state_dict at 3 -> load error -> the demo fell back to V2).
Fixed by inferring `mov_feat_dim` from the checkpoint's own weights;
re-verified the runner loads V1/V2/V3 and runs V3 as the active model.
A regression test guards it (`ai/v3/tests/test_frap_q.py`).

## Self-containment (why not a shared JS module)

Each frontend stays a single self-contained HTML opened directly /
served statically (no build step) — the codebase's established
convention and how the live site serves them. The overlay is therefore
inlined per file rather than extracted to a shared `.js` sidecar; the
delimited block + this spec keep the three copies reconcilable.

## Verification

1. JS syntax check on each edited frontend.
2. Launch `python run_websocket_ai.py`; open each updated frontend:
   AI badge populates, decisions panel streams, switching the model
   dropdown changes the active controller, per-light chips update, lights
   animate. (Live visual confirm is the user's exam on their machine.)

## Delivery (fork-first)

Branch `frontend-ai-integration` (off `v4-stage0`) → implement + verify →
merge into fork `main` (consolidating the V4 AI work + this integration)
→ push `origin/main` → **stop for user examination**. The focused PR into
the team (`upstream`) repo is deferred until after that review.
