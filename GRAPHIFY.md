# Running graphify on this repo

[`graphify`](https://github.com/safishamsi/graphify) maps a codebase into
a queryable knowledge graph (interactive HTML + Markdown report + JSON).

Run it locally — the repo is already cloned, `graphify` is a CLI you
install once on your machine.

## One-time setup

```bash
pipx install graphifyy        # or: uv tool install graphifyy
graphify install              # registers the /graphify skill in your IDE

# Pick one backend; export the matching key:
export ANTHROPIC_API_KEY=...   # backend=claude (default if set)
# or
export GEMINI_API_KEY=...      # backend=gemini
# or
export OPENAI_API_KEY=...      # backend=openai
```

## Run it on this repo

```bash
cd /path/to/AI-Traffic
graphify extract . --out graphify-out
```

Outputs land in `graphify-out/`:
- `graph.html` — open in a browser; clickable nodes + search
- `GRAPH_REPORT.md` — narrative summary + suggested questions
- `graph.json` — full graph, queryable via `graphify query "..."`

`graphify-out/` is ignored by git (no commit needed unless you want
the rendered graph shipped in a release branch).

## What to look at first

The architecture work this branch builds is concentrated in two
places — point graphify at those when you want a focused subgraph:

```bash
graphify extract SUMO/v2/ai/    --out graphify-out-ai     # V1 + V2 RL code
graphify extract SUMO/v2/ai/v2/ --out graphify-out-v2     # V2 stack only
```

The V2 stack is six modules (`frap_encoder`, `colight_gat`,
`shared_policy`, `centralized_critic`, `mappo_trainer`,
`inference_adapter`) plus the inference / DR / live-runner helpers.
The graph should pick up the FRAP → GAT → SharedActor → CentralCritic
flow and the V1 `MultiTlsEnv` → V2 `get_state_frap_batch` → trainer
edges.

## Worth excluding (large or generated)

`graphify` respects `.gitignore`, so these stay out by default:

- `SUMO/v2/ai/runs/`      — training-run artifacts + `.pth` checkpoints
- `SUMO/v2/ai/checkpoints/`
- `SUMO/v2/ai/logs/`
- `SUMO/v2/*.rou.xml`     — multi-MB generated route files
- `graphify-out*/`        — graphify's own outputs

If any of those somehow show up as graph nodes, add them to
`.graphify-ignore` (same syntax as `.gitignore`) at the repo root.
