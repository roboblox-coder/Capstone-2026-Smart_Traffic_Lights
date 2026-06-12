# ATLAS V2 Retrain — Design Spec

**Date:** 2026-05-29
**Branch:** `atlas-v2`
**Owner:** Duke Schnepf
**Status:** Approved, awaiting implementation plan

---

## 1. Problem

The completed V2 MAPPO run (committed as `ai/runs/v2_mappo/checkpoints/best.pth`,
ep 180 snapshot, plateau-stopped at ep 300) does not work as supposed to:

- **The CoLight learned-attention — V2's headline architectural advantage —
  never trained.** `gat_freeze_until_step = 50000`
  ([`mappo_trainer.py:80`](../../../SUMO/v2/ai/v2/mappo_trainer.py)) and the run
  reached only 1,000 gradient steps. `gat_lr: 0.0` on every log line. The
  threshold is ~10× beyond even the planned 1500-episode budget.
- **Training was unstable.** Training-time eval bounced from wpv 7,052
  (ep 180) to 120,333 (ep 210) — a catastrophic policy collapse — and never
  recovered. Critic learned cleanly (val_loss 275 → 30); the policy did not.
- **Quantitative result.** 3-seed eval on `sim_calibrated.sumocfg`:
  V2 throughput 1639 ± 69 vs native 2046 ± 57 (**−20%**, 0/3 seeds),
  V2 wait 7690 ± 2190 vs native 5938 ± 841 (**+29% worse**, 0/3 seeds).
- **What works.** Live demo wiring is correct: `run_websocket_ai.py`
  auto-selects the calibrated config, loads `best.pth`, drives all 12 lights,
  and broadcasts `aiSummary.mode = "v2"`. Inference path is healthy.

## 2. Goals

1. Make the V2 architecture actually train (GAT learns attention).
2. Make training stable enough that the best checkpoint reflects the
   policy's real capability, not a one-shot outlier.
3. Produce a new `best.pth` that clears the **honest gate vs native**:
   ≥+6% throughput AND ≥−12% wait/veh on a 5-seed eval.
4. Promote the new model into the live demo only on clearing the gate.

## 3. Non-goals

- Architectural changes (FRAP, GAT, MAPPO algorithm stay).
- Domain-randomized training (Phase 1.3, separate follow-up).
- Touching V1 (`runs/coordinated/`) or its fallback path.
- Repo-history reclamation of the deleted duplicate checkpoint blob.

## 4. Design

### 4.1 Strategy

Single comprehensive retrain. Fix the three issues (GAT schedule, actor LR,
entropy floor) plus one ancillary code bug in one run. Risk of mixing
variables is mitigated by per-update observability (§4.5) so a failure
mode is readable from the log.

### 4.2 Training budget

~24 hours CPU wall-clock, **1200 episodes** (≈70 s/episode on prior run).

| Episode | Grad step | What happens |
|---:|---:|---|
| 0 | 0 | GAT frozen-uniform, actor LR 3e-4, entropy 0.01 |
| ~450 | 1500 | GAT unfreeze begins (linear ramp) |
| ~600 | 2000 | GAT fully active |
| 1200 | ~4000 | End: LR 5e-5 (cosine), entropy 0.005 |

GAT gets ~2000 gradient steps of trained attention — 3× V1's effective
budget for the equivalent feature. Freeze covers ~37% of the run (standard
CoLight schedule), giving FRAP + critic time to stabilize first.

### 4.3 Concrete changes

| File | Change | Rationale |
|---|---|---|
| `SUMO/v2/ai/v2/mappo_trainer.py` `MAPPOConfig` | `gat_freeze_until_step: 50_000 → 1500` and `gat_ramp_end_step: 60_000 → 2000` | Land freeze inside the achievable gradient-step budget |
| `SUMO/v2/ai/v2/mappo_trainer.py` `MAPPOConfig` | `entropy_coef_final: 0.001 → 0.005` | Stop exploration from collapsing too aggressively |
| `SUMO/v2/ai/v2/mappo_trainer.py` actor optimizer + update loop | Add cosine actor-LR schedule `3e-4 → 5e-5` over `total_episodes` (CosineAnnealingLR or hand-rolled) | Tame the eval-instability collapse |
| `SUMO/v2/ai/v2/mappo_trainer.py` `_log_jsonl` per-update payload | Add `gat_attention_entropy` (mean over heads, from `self.gat.attention_entropy()`) and `actor_lr` | Diagnostic for "did attention actually learn" |
| `SUMO/v2/ai/eval_network.py` end-of-run verdict block | Add a V2-vs-native block alongside the existing V1-vs-native one ([line 244](../../../SUMO/v2/ai/eval_network.py)) | Bug: current printed verdict ignores V2 entirely; misleading |

No new files. No env / FRAP / GAT / actor / critic changes. CLI surface
of the trainer unchanged (new constants are config defaults; existing
`--plateau-episodes`, `--eval-seeds` etc. all still work).

### 4.4 Run & artifact layout

- **Output dir:** `SUMO/v2/ai/runs/v2_mappo_retrain/` (new dir,
  preserves the current run's artifacts for postmortem).
- **Live demo during retrain:** continues pointing at the old
  `runs/v2_mappo/best.pth` — demo stays usable, no swap until gate clears.
- **Promotion:** if the new `best.pth` clears the gate (§4.6), update
  `V2_CKPT_PATH` in
  [`SUMO/v2/run_websocket_ai.py:66`](../../../SUMO/v2/run_websocket_ai.py)
  to the retrain dir (or copy the new `best.pth` over the old path) and
  commit. Else: keep the old V2 as the demo model, archive the retrain
  artifacts for a postmortem.
- **Resume safety:** `--resume` still supported; existing GAT-schedule
  snap-on-resume ([`mappo_trainer.py:544`](../../../SUMO/v2/ai/v2/mappo_trainer.py))
  works with the new thresholds without modification.

### 4.5 Observability additions

Per-update JSONL entry already logs: `pol_loss`, `val_loss`, `entropy`,
`approx_kl`, `gat_lr`, `entropy_coef`. Add:

- `gat_attention_entropy` — mean over heads of `gat.attention_entropy()`.
  Frozen-uniform produces ~`log(n_neighbors)`; learned attention should
  drift noticeably below that after unfreeze. If it stays at the
  frozen-uniform value post-unfreeze, attention isn't learning.
- `actor_lr` — explicit log of the cosine-decayed actor learning rate
  (currently only the GAT LR is logged via `param_groups[1]`).

These let a single post-run log read distinguish:
- "Attention learned + policy collapsed anyway" → LR/entropy still wrong
- "Attention never moved" → GAT schedule or freeze flag bug
- "Both fine but no improvement" → architectural ceiling

### 4.6 Acceptance criteria (gate to promote)

**Hard floor (must clear, else don't promote):**
- 5-seed eval on `sim_calibrated.sumocfg`, 1200 s each.
- V2 vs SUMO native actuated:
  - Throughput improvement ≥ **+6%**, AND
  - Wait/veh improvement ≥ **−12%**.
- (This is the RUN_AI.md "honest gate." Strictly weaker than the PLAN_V2
  §5 Phase-1 "beat V1 5/5" gate; on the calibrated env V1 doesn't beat
  round-robin, so beating V1 is a degenerate test. The honest bar is
  vs native.)

**Diagnostic checks (read from log, not blocking):**
- `gat_attention_entropy` post-unfreeze differs from frozen-uniform
  baseline by ≥ ~0.1 nats on at least one head.
- No single training-eval `wait_per_vehicle_mean` exceeds 2× the rolling
  3-eval median (no ep-210-style collapse).

If hard floor fails: leave V1 / old V2 in place; archive retrain dir.

### 4.7 Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| GAT unfreeze shocks policy mid-run | Medium | LR ramp 0 → 3e-4 over 500 grad steps already in the schedule; log entropy + KL spike at update ~75–100 to spot it |
| Cosine LR + slower entropy still aren't enough | Medium | Diagnostic logs distinguish failure modes; second pass adjusts LR or adds value clip retuning |
| Plateau-stop fires too early again | Low | With GAT now training, eval should be non-monotone but improving; `plateau_episodes` stays at 100 (default), but `--plateau-episodes 0` is an escape hatch if needed |
| 24-hour CPU run interrupted | Low-Medium | `--resume` works; design preserves it |
| Eval verdict bug fix introduces regression in V1-vs-native printout | Low | Bug fix is additive (new block, not modified block); existing V1-vs-native print unchanged |

### 4.8 Out of scope (deferred follow-ups)

- Domain-randomized retrain (RUN_AI.md Phase 1.3).
- Repo-history reclamation of the deleted `best_ep200.pth` blob via
  `git filter-repo` — separate operation, only worth it if clone size
  becomes painful.
- Updating `PLAN_V2.md` §1.2 R2 prose if the new thresholds make the
  documented schedule misleading.

## 5. Open questions

None. All design choices locked.

## 6. Glossary

- **GAT / CoLight** — graph attention layer over neighboring intersections,
  the parameter that lets V2 learn *which* neighbor to listen to under
  *which* phase. V1 used a fixed 6-float neighbor summary.
- **Frozen-uniform** — `gat.set_frozen_uniform(True)`: attention weights
  forced to uniform mean-pool. The "training wheels" mode before unfreeze.
- **FRAP** — phase-invariant movement encoder, parameter-shared across all
  12 lights. Independent of the GAT schedule.
- **Honest gate** — the RUN_AI.md acceptance threshold for promoting V2
  over native actuated: ≥6pp throughput and ≥12pp wait improvement.

---

*Implementation plan to follow via `superpowers:writing-plans` skill.*
