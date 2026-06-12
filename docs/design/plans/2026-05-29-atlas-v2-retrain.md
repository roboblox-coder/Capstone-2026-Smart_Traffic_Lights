# ATLAS V2 Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the V2 MAPPO training so the CoLight attention actually trains, the policy doesn't collapse during eval, and the resulting `best.pth` clears the honest gate against SUMO's native actuated controller.

**Architecture:** Four narrowly-scoped code changes (config defaults, cosine actor-LR schedule, additional JSONL observability fields, eval_network.py V2-vs-native verdict block), followed by a smoke run, a ~24-hour retrain on `sim_calibrated.sumocfg`, a 5-seed eval, and a conditional promotion of the new checkpoint into the live demo.

**Tech Stack:** Python 3.13, PyTorch 2.11 (CPU), Eclipse SUMO 1.26 via TraCI/sumolib, dataclass-based config, JSONL training logs.

**Spec:** `docs/superpowers/specs/2026-05-29-atlas-v2-retrain-design.md`

---

## File Structure

| File | Responsibility | Touch type |
|---|---|---|
| `SUMO/v2/ai/v2/mappo_trainer.py` | Training driver: config defaults, LR schedule, JSONL logging | **Modify** (lines 55–92, 169–219, 655–667) |
| `SUMO/v2/ai/v2/tests/test_smoke.py` | Smoke tests; add cosine_lr unit test | **Modify** (append new test) |
| `SUMO/v2/ai/eval_network.py` | Network eval: add V2-vs-native verdict block | **Modify** (after line 285) |
| `SUMO/v2/run_websocket_ai.py` | Live runner: V2_CKPT_PATH constant | **Modify** (line 66, conditional on Task 9 gate) |
| `SUMO/v2/ai/runs/v2_mappo_retrain/` | New training-run artifacts directory | **Create** (runtime) |

No new source files. Tests live in the existing `test_smoke.py` since the cosine LR helper is a pure function that doesn't require SUMO.

---

## Task 0: Pre-flight check

**Files:** none (verification only)

- [ ] **Step 1: Confirm branch and clean working tree**

Run from repo root:

```bash
git status --short
git log --oneline -3
```

Expected: empty working tree, top commit is `b22bc3c Design spec: V2 MAPPO retrain to activate CoLight attention` or later.

- [ ] **Step 2: Confirm environment is ready**

Run from `SUMO/v2`:

```bash
python -c "import torch, numpy, traci, sumolib, websockets; print('deps ok')"
python -c "import sumolib; print('sumo:', sumolib.checkBinary('sumo'))"
```

Expected: `deps ok` and a path to `sumo.exe` printing without error.

- [ ] **Step 3: Confirm disk space**

Run:

```bash
df -h . 2>/dev/null || (echo "Windows: skipping df; need >2GB free for retrain artifacts")
```

The retrain run will write ~50 MB of checkpoints + ~50 KB JSONL. Plenty of margin under 2 GB.

---

## Task 1: Update MAPPOConfig defaults

**Files:**
- Modify: `SUMO/v2/ai/v2/mappo_trainer.py` (lines 63, 80–81; add new field)

- [ ] **Step 1: Edit MAPPOConfig dataclass**

In `SUMO/v2/ai/v2/mappo_trainer.py`, change the existing values:

```python
# was: entropy_coef_final: float = 0.001
entropy_coef_final: float = 0.005

# was: gat_freeze_until_step: int = 50_000
gat_freeze_until_step: int = 1_500

# was: gat_ramp_end_step: int = 60_000
gat_ramp_end_step: int = 2_000
```

Then add a new field directly below `actor_lr: float = 3e-4` in the **Optimization** block:

```python
# Optimization
actor_lr: float = 3e-4
actor_lr_final: float = 5e-5  # cosine decay endpoint over total_episodes
critic_lr: float = 1e-3
```

- [ ] **Step 2: Verify the file still parses**

Run from `SUMO/v2`:

```bash
python -c "from ai.v2.mappo_trainer import MAPPOConfig; c = MAPPOConfig(); print(c.gat_freeze_until_step, c.gat_ramp_end_step, c.entropy_coef_final, c.actor_lr_final)"
```

Expected output:

```
1500 2000 0.005 5e-05
```

- [ ] **Step 3: Commit**

```bash
git add SUMO/v2/ai/v2/mappo_trainer.py
git commit -m "$(cat <<'EOF'
MAPPOConfig: rescale GAT freeze schedule + softer entropy floor

gat_freeze_until_step 50_000 -> 1_500, gat_ramp_end_step 60_000 -> 2_000.
The old thresholds were ~10x beyond the achievable gradient-step budget,
so the CoLight attention never trained. New thresholds put unfreeze
inside a 1200-episode run.

entropy_coef_final 0.001 -> 0.005 keeps exploration alive past the
midpoint to avoid the eval-instability collapse seen at ep 210 of the
last run.

Adds actor_lr_final = 5e-5 as the endpoint for the cosine LR schedule
wired in the next commit.
EOF
)"
```

---

## Task 2: Add cosine_lr helper with unit test

**Files:**
- Modify: `SUMO/v2/ai/v2/mappo_trainer.py` (add module-level helper near top)
- Modify: `SUMO/v2/ai/v2/tests/test_smoke.py` (add test function + call in main)

- [ ] **Step 1: Write the failing test**

Append to `SUMO/v2/ai/v2/tests/test_smoke.py` (after `test_inference_adapter_roundtrip`):

```python
def test_cosine_lr_schedule() -> None:
    """cosine_lr returns lr_start at progress=0, lr_final at progress=1,
    and the midpoint average between them. Must clamp progress to [0, 1]."""
    print("  test_cosine_lr_schedule ... ", end="")
    from v2.mappo_trainer import cosine_lr

    lr0 = 3e-4
    lrN = 5e-5
    # Endpoints
    assert abs(cosine_lr(0.0, lr0, lrN) - lr0) < 1e-12, \
        f"progress=0 should give lr_start, got {cosine_lr(0.0, lr0, lrN)}"
    assert abs(cosine_lr(1.0, lr0, lrN) - lrN) < 1e-12, \
        f"progress=1 should give lr_final, got {cosine_lr(1.0, lr0, lrN)}"
    # Midpoint of cosine (1+cos(pi/2))/2 = 0.5, so result is the average
    mid_expected = (lr0 + lrN) / 2
    assert abs(cosine_lr(0.5, lr0, lrN) - mid_expected) < 1e-12, \
        f"progress=0.5 should give mean, got {cosine_lr(0.5, lr0, lrN)}"
    # Clamping below 0 / above 1
    assert cosine_lr(-0.5, lr0, lrN) == cosine_lr(0.0, lr0, lrN)
    assert cosine_lr(2.0, lr0, lrN) == cosine_lr(1.0, lr0, lrN)
    print("OK")
```

And add the call inside `main()`:

```python
def main() -> int:
    print("V2 smoke tests:")
    test_frap_only()
    test_frap_batched_matches_per_tls()
    test_full_stack_forward()
    test_full_stack_backward()
    test_gat_attention_modes()
    test_batched_minibatch_shapes()
    test_inference_adapter_roundtrip()
    test_cosine_lr_schedule()       # <-- add this line
    print("\nAll smoke tests passed.")
    return 0
```

- [ ] **Step 2: Run test to verify it fails**

Run from `SUMO/v2`:

```bash
python -m ai.v2.tests.test_smoke
```

Expected: tests up through `test_inference_adapter_roundtrip` pass; `test_cosine_lr_schedule` fails with `ImportError: cannot import name 'cosine_lr' from 'v2.mappo_trainer'`.

- [ ] **Step 3: Add the helper to mappo_trainer.py**

In `SUMO/v2/ai/v2/mappo_trainer.py`, add at module level **between** the imports block (after line 50) and the `# ---------- hyperparameters ----------` comment (before line 53):

```python
import math as _math  # noqa: E402 — used by cosine_lr below

def cosine_lr(progress: float, lr_start: float, lr_final: float) -> float:
    """Cosine decay from ``lr_start`` to ``lr_final`` over ``progress`` in
    [0, 1]. Out-of-range progress is clamped. At progress=0 returns
    lr_start; at progress=1 returns lr_final; at progress=0.5 returns the
    arithmetic mean.
    """
    p = max(0.0, min(1.0, progress))
    return lr_final + 0.5 * (lr_start - lr_final) * (1.0 + _math.cos(_math.pi * p))
```

- [ ] **Step 4: Run test to verify it passes**

Run from `SUMO/v2`:

```bash
python -m ai.v2.tests.test_smoke
```

Expected: all 8 tests pass, ending with `All smoke tests passed.`

- [ ] **Step 5: Commit**

```bash
git add SUMO/v2/ai/v2/mappo_trainer.py SUMO/v2/ai/v2/tests/test_smoke.py
git commit -m "$(cat <<'EOF'
Add cosine_lr helper for actor learning-rate schedule

Pure function: cosine decay from lr_start to lr_final over progress in
[0, 1], clamped at endpoints. Used by the actor-LR schedule wired in
the next commit. Unit-tested in the V2 smoke test (no SUMO dependency).
EOF
)"
```

---

## Task 3: Wire cosine actor-LR into the per-step schedule

**Files:**
- Modify: `SUMO/v2/ai/v2/mappo_trainer.py` (lines 200–218: extend `_update_gat_schedule`)

- [ ] **Step 1: Edit `_update_gat_schedule` to also update actor LR**

Replace the body of `_update_gat_schedule` (lines 200–218) with:

```python
def _update_gat_schedule(self) -> None:
    """Called once per gradient step. Implements the plan's frozen-
    uniform (0..gat_freeze_until_step) -> linear ramp -> full-rate
    schedule for the GAT, and a cosine decay actor_lr -> actor_lr_final
    schedule for the encoder + actor (param_groups [0] and [2]).
    Progress for the cosine decay is measured in episodes (matching
    entropy_coef), not gradient steps.
    """
    s = self._gradient_steps
    cfg = self.cfg

    # --- GAT schedule (unchanged logic, new thresholds) ---
    if s < cfg.gat_freeze_until_step:
        self.gat.set_frozen_uniform(True)
        gat_lr = 0.0
    elif s < cfg.gat_ramp_end_step:
        self.gat.set_frozen_uniform(False)
        f = ((s - cfg.gat_freeze_until_step) /
             max(1, cfg.gat_ramp_end_step - cfg.gat_freeze_until_step))
        gat_lr = f * cfg.actor_lr
    else:
        self.gat.set_frozen_uniform(False)
        gat_lr = cfg.actor_lr
    self.actor_opt.param_groups[1]["lr"] = gat_lr

    # --- Actor LR cosine decay (encoder + actor head) ---
    progress = (self._episodes_done /
                max(1, self.cfg.total_episodes))
    new_actor_lr = cosine_lr(progress, cfg.actor_lr, cfg.actor_lr_final)
    self.actor_opt.param_groups[0]["lr"] = new_actor_lr  # encoder
    self.actor_opt.param_groups[2]["lr"] = new_actor_lr  # actor head
```

(`cosine_lr` is the module-level helper added in Task 2.)

- [ ] **Step 2: Verify the trainer still constructs cleanly**

Run from `SUMO/v2`:

```bash
python -c "
from ai.v2.mappo_trainer import MAPPOConfig, cosine_lr
cfg = MAPPOConfig(total_episodes=1200)
# Verify endpoints of the cosine decay
print('lr at 0%:', cosine_lr(0.0, cfg.actor_lr, cfg.actor_lr_final))
print('lr at 50%:', cosine_lr(0.5, cfg.actor_lr, cfg.actor_lr_final))
print('lr at 100%:', cosine_lr(1.0, cfg.actor_lr, cfg.actor_lr_final))
"
```

Expected:

```
lr at 0%: 0.0003
lr at 50%: 0.000175
lr at 100%: 5e-05
```

- [ ] **Step 3: Commit**

```bash
git add SUMO/v2/ai/v2/mappo_trainer.py
git commit -m "$(cat <<'EOF'
Wire cosine actor-LR schedule into per-step update

Extends _update_gat_schedule to drive encoder + actor head LRs via
cosine_lr(progress, actor_lr, actor_lr_final), with progress measured
in episodes (matches entropy_coef cadence). GAT schedule logic is
unchanged; only the new thresholds from MAPPOConfig take effect.

This is the mitigation for the eval-instability collapse seen at ep 210
of the previous run, where the policy collapsed once entropy was low
and actor LR stayed high.
EOF
)"
```

---

## Task 4: Add gat_attention_entropy + actor_lr to JSONL logging

**Files:**
- Modify: `SUMO/v2/ai/v2/mappo_trainer.py` (lines 655–667: per-update JSONL payload)

- [ ] **Step 1: Edit the per-update JSONL log dict**

Replace the existing `self._log_jsonl({...})` call at lines 655–667 with:

```python
self._log_jsonl({
    "kind": "update",
    "update": n_updates,
    "episodes_done": self._episodes_done,
    "gradient_steps": self._gradient_steps,
    "update_seconds": update_seconds,
    "elapsed_seconds": elapsed,
    "eta_seconds": eta_seconds,
    "reward_per_episode": traj["episode_reward_mean"],
    "gat_lr": self.actor_opt.param_groups[1]["lr"],
    "actor_lr": self.actor_opt.param_groups[0]["lr"],
    "gat_attention_entropy": float(
        self.gat.attention_entropy().mean().item()),
    "entropy_coef": self._current_entropy_coef(),
    **logs,
})
```

(`self.gat.attention_entropy()` already exists — `colight_gat.py` exposes it; the smoke test exercises it at `test_gat_attention_modes`.)

- [ ] **Step 2: Sanity-check by inspecting the existing GAT API**

Run from `SUMO/v2`:

```bash
python -c "
from v2.colight_gat import CoLightGAT
import torch
g = CoLightGAT(embed_dim=128)
g.set_frozen_uniform(True)
le = torch.randn(12, 128)
adj = torch.eye(12).bool()
for i in range(11):
    adj[i, i+1] = True; adj[i+1, i] = True
_ = g(le, adj)
ent = g.attention_entropy()
print('shape:', ent.shape, 'mean:', float(ent.mean()))
" 
```

Expected: a tensor of shape `torch.Size([4])` (4 GAT heads) and a positive float mean (frozen-uniform attention has nonzero entropy).

- [ ] **Step 3: Commit**

```bash
git add SUMO/v2/ai/v2/mappo_trainer.py
git commit -m "$(cat <<'EOF'
Log gat_attention_entropy + actor_lr per update in JSONL

After a run, the JSONL alone tells us:
- whether the attention is actually learning (entropy diverges from the
  frozen-uniform baseline after unfreeze)
- the effective actor LR at each update (cosine schedule trace)

Without these, "the GAT never trained" failure mode (as happened last
run) is invisible until you reconstruct the schedule from constants.
EOF
)"
```

---

## Task 5: Add V2-vs-native verdict in eval_network.py (TDD)

**Files:**
- Modify: `SUMO/v2/ai/eval_network.py` (add helper + new verdict block after line 285)

- [ ] **Step 1: Write the failing test**

Create `SUMO/v2/ai/tests/__init__.py` (empty) and `SUMO/v2/ai/tests/test_eval_network.py`:

```python
"""Tests for the eval_network.py verdict helpers."""
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from eval_network import compute_vs_native_verdict  # noqa: E402


def test_v2_beats_native_on_both_metrics():
    """When v2 throughput > native and v2 wait < native on all seeds,
    the verdict should report +pct on throughput and +pct (lower) on wait
    with full seed wins."""
    v2_runs = [
        {"arrived": 2200, "wait_per_vehicle": 5000.0},
        {"arrived": 2300, "wait_per_vehicle": 4800.0},
        {"arrived": 2150, "wait_per_vehicle": 5100.0},
    ]
    nat_runs = [
        {"arrived": 2046, "wait_per_vehicle": 5938.0},
        {"arrived": 2076, "wait_per_vehicle": 5503.0},
        {"arrived": 1966, "wait_per_vehicle": 7113.0},
    ]
    v = compute_vs_native_verdict(v2_runs, nat_runs)
    assert v["throughput_pct"] > 0
    assert v["wait_pct"] > 0  # positive = lower wait
    assert v["throughput_wins"] == 3
    assert v["wait_wins"] == 3
    assert v["n_seeds"] == 3


def test_v2_loses_to_native():
    """V2 numbers from the last run (3-seed eval). Confirms the verdict
    correctly reports negative percentages and 0 wins."""
    v2_runs = [
        {"arrived": 1632, "wait_per_vehicle": 7534.59},
        {"arrived": 1558, "wait_per_vehicle": 10446.99},
        {"arrived": 1726, "wait_per_vehicle": 5089.16},
    ]
    nat_runs = [
        {"arrived": 2095, "wait_per_vehicle": 5196.23},
        {"arrived": 2076, "wait_per_vehicle": 5503.70},
        {"arrived": 1966, "wait_per_vehicle": 7113.79},
    ]
    v = compute_vs_native_verdict(v2_runs, nat_runs)
    assert v["throughput_pct"] < 0          # v2 throughput < native
    assert v["wait_pct"] < 0                # v2 wait > native (so pct lower < 0)
    assert v["throughput_wins"] == 0
    # one seed (seed 3) has v2 wait < native wait
    assert v["wait_wins"] == 1


if __name__ == "__main__":
    test_v2_beats_native_on_both_metrics()
    test_v2_loses_to_native()
    print("eval_network tests OK")
```

- [ ] **Step 2: Run the test to verify it fails**

Run from `SUMO/v2`:

```bash
python -m ai.tests.test_eval_network
```

Expected: `ImportError: cannot import name 'compute_vs_native_verdict' from 'eval_network'`.

- [ ] **Step 3: Add the helper and the V2 verdict block to eval_network.py**

In `SUMO/v2/ai/eval_network.py`, add the helper near the top (after imports, before `run_controlled`):

```python
def compute_vs_native_verdict(policy_runs: list, native_runs: list) -> dict:
    """Pure helper: compute mean throughput / wait, per-seed wins, and
    pct improvement of ``policy_runs`` vs ``native_runs``. Caller passes
    parallel lists of per-episode result dicts with keys ``arrived`` and
    ``wait_per_vehicle``. Returns:

        {
          "throughput_pct": float,   # positive = policy beats native
          "wait_pct":       float,   # positive = policy has lower wait
          "throughput_wins": int,    # seeds where policy > native arrived
          "wait_wins":      int,     # seeds where policy < native wait
          "n_seeds":        int,
          "policy_arr":     float,   # mean arrived
          "policy_wpv":     float,   # mean wait per vehicle
          "native_arr":     float,
          "native_wpv":     float,
        }
    """
    n = min(len(policy_runs), len(native_runs))
    if n == 0:
        return {"throughput_pct": 0.0, "wait_pct": 0.0,
                "throughput_wins": 0, "wait_wins": 0, "n_seeds": 0,
                "policy_arr": 0.0, "policy_wpv": 0.0,
                "native_arr": 0.0, "native_wpv": 0.0}
    p_arr = sum(r["arrived"] for r in policy_runs[:n]) / n
    p_wpv = sum(r["wait_per_vehicle"] for r in policy_runs[:n]) / n
    n_arr = sum(r["arrived"] for r in native_runs[:n]) / n
    n_wpv = sum(r["wait_per_vehicle"] for r in native_runs[:n]) / n
    arr_pct = (p_arr - n_arr) / max(1e-9, n_arr) * 100.0
    wait_pct = (n_wpv - p_wpv) / max(1e-9, n_wpv) * 100.0
    arr_wins = sum(1 for i in range(n)
                   if policy_runs[i]["arrived"] > native_runs[i]["arrived"])
    wait_wins = sum(1 for i in range(n)
                    if policy_runs[i]["wait_per_vehicle"]
                    < native_runs[i]["wait_per_vehicle"])
    return {
        "throughput_pct": arr_pct,
        "wait_pct": wait_pct,
        "throughput_wins": arr_wins,
        "wait_wins": wait_wins,
        "n_seeds": n,
        "policy_arr": p_arr,
        "policy_wpv": p_wpv,
        "native_arr": n_arr,
        "native_wpv": n_wpv,
    }
```

Then, after the existing V1-vs-native block (after line 285, before the `out_path = Path(args.out)` line), add the V2 verdict block:

```python
    # V2-vs-native verdict (parallel to the V1 block above). Only emitted
    # when the V2 spec ran in this eval (i.e. v2_choose was not None).
    if "coordinated_v2_frap" in results and results["coordinated_v2_frap"]:
        v2 = compute_vs_native_verdict(
            results["coordinated_v2_frap"], results[base])
        emit("\n=== coordinated_v2_frap vs all_native_actuated (network) ===")
        emit(f"  throughput : {v2['policy_arr']:8.1f} vs "
             f"{v2['native_arr']:8.1f}  "
             f"({v2['throughput_pct']:+.1f}%, beats native on "
             f"{v2['throughput_wins']}/{v2['n_seeds']} seeds)")
        emit(f"  wait/veh   : {v2['policy_wpv']:8.2f} vs "
             f"{v2['native_wpv']:8.2f}  "
             f"({v2['wait_pct']:+.1f}%, beats native on "
             f"{v2['wait_wins']}/{v2['n_seeds']} seeds)")
        emit("")
        # Honest gate from RUN_AI.md: +6pp throughput AND -12pp wait.
        passes_throughput = v2["throughput_pct"] >= 6.0
        passes_wait = v2["wait_pct"] >= 12.0
        if passes_throughput and passes_wait:
            emit("Verdict: V2 clears the honest gate vs native "
                 f"(throughput +{v2['throughput_pct']:.1f}% >= +6%, "
                 f"wait +{v2['wait_pct']:.1f}% lower >= +12%). "
                 "Eligible for promotion to live demo.")
        else:
            reasons = []
            if not passes_throughput:
                reasons.append(f"throughput {v2['throughput_pct']:+.1f}% < +6%")
            if not passes_wait:
                reasons.append(f"wait {v2['wait_pct']:+.1f}% < +12% lower")
            emit("Verdict: V2 does NOT clear the honest gate vs native "
                 f"({'; '.join(reasons)}). Do not promote.")
```

- [ ] **Step 4: Run the test to verify it passes**

Run from `SUMO/v2`:

```bash
python -m ai.tests.test_eval_network
```

Expected: `eval_network tests OK`.

- [ ] **Step 5: Commit**

```bash
git add SUMO/v2/ai/eval_network.py SUMO/v2/ai/tests/__init__.py SUMO/v2/ai/tests/test_eval_network.py
git commit -m "$(cat <<'EOF'
eval_network: add V2-vs-native verdict block

The existing end-of-run "Verdict:" line only compares coordinated_dqn
(V1) to native -- when V2 is present in the results, its numbers are
in the summary table but the printed verdict ignores them, which is
misleading.

Add a parallel block that applies the RUN_AI.md honest gate
(+6pp throughput, -12pp wait) to V2-vs-native and prints a clear
promote-or-don't verdict. Pure-helper computation is unit-tested in
ai/tests/test_eval_network.py.
EOF
)"
```

---

## Task 6: Pre-flight smoke run (12-episode trainer sanity check, ~5–7 min)

**Files:** none committed; output to `SUMO/v2/ai/runs/v2_smoke_preflight/` (already gitignored as `v2_smoke*`).

- [ ] **Step 1: Run a 2-update smoke training run**

From `SUMO/v2`:

```bash
PYTHONIOENCODING=utf-8 python ai/v2/mappo_trainer.py \
    --sumo-cfg sim_calibrated.sumocfg \
    --episodes 12 --time-limit 600 \
    --rollout-episodes 6 \
    --out-dir ai/runs/v2_smoke_preflight \
    --eval-every 999 \
    --plateau-episodes 0
```

Expected: trainer prints two `[update N] ep=6/12 ...` then `[update 2] ep=12/12 ...` lines and exits cleanly. ~5–7 min wall clock on CPU.

(`--eval-every 999` effectively skips eval — at 2 updates, `n_updates % 999` is never 0. Do NOT pass `--eval-every 0`: the trainer does `n_updates % cfg.eval_every_updates` with no zero-guard, which would crash with `ZeroDivisionError`. `--plateau-episodes 0` is the proper "disable" sentinel for plateau detection.)

- [ ] **Step 2: Verify the JSONL contains the new fields**

```bash
python -c "
import json
lines = [json.loads(l) for l in open('ai/runs/v2_smoke_preflight/train_log.jsonl')]
upds = [l for l in lines if l.get('kind') == 'update']
assert len(upds) >= 1, 'expected at least one update entry'
for u in upds:
    assert 'gat_attention_entropy' in u, 'missing gat_attention_entropy'
    assert 'actor_lr' in u, 'missing actor_lr'
print('OK', len(upds), 'updates,',
      'gat_ent=', round(upds[-1]['gat_attention_entropy'], 3),
      'actor_lr=', upds[-1]['actor_lr'])
"
```

Expected: prints `OK 2 updates, gat_ent= <positive number> actor_lr= <approx 0.0003>` (cosine has barely moved at 12/1200 episodes).

- [ ] **Step 3: Verify the GAT actually unfroze in the smoke run**

The smoke run did 2 updates × 20 grad steps = 40 grad steps, which is < 1500 = `gat_freeze_until_step`. So gat_lr should still be 0.0 in the smoke. Confirm:

```bash
python -c "
import json
lines = [json.loads(l) for l in open('ai/runs/v2_smoke_preflight/train_log.jsonl')]
upds = [l for l in lines if l.get('kind') == 'update']
last = upds[-1]
assert last['gat_lr'] == 0.0, f'GAT should still be frozen at 40 grad steps; got lr={last[\"gat_lr\"]}'
print('OK GAT still frozen at', last['gradient_steps'], 'grad steps as expected')
"
```

Expected: `OK GAT still frozen at 40 grad steps as expected`. (Confirms the schedule is wired correctly — we are NOT yet past the unfreeze; that's what we want for a smoke. The retrain in Task 7 will cross the threshold.)

- [ ] **Step 4: Do NOT commit the smoke artifacts**

The output dir is in `.gitignore` (pattern `SUMO/v2/ai/runs/v2_smoke*/`). Move on. (You can `rm -rf SUMO/v2/ai/runs/v2_smoke_preflight` to reclaim ~40 MB if you want.)

---

## Task 7: Run the long retrain (~24 hours)

**Files:** none committed; output to `SUMO/v2/ai/runs/v2_mappo_retrain/`.

⚠️ **This task is NOT a 2–5 min step.** It launches a long-running process and monitors it. Sub-steps cover kickoff, periodic check-ins, and resume-on-interrupt.

- [ ] **Step 1: Kick off the retrain in the background**

From `SUMO/v2`:

```bash
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 python ai/v2/mappo_trainer.py \
    --sumo-cfg sim_calibrated.sumocfg \
    --episodes 1200 --time-limit 1200 \
    --rollout-episodes 6 \
    --out-dir ai/runs/v2_mappo_retrain \
    --eval-every 5 \
    --eval-seeds 1042 1043 1044 \
    --plateau-episodes 100 \
    > /tmp/v2_retrain.log 2>&1 &
echo "STARTED PID=$!"
```

(On Windows shells use `&` differently — if running interactively, drop the trailing `&` and run it in a separate terminal, or use `start /B`.)

Expected: a single line `STARTED PID=<n>`. Trainer prints update lines into `/tmp/v2_retrain.log` every ~70s.

- [ ] **Step 2: Confirm the run is alive (every few hours)**

```bash
ps -p <PID> -o pid,etime,pcpu,comm 2>/dev/null || echo "EXITED"
tail -3 /tmp/v2_retrain.log
ls -lh ai/runs/v2_mappo_retrain/checkpoints/ 2>/dev/null
```

Expected (mid-run): process alive, log grows with `[update N] ep=X/1200 ... gat_lr=...` lines, `best.pth` + `last.pth` exist in checkpoints dir.

The `gat_lr` field in the log should transition from `0.00e+00` to non-zero around update 75 (gradient step 1500). If it never does, the schedule is wired wrong — stop and fix.

- [ ] **Step 3: Resume if interrupted**

If the run dies for any reason (OOM, reboot, you killed it), re-launch with `--resume`:

```bash
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 python ai/v2/mappo_trainer.py \
    --sumo-cfg sim_calibrated.sumocfg \
    --episodes 1200 --time-limit 1200 \
    --rollout-episodes 6 \
    --out-dir ai/runs/v2_mappo_retrain \
    --eval-every 5 \
    --eval-seeds 1042 1043 1044 \
    --plateau-episodes 100 \
    --resume ai/runs/v2_mappo_retrain/checkpoints/last.pth \
    > /tmp/v2_retrain.log 2>&1 &
```

The resume path restores nets, optimizers, episode count, gradient-step count, plateau tracker, AND snaps the GAT/actor LR schedule onto the current gradient step ([mappo_trainer.py:544](../../SUMO/v2/ai/v2/mappo_trainer.py)).

- [ ] **Step 4: Confirm successful completion**

When the run exits:

```bash
tail -20 /tmp/v2_retrain.log
python -c "
import json
lines = [json.loads(l) for l in open('ai/runs/v2_mappo_retrain/train_log.jsonl')]
upds = [l for l in lines if l.get('kind') == 'update']
evs = [l for l in lines if l.get('kind') == 'eval']
last_ep = upds[-1]['episodes_done'] if upds else 0
plateau = [l for l in lines if l.get('kind') == 'plateau_stop']
print(f'Run ended at episode {last_ep}, n_updates={len(upds)}, n_evals={len(evs)}, plateau_stopped={len(plateau) > 0}')
print(f'Best eval wait: {min(e[\"wait_per_vehicle_mean\"] for e in evs):.2f} (lower is better)')
"
```

Expected: either `last_ep=1200, plateau_stopped=False` (full run) or `plateau_stopped=True` at some lower episode (acceptable if it happens after GAT unfreezes, i.e. ep_done ≥ ~600).

- [ ] **Step 5: Confirm GAT actually trained**

```bash
python -c "
import json
lines = [json.loads(l) for l in open('ai/runs/v2_mappo_retrain/train_log.jsonl')]
upds = [l for l in lines if l.get('kind') == 'update']
# First 50 updates are frozen-uniform baseline; collect the median entropy.
import statistics
pre = [u['gat_attention_entropy'] for u in upds if u['gradient_steps'] < 1500]
post = [u['gat_attention_entropy'] for u in upds if u['gradient_steps'] > 2500]
print(f'Frozen-uniform GAT entropy (n={len(pre)}): median={statistics.median(pre):.4f}')
if post:
    print(f'Post-unfreeze GAT entropy  (n={len(post)}): median={statistics.median(post):.4f}')
    diff = abs(statistics.median(pre) - statistics.median(post))
    print(f'|delta| = {diff:.4f}  ({'> 0.1 ok' if diff > 0.1 else 'WARNING: attention did not move'})')
else:
    print('WARNING: run did not get past gradient step 2500 (GAT only partially trained)')
"
```

Expected: post-unfreeze entropy differs from frozen-uniform by ≥ 0.1 nats on at least the median. If `delta < 0.1`, attention didn't actually learn — that's a failure mode the spec §4.5 anticipates; record it in the postmortem.

- [ ] **Step 6: Do NOT commit retrain artifacts during this task**

`runs/v2_mappo_retrain/` will be committed (or its `best.pth` copied into `runs/v2_mappo/`) **only** if Task 9 passes. Hold off until then.

---

## Task 8: Post-retrain eval at 5 seeds (~30–45 min)

**Files:** none committed; output to `SUMO/v2/ai/logs/eval_network_retrain.txt`.

- [ ] **Step 1: Run the 5-seed eval against the new checkpoint**

From `SUMO/v2`:

```bash
PYTHONIOENCODING=utf-8 python ai/eval_network.py \
    --sumo-cfg sim_calibrated.sumocfg \
    --v2-ckpt ai/runs/v2_mappo_retrain/checkpoints/best.pth \
    --episodes 5 \
    --time-limit 1200 \
    --out ai/logs/eval_network_retrain.txt \
    > /tmp/v2_eval_retrain.log 2>&1 &
echo "STARTED PID=$!"
```

Expected: backgrounded; ~30–45 min wall-clock for 5 episodes × 4 policies × 1200 sim-seconds.

- [ ] **Step 2: When complete, read the verdict**

```bash
grep -A 20 "coordinated_v2_frap vs all_native_actuated" ai/logs/eval_network_retrain.txt
```

Expected (success): lines like

```
=== coordinated_v2_frap vs all_native_actuated (network) ===
  throughput :   <X>  vs   <Y>   (+<P>%, beats native on <W>/5 seeds)
  wait/veh   :   <X>  vs   <Y>   (+<P>%, beats native on <W>/5 seeds)

Verdict: V2 clears the honest gate vs native (...).
```

Expected (failure): same block but with a `Verdict: V2 does NOT clear the honest gate vs native (...)` line listing which metric(s) missed.

- [ ] **Step 3: Save the eval log (gitignored — for your records)**

`ai/logs/eval_network_retrain.txt` lives under the gitignored `ai/logs/` dir; nothing else to do. Keep it for the postmortem if Task 9 fails.

---

## Task 9: Acceptance-gate decision

**Files:** none modified in this task (decision only).

- [ ] **Step 1: Apply the hard floor**

From the verdict in Task 8 Step 2:

- **Pass criteria (both must hold):** `throughput_pct >= 6.0` AND `wait_pct >= 12.0`
- **Pass → proceed to Task 10**
- **Fail → STOP. Skip Task 10.** Archive the retrain dir for postmortem and decide whether to launch another retrain (separate cycle).

- [ ] **Step 2 (on fail): Capture the postmortem signal**

Run:

```bash
python -c "
import json
lines = [json.loads(l) for l in open('ai/runs/v2_mappo_retrain/train_log.jsonl')]
upds = [l for l in lines if l.get('kind') == 'update']
evs  = [l for l in lines if l.get('kind') == 'eval']
print('=== training-time eval trajectory ===')
for e in evs:
    print(f\"  ep={e['episodes_done']:>4d}  wpv={e['wait_per_vehicle_mean']:>9.2f}  thr={e['throughput_mean']:>7.1f}\")
print('=== GAT entropy after unfreeze (sample) ===')
post = [u for u in upds if u['gradient_steps'] > 2500]
for u in post[::max(1, len(post)//8)]:
    print(f\"  step={u['gradient_steps']:>5d}  ent={u['gat_attention_entropy']:.4f}  actor_lr={u['actor_lr']:.2e}\")
"
```

Pin which failure mode (from spec §4.5):

- Attention never moved → GAT schedule / freeze flag bug
- Attention moved but eval still volatile → LR / entropy still wrong
- Both fine + eval stable but doesn't beat native → architectural ceiling; needs DR or different reward

Document the failure mode and stop. (Drafting a postmortem doc is out of scope for this plan; it would be input to a next plan.)

- [ ] **Step 3 (on pass): Proceed to Task 10**

No commit yet — the promotion commit is part of Task 10.

---

## Task 10 (conditional): Promote new checkpoint into the live demo

**Files:**
- Modify: `SUMO/v2/run_websocket_ai.py` (line 66)
- (Optional) copy `runs/v2_mappo_retrain/checkpoints/best.pth` → `runs/v2_mappo/checkpoints/best.pth` to keep the canonical path stable

⚠️ Execute this task **only if Task 9 Step 1 passed**.

- [ ] **Step 1: Decide on layout — pointer-swap or file-copy**

Two options, your choice:

| Option | Action | Pro | Con |
|---|---|---|---|
| **A: Pointer-swap** | Edit `run_websocket_ai.py` `V2_CKPT_PATH` to point at `ai/runs/v2_mappo_retrain/checkpoints/best.pth` | Both runs traceable in git history | Two model paths in the tree |
| **B: File-copy** | `cp ai/runs/v2_mappo_retrain/checkpoints/best.pth ai/runs/v2_mappo/checkpoints/best.pth`, leave the constant alone | Single canonical path stays the same | Old checkpoint overwritten; only `runs/v2_mappo_retrain/` log retains the lineage |

Recommended: **A (pointer-swap)** — leaves both checkpoints in the repo for traceability, and the diff makes the promotion explicit.

- [ ] **Step 2 (option A): Update V2_CKPT_PATH**

In `SUMO/v2/run_websocket_ai.py`, replace line 66:

```python
# was:
V2_CKPT_PATH = os.path.join(
    os.path.dirname(__file__), "ai", "runs", "v2_mappo", "checkpoints", "best.pth"
)
```

with:

```python
V2_CKPT_PATH = os.path.join(
    os.path.dirname(__file__), "ai", "runs", "v2_mappo_retrain", "checkpoints", "best.pth"
)
```

- [ ] **Step 2 (option B): Copy the new checkpoint**

```bash
cp SUMO/v2/ai/runs/v2_mappo_retrain/checkpoints/best.pth \
   SUMO/v2/ai/runs/v2_mappo/checkpoints/best.pth
```

- [ ] **Step 3: Live verification (replays the websocket-client check)**

From `SUMO/v2`, in one shell:

```bash
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 python run_websocket_ai.py > /tmp/v2_live_after.log 2>&1 &
echo "STARTED PID=$!"
# wait for "Loaded 12/12 agents" line
for i in $(seq 1 30); do grep -q "Loaded [0-9]*/12 agents" /tmp/v2_live_after.log && break; sleep 1; done
grep -E "V2 inference|Loaded" /tmp/v2_live_after.log
```

Expected: `V2 inference: loaded ... best.pth (corridor-level FRAP/GAT/MAPPO).` referencing the **new** checkpoint path (option A) or the same path (option B). Then `Loaded 12/12 agents`.

Then run a tiny WS client in another shell (mirror of the verification done in this conversation):

```bash
python -c "
import asyncio, json
import websockets
async def go():
    async with websockets.connect('ws://localhost:8765', max_size=None) as ws:
        for _ in range(40):
            msg = json.loads(await ws.recv())
            if msg.get('type') == 'step':
                last = msg
        s = last['aiSummary']
        print('mode=', s['mode'], 'active=', s['active'], '/', s['total'])
asyncio.run(go())
"
```

Expected: `mode= v2 active= 12 / 12`.

Then stop the live runner:

```bash
# PowerShell or via taskkill matching cmdline; or Ctrl+C if running foreground
```

- [ ] **Step 4: Commit the promotion**

```bash
git add SUMO/v2/run_websocket_ai.py     # option A
# OR: git add SUMO/v2/ai/runs/v2_mappo/checkpoints/best.pth  # option B

git commit -m "$(cat <<'EOF'
Promote V2 retrain checkpoint to live demo

The 1200-episode retrain (ai/runs/v2_mappo_retrain/checkpoints/best.pth)
clears the honest gate vs SUMO native actuated:
  throughput: +<X>% (>= +6%), wait/veh: -<Y>% (>= -12%)
on a 5-seed eval (eval_network_retrain.txt for raw numbers).

The CoLight attention is now actually trained (gat_attention_entropy
diverges from frozen-uniform baseline by <Z> nats after unfreeze),
and the policy is stable through training-eval cadence (no >2x median
collapse).

Live demo verified post-swap: aiSummary.mode = "v2", 12/12 lights
active, decisions incrementing.
EOF
)"
```

(Fill in the `<X>`, `<Y>`, `<Z>` numbers from Task 8 / Task 7 Step 5 output before committing.)

- [ ] **Step 5: Optional — archive the retrain dir explicitly**

If you went with option B (file-copy), the `runs/v2_mappo_retrain/` dir is now redundant for the live demo. Decide whether to keep it (recommended — it has the training log) or remove. If keeping, no action; it's not gitignored, so it stays in the repo. If removing: `git rm -r SUMO/v2/ai/runs/v2_mappo_retrain/checkpoints/` (keep the `train_log.jsonl` + `config.json`) and commit.

---

## Done

After Task 10 commits, the V2 corridor policy in the live demo is the retrained one. Working tree is clean. The next reasonable cycle would be Phase 1.3 (domain-randomized retrain for sim-to-real robustness), which is **out of scope for this plan**.
