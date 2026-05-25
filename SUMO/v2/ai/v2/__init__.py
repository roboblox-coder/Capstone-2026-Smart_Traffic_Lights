"""V2 architecture: FRAP encoder + CoLight GAT + MAPPO CTDE.

Component layout:
  frap_encoder.py        - phase-symmetry-invariant per-light encoder
  colight_gat.py         - graph attention over upstream/downstream lights
  shared_policy.py       - parameter-shared actor with masked-categorical head
  centralized_critic.py  - MAPPO joint-state critic, training-only
  mappo_trainer.py       - rollout buffer + PPO update orchestration
  inference_adapter.py   - V1-compatible Agent surface for the live runner

See SUMO/v2/ai/PLAN_V2.md for the architectural rationale and acceptance
gates.
"""
