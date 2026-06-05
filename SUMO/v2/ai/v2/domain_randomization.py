"""Domain randomization wrapper around MultiTlsEnv.

Sampled per episode in ``reset()`` and applied across the rollout:

  demand_mult     - log-uniform[0.7, 1.3]; passed to SUMO as --scale
                    so every vehicle insertion is scaled, no route-file
                    regeneration needed.
  detector_fn     - uniform[0, 0.15] false-negative rate; applied at
                    state-egress only (the reward path continues to
                    read ground-truth TraCI queues -- PLAN_V2.md §1.3
                    pre-commits this trade-off).
  truck_frac      - uniform[0, 0.15]. Per-OD truck injection requires
                    either a separate route file or TraCI vehicle.add
                    plumbing that the corridor route generator doesn't
                    expose yet. STUBBED: stored on the wrapper so the
                    reporting still surfaces the value, but no
                    behavior change vs. truck_frac=0 until the OD
                    pipeline lands.
  sigma_jitter    - +/- 20% on Krauss sigma. STUBBED for the same
                    reason: requires writing a fresh vTypes file per
                    episode or using vehicletype.setImperfection after
                    the first vehicle is inserted (timing-fragile).
                    Defer until per-episode vType generation lands.

Train::

    python ai/v2/mappo_trainer.py --sumo-cfg sim_calibrated.sumocfg \\
        --randomize --episodes 1500 --out-dir ai/runs/v2_mappo_dr

Eval on the V2-DR vs. V1 stress test::

    python ai/eval_network.py --sumo-cfg sim_stress.sumocfg \\
        --v2-ckpt ai/runs/v2_mappo_dr/checkpoints/best.pth \\
        --episodes 10
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from multi_env import MultiTlsEnv  # noqa: E402


class DRWrapper(MultiTlsEnv):
    """MultiTlsEnv subclass injecting per-episode demand + obs noise.

    See module docstring for which knobs are live today vs. stubbed.

    Reward path is untouched: the centralized critic still sees
    ground-truth TraCI queues. The policy learns to handle noisy
    observations on input only -- the "honest reward" choice.
    """

    def __init__(
        self,
        *args,
        dr_seed: int = 4242,
        demand_range: tuple = (0.7, 1.3),
        truck_range: tuple = (0.0, 0.15),
        detector_fn_range: tuple = (0.0, 0.15),
        sigma_jitter: float = 0.20,
        log_per_episode: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._dr_rng = np.random.default_rng(dr_seed)
        self.demand_range = tuple(demand_range)
        self.truck_range = tuple(truck_range)
        self.detector_fn_range = tuple(detector_fn_range)
        self.sigma_jitter = float(sigma_jitter)
        self.log_per_episode = bool(log_per_episode)
        self.extra_cli_args: list = []
        self._current_noise: dict = self._sample_noise_for_episode()
        self._episode_idx = 0

    def _sample_noise_for_episode(self) -> dict:
        # Log-uniform demand so 0.7..1.3 is symmetric in log-space.
        log_lo = math.log(self.demand_range[0])
        log_hi = math.log(self.demand_range[1])
        demand_mult = float(math.exp(self._dr_rng.uniform(log_lo, log_hi)))
        truck_frac = float(self._dr_rng.uniform(*self.truck_range))
        fn_rate = float(self._dr_rng.uniform(*self.detector_fn_range))
        sigma_delta = float(self._dr_rng.uniform(-self.sigma_jitter,
                                                 self.sigma_jitter))
        return {
            "demand_mult": demand_mult,
            "truck_frac": truck_frac,    # stubbed
            "fn_rate": fn_rate,
            "sigma_delta": sigma_delta,  # stubbed
        }

    @property
    def current_noise(self) -> dict:
        return dict(self._current_noise)

    def reset(self) -> dict:
        self._current_noise = self._sample_noise_for_episode()
        self._episode_idx += 1

        # Inject --scale before super().reset() restarts SUMO.
        self.extra_cli_args = [
            "--scale", f"{self._current_noise['demand_mult']:.4f}",
        ]

        if self.log_per_episode:
            n = self._current_noise
            stubs = []
            if n["truck_frac"] > 0:
                stubs.append(f"truck_frac={n['truck_frac']:.2f}(stub)")
            if abs(n["sigma_delta"]) > 1e-6:
                stubs.append(f"sigma_delta={n['sigma_delta']:+.2f}(stub)")
            stub_str = f"  [stubbed: {', '.join(stubs)}]" if stubs else ""
            print(f"[DR ep {self._episode_idx}] "
                  f"demand_mult={n['demand_mult']:.3f}  "
                  f"fn_rate={n['fn_rate']:.3f}{stub_str}")

        return super().reset()

    # ---------- observation-side noise ----------

    def _apply_detector_noise(self, mov_feats: np.ndarray,
                              mov_mask: np.ndarray) -> np.ndarray:
        """Bernoulli-drop a fraction of the per-movement halting /
        vehicle counts. Per-movement (not per-vehicle, which we don't
        have addressable handles for at this layer): each movement
        row's queue + vehicle counts are scaled by a sampled
        keep-fraction. waiting_time is left as-is -- a missed-detection
        camera wouldn't accumulate wait on the missed cars but it also
        wouldn't fabricate wait; leaving it alone biases the noise
        toward queue + vehicle counts, which dominate the FRAP signal."""
        fn = self._current_noise["fn_rate"]
        if fn <= 0.0:
            return mov_feats
        out = mov_feats.copy()
        # Sample keep-fraction per real movement; pads get untouched
        # (and are 0 anyway, so the scaling is a no-op).
        n_tls, m_max = mov_feats.shape[:2]
        # Per-movement keep mask via Bernoulli on each "vehicle slot"
        # would require per-vehicle handles; instead we scale by a
        # uniform[1-fn, 1] keep-fraction per movement row, which has
        # the same expected drop rate over the rollout while being
        # cheap per step.
        keep = 1.0 - self._dr_rng.uniform(0.0, fn,
                                          size=(n_tls, m_max))
        keep = np.where(mov_mask, keep, 1.0).astype(np.float32)
        # Apply to queue (col 0) and vehicle count (col 1); leave
        # waiting time (col 2) alone (see docstring above).
        out[..., 0] = out[..., 0] * keep
        out[..., 1] = out[..., 1] * keep
        return out

    def get_state_frap_batch(self) -> dict:
        batch = super().get_state_frap_batch()
        if self._current_noise["fn_rate"] > 0.0:
            batch["movement_features"] = self._apply_detector_noise(
                batch["movement_features"], batch["movement_mask"])
        return batch


def make_dr_env_from_config(env_kwargs: dict,
                            dr_kwargs: Optional[dict] = None
                            ) -> DRWrapper:
    """Convenience builder used by the trainer's --randomize path."""
    return DRWrapper(**env_kwargs, **(dr_kwargs or {}))
