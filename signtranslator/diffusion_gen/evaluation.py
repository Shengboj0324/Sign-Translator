"""Evaluation metrics for diffusion motion generation (docs/DIFFUSION_GEN.md §10).

* ``multimodality`` — mean pairwise distance among samples for one condition
  (0 for a deterministic model, > 0 for a stochastic one).
* **Innovation — ``semantic_preservation_verified_multimodality``** — diversity
  measured ONLY over samples that pass a meaning check, directly answering the
  document's "demonstrate that stochastic samples preserve meaning".
* ``jerk`` — mean magnitude of the third temporal difference (smoothness).
* Collision / contact reuse Docs 04-05; ``p95_generation_time`` a latency
  percentile; ``semantic_accuracy_diagnostic`` a flagged diagnostic only.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import torch


def multimodality(samples: torch.Tensor) -> torch.Tensor:
    """Mean pairwise L2 distance among ``K`` samples. ``samples`` (K, ...)."""
    K = samples.shape[0]
    if K < 2:
        return samples.new_zeros(())
    flat = samples.reshape(K, -1)
    d = torch.cdist(flat, flat)                              # (K, K)
    iu = torch.triu_indices(K, K, offset=1)
    return d[iu[0], iu[1]].mean()


def semantic_preservation_verified_multimodality(
        samples: torch.Tensor, preserved: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Diversity restricted to meaning-preserving samples.

    ``preserved`` (K,) bool marks samples whose meaning check passed. Returns
    (diversity over the preserved subset, fraction preserved). A model that is
    diverse *and* meaning-preserving scores high diversity with a high fraction.
    """
    keep = samples[preserved.bool()]
    frac = float(preserved.float().mean())
    return multimodality(keep), frac


def jerk(x: torch.Tensor, time_dim: int = -2) -> torch.Tensor:
    """Mean magnitude of the third temporal difference (Δ³x). Lower = smoother.

    ``x`` (..., T, D); Δ³x_t = x_{t+3} − 3x_{t+2} + 3x_{t+1} − x_t.
    """
    x = x.movedim(time_dim, -2)
    d3 = x[..., 3:, :] - 3 * x[..., 2:-1, :] + 3 * x[..., 1:-2, :] - x[..., :-3, :]
    return d3.abs().mean()


def p95_generation_time(times: Sequence[float]) -> float:
    """95th-percentile of a list of per-sample generation times (seconds)."""
    if not times:
        return 0.0
    s = sorted(times)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return float(s[idx])


def semantic_accuracy_diagnostic(pred_labels: torch.Tensor,
                                 target_labels: torch.Tensor) -> float:
    """DIAGNOSTIC ONLY (per the document): fraction of generated motions whose
    recognised label matches the target. A recognizer cycle score can reward
    adversarial/unnatural motion, so this must never be a primary training loss.
    """
    return float((pred_labels == target_labels).float().mean())


def compare_generators(diversity_by_model: dict) -> dict:
    """Honest comparison scaffold: given {model_name: samples (K, ...)}, report the
    multimodality of each. A deterministic model has ~0; diffusion has > 0."""
    return {name: float(multimodality(s)) for name, s in diversity_by_model.items()}
