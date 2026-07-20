"""Classifier-free guidance (docs/DIFFUSION_GEN.md §3).

Ho & Salimans (arXiv:2207.12598): train with condition dropout so the network
learns both the conditional and unconditional predictions, then at sampling

    ε̂ = (1+w) ε_θ(x_t, c) − w ε_θ(x_t, ∅)
       = ε_θ(x_t, ∅) + (1+w)(ε_θ(x_t, c) − ε_θ(x_t, ∅)).

``w=0`` recovers the conditional model; the combination is a convex extrapolation
and is **parameterization-equivariant** (the same formula in ε-, x₀-, or v-space
gives the corresponding guided quantity). **Innovation — guidance annealing:** a
per-step schedule ``w(t)`` (higher early for semantics, lower late for diversity /
naturalness) preserves meaning while keeping multimodality.
"""

from __future__ import annotations

from typing import Optional

import torch


def drop_condition_mask(batch: int, p_uncond: float,
                        generator: Optional[torch.Generator] = None,
                        device=None) -> torch.Tensor:
    """(batch,) bool mask; True = drop the condition (use ∅) for that sample."""
    return torch.rand(batch, generator=generator, device=device) < p_uncond


def apply_condition_dropout(cond: torch.Tensor, null_cond: torch.Tensor,
                            drop_mask: torch.Tensor) -> torch.Tensor:
    """Replace dropped samples' conditioning with the learned null embedding.

    ``cond`` (B, ...), ``null_cond`` (...,) or (1, ...), ``drop_mask`` (B,) bool.
    """
    m = drop_mask.reshape((-1,) + (1,) * (cond.dim() - 1)).to(cond.dtype)
    null = null_cond.expand_as(cond)
    return (1 - m) * cond + m * null


def classifier_free_guidance(pred_cond: torch.Tensor, pred_uncond: torch.Tensor,
                             w: float) -> torch.Tensor:
    """ε̂ = (1+w) ε_c − w ε_∅  (works identically in ε/x₀/v space)."""
    return pred_uncond + (1.0 + w) * (pred_cond - pred_uncond)


def guidance_weight_schedule(t: torch.Tensor, num_timesteps: int,
                             w_high: float = 4.0, w_low: float = 0.5) -> torch.Tensor:
    """Innovation: anneal the guidance weight over the sampling trajectory.

    ``t`` near ``num_timesteps`` (early, noisy) -> ``w_high`` (lock in semantics);
    ``t`` near 0 (late, clean) -> ``w_low`` (preserve diversity/naturalness). Linear
    in ``t``. Returns a per-sample weight tensor.
    """
    frac = t.to(torch.get_default_dtype()) / max(1, num_timesteps - 1)   # 0..1
    return w_low + (w_high - w_low) * frac
