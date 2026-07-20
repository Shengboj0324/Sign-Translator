"""Part-aware noise schedules and loss weights (docs/DIFFUSION_GEN.md §5).

The document asks for part-aware noise for hands/face. We provide:

* **Per-part loss weights** ``λ_p`` — up-weight semantically critical, high-frequency
  parts (hands/face) so their reconstruction error dominates.
* **Per-part noise schedules** — any monotone ``ᾱ: 1→0`` is a valid forward. We
  scale a base schedule per part.
* **Innovation — SNR capacity allocation** — give a part a schedule that retains
  more signal (higher SNR) at each ``t`` by raising ``ᾱ`` toward 1 with an exponent
  ``κ_p ≤ 1`` (``ᾱ^{κ}`` with ``ᾱ∈(0,1)`` is ≥ ``ᾱ`` and still monotone), so the
  denoiser resolves that part more precisely.
"""

from __future__ import annotations

from typing import Dict

import torch

from .schedule import NoiseSchedule


def part_loss_weights(part_slices: Dict[str, slice], weights: Dict[str, float],
                      total_dim: int, dtype=torch.get_default_dtype()) -> torch.Tensor:
    """Build a ``(total_dim,)`` per-channel weight vector from per-part weights."""
    w = torch.ones(total_dim, dtype=dtype)
    for name, sl in part_slices.items():
        w[sl] = weights.get(name, 1.0)
    return w


def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 channel_weights: torch.Tensor) -> torch.Tensor:
    """Per-channel-weighted MSE over the last (channel) axis.

    ``pred``/``target`` (..., C); ``channel_weights`` (C,). Weights are normalised
    to mean 1 so the overall loss scale is comparable to unweighted MSE.
    """
    w = channel_weights / channel_weights.mean()
    return (((pred - target) ** 2) * w).mean()


class PartAwareSchedule:
    """A base schedule plus a per-part SNR exponent ``κ_p`` (≤ 1 keeps more signal)."""

    def __init__(self, base: NoiseSchedule, part_kappa: Dict[str, float]) -> None:
        self.base = base
        self.part_kappa = part_kappa
        # precompute per-part alpha_bar = base_alpha_bar ** kappa (monotone, in (0,1])
        self.part_alpha_bar: Dict[str, torch.Tensor] = {}
        for name, kappa in part_kappa.items():
            if kappa <= 0:
                raise ValueError("kappa must be > 0")
            self.part_alpha_bar[name] = base.alpha_bar ** kappa

    def part_a_b(self, name: str, t: torch.Tensor):
        """(a, b) = (√ᾱ^κ, √(1−ᾱ^κ)) for a part at timesteps ``t``."""
        ab = self.part_alpha_bar[name].to(t.device)[t]
        return torch.sqrt(ab), torch.sqrt(1.0 - ab)

    def part_snr(self, name: str, t: torch.Tensor) -> torch.Tensor:
        ab = self.part_alpha_bar[name].to(t.device)[t]
        return ab / (1.0 - ab)

    def is_valid(self, name: str) -> bool:
        """A valid forward: ᾱ monotone non-increasing and within (0, 1]."""
        ab = self.part_alpha_bar[name]
        return bool(torch.all(ab[1:] <= ab[:-1] + 1e-12) and ab.max() <= 1.0
                    and ab.min() >= 0.0)
