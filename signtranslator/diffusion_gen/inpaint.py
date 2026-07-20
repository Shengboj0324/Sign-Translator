"""Diffusion inpainting for streaming overlap and user correction (§6).

RePaint-style (Lugmayr et al.): the **known** region is overwritten at every
reverse step with the forward-diffused ground truth, while the **unknown** region
is sampled by the model:

    x_{t−1}^known = √ᾱ_{t−1} x₀^known + √(1−ᾱ_{t−1}) ε,
    x_{t−1}       = m ⊙ x_{t−1}^known + (1−m) ⊙ x_{t−1}^sampled,

with ``m = 1`` on known frames/joints. The known region therefore always equals
the forward-diffused ground truth, and at ``t=0`` it equals ``x₀^known`` exactly —
so a streaming overlap (previous chunk's tail) or a user's fixed keyframes are
preserved while the rest is generated to be coherent with them.
"""

from __future__ import annotations

from typing import Optional

import torch

from .schedule import NoiseSchedule


def forward_diffuse_known(schedule: NoiseSchedule, x0_known: torch.Tensor,
                         t: torch.Tensor,
                         eps: Optional[torch.Tensor] = None) -> torch.Tensor:
    """√ᾱ_t x₀ + √(1−ᾱ_t) ε for the known region (ε=0 gives the noise-free anchor)."""
    if eps is None:
        eps = torch.zeros_like(x0_known)
    return schedule.q_sample(x0_known, t, eps)


def merge_known_unknown(x_known: torch.Tensor, x_sampled: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """m ⊙ x_known + (1−m) ⊙ x_sampled. ``mask`` broadcasts to the data shape."""
    m = mask.to(x_known.dtype)
    return m * x_known + (1.0 - m) * x_sampled


def inpaint_step(schedule: NoiseSchedule, x_t: torch.Tensor, t: torch.Tensor,
                 x0_known: torch.Tensor, mask: torch.Tensor,
                 x_prev_sampled: torch.Tensor,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """One RePaint reverse step: replace known region with forward-diffused GT.

    ``x_prev_sampled`` is the model's proposed ``x_{t−1}`` for the unknown region.
    ``t`` is the *destination* timestep (t-1 index); at ``t=0`` no noise is added so
    the known region equals ``x₀^known`` exactly.
    """
    if int(t.min()) == 0:
        x_known = x0_known                                   # exact anchor at t=0
    else:
        eps = torch.randn(x0_known.shape, generator=generator,
                          dtype=x0_known.dtype, device=x0_known.device)
        x_known = schedule.q_sample(x0_known, t, eps)
    return merge_known_unknown(x_known, x_prev_sampled, mask)


def streaming_overlap_mask(num_frames: int, overlap: int, data_shape,
                           dtype=torch.get_default_dtype()) -> torch.Tensor:
    """Mask that fixes the first ``overlap`` frames (a previous chunk's tail).

    Returns a (num_frames, *feat) mask that is 1 on the overlap frames, 0 elsewhere
    -- so the diffusion regenerates the new frames coherent with the fixed overlap.
    """
    mask = torch.zeros(num_frames, *data_shape, dtype=dtype)
    mask[:overlap] = 1.0
    return mask
