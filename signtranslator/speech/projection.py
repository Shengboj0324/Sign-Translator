"""Temporal resampling and gated projection into the planner width.

Implements the recommended design of ``01_speech_foundation_layer.md``:

    H~  = Resample(H^a)
    G   = sigma(W_g H~)
    H^p = G (*) W_1 H~ + (1 - G) (*) W_2 H~

The gate is a **learned per-dimension interpolation between two projections** --
not attention, not a residual. Two structural consequences follow and are
tested:

* *Convexity collapse.* If ``W_1 == W_2`` the output is that shared projection
  for **every** gate value, so it becomes independent of ``W_g``. A test that
  passes only because the gate happens to sit near 0.5 would miss a broken
  gate; this identity does not.
* *Elementwise convex bound.* Since ``G in (0,1)`` componentwise, every output
  element lies between the two projections' corresponding elements. The layer
  can therefore never extrapolate beyond what either projection produces.

The three pathways the specification requires be *retained* -- lexical
posterior, acoustic embedding, prosody -- are kept as separate fields of
:class:`SpeechPathways` rather than being summed into one tensor, so downstream
components can weigh (or ablate) them independently. The acceptance criteria
call for exactly that ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResampler(nn.Module):
    """Resample a ``(B, T, D)`` sequence along time.

    ``mode="linear"`` interpolates to a target length (or by a fixed ratio) and
    has no parameters -- appropriate when the encoder frame rate merely needs to
    be matched. ``mode="conv"`` learns a strided 1-D convolution, appropriate
    when the rate change should also be a feature transform.
    """

    def __init__(self, dim: int, mode: Literal["linear", "conv"] = "linear",
                 ratio: float = 1.0, stride: int = 2,
                 kernel_size: int = 3) -> None:
        super().__init__()
        if mode not in {"linear", "conv"}:
            raise ValueError("mode must be 'linear' or 'conv'")
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        self.dim = dim
        self.mode = mode
        self.ratio = ratio
        self.stride = stride
        if mode == "conv":
            if stride < 1:
                raise ValueError("stride must be >= 1")
            self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                  stride=stride, padding=kernel_size // 2)

    def output_length(self, t_in: int) -> int:
        if self.mode == "conv":
            return (t_in + 2 * (self.conv.kernel_size[0] // 2)
                    - self.conv.kernel_size[0]) // self.stride + 1
        return max(1, int(round(t_in * self.ratio)))

    def forward(self, x: torch.Tensor,
                target_length: Optional[int] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("expected (B, T, D)")
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {x.shape[-1]}")
        h = x.transpose(1, 2)                                  # (B, D, T)
        if self.mode == "conv":
            if target_length is not None:
                raise ValueError("target_length is only valid for mode='linear'")
            h = self.conv(h)
        else:
            t_out = target_length if target_length is not None \
                else self.output_length(x.shape[1])
            if t_out != x.shape[1]:
                h = F.interpolate(h, size=t_out, mode="linear",
                                  align_corners=True)
        return h.transpose(1, 2).contiguous()


class GatedProjection(nn.Module):
    """``H^p = G (*) W_1 H~ + (1 - G) (*) W_2 H~`` with ``G = sigma(W_g H~)``.

    All three maps take ``in_dim -> out_dim``: the gate must share the *output*
    width so the elementwise products are well defined.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w1 = nn.Linear(in_dim, out_dim, bias=bias)
        self.w2 = nn.Linear(in_dim, out_dim, bias=bias)
        self.wg = nn.Linear(in_dim, out_dim, bias=bias)

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        """The mixing coefficients ``G``, strictly inside (0, 1)."""
        return torch.sigmoid(self.wg(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        return g * self.w1(x) + (1.0 - g) * self.w2(x)


@dataclass
class SpeechPathways:
    """The three pathways, deliberately kept separate.

    Attributes:
        acoustic: ``(B, T_p, d_p)`` projected acoustic states ``H^p``.
        prosody: ``(B, T_p, n_prosody)`` pitch/energy/voicing/aperiodicity,
            resampled onto the same grid. Per the specification this is
            *conditioning evidence* for discourse and affect, not a
            deterministic mapping to any non-manual marker.
        lexical: optional ``(B, T_l, V)`` token posteriors from the ASR head.
            ``None`` until the recognition stage is wired in (stage 2).
    """

    acoustic: torch.Tensor
    prosody: Optional[torch.Tensor] = None
    lexical: Optional[torch.Tensor] = None

    def as_fused(self) -> torch.Tensor:
        """Concatenate acoustic and prosody along the feature axis.

        Provided for the "fused" arm of the required transcript-only /
        acoustic-only / fused ablation. Callers wanting a single tensor must ask
        for it explicitly -- fusing by default would make the ablation
        impossible to run.
        """
        if self.prosody is None:
            return self.acoustic
        if self.prosody.shape[:2] != self.acoustic.shape[:2]:
            raise ValueError("prosody and acoustic must share (batch, time)")
        return torch.cat([self.acoustic, self.prosody], dim=-1)


class SpeechProjector(nn.Module):
    """Resample encoder states, gate-project them, and carry prosody alongside."""

    def __init__(self, encoder_dim: int, planner_dim: int,
                 resample_mode: Literal["linear", "conv"] = "linear",
                 ratio: float = 1.0, stride: int = 2) -> None:
        super().__init__()
        self.resampler = TemporalResampler(encoder_dim, mode=resample_mode,
                                           ratio=ratio, stride=stride)
        self.projection = GatedProjection(encoder_dim, planner_dim)

    def forward(self, acoustic_states: torch.Tensor,
                prosody: Optional[torch.Tensor] = None,
                lexical: Optional[torch.Tensor] = None,
                target_length: Optional[int] = None) -> SpeechPathways:
        kwargs = {} if self.resampler.mode == "conv" else {"target_length": target_length}
        h_tilde = self.resampler(acoustic_states, **kwargs)
        h_p = self.projection(h_tilde)
        if prosody is not None:
            if prosody.dim() != 3:
                raise ValueError("prosody must be (B, T, C)")
            if prosody.shape[1] != h_p.shape[1]:
                prosody = F.interpolate(prosody.transpose(1, 2),
                                        size=h_p.shape[1], mode="linear",
                                        align_corners=True).transpose(1, 2)
        return SpeechPathways(acoustic=h_p, prosody=prosody, lexical=lexical)
