"""Multi-scale temporal pyramid and confidence-aware masked temporal conv.

See docs/HAND_GRAPH.md §6-7. Signing mixes rapid finger articulation (short time
scale), transitions (medium), and holds (long). A **temporal pyramid** of parallel
dilated 1-D convolutions covers all three: for a branch with kernel ``k`` and
dilation ``d`` the (symmetric) receptive field is ``1 + (k-1) d``.

Occluded landmarks carry confidence 0. A **normalised (masked) convolution**

    y_t = Σ_τ w_τ c_{t+τ} x_{t+τ}  /  Σ_τ w_τ c_{t+τ}

lets an occluded frame contribute nothing and renormalises over the visible frames
in the window; an all-occluded window yields 0 (graceful fallback, no NaN).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPyramid(nn.Module):
    """Parallel dilated temporal convolutions summed into one output.

    Operates on ``(N, C, T)``. Each branch preserves length via symmetric padding
    ``(k-1)//2 * d``. The pyramid receptive field is the max branch receptive
    field ``1 + (k-1) * max(dilations)``.
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilations: Sequence[int] = (1, 2, 4), residual: bool = True) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding")
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size,
                      padding=((kernel_size - 1) // 2) * d, dilation=d)
            for d in self.dilations
        ])
        self.residual = residual

    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * max(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = sum(branch(x) for branch in self.branches)
        if self.residual:
            out = out + x
        return out


def masked_normalized_conv1d(x: torch.Tensor, weight: torch.Tensor,
                             conf: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Confidence-normalised depthwise temporal smoothing.

    ``x`` (N, C, T); ``weight`` (K,) a non-negative smoothing kernel;
    ``conf`` (N, T) in [0, 1]. Returns (N, C, T):

        y = conv(x * c, w) / conv(c, w)

    Occluded frames (c=0) contribute nothing; an all-occluded window -> 0.
    """
    N, C, T = x.shape
    K = weight.shape[0]
    pad = (K - 1) // 2
    w = weight.view(1, 1, K).to(x.dtype)
    c = conf.view(N, 1, T).to(x.dtype)
    num = F.conv1d((x * c), w.expand(C, 1, K), padding=pad, groups=C)   # (N,C,T)
    den = F.conv1d(c, w, padding=pad)                                    # (N,1,T)
    return num / den.clamp_min(eps)


class MaskedTemporalConv(nn.Module):
    """A learnable-kernel version of the masked normalized conv (per channel).

    The kernel is passed through softplus to stay non-negative (so the
    normalisation denominator is a genuine visible-weight sum).
    """

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.raw = nn.Parameter(torch.zeros(kernel_size))   # softplus(0)=~0.69 uniform-ish

    def forward(self, x: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        w = F.softplus(self.raw)
        return masked_normalized_conv1d(x, w, conf)


def to_time_series(x: torch.Tensor) -> torch.Tensor:
    """(N, C, T, V) -> (N*V, C, T) so temporal modules see each joint's series."""
    n, c, t, v = x.shape
    return x.permute(0, 3, 1, 2).reshape(n * v, c, t)


def from_time_series(y: torch.Tensor, n: int, v: int) -> torch.Tensor:
    """Inverse of :func:`to_time_series`: (N*V, C, T) -> (N, C, T, V)."""
    nv, c, t = y.shape
    return y.reshape(n, v, c, t).permute(0, 2, 3, 1)
