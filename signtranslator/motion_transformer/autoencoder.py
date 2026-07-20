"""Motion VQ-VAE autoencoder and the full motion loss (docs/MOTION_TRANSFORMER.md §4).

A temporal (1-D conv) encoder downsamples a pose-feature sequence to a latent
sequence, a (residual) vector quantiser tokenises it, and a transpose-conv decoder
reconstructs the motion. The loss combines a rotation-geodesic term, L1 velocity
and acceleration terms (to preserve high-frequency motion), an optional contact
term, and the quantiser commitment.

Motion features are 6D rotations laid out ``(N, C, T)`` with ``C = J·6``. The
geodesic term reshapes to ``(N, T, J, 6)`` and uses the Doc-04 SO(3) distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pose.rotations import rotation_6d_to_matrix, geodesic_distance
from .quantizer import VectorQuantizer
from .residual_vq import ResidualVQ


# ---------------------------------------------------------------------------
# temporal derivatives + loss terms
# ---------------------------------------------------------------------------
def velocity(x: torch.Tensor) -> torch.Tensor:
    """Δx along the time axis (assumed axis -1). (..., T) -> (..., T-1)."""
    return x[..., 1:] - x[..., :-1]


def acceleration(x: torch.Tensor) -> torch.Tensor:
    """Δ²x along the time axis. (..., T) -> (..., T-2)."""
    return x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]


def velocity_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (velocity(pred) - velocity(target)).abs().mean()


def acceleration_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (acceleration(pred) - acceleration(target)).abs().mean()


def geodesic_motion_loss(pred6d: torch.Tensor, target6d: torch.Tensor) -> torch.Tensor:
    """Mean SO(3) geodesic between predicted/target rotations. (N, T, J, 6)."""
    Rp = rotation_6d_to_matrix(pred6d)
    Rt = rotation_6d_to_matrix(target6d)
    return geodesic_distance(Rp, Rt).mean()


@dataclass
class MotionLossWeights:
    geodesic: float = 1.0
    velocity: float = 1.0
    acceleration: float = 1.0
    contact: float = 0.0
    commit: float = 1.0


def motion_loss(pred: torch.Tensor, target: torch.Tensor, num_joints: int,
                commit_loss: torch.Tensor,
                weights: MotionLossWeights = MotionLossWeights(),
                contact_loss: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
    """Assemble the motion loss. ``pred``/``target`` (N, C, T) with C = J·6."""
    N, C, T = pred.shape
    assert C == num_joints * 6, "channels must be num_joints*6 (6D rotations)"
    # (N, C, T) -> (N, T, J, 6) for the geodesic term
    p6 = pred.permute(0, 2, 1).reshape(N, T, num_joints, 6)
    t6 = target.permute(0, 2, 1).reshape(N, T, num_joints, 6)
    geo = geodesic_motion_loss(p6, t6)
    vel = velocity_l1(pred, target)
    acc = acceleration_l1(pred, target)
    total = (weights.geodesic * geo + weights.velocity * vel
             + weights.acceleration * acc + weights.commit * commit_loss)
    terms = {"geodesic": geo, "velocity": vel, "acceleration": acc,
             "commit": commit_loss}
    if contact_loss is not None and weights.contact > 0:
        total = total + weights.contact * contact_loss
        terms["contact"] = contact_loss
    terms["total"] = total
    return terms


# ---------------------------------------------------------------------------
# temporal encoder / decoder
# ---------------------------------------------------------------------------
class TemporalEncoder(nn.Module):
    """Strided 1-D conv stack; downsamples time by 2**num_downsamples."""

    def __init__(self, in_channels: int, dim: int, num_downsamples: int = 2) -> None:
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(num_downsamples):
            layers += [nn.Conv1d(c, dim, kernel_size=4, stride=2, padding=1),
                       nn.GELU()]
            c = dim
        layers.append(nn.Conv1d(dim, dim, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalDecoder(nn.Module):
    """Transpose-conv stack; upsamples time by 2**num_upsamples (inverse of encoder)."""

    def __init__(self, dim: int, out_channels: int, num_upsamples: int = 2) -> None:
        super().__init__()
        layers = []
        for _ in range(num_upsamples):
            layers += [nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1),
                       nn.GELU()]
        layers.append(nn.Conv1d(dim, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MotionVQVAE(nn.Module):
    """Encoder -> (residual) VQ -> decoder over motion-feature sequences."""

    def __init__(self, in_channels: int, dim: int = 128, num_codes: int = 256,
                 num_downsamples: int = 2, residual_stages: int = 1,
                 ema: bool = True) -> None:
        super().__init__()
        self.downsample = 2 ** num_downsamples
        self.encoder = TemporalEncoder(in_channels, dim, num_downsamples)
        self.decoder = TemporalDecoder(dim, in_channels, num_downsamples)
        self.quantizer = ResidualVQ(residual_stages, num_codes, dim, ema=ema)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``x`` (N, C, T), ``T`` divisible by the downsample factor."""
        if x.shape[-1] % self.downsample != 0:
            raise ValueError(f"T={x.shape[-1]} must be divisible by {self.downsample}")
        z_e = self.encoder(x)                                # (N, dim, T')
        q = self.quantizer(z_e.transpose(1, 2))              # quantise (N, T', dim)
        z_q = q["z_q"].transpose(1, 2)                       # (N, dim, T')
        recon = self.decoder(z_q)                            # (N, C, T)
        return recon, q

    @torch.no_grad()
    def init_codebook(self, x: torch.Tensor) -> None:
        z_e = self.encoder(x).transpose(1, 2).reshape(-1, self.encoder.net[-1].out_channels)
        self.quantizer.init_from_data(z_e)
