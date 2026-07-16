"""CLIP-style contrastive alignment into a shared motion-language manifold.

Motion embeddings (from the ST-GCN encoder) and language embeddings (from the
text/speech encoder) live in different spaces with different dimensionality. We
project each into a common ``latent_dim`` space, L2-normalise onto the unit
hypersphere, and align them with a symmetric InfoNCE (a.k.a. NT-Xent) loss.

Given a batch of ``N`` paired examples with unit-norm projections
``z^m_i`` (motion) and ``z^l_i`` (language), define logits

    S_{ij} = <z^m_i, z^l_j> / tau .

The loss is the average of two cross-entropies where the correct match for row
``i`` is column ``i``:

    L = 1/2 [ CE(softmax_row(S), I) + CE(softmax_col(S), I) ] .

At the optimum, paired motion/language embeddings are mutually nearest
neighbours on the sphere -- i.e. English meaning and 3D motion occupy the same
latent manifold, which is exactly the property the generator conditions on.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection to the shared manifold followed by L2 normalisation."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None,
                 normalize: bool = True) -> None:
        super().__init__()
        hidden = hidden or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.normalize:
            z = F.normalize(z, dim=-1, eps=1e-8)
        return z


def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor,
                  temperature: float = 0.07) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric InfoNCE loss for L2-normalised embeddings.

    Args:
        z_a, z_b: (N, D) unit-norm embeddings of paired examples.
        temperature: softmax temperature ``tau`` (> 0).

    Returns:
        (loss, logits) where ``logits`` is the (N, N) similarity matrix / tau.
    """
    if z_a.shape != z_b.shape:
        raise ValueError("paired embeddings must share shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    n = z_a.shape[0]
    logits = (z_a @ z_b.t()) / temperature  # (N, N)
    targets = torch.arange(n, device=z_a.device)
    loss_a = F.cross_entropy(logits, targets)       # match motion->language
    loss_b = F.cross_entropy(logits.t(), targets)   # match language->motion
    return 0.5 * (loss_a + loss_b), logits


class ContrastiveAligner(nn.Module):
    """Two projection heads + a learnable log-temperature (CLIP convention)."""

    def __init__(self, motion_dim: int, language_dim: int, latent_dim: int,
                 init_temperature: float = 0.07,
                 max_log_scale: float = 4.6052) -> None:  # ln(100), CLIP clamp
        super().__init__()
        self.motion_head = ProjectionHead(motion_dim, latent_dim)
        self.language_head = ProjectionHead(language_dim, latent_dim)
        # Parameterise as log(1/temperature) = log-scale for stable optimisation.
        init_log_scale = torch.log(torch.tensor(1.0 / init_temperature))
        self.log_scale = nn.Parameter(init_log_scale)
        self.max_log_scale = max_log_scale

    def encode(self, motion_feat: torch.Tensor,
               language_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.motion_head(motion_feat), self.language_head(language_feat)

    def forward(self, motion_feat: torch.Tensor,
                language_feat: torch.Tensor) -> dict:
        z_m, z_l = self.encode(motion_feat, language_feat)
        scale = self.log_scale.clamp(max=self.max_log_scale).exp()
        temperature = 1.0 / scale
        loss, logits = info_nce_loss(z_m, z_l, temperature=float(temperature.detach()))
        # Recompute logits with the differentiable scale so the temperature trains.
        logits = scale * (z_m @ z_l.t())
        targets = torch.arange(z_m.shape[0], device=z_m.device)
        loss = 0.5 * (F.cross_entropy(logits, targets)
                      + F.cross_entropy(logits.t(), targets))
        return {"loss": loss, "logits": logits, "z_motion": z_m, "z_language": z_l}
