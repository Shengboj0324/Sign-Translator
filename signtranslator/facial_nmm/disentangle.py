"""Linguistic vs affect/identity disentanglement (docs/FACIAL_NMM.md §6).

A raised brow is a yes/no-question marker, not "surprise": the linguistic-marker
representation ``z_ling`` must be independent of affect ``a`` and identity ``id``.
We enforce and **certify** this with the Doc-04 leakage-probe method — a probe
cannot recover affect from ``z_ling`` (normalised error ~1), yet the SAME probe
recovers it when affect is folded in (~0), so the guard has power.

Enforcement uses a gradient-reversal adversary: an affect classifier is trained on
``z_ling`` while the reversed gradient pushes ``z_ling`` to be affect-invariant.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

# reuse the audited leakage probe from Doc-04
from ..pose.leakage import LinearProbe, normalised_recovery_error


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Identity forward; negates and scales the gradient backward (DANN)."""
    return _GradReverse.apply(x, lambd)


class AffectAdversary(nn.Module):
    """An affect classifier fed through a gradient-reversal layer.

    Minimising its loss trains it to read affect; the reversed gradient trains the
    upstream encoder to REMOVE affect from ``z_ling`` -- adversarial disentanglement.
    """

    def __init__(self, dim: int, num_affect: int, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = lambd
        self.fc = nn.Linear(dim, num_affect)

    def forward(self, z_ling: torch.Tensor) -> torch.Tensor:
        return self.fc(grad_reverse(z_ling, self.lambd))


def affect_leakage(z_ling: torch.Tensor, affect: torch.Tensor,
                   ntrain: int, l2: float = 1.0) -> float:
    """Normalised error of a linear probe recovering affect from ``z_ling``.

    ~1.0 means affect is NOT recoverable (disentangled); ~0.0 means it leaks.
    """
    probe = LinearProbe(l2=l2).fit(z_ling[:ntrain], affect[:ntrain])
    return normalised_recovery_error(probe.predict(z_ling[ntrain:]), affect[ntrain:])
