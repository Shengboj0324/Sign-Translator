"""Numerically-certified optimization gate (Doc-13 §5).

Every optimization transform (FP16, INT8, a chunked-attention kernel, a distilled
model) must pass BOTH a numerical-equivalence gate (vs eager execution) AND a
quality-non-regression gate (a Doc-12 contract) or be rejected. Online-softmax is
the worked chunked-attention example: blockwise softmax equals full softmax exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..eval_framework.contracts import Contract, Direction


# ---------------------------------------------------------------------------
# numerical equivalence
# ---------------------------------------------------------------------------
def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return float((a - b).abs().max()) if a.numel() else 0.0


def max_rel_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    denom = a.abs().clamp_min(eps)
    return float(((a - b).abs() / denom).max()) if a.numel() else 0.0


@dataclass(frozen=True)
class TransformCertificate:
    max_abs: float
    max_rel: float
    numerically_equivalent: bool
    quality_preserved: bool

    @property
    def accepted(self) -> bool:
        """A transform is accepted iff it passes BOTH gates."""
        return self.numerically_equivalent and self.quality_preserved


def certify_optimization(eager_out: torch.Tensor, optimized_out: torch.Tensor,
                         quality_eager: float, quality_optimized: float,
                         atol: float = 1e-3, rtol: float = 1e-2,
                         quality_tolerance: float = 0.0,
                         quality_higher_is_better: bool = True) -> TransformCertificate:
    """Gate an optimization on numerical equivalence AND quality non-regression."""
    ma = max_abs_error(eager_out, optimized_out)
    mr = max_rel_error(eager_out, optimized_out)
    numeric = torch.allclose(eager_out, optimized_out, atol=atol, rtol=rtol)
    # quality non-regression expressed as a Doc-12 falsifiable contract.
    if quality_higher_is_better:
        threshold = quality_eager - quality_tolerance
        direction = Direction.GE
    else:
        threshold = quality_eager + quality_tolerance
        direction = Direction.LE
    quality_contract = Contract(
        "quality_non_regression", "distribution", quality_optimized, threshold,
        direction, caveat="embedding choice can bias results")
    return TransformCertificate(ma, mr, numeric, quality_contract.passed)


# ---------------------------------------------------------------------------
# online softmax (chunked-attention exactness)
# ---------------------------------------------------------------------------
def online_softmax(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Blockwise softmax with a running max + rescaled running sum.

    Equals the full softmax exactly (to float rounding): the FlashAttention identity.
    """
    n = x.shape[0]
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    m = torch.tensor(float("-inf"), dtype=x.dtype)
    ell = torch.zeros((), dtype=x.dtype)
    for s in range(0, n, block_size):
        blk = x[s:s + block_size]
        m_new = torch.maximum(m, blk.max())
        correction = torch.exp(m - m_new) if torch.isfinite(m) else torch.zeros((), dtype=x.dtype)
        ell = ell * correction + torch.exp(blk - m_new).sum()
        m = m_new
    return torch.exp(x - m) / ell


def online_softmax_attention(scores: torch.Tensor, values: torch.Tensor,
                             block_size: int) -> torch.Tensor:
    """FlashAttention-style tiled attention output; equals softmax(scores) @ values.

    ``scores`` (n,), ``values`` (n, d). Maintains running (max m, sum ell, output o)
    and rescales the accumulator by exp(m_old - m_new) per block.
    """
    n, d = values.shape
    if scores.shape[0] != n:
        raise ValueError("scores and values must align on n")
    m = torch.tensor(float("-inf"), dtype=scores.dtype)
    ell = torch.zeros((), dtype=scores.dtype)
    o = torch.zeros(d, dtype=values.dtype)
    for s in range(0, n, block_size):
        sc = scores[s:s + block_size]
        v = values[s:s + block_size]
        m_new = torch.maximum(m, sc.max())
        correction = torch.exp(m - m_new) if torch.isfinite(m) else torch.zeros((), dtype=scores.dtype)
        p = torch.exp(sc - m_new)                      # (b,)
        ell = ell * correction + p.sum()
        o = o * correction + (p.unsqueeze(-1) * v).sum(0)
        m = m_new
    return o / ell
