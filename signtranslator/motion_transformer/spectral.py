"""Anti-oversmoothing diagnostics (docs/MOTION_TRANSFORMER.md §5).

The document warns that velocity loss alone can favour small motion; the remedy is
to *measure* — and (our innovation) to *optimise against* — lost high-frequency
energy per body part.

* ``power_spectrum`` / ``parseval_energy`` — the one-sided power spectrum and the
  exact Parseval identity ``Σ_t x_t² = (1/N) Σ_k w_k |X_k|²`` (verified).
* ``band_energy`` — energy in low/mid/high frequency bands.
* ``spectral_energy_matching_loss`` — a differentiable penalty on the per-part,
  per-band energy gap between prediction and reference (INNOVATION beyond the
  document's "report it").
* ``duration_calibration_error`` — reliability of predicted event durations.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _parseval_weights(n: int, device, dtype) -> torch.Tensor:
    """One-sided-spectrum weights so ``(w·|X|²).sum()/N == Σ x²`` (Parseval)."""
    m = n // 2 + 1
    w = torch.full((m,), 2.0, device=device, dtype=dtype)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0                                          # Nyquist bin not doubled
    return w


def power_spectrum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """|rFFT(x)|² along ``dim`` (one-sided). (..., T) -> (..., T//2+1)."""
    X = torch.fft.rfft(x, dim=dim)
    return (X.real ** 2 + X.imag ** 2)


def parseval_energy(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Total time-domain energy Σ_t x_t² computed from the spectrum (exact)."""
    n = x.shape[dim]
    p = power_spectrum(x, dim=dim)
    w = _parseval_weights(n, x.device, x.dtype)
    shape = [1] * p.dim(); shape[dim] = w.shape[0]
    return (p * w.reshape(shape)).sum(dim=dim) / n


def band_energy(x: torch.Tensor, num_bands: int = 3, dim: int = -1) -> torch.Tensor:
    """Energy in ``num_bands`` contiguous frequency bands. (..., T) -> (..., num_bands).

    Bands partition the one-sided spectrum bins evenly (low..high). Each band's
    energy uses the Parseval weights so the bands sum to the total energy.
    """
    n = x.shape[dim]
    p = power_spectrum(x, dim=dim)                           # (..., F)
    w = _parseval_weights(n, x.device, x.dtype)
    p = p.movedim(dim, -1) * w                               # (..., F), weighted
    F = p.shape[-1]
    edges = torch.linspace(0, F, num_bands + 1).round().long().tolist()
    bands = [p[..., edges[b]:edges[b + 1]].sum(-1) for b in range(num_bands)]
    return torch.stack(bands, dim=-1) / n                    # (..., num_bands)


def spectral_energy_by_part(x: torch.Tensor, part_slices: Dict[str, slice],
                            num_bands: int = 3) -> Dict[str, torch.Tensor]:
    """Per-part band energy. ``x`` (N, C, T); returns {part: (N, num_bands)}."""
    out = {}
    for name, sl in part_slices.items():
        xe = band_energy(x[:, sl, :], num_bands=num_bands, dim=-1)   # (N, Cpart, bands)
        out[name] = xe.mean(dim=1)                          # mean over the part's channels
    return out


def spectral_energy_matching_loss(pred: torch.Tensor, real: torch.Tensor,
                                  part_slices: Optional[Dict[str, slice]] = None,
                                  num_bands: int = 3) -> torch.Tensor:
    """Σ_{part,band} |E_band(pred) − E_band(real)| (differentiable). (N, C, T)."""
    if part_slices is None:
        part_slices = {"all": slice(0, pred.shape[1])}
    ep = spectral_energy_by_part(pred, part_slices, num_bands)
    er = spectral_energy_by_part(real, part_slices, num_bands)
    loss = pred.new_zeros(())
    for name in part_slices:
        loss = loss + (ep[name] - er[name]).abs().mean()
    return loss


def duration_calibration_error(pred: torch.Tensor, true: torch.Tensor,
                               num_buckets: int = 10) -> torch.Tensor:
    """Reliability of predicted durations: bucket by predicted value and average
    |mean(pred) − mean(true)| weighted by bucket count. 0 == perfectly calibrated.
    """
    if pred.shape != true.shape:
        raise ValueError("pred and true must share shape")
    pred = pred.reshape(-1).to(torch.float64)
    true = true.reshape(-1).to(torch.float64)
    lo, hi = float(pred.min()), float(pred.max())
    if hi - lo < 1e-12:
        return (pred.mean() - true.mean()).abs()
    edges = torch.linspace(lo, hi, num_buckets + 1)
    total = pred.new_zeros(())
    n = pred.numel()
    for b in range(num_buckets):
        left, right = edges[b], edges[b + 1]
        m = (pred >= left) & (pred <= right if b == num_buckets - 1 else pred < right)
        if m.any():
            gap = (pred[m].mean() - true[m].mean()).abs()
            total = total + gap * m.sum() / n
    return total
