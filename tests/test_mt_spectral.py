"""Verification of anti-oversmoothing diagnostics.

Proves the Parseval identity exactly (even and odd length), band energy summing to
total, that smoothing lowers high-frequency energy (oversmoothing detection), the
spectral-energy-matching loss (zero at equality, positive under smoothing), and
duration calibration.
"""

import pytest
import torch

from signtranslator.motion_transformer.spectral import (
    power_spectrum, parseval_energy, band_energy, spectral_energy_by_part,
    spectral_energy_matching_loss, duration_calibration_error,
)


# ---------------------------------------------------------------------------
# Parseval
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [16, 17, 64, 65])
def test_parseval_identity_exact(T):
    x = torch.randn(3, T, dtype=torch.float64)
    time_energy = (x ** 2).sum(-1)
    freq_energy = parseval_energy(x, dim=-1)
    assert torch.allclose(time_energy, freq_energy, atol=1e-9)


def test_band_energy_sums_to_total():
    x = torch.randn(4, 32, dtype=torch.float64)
    be = band_energy(x, num_bands=3, dim=-1)
    total = parseval_energy(x, dim=-1)
    assert torch.allclose(be.sum(-1), total, atol=1e-9)


# ---------------------------------------------------------------------------
# oversmoothing detection
# ---------------------------------------------------------------------------
def test_smoothing_lowers_high_frequency_energy():
    torch.manual_seed(0)
    T = 128
    x = torch.randn(1, T, dtype=torch.float64)              # broadband signal
    # moving-average smooth (low-pass) -> should kill high-band energy
    k = torch.ones(1, 1, 5, dtype=torch.float64) / 5.0
    xs = torch.nn.functional.conv1d(x.view(1, 1, T), k, padding=2).view(1, T)
    hi_orig = band_energy(x, num_bands=3)[..., 2]
    hi_smooth = band_energy(xs, num_bands=3)[..., 2]
    assert float(hi_smooth) < float(hi_orig)                # oversmoothing is visible


# ---------------------------------------------------------------------------
# spectral energy matching loss (innovation)
# ---------------------------------------------------------------------------
def test_spectral_matching_zero_at_equality_positive_under_smoothing():
    torch.manual_seed(1)
    real = torch.randn(2, 6, 64, dtype=torch.float64)
    assert spectral_energy_matching_loss(real, real.clone()).item() < 1e-9
    # smoothed prediction (low high-freq energy) -> positive matching loss
    k = torch.ones(6, 1, 7, dtype=torch.float64) / 7.0
    pred = torch.nn.functional.conv1d(real, k, padding=3, groups=6)
    assert spectral_energy_matching_loss(pred, real).item() > 0


def test_spectral_matching_is_differentiable():
    real = torch.randn(2, 4, 32, dtype=torch.float64)
    pred = torch.randn(2, 4, 32, dtype=torch.float64, requires_grad=True)
    spectral_energy_matching_loss(pred, real).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_spectral_energy_by_part_splits_channels():
    x = torch.randn(2, 9, 32, dtype=torch.float64)
    parts = {"torso": slice(0, 3), "hands": slice(3, 9)}
    out = spectral_energy_by_part(x, parts, num_bands=3)
    assert set(out) == {"torso", "hands"}
    assert out["torso"].shape == (2, 3) and out["hands"].shape == (2, 3)


# ---------------------------------------------------------------------------
# duration calibration
# ---------------------------------------------------------------------------
def test_duration_calibration_zero_when_perfect():
    d = torch.linspace(1, 10, 40)
    assert duration_calibration_error(d, d.clone()).item() < 1e-9


def test_duration_calibration_positive_when_biased():
    pred = torch.linspace(1, 10, 40)
    true = pred + 2.0                                        # systematically longer
    assert duration_calibration_error(pred, true).item() > 1.0


def test_duration_calibration_validates_shape():
    with pytest.raises(ValueError):
        duration_calibration_error(torch.zeros(3), torch.zeros(4))
