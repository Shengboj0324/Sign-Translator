"""Verification of part-aware schedules and loss weights.

Proves the per-channel weight construction, that up-weighting a part increases its
loss contribution, that per-part schedules remain valid forwards, and the SNR
capacity-allocation property (kappa<1 retains more signal / higher SNR).
"""

import pytest
import torch

from signtranslator.diffusion_gen.schedule import NoiseSchedule
from signtranslator.diffusion_gen.part_aware import (
    part_loss_weights, weighted_mse, PartAwareSchedule,
)


PARTS = {"torso": slice(0, 3), "hands": slice(3, 9), "face": slice(9, 11)}


# ---------------------------------------------------------------------------
# loss weights
# ---------------------------------------------------------------------------
def test_part_loss_weights_vector():
    w = part_loss_weights(PARTS, {"hands": 5.0, "face": 3.0}, total_dim=11)
    assert torch.allclose(w[0:3], torch.ones(3))             # torso default 1
    assert torch.allclose(w[3:9], torch.full((6,), 5.0))     # hands up-weighted
    assert torch.allclose(w[9:11], torch.full((2,), 3.0))


def test_weighted_mse_emphasises_upweighted_part():
    pred = torch.zeros(4, 11, dtype=torch.float64)
    target = torch.zeros(4, 11, dtype=torch.float64)
    target[:, 3:9] = 1.0                                     # error only on hands
    w_flat = part_loss_weights(PARTS, {}, 11, dtype=torch.float64)          # all 1
    w_hands = part_loss_weights(PARTS, {"hands": 10.0}, 11, dtype=torch.float64)
    assert weighted_mse(pred, target, w_hands) > weighted_mse(pred, target, w_flat)


def test_weighted_mse_reduces_to_plain_mse_when_uniform():
    pred = torch.randn(3, 5, dtype=torch.float64)
    target = torch.randn(3, 5, dtype=torch.float64)
    w = torch.ones(5, dtype=torch.float64)
    assert torch.allclose(weighted_mse(pred, target, w),
                          ((pred - target) ** 2).mean(), atol=1e-12)


# ---------------------------------------------------------------------------
# part-aware schedule validity + SNR allocation
# ---------------------------------------------------------------------------
def test_part_schedules_are_valid_forwards():
    base = NoiseSchedule(num_timesteps=1000)
    pas = PartAwareSchedule(base, {"torso": 1.0, "hands": 0.5, "face": 0.7})
    for name in ("torso", "hands", "face"):
        assert pas.is_valid(name)


def test_kappa_one_matches_base_schedule():
    base = NoiseSchedule(num_timesteps=500)
    pas = PartAwareSchedule(base, {"torso": 1.0})
    assert torch.allclose(pas.part_alpha_bar["torso"], base.alpha_bar, atol=1e-12)


def test_lower_kappa_retains_more_signal_and_higher_snr():
    base = NoiseSchedule(num_timesteps=1000)
    pas = PartAwareSchedule(base, {"hands": 0.5, "torso": 1.0})
    t = torch.arange(50, 950, 100)
    # hands (kappa=0.5) keep MORE signal (higher alpha_bar) than torso (kappa=1)
    ab_hands = pas.part_alpha_bar["hands"][t]
    ab_torso = pas.part_alpha_bar["torso"][t]
    assert torch.all(ab_hands > ab_torso)
    # and therefore higher SNR at every t
    assert torch.all(pas.part_snr("hands", t) > pas.part_snr("torso", t))


def test_part_a_b_satisfies_unit_circle():
    base = NoiseSchedule(num_timesteps=300)
    pas = PartAwareSchedule(base, {"hands": 0.6})
    t = torch.arange(0, 300, 30)
    a, b = pas.part_a_b("hands", t)
    assert torch.allclose(a ** 2 + b ** 2, torch.ones_like(a), atol=1e-10)


def test_invalid_kappa_rejected():
    base = NoiseSchedule(num_timesteps=100)
    with pytest.raises(ValueError):
        PartAwareSchedule(base, {"hands": 0.0})
