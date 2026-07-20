"""Verification of the motion autoencoder and full motion loss.

Proves the temporal-derivative loss terms, the geodesic term, the documented
oversmoothing insight (a constant prediction has zero velocity/acceleration error
yet large geodesic error), autoencoder shape/round-trip, and a real overfit of a
tiny motion clip (reconstruction collapses).
"""

import math

import pytest
import torch

from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d
from signtranslator.motion_transformer.autoencoder import (
    velocity, acceleration, velocity_l1, acceleration_l1, geodesic_motion_loss,
    motion_loss, MotionLossWeights, MotionVQVAE,
)


# ---------------------------------------------------------------------------
# temporal derivatives
# ---------------------------------------------------------------------------
def test_velocity_and_acceleration_finite_difference():
    x = torch.tensor([[0.0, 1.0, 3.0, 6.0]], dtype=torch.float64)  # (1, T=4)
    v = velocity(x)
    assert torch.allclose(v, torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64))
    a = acceleration(x)
    assert torch.allclose(a, torch.tensor([[1.0, 1.0]], dtype=torch.float64))


def test_velocity_accel_l1_zero_when_identical():
    x = torch.randn(2, 5, 10, dtype=torch.float64)
    assert velocity_l1(x, x.clone()).item() == 0.0
    assert acceleration_l1(x, x.clone()).item() == 0.0


# ---------------------------------------------------------------------------
# geodesic term
# ---------------------------------------------------------------------------
def test_geodesic_motion_loss_zero_when_identical():
    six = matrix_to_rotation_6d(axis_angle_to_matrix(torch.randn(2, 4, 3, 3, dtype=torch.float64)))
    assert geodesic_motion_loss(six, six.clone()).item() < 1e-7


# ---------------------------------------------------------------------------
# the oversmoothing insight
# ---------------------------------------------------------------------------
def test_constant_prediction_has_zero_velocity_but_large_geodesic():
    """A constant (fully smoothed) prediction trivially minimises velocity AND
    acceleration error, yet is geodesically wrong -- exactly why velocity loss
    alone is insufficient and the geodesic/spectral terms are needed."""
    torch.manual_seed(0)
    # a genuinely moving target: rotations sweeping about z
    T, J = 12, 2
    angles = torch.linspace(0, math.pi, T, dtype=torch.float64)
    Rt = torch.stack([axis_angle_to_matrix(torch.tensor([0.0, 0, float(a)], dtype=torch.float64))
                      for a in angles])                      # (T,3,3)
    target6d = matrix_to_rotation_6d(Rt).reshape(T, 1, 6).expand(T, J, 6)  # (T,J,6)
    target = target6d.reshape(T, J * 6).permute(1, 0).unsqueeze(0)  # (1, C, T)
    # constant prediction = the first frame repeated
    const = target[..., :1].expand_as(target).contiguous()

    # velocity & acceleration errors of the constant predictor
    assert velocity_l1(const, target).item() > 0             # it is NOT zero vs a moving target
    # but a predictor that MATCHES velocity yet is offset (target + const shift) has 0 vel error
    shifted = target + 5.0
    assert velocity_l1(shifted, target).item() < 1e-12       # velocity blind to constant offset
    assert acceleration_l1(shifted, target).item() < 1e-12
    # geodesic (position) DOES see the error -> this is why it must be in the loss
    tv = target.permute(0, 2, 1).reshape(1, T, J, 6)
    sv = shifted.permute(0, 2, 1).reshape(1, T, J, 6)
    assert geodesic_motion_loss(sv, tv).item() > 0.1


# ---------------------------------------------------------------------------
# loss assembly
# ---------------------------------------------------------------------------
def test_motion_loss_assembles_and_weights():
    N, J, T = 2, 3, 8
    pred = torch.randn(N, J * 6, T, dtype=torch.float64)
    target = torch.randn(N, J * 6, T, dtype=torch.float64)
    commit = torch.tensor(0.5, dtype=torch.float64)
    terms = motion_loss(pred, target, num_joints=J, commit_loss=commit,
                        weights=MotionLossWeights(geodesic=2.0, velocity=1.0,
                                                  acceleration=0.5, commit=1.0))
    assert torch.isfinite(terms["total"])
    manual = (2.0 * terms["geodesic"] + 1.0 * terms["velocity"]
              + 0.5 * terms["acceleration"] + 1.0 * commit)
    assert torch.allclose(terms["total"], manual, atol=1e-9)


# ---------------------------------------------------------------------------
# autoencoder
# ---------------------------------------------------------------------------
def test_autoencoder_preserves_length_and_shape():
    model = MotionVQVAE(in_channels=12, dim=32, num_codes=64, num_downsamples=2).double()
    x = torch.randn(2, 12, 16, dtype=torch.float64)          # T=16 divisible by 4
    recon, q = model(x)
    assert recon.shape == x.shape
    assert "z_q" in q and torch.isfinite(recon).all()


def test_autoencoder_rejects_bad_length():
    model = MotionVQVAE(in_channels=6, dim=16, num_downsamples=2)
    with pytest.raises(ValueError):
        model(torch.randn(1, 6, 15))                         # 15 not divisible by 4


def test_autoencoder_overfits_a_tiny_clip():
    torch.manual_seed(1)
    model = MotionVQVAE(in_channels=12, dim=48, num_codes=64, num_downsamples=1,
                        ema=False).double()                  # param codebook -> trainable
    x = torch.randn(1, 12, 16, dtype=torch.float64) * 0.5
    model.init_codebook(x)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    l0 = None
    for step in range(400):
        opt.zero_grad()
        recon, q = model(x)
        loss = torch.nn.functional.mse_loss(recon, x) + q["loss"]
        loss.backward(); opt.step()
        if step == 0:
            l0 = float(torch.nn.functional.mse_loss(recon, x))
    final = float(torch.nn.functional.mse_loss(model(x)[0], x))
    assert final < 0.3 * l0                                  # reconstruction collapses
