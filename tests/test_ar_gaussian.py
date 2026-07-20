"""Verification of the 3D Gaussian Splatting rasterizer.

Proves Σ = R S Sᵀ Rᵀ is PSD, the projection Jacobian (vs finite differences), the
2D covariance transform, the 2D Gaussian normalisation ∫G = 2π√|Σ'|, the
front-to-back alpha "over" operator with occlusion, and depth-order correctness.
"""

import math

import pytest
import torch

from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_quaternion
from signtranslator.avatar_render.gaussian import (
    covariance_3d, projection_jacobian, covariance_2d, gaussian_2d_value,
    alpha_composite, render_pixel,
)


# ---------------------------------------------------------------------------
# 3D covariance
# ---------------------------------------------------------------------------
def test_covariance_3d_is_psd_and_symmetric():
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(20, 4, dtype=torch.float64), dim=-1)
    s = torch.rand(20, 3, dtype=torch.float64) + 0.1
    cov = covariance_3d(q, s)
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-12)   # symmetric
    eig = torch.linalg.eigvalsh(cov)
    assert torch.all(eig >= -1e-10)                                 # PSD


def test_covariance_3d_identity_rotation_is_scale_squared():
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)     # identity rotation
    s = torch.tensor([2.0, 3.0, 0.5], dtype=torch.float64)
    cov = covariance_3d(q, s)
    assert torch.allclose(cov, torch.diag(s ** 2), atol=1e-12)


# ---------------------------------------------------------------------------
# projection Jacobian
# ---------------------------------------------------------------------------
def test_projection_jacobian_matches_finite_difference():
    mu = torch.tensor([0.3, -0.2, 2.0], dtype=torch.float64)
    fx, fy = 500.0, 480.0
    J = projection_jacobian(mu, fx, fy)

    def proj(p):
        return torch.stack((fx * p[0] / p[2], fy * p[1] / p[2]))

    eps = 1e-6
    Jnum = torch.zeros(2, 3, dtype=torch.float64)
    for k in range(3):
        d = torch.zeros(3, dtype=torch.float64); d[k] = eps
        Jnum[:, k] = (proj(mu + d) - proj(mu - d)) / (2 * eps)
    assert torch.allclose(J, Jnum, atol=1e-4)


def test_covariance_2d_is_symmetric_psd():
    cov3d = covariance_3d(torch.tensor([1.0, 0, 0, 0], dtype=torch.float64),
                          torch.tensor([0.1, 0.2, 0.1], dtype=torch.float64))
    mu = torch.tensor([0.1, 0.1, 2.0], dtype=torch.float64)
    W = torch.eye(3, dtype=torch.float64)
    cov2d = covariance_2d(cov3d, mu, W, 500.0, 500.0)
    assert torch.allclose(cov2d, cov2d.T, atol=1e-12)
    assert torch.all(torch.linalg.eigvalsh(cov2d) >= -1e-10)


# ---------------------------------------------------------------------------
# 2D Gaussian
# ---------------------------------------------------------------------------
def test_gaussian_2d_peak_and_decay():
    mu = torch.tensor([10.0, 20.0], dtype=torch.float64)
    cov = torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    assert abs(float(gaussian_2d_value(mu, mu, cov)) - 1.0) < 1e-12  # peak = 1 at centre
    far = mu + torch.tensor([10.0, 0.0], dtype=torch.float64)
    assert float(gaussian_2d_value(far, mu, cov)) < 0.01


def test_gaussian_2d_integrates_to_2pi_sqrt_det():
    mu = torch.tensor([0.0, 0.0], dtype=torch.float64)
    cov = torch.tensor([[0.5, 0.1], [0.1, 0.3]], dtype=torch.float64)
    xs = torch.linspace(-6, 6, 400, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    vals = gaussian_2d_value(pts, mu, cov)
    cell = (xs[1] - xs[0]) ** 2
    integral = float(vals.sum() * cell)
    expected = 2 * math.pi * math.sqrt(float(torch.linalg.det(cov)))
    assert abs(integral - expected) / expected < 1e-3


# ---------------------------------------------------------------------------
# alpha compositing
# ---------------------------------------------------------------------------
def test_alpha_composite_over_operator():
    colors = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], dtype=torch.float64)
    alphas = torch.tensor([0.5, 0.8], dtype=torch.float64)
    C, acc = alpha_composite(colors, alphas)
    expected = colors[0] * 0.5 + colors[1] * 0.8 * (1 - 0.5)
    assert torch.allclose(C, expected, atol=1e-12)
    assert abs(float(acc) - (1 - (1 - 0.5) * (1 - 0.8))) < 1e-12


def test_opaque_front_gaussian_occludes_back():
    colors = torch.tensor([[1.0, 0, 0], [0, 0, 1.0]], dtype=torch.float64)
    alphas = torch.tensor([1.0, 1.0], dtype=torch.float64)   # front fully opaque
    C, _ = alpha_composite(colors, alphas)
    assert torch.allclose(C, colors[0], atol=1e-12)          # only the front is seen


def test_render_pixel_respects_depth_order():
    pixel = torch.tensor([0.0, 0.0], dtype=torch.float64)
    mu2d = torch.zeros(2, 2, dtype=torch.float64)            # both centred on the pixel
    cov2d = torch.eye(2, dtype=torch.float64).expand(2, 2, 2)
    opacity = torch.ones(2, dtype=torch.float64)             # g=1 -> alpha=1 each
    colors = torch.tensor([[1.0, 0, 0], [0, 0, 1.0]], dtype=torch.float64)
    near_red = render_pixel(pixel, mu2d, cov2d, opacity, colors,
                            depth=torch.tensor([1.0, 2.0], dtype=torch.float64))[0]
    assert torch.allclose(near_red, colors[0], atol=1e-9)    # red is nearer
    near_blue = render_pixel(pixel, mu2d, cov2d, opacity, colors,
                             depth=torch.tensor([2.0, 1.0], dtype=torch.float64))[0]
    assert torch.allclose(near_blue, colors[1], atol=1e-9)   # swapping depth swaps result
