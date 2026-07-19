"""Verification of camera projection and the robust re-projection term.

Projection is checked against hand-computed pinhole values; the Geman-McClure
robustifier is checked for boundedness and a redescending influence (its whole
purpose: outliers cannot dominate the fit); the loss is zero at a perfect fit and
ignores zero-confidence detections.
"""

import math

import pytest
import torch

from signtranslator.pose.camera import (
    PerspectiveCamera, WeakPerspectiveCamera,
    geman_mcclure, geman_mcclure_influence, reprojection_loss,
)


def _cam_at_origin(dtype=torch.float64):
    mk = lambda x: torch.tensor(x, dtype=dtype)
    return PerspectiveCamera(mk(500.0), mk(500.0), mk(320.0), mk(240.0),
                             torch.eye(3, dtype=dtype), torch.zeros(3, dtype=dtype))


# ---------------------------------------------------------------------------
# projection correctness
# ---------------------------------------------------------------------------
def test_perspective_matches_hand_computation():
    cam = _cam_at_origin()
    X = torch.tensor([[0.1, -0.2, 2.0]], dtype=torch.float64)
    px, in_front = cam.project(X)
    expected = torch.tensor([[500.0 * 0.1 / 2.0 + 320.0,
                              500.0 * -0.2 / 2.0 + 240.0]], dtype=torch.float64)
    assert torch.allclose(px, expected, atol=1e-12)
    assert in_front.all()


def test_point_on_axis_projects_to_principal_point():
    cam = _cam_at_origin()
    X = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float64)
    px, _ = cam.project(X)
    assert torch.allclose(px, torch.tensor([[320.0, 240.0]], dtype=torch.float64),
                          atol=1e-12)


def test_look_at_projects_target_to_principal_point():
    cam = PerspectiveCamera.look_at(600, 600, 100, 100,
                                    eye=(0.0, 0.0, -5.0), target=(0.0, 0.0, 0.0))
    px, in_front = cam.project(torch.zeros(1, 3, dtype=torch.float64))
    assert in_front.all()
    assert torch.allclose(px, torch.tensor([[100.0, 100.0]], dtype=torch.float64),
                          atol=1e-9)


def test_point_behind_camera_flagged():
    cam = _cam_at_origin()
    X = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
    _, in_front = cam.project(X)
    assert not in_front.any()


def test_projection_is_differentiable():
    cam = _cam_at_origin()
    X = torch.tensor([[0.1, 0.2, 2.0]], dtype=torch.float64, requires_grad=True)
    px, _ = cam.project(X)
    px.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()


def test_weak_perspective_is_linear():
    cam = WeakPerspectiveCamera(s=torch.tensor(2.0), tx=torch.tensor(10.0),
                                ty=torch.tensor(-5.0))
    X = torch.tensor([[1.0, 2.0, 99.0], [0.0, 0.0, 3.0]])
    px = cam.project(X)
    assert torch.allclose(px, torch.tensor([[12.0, -1.0], [10.0, -5.0]]))


# ---------------------------------------------------------------------------
# Geman-McClure robustifier
# ---------------------------------------------------------------------------
def test_geman_mcclure_zero_at_zero_and_bounded():
    r = torch.linspace(0, 1e6, 1000, dtype=torch.float64)
    rho = geman_mcclure(r, sigma=100.0)
    assert rho[0] == 0.0
    assert torch.all(rho >= 0) and torch.all(rho < 1.0)
    # monotone non-decreasing in |r|
    assert torch.all(rho[1:] - rho[:-1] >= -1e-15)


def test_geman_mcclure_saturates_unlike_l2():
    sigma = 100.0
    small = geman_mcclure(torch.tensor(10.0, dtype=torch.float64), sigma)
    huge = geman_mcclure(torch.tensor(1e5, dtype=torch.float64), sigma)
    # robust error of a gross outlier is ~1, i.e. O(1), not O(r^2)
    assert huge < 1.0
    assert huge - small < 1.0
    # the non-robust L2 of the same outlier is enormous
    assert (1e5 ** 2) > 1e6 * huge


def test_geman_mcclure_influence_redescends_and_matches_autograd():
    r = torch.linspace(0.1, 5000, 400, dtype=torch.float64, requires_grad=True)
    rho = geman_mcclure(r, sigma=100.0)
    grad, = torch.autograd.grad(rho.sum(), r)
    analytic = geman_mcclure_influence(r.detach(), sigma=100.0)
    assert torch.allclose(grad, analytic, atol=1e-9)
    # influence goes to ~0 for large residual (redescending) -> outliers ignored
    assert analytic[-1].abs() < analytic[len(analytic) // 20].abs()
    assert analytic[-1].abs() < 1e-6


# ---------------------------------------------------------------------------
# re-projection loss
# ---------------------------------------------------------------------------
def test_reprojection_zero_at_perfect_fit():
    proj = torch.randn(5, 2, dtype=torch.float64)
    loss = reprojection_loss(proj, proj.clone(), sigma=100.0)
    assert loss.item() == 0.0


def test_zero_confidence_keypoints_are_ignored():
    proj = torch.zeros(3, 2, dtype=torch.float64)
    kp = torch.tensor([[1e4, 1e4], [5.0, 5.0], [0.0, 0.0]], dtype=torch.float64)
    conf = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    loss = reprojection_loss(proj, kp, conf, sigma=100.0)
    # the huge-residual keypoint has zero confidence -> contributes nothing
    loss_no_outlier = reprojection_loss(proj[1:], kp[1:], conf[1:], sigma=100.0)
    assert torch.allclose(loss, loss_no_outlier, atol=1e-12)


def test_robust_loss_bounds_outlier_influence_on_gradient():
    """A single gross outlier exerts bounded pull under GM, unbounded under L2."""
    proj = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    kp = torch.tensor([[1e5, 0.0]], dtype=torch.float64)
    robust = reprojection_loss(proj, kp, sigma=100.0, robust=True)
    g_robust, = torch.autograd.grad(robust, proj, retain_graph=False)

    proj2 = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    l2 = reprojection_loss(proj2, kp, sigma=100.0, robust=False)
    g_l2, = torch.autograd.grad(l2, proj2)

    assert g_robust.abs().max() < 1.0                        # redescended
    assert g_l2.abs().max() > 1e4                            # L2 blows up


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        reprojection_loss(torch.zeros(3, 2), torch.zeros(4, 2))
