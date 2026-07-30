"""Adversarial tests for triangulation + weighted reprojection (Doc-10 10c)."""

import pytest
import torch

from signtranslator.pose.camera import PerspectiveCamera
from signtranslator.data_engineering.quality import (
    projection_matrix, triangulate_dlt, triangulation_confidence,
    weighted_reprojection_residual,
)

torch.manual_seed(0)


def _cam(eye, target=(0.0, 0.0, 0.0)):
    return PerspectiveCamera.look_at(500.0, 500.0, 320.0, 240.0, eye, target,
                                     dtype=torch.float64)


def _project(cam, X):
    p, in_front = cam.project(X)
    assert bool(in_front)
    return p


def test_projection_matrix_matches_pinhole_projection():
    cam = _cam((0.0, 0.0, -3.0))
    X = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    P = projection_matrix(cam)
    Xh = torch.cat((X, torch.ones(1, dtype=torch.float64)))
    uvw = P @ Xh
    uv_dlt = uvw[:2] / uvw[2]
    assert torch.allclose(uv_dlt, _project(cam, X), atol=1e-9)


def test_dlt_recovers_known_point_from_two_views():
    cams = [_cam((0.0, 0.0, -3.0)), _cam((3.0, 0.0, 0.0))]
    X = torch.tensor([0.15, 0.25, -0.1], dtype=torch.float64)
    obs = torch.stack([_project(c, X) for c in cams])
    Xhat = triangulate_dlt(cams, obs)
    assert torch.allclose(Xhat, X, atol=1e-8)


def test_dlt_recovers_from_many_views_with_confidence():
    cams = [_cam(e) for e in [(0, 0, -3), (3, 0, 0), (0, 3, -0.5), (-3, 0.5, -1)]]
    X = torch.tensor([-0.2, 0.1, 0.3], dtype=torch.float64)
    obs = torch.stack([_project(c, X) for c in cams])
    conf = torch.tensor([0.9, 0.8, 0.95, 0.7], dtype=torch.float64)
    Xhat = triangulate_dlt(cams, obs, conf)
    assert torch.allclose(Xhat, X, atol=1e-8)


def test_triangulation_confidence_increases_with_confidence():
    r = torch.tensor([1.0, 1.0])
    low = triangulation_confidence(r, torch.tensor([0.3, 0.3]))
    high = triangulation_confidence(r, torch.tensor([0.9, 0.9]))
    assert high > low


def test_triangulation_confidence_decreases_with_residual():
    c = torch.tensor([0.8, 0.8])
    small = triangulation_confidence(torch.tensor([0.1, 0.1]), c)
    large = triangulation_confidence(torch.tensor([5.0, 5.0]), c)
    assert small > large
    assert 0.0 <= float(large) <= float(small) <= 1.0


def test_weighted_residual_zero_when_projection_matches():
    cam = _cam((0.0, 0.0, -3.0))
    X = torch.tensor([0.1, 0.2, 0.0], dtype=torch.float64)
    k = _project(cam, X)
    e = weighted_reprojection_residual(cam, X, k, torch.tensor(1.0))
    assert float(e) < 1e-12


def test_weighted_residual_scales_with_confidence():
    cam = _cam((0.0, 0.0, -3.0))
    X = torch.tensor([0.1, 0.2, 0.0], dtype=torch.float64)
    k = _project(cam, X) + torch.tensor([30.0, 0.0], dtype=torch.float64)
    e_full = weighted_reprojection_residual(cam, X, k, torch.tensor(1.0))
    e_half = weighted_reprojection_residual(cam, X, k, torch.tensor(0.5))
    e_zero = weighted_reprojection_residual(cam, X, k, torch.tensor(0.0))
    assert torch.allclose(e_half, 0.5 * e_full, atol=1e-12)
    assert float(e_zero) == 0.0            # zero confidence => no contribution


def test_weighted_residual_is_bounded_by_confidence():
    # Geman-McClure rho < 1 => e = c*rho < c even for a gross outlier.
    cam = _cam((0.0, 0.0, -3.0))
    X = torch.tensor([0.1, 0.2, 0.0], dtype=torch.float64)
    k = _project(cam, X) + torch.tensor([1e4, 1e4], dtype=torch.float64)
    e = weighted_reprojection_residual(cam, X, k, torch.tensor(1.0), sigma=100.0)
    assert 0.0 < float(e) < 1.0


def test_dlt_rejects_one_effective_view_and_invalid_confidence():
    cams = [_cam((0.0, 0.0, -3.0)), _cam((3.0, 0.0, 0.0))]
    X = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    obs = torch.stack([_project(camera, X) for camera in cams])
    with pytest.raises(ValueError, match="positive-confidence"):
        triangulate_dlt(cams, obs, torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        triangulate_dlt(cams, obs, torch.tensor([1.0, 1.1]))


def test_dlt_rejects_coincident_camera_geometry():
    cams = [_cam((0.0, 0.0, -3.0)), _cam((0.0, 0.0, -3.0))]
    X = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    obs = torch.stack([_project(camera, X) for camera in cams])
    with pytest.raises(ValueError, match="degenerate camera geometry"):
        triangulate_dlt(cams, obs)


def test_dlt_rejects_nonfinite_observations():
    cams = [_cam((0.0, 0.0, -3.0)), _cam((3.0, 0.0, 0.0))]
    obs = torch.tensor([[320.0, 240.0], [float("nan"), 240.0]])
    with pytest.raises(ValueError, match="finite"):
        triangulate_dlt(cams, obs)


def test_triangulation_confidence_validates_its_mathematical_domain():
    with pytest.raises(ValueError, match="tau"):
        triangulation_confidence(torch.ones(2), torch.ones(2), tau=0.0)
    with pytest.raises(ValueError, match="valid domains"):
        triangulation_confidence(torch.tensor([-1.0, 0.0]), torch.ones(2))
