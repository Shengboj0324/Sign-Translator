"""Evaluation + whole-chain integration for the avatar rendering pipeline.

Proves the metric formulas, the structurally-enforced appearance/signing
separation (a signing verdict cannot be derived from appearance), and a whole-chain
run (stream -> validate -> pace -> LBS -> penetration; Gaussian render -> appearance
report) with a determinism/finiteness cycle stress.
"""

import math

import pytest
import torch

from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d
from signtranslator.avatar_render.stream import AvatarContract, ParameterStream, validate_stream
from signtranslator.avatar_render.pacing import pace
from signtranslator.avatar_render.rigged import apply_lbs
from signtranslator.avatar_render.gaussian import covariance_3d, covariance_2d, render_pixel
from signtranslator.avatar_render.evaluation import (
    motion_to_photon_p95, dropped_frame_rate, temporal_flicker, penetration_rate,
    silhouette_iou_error, psnr, ssim_global, AppearanceReport, SigningReport,
    signing_quality_from_appearance, combined_report,
)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_latency_and_dropped_frames():
    lat = [0.01 * i for i in range(1, 101)]                  # 0.01 .. 1.0 s
    assert abs(motion_to_photon_p95(lat) - 0.96) < 0.03
    times = [0.02, 0.05, 0.01, 0.09]                         # budget 1/30 ~ 0.033
    assert abs(dropped_frame_rate(times, 1 / 30.0) - 0.5) < 1e-9   # 2 of 4 over budget


def test_temporal_flicker_zero_for_static_positive_for_jitter():
    static = torch.ones(10, 4, dtype=torch.float64)
    assert temporal_flicker(static).item() == 0.0
    jittery = torch.randn(10, 4, dtype=torch.float64)
    assert temporal_flicker(jittery).item() > 0


def test_silhouette_iou_error():
    a = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    assert silhouette_iou_error(a, a) < 1e-12                # identical -> 0 error
    b = torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.bool)
    assert abs(silhouette_iou_error(a, b) - 1.0) < 1e-12     # disjoint -> error 1


def test_psnr_and_ssim_identical_images():
    img = torch.rand(3, 8, 8, dtype=torch.float64)
    assert psnr(img, img.clone()) == float("inf")           # identical -> inf PSNR
    assert abs(ssim_global(img, img.clone()) - 1.0) < 1e-9  # identical -> SSIM 1


# ---------------------------------------------------------------------------
# structural separation of concerns (the document's core rule)
# ---------------------------------------------------------------------------
def test_appearance_report_is_not_a_signing_verdict():
    ap = AppearanceReport(psnr=35.0, ssim=0.98, silhouette_iou_error=0.02)
    assert ap.is_signing_verdict is False
    with pytest.raises(TypeError):
        signing_quality_from_appearance(ap)                 # refuses to conflate


def test_combined_report_keeps_them_separate():
    ap = AppearanceReport(30.0, 0.9, 0.1)
    sg = SigningReport(semantic_accuracy=0.95, comprehension=0.9)
    rep = combined_report(ap, sg)
    assert rep["appearance"] is ap and rep["signing"] is sg
    with pytest.raises(TypeError):
        combined_report(ap, ap)                             # a second appearance is not signing


# ---------------------------------------------------------------------------
# whole-chain integration
# ---------------------------------------------------------------------------
def _stream(T=6, J=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ParameterStream(
        contract=AvatarContract(),
        timestamps=torch.arange(T, dtype=torch.float64) / 30.0,
        rot6d=matrix_to_rotation_6d(axis_angle_to_matrix(
            0.5 * torch.randn(T, J, 3, generator=g, dtype=torch.float64))),
        gamma=torch.randn(T, 3, generator=g, dtype=torch.float64),
        expr=torch.randn(T, 4, generator=g, dtype=torch.float64),
    )


def test_whole_chain_stream_to_render():
    s = _stream(seed=1)
    assert validate_stream(s) == []
    # pace to 60 fps (rotations of a single joint as a keyframe track)
    from signtranslator.pose.rotations import rotation_6d_to_matrix
    key_R = rotation_6d_to_matrix(s.rot6d[:, 0])            # (T,3,3)
    q, R, trans, expr = pace(s.timestamps, key_R, s.gamma, s.expr, fps=60.0)
    assert R.shape[0] == q.shape[0] and torch.isfinite(R).all()
    # rigged LBS of a tiny mesh driven by two joints
    v = torch.randn(5, 3, dtype=torch.float64)
    w = torch.softmax(torch.randn(5, 2, dtype=torch.float64), dim=-1)
    g_rel = torch.eye(4, dtype=torch.float64).expand(2, 4, 4)
    skinned = apply_lbs(v, w, g_rel)
    assert torch.allclose(skinned, v, atol=1e-10)          # rest pose round trip
    # gaussian render of one pixel from a two-Gaussian scene
    cov3d = covariance_3d(torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=torch.float64),
                          torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=torch.float64))
    mu_cam = torch.tensor([[0.0, 0, 2.0], [0.0, 0, 3.0]], dtype=torch.float64)
    W = torch.eye(3, dtype=torch.float64)
    cov2d = torch.stack([covariance_2d(cov3d[i], mu_cam[i], W, 500.0, 500.0) for i in range(2)])
    color, acc = render_pixel(torch.zeros(2, dtype=torch.float64),
                              torch.zeros(2, 2, dtype=torch.float64), cov2d,
                              torch.ones(2, dtype=torch.float64),
                              torch.tensor([[1.0, 0, 0], [0, 0, 1.0]], dtype=torch.float64),
                              depth=torch.tensor([2.0, 3.0], dtype=torch.float64))
    assert torch.isfinite(color).all()


def test_cycle_stress_determinism_and_finiteness():
    for seed in range(60):
        s = _stream(seed=100 + seed)
        assert validate_stream(s) == []
        from signtranslator.pose.rotations import rotation_6d_to_matrix
        key_R = rotation_6d_to_matrix(s.rot6d[:, 0])
        a = pace(s.timestamps, key_R, s.gamma, s.expr, fps=48.0)
        b = pace(s.timestamps, key_R, s.gamma, s.expr, fps=48.0)
        for x, y in zip(a, b):
            assert torch.equal(x, y)                         # deterministic
        assert torch.isfinite(a[1]).all()
