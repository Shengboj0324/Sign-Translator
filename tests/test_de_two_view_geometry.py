"""Two-view self-calibration: exact geometry, fixating cameras, bone-constrained focal recovery."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from signtranslator.data_engineering.two_view_geometry import (
    TwoViewObservations, bone_length_cv, calibrate, eight_point, intrinsics, project, sampson,
    distort_division, triangulate, triangulate_session, undistort_division, view_offset,
)

SIZE = (1280, 720)


def skeleton(frames=60, seed=0):
    """A chain-structured articulated body with rigid bones and varied poses."""
    rng = np.random.default_rng(seed)
    parents = [-1, 0, 1, 2, 1, 4, 5, 1, 7, 8, 0, 10, 0, 12]
    lengths = rng.uniform(.15, .35, len(parents))
    out = np.zeros((frames, len(parents), 3))
    for t in range(frames):
        out[t, 0] = [.05 * np.sin(t / 7), .02 * np.cos(t / 5), 0.]
        for j, p in enumerate(parents):
            if p < 0:
                continue
            direction = rng.normal(size=3) + [0, -1 if j > 9 else 1, 0]
            out[t, j] = out[t, p] + lengths[j] * direction / np.linalg.norm(direction)
    bones = tuple((p, j) for j, p in enumerate(parents) if p >= 0)
    return out, bones


def cameras(f1=1100., f2=1450., angle=70.):
    """Camera 1 at the origin looking +z; camera 2 fixating the same target from the side."""
    target = np.array([0., 0., 3.])
    rotation = Rotation.from_euler("y", -angle, degrees=True).as_matrix()
    centre2 = target - rotation.T @ np.array([0., 0., 3.2])
    t2 = -rotation @ centre2
    scale = np.linalg.norm(t2)
    p1 = intrinsics(f1, SIZE) @ np.c_[np.eye(3), np.zeros(3)]
    p2 = intrinsics(f2, SIZE) @ np.c_[rotation, t2]
    return p1, p2, rotation, t2 / scale, target


def observations(noise=.7, seed=1):
    body, bones = skeleton()
    p1, p2, rotation, baseline, target = cameras()
    world = body + target
    rng = np.random.default_rng(seed)
    flat = world.reshape(-1, 3)
    v1 = project(p1, flat).reshape(body.shape[:2] + (2,)) + rng.normal(scale=noise, size=body.shape[:2] + (2,))
    v2 = project(p2, flat).reshape(body.shape[:2] + (2,)) + rng.normal(scale=noise, size=body.shape[:2] + (2,))
    return TwoViewObservations(v1, v2, np.ones(body.shape[:2], bool), SIZE, SIZE, bones), rotation, baseline


def test_eight_point_and_triangulation_are_exact_without_noise():
    obs, _, _ = observations(noise=0.)
    x1, x2 = obs.view1.reshape(-1, 2), obs.view2.reshape(-1, 2)
    f = eight_point(x1, x2)
    assert np.max(sampson(f, x1, x2)) < 1e-6
    p1, p2, *_ = cameras()
    body, _ = skeleton()
    np.testing.assert_allclose(triangulate(p1, p2, x1, x2), (body + [0, 0, 3]).reshape(-1, 3), atol=1e-8)


def test_bone_constrained_bundle_adjustment_recovers_focal_lengths_of_fixating_cameras():
    obs, rotation, baseline = observations()
    calibration = calibrate(obs, focal_guess=1280.)
    assert calibration.converged
    assert abs(calibration.focal1 / 1100. - 1) < .03 and abs(calibration.focal2 / 1450. - 1) < .03
    angle = np.degrees(np.arccos(np.clip((np.trace(calibration.rotation.T @ rotation) - 1) / 2, -1, 1)))
    assert angle < 1.
    assert np.degrees(np.arccos(np.clip(calibration.baseline @ baseline, -1, 1))) < 2.
    assert max(calibration.median_reprojection_px) < 1.5
    points = triangulate_session(calibration, obs)
    assert bone_length_cv(points, obs.bones) < .02


def test_view_offset_recovers_a_known_lag():
    rng = np.random.default_rng(4)
    signal = np.cumsum(rng.normal(size=300))
    lag, r = view_offset(np.diff(signal[:-3]), np.diff(signal[3:]) * 2. + .1)
    assert lag == -3 and r > .999
    with pytest.raises(ValueError):
        eight_point(np.zeros((5, 2)), np.zeros((5, 2)))


def test_reprojection_alone_cannot_detect_the_fixating_degeneracy():
    """Without the bone term, focal lengths are badly wrong yet reprojection still looks perfect."""
    obs, _, _ = observations(seed=2)
    unconstrained = calibrate(obs, focal_guess=1280., bone_weight=0.)
    constrained = calibrate(obs, focal_guess=1280.)
    assert max(unconstrained.median_reprojection_px) < 1.  # looks fine...
    assert abs(unconstrained.focal1 / 1100. - 1) > .2       # ...but is not
    assert unconstrained.bone_length_cv > 10 * constrained.bone_length_cv


def test_division_model_distortion_is_recovered_and_removes_its_bias():
    obs, rotation, _ = observations(noise=.5, seed=3)
    distorted = TwoViewObservations(distort_division(obs.view1, SIZE, 1100., .25),
                                    distort_division(obs.view2, SIZE, 1450., .15),
                                    obs.usable, SIZE, SIZE, obs.bones)
    np.testing.assert_allclose(undistort_division(distorted.view1, SIZE, 1100., .25), obs.view1, atol=1e-3)
    pinhole = calibrate(distorted, focal_guess=1280.)
    modelled = calibrate(distorted, focal_guess=1280., distortion=True)
    assert abs(modelled.distortion[0] - .25) < .05 and abs(modelled.distortion[1] - .15) < .05
    assert abs(modelled.focal1 / 1100. - 1) < .03 and abs(modelled.focal2 / 1450. - 1) < .03
    assert max(modelled.median_reprojection_px) < max(pinhole.median_reprojection_px)
    assert bone_length_cv(triangulate_session(modelled, distorted), obs.bones) < \
        bone_length_cv(triangulate_session(pinhole, distorted), obs.bones)


def test_weighted_triangulation_is_exact_without_noise_and_follows_the_trusted_view():
    obs, _, _ = observations(noise=0.)
    p1, p2, *_ = cameras()
    x1, x2 = obs.view1.reshape(-1, 2), obs.view2.reshape(-1, 2)
    exact = triangulate(p1, p2, x1, x2)
    ones = np.ones(len(x1))
    np.testing.assert_allclose(triangulate(p1, p2, x1, x2, ones, ones), exact, atol=1e-8)
    shifted = x2 + [25., 0.]                       # corrupt view 2
    trusted = triangulate(p1, p2, x1, shifted, ones, .05 * ones)
    plain = triangulate(p1, p2, x1, shifted)
    assert np.median(np.linalg.norm(project(p1, trusted) - x1, axis=1)) < \
        .2 * np.median(np.linalg.norm(project(p1, plain) - x1, axis=1))
