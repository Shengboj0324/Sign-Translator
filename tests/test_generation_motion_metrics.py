"""Motion metrics: exact geodesics and speeds, ratios of 1 for identical sets, damped motion detected."""
from dataclasses import replace

import numpy as np
import torch

from signtranslator.generation.motion_metrics import (
    angular_speeds, clip_features, compare, frechet_distance, geodesic_degrees,
)
from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d
from tests.test_generation_text_to_motion import state


def six(axis_angle):
    return matrix_to_rotation_6d(axis_angle_to_matrix(torch.as_tensor(axis_angle, dtype=torch.float64))).numpy()


def test_geodesic_angles_and_speeds_are_exact():
    a, b = six([[0, 0, 0.]]), six([[0, 0, np.radians(30)]])
    np.testing.assert_allclose(geodesic_degrees(a, b), [30.], atol=1e-9)
    s = state(0, 24)
    body = s.channels["body"]
    angles = np.zeros((24, body.values.shape[1], 3))
    angles[..., 2] = np.radians(10) * np.arange(24)[:, None]           # 10° per frame at 24 fps
    s.channels["body"] = replace(body, values=six(angles))
    np.testing.assert_allclose(angular_speeds(s, "body"), 240., atol=1e-6)


def test_identical_sets_score_perfectly_and_damped_motion_is_detected():
    reference = [state(i, 30 + 6 * i) for i in range(12)]
    same = compare(reference, reference)
    for name in ("body", "head_rotation"):
        assert same[name]["geodesic_deg"] < 1e-6
        assert abs(same[name]["speed_ratio"] - 1) < 1e-9 and abs(same[name]["variation_ratio"] - 1) < 1e-9
    assert abs(same["frechet"]) < 1e-6
    damped = []
    for s in reference:                       # halve every rotation angle: averaged-looking motion
        channels = dict(s.channels)
        for name in ("body", "head_rotation"):
            values = torch.from_numpy(s.channels[name].values)
            from signtranslator.pose.rotations import rotation_6d_to_matrix, matrix_to_axis_angle
            half = matrix_to_rotation_6d(axis_angle_to_matrix(matrix_to_axis_angle(rotation_6d_to_matrix(values)) / 2))
            channels[name] = replace(s.channels[name], values=half.numpy())
        damped.append(replace(s, channels=channels))
    worse = compare(damped, reference)
    assert worse["body"]["speed_ratio"] < .7 and worse["body"]["variation_ratio"] < .7
    assert worse["frechet"] > same["frechet"] + 1e-3


def test_frechet_distance_grows_with_a_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(200, 10))
    assert frechet_distance(reference + rng.normal(scale=.01, size=reference.shape), reference) < .05
    assert frechet_distance(reference + 1., reference) > 5.
    assert np.isfinite(clip_features(state(1, 30))).any()
