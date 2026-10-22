import numpy as np
import pytest

from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from signtranslator.pose.resample import resample_motion


def state(times):
    t = np.asarray(times, dtype=np.float64)
    n = len(t)
    channels = {}
    for name, (kind, width, units) in CHANNELS.items():
        if kind == "rotation_6d_columns":
            ang = np.linspace(0, np.pi / 2, n)              # rotation about z from 0 to 90 degrees
            values = np.stack([np.cos(ang), np.sin(ang), 0 * ang, -np.sin(ang), np.cos(ang), 0 * ang], -1)[:, None, :]
        elif kind == "unit_vector":
            values = np.tile([[[0., 0., 1.]]], (n, 1, 1))
        else:
            values = np.linspace(0, 1, n)[:, None, None].repeat(width, -1)
        valid = np.ones((n, 1), dtype=bool)
        channels[name] = MotionChannel(values, valid, valid.copy(), np.ones((n, 1)), (name,), "f", units, "src",
                                       "a" * 64, "c", inference_method="solve")
    return MultichannelMotion(t, "clock", "s", channels)


def test_60_fps_clip_resampled_to_24_fps_keeps_its_duration_and_endpoints():
    s = state(np.arange(121) / 60)                           # 2.0 s at 60 fps
    r = resample_motion(s, 24)
    assert len(r.timestamps) == 49 and r.timestamps[-1] == pytest.approx(2.0)
    np.testing.assert_allclose(r.channels["root_translation"].values[[0, -1]], s.channels["root_translation"].values[[0, -1]])


def test_rotations_stay_orthonormal_and_unit_vectors_unit():
    r = resample_motion(state([0., .1, .25, .3, .7, 1.]), 24)  # variable input rate
    v = r.channels["body"].values
    a, b = v[..., :3], v[..., 3:]
    np.testing.assert_allclose(np.linalg.norm(a, axis=-1), 1, atol=1e-9)
    np.testing.assert_allclose((a * b).sum(-1), 0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(r.channels["left_gaze"].values, axis=-1), 1, atol=1e-9)
    r.validate()


def test_a_frame_is_valid_only_if_both_neighbours_are():
    s = state(np.arange(5) / 10.)                            # 10 fps
    s.channels["face"].valid[2] = False
    s.channels["face"].observed[2] = False
    s.channels["face"].confidence[2] = 0.
    s.channels["face"].values[2] = 0.
    r = resample_motion(s, 20)                               # new frames at 0, .05, .10, ...
    face = r.channels["face"].valid[:, 0]
    assert face.tolist() == [True, True, True, False, False, False, True, True, True]
