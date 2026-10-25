"""Rotation smoothing of generated states: SO(3) output, slow motion kept, jitter removed, invalid frames untouched."""
from dataclasses import replace

import numpy as np
import torch

from signtranslator.generation.postprocess import damp_rotation_channel, project_to_rotation, smooth_rotation_channel
from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix
from tests.test_generation_text_to_motion import state


def with_head(angles, valid=None):
    s = state(0, len(angles))
    head = s.channels["head_rotation"]
    aa = np.zeros((len(angles), 1, 3)); aa[:, 0, 1] = angles                       # yaw
    values = matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(aa))).numpy()
    v = np.ones((len(angles), 1), bool) if valid is None else valid
    values = np.where(v[..., None], values, 0.)                                     # unavailable payload is zero
    return replace(s, channels={**s.channels, "head_rotation": replace(head, values=values, valid=v,
                                                                       confidence=v.astype(float))})


def yaw(st):
    m = rotation_6d_to_matrix(torch.from_numpy(st.channels["head_rotation"].values[:, 0])).numpy()
    return np.arctan2(m[:, 0, 2], m[:, 2, 2])


def test_projection_returns_rotations():
    rng = np.random.default_rng(0)
    r = project_to_rotation(rng.normal(size=(5, 3, 3)))
    np.testing.assert_allclose(r @ np.swapaxes(r, -1, -2), np.eye(3)[None].repeat(5, 0), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(r), 1., atol=1e-10)


def test_constant_and_slow_motion_are_kept_and_jitter_is_removed():
    t = np.arange(96) / 24.
    const = with_head(np.full(96, .3))
    np.testing.assert_allclose(yaw(smooth_rotation_channel(const, "head_rotation", .25)), .3, atol=1e-9)
    slow = with_head(.3 * np.sin(2 * np.pi * .5 * t))                               # a 0.5 Hz turn
    out = yaw(smooth_rotation_channel(slow, "head_rotation", .25))
    # a centred moving average of width W attenuates frequency f by sinc(f W): 3.5% at 0.5 Hz, W = 0.29 s
    assert np.max(np.abs(out[6:-6] - yaw(slow)[6:-6])) < .015
    rng = np.random.default_rng(1)
    noisy = with_head(.3 * np.sin(2 * np.pi * .5 * t) + rng.normal(scale=.05, size=96))
    clean = np.abs(np.diff(yaw(slow))).mean()                                       # the turn's own speed
    before = np.abs(np.diff(yaw(noisy))).mean()
    after = np.abs(np.diff(yaw(smooth_rotation_channel(noisy, "head_rotation", .25)))).mean()
    assert before > 2 * clean and after < 1.3 * clean                               # jitter gone, the turn kept


def test_invalid_frames_are_untouched_and_excluded():
    angles = np.zeros(30)
    valid = np.ones((30, 1), bool); valid[10] = False
    s = with_head(angles, valid)
    out = smooth_rotation_channel(s, "head_rotation", .5)
    np.testing.assert_array_equal(out.channels["head_rotation"].values[10], s.channels["head_rotation"].values[10])
    ok = np.delete(np.arange(30), 10)
    m = rotation_6d_to_matrix(torch.from_numpy(out.channels["head_rotation"].values[ok, 0])).numpy()
    np.testing.assert_allclose(m, np.eye(3)[None].repeat(len(ok), 0), atol=1e-9)       # neighbours never saw it
    assert smooth_rotation_channel(s, "head_rotation", 0.) is s


def test_face_calibration_restores_the_real_range_and_never_shrinks_it():
    from dataclasses import replace
    import numpy as np
    from signtranslator.generation.postprocess import calibrate_face, estimate_face_calibration
    from tests.test_pose_resample import state
    rng = np.random.default_rng(0)

    def with_face(values):
        s = state(np.arange(len(values)) / 24.)
        f = s.channels["face"]
        v = np.asarray(values, float)[:, None, None]
        return replace(s, channels=dict(s.channels, face=replace(f, values=v, valid=np.ones((len(v), 1), bool),
                                                                observed=np.ones((len(v), 1), bool),
                                                                confidence=np.ones((len(v), 1)))))
    real = [with_face(np.clip(.3 + .2 * rng.standard_normal(40), 0, 1)) for _ in range(6)]
    flat = [with_face(np.clip(.1 + .05 * rng.standard_normal(40), 0, 1)) for _ in range(6)]
    cal = estimate_face_calibration(flat, real)
    assert 3 < cal["gain"][0] <= 4 and abs(cal["center_real"][0] - .3) < .05
    fixed = [calibrate_face(s, cal) for s in flat]
    ratio = np.mean([s.channels["face"].values[:, 0, 0].std() for s in fixed]) / np.mean(
        [s.channels["face"].values[:, 0, 0].std() for s in real])
    assert .8 < ratio < 1.2 and all(((s.channels["face"].values >= 0) & (s.channels["face"].values <= 1)).all() for s in fixed)
    assert estimate_face_calibration(real, flat)["gain"][0] == 1.          # never reduces the range


def test_damping_shrinks_excursions_around_the_mean_and_keeps_timing():
    t = np.arange(96) / 24.
    turn = with_head(.2 + .3 * np.sin(2 * np.pi * .5 * t))       # mean 0.2 rad, +-0.3 excursions
    out = yaw(damp_rotation_channel(turn, "head_rotation", .5))
    np.testing.assert_allclose(out - .2, .5 * (yaw(turn) - .2), atol=5e-3)   # half the excursion, same phase
    assert damp_rotation_channel(turn, "head_rotation", 1.) is turn
    gaps = np.ones((96, 1), bool); gaps[10:20] = False
    held = with_head(.3 * np.sin(2 * np.pi * .5 * t), gaps)
    damped = damp_rotation_channel(held, "head_rotation", .5)
    np.testing.assert_array_equal(damped.channels["head_rotation"].values[10:20], held.channels["head_rotation"].values[10:20])


def with_head_translation(x, valid=None):
    s = state(0, len(x))
    ch = s.channels["head_translation"]
    values = np.zeros((len(x), 1, 3)); values[:, 0, 0] = x
    v = np.ones((len(x), 1), bool) if valid is None else valid
    return replace(s, channels={**s.channels, "head_translation": replace(ch, values=np.where(v[..., None], values, 0.),
                                                                          valid=v, observed=v,
                                                                          confidence=v.astype(float))})


def test_vector_smoothing_keeps_slow_motion_removes_jitter_and_lowers_the_high_frequency_rms():
    from signtranslator.generation.motion_metrics import high_frequency_rms
    from signtranslator.generation.postprocess import smooth_vector_channel
    t = np.arange(120) / 24.
    slow = .02 * np.sin(2 * np.pi * .5 * t)                                         # a 0.5 Hz sway, 2 cm
    jitter = .004 * (-1.) ** np.arange(120)                                         # frame-to-frame shake at Nyquist
    noisy = with_head_translation(slow + jitter)
    out = smooth_vector_channel(noisy, "head_translation", .25)
    x = out.channels["head_translation"].values[:, 0, 0]
    assert np.abs(x[12:-12] - slow[12:-12]).max() < .003                            # slow sway kept, shake gone
    before = high_frequency_rms(noisy, "head_translation")
    after = high_frequency_rms(out, "head_translation")
    assert before > .003 and after < .25 * before
    assert high_frequency_rms(with_head_translation(slow), "head_translation") < .05 * before   # leakage only
    assert smooth_vector_channel(noisy, "head_translation", 0.) is noisy


def test_vector_smoothing_leaves_invalid_frames_and_uses_only_valid_ones():
    from signtranslator.generation.postprocess import smooth_vector_channel
    x = np.full(48, .01); x[20] = 5.                                               # an outlier on an invalid frame
    v = np.ones((48, 1), bool); v[20] = False
    out = smooth_vector_channel(with_head_translation(x, v), "head_translation", .25)
    vals = out.channels["head_translation"].values[:, 0, 0]
    assert vals[20] == 0. and np.allclose(vals[v[:, 0]], .01)
