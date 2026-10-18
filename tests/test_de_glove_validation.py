"""Glove validation: exact twist, exact timecode grid, one offset for both hands."""
import json

import numpy as np
import pytest
import torch

from signtranslator.data_engineering.glove_validation import (
    GLOVE_COLUMNS, VALIDATED_JOINTS, parse_glove_csv, parse_glove_metadata, twist_degrees,
    validate_hands,
)
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d

FPS = 24
FRAMES = 60
OFFSET = 37  # glove frames between glove start and skeleton frame 0


def flexion(t, joint):
    """Aperiodic, joint-specific flexion in degrees."""
    return 30 + 25 * np.sin(1.7 * t + joint) + 8 * np.sign(t) * np.abs(t) ** 1.5 * np.cos(.6 * joint)


def rot(axis, degrees):
    aa = torch.from_numpy(np.asarray(axis, float)[None] * np.radians(np.atleast_1d(degrees))[:, None])
    return axis_angle_to_matrix(aa).numpy()


def state(*, corrupt=None):
    times = np.arange(FRAMES) / FPS
    rng = np.random.default_rng(0)
    channels = {}
    for name in CHANNELS:
        _, width, units = CHANNELS[name]
        if name in ("left_hand", "right_hand"):
            side = "Left" if name == "left_hand" else "Right"
            labels = tuple(f"{side}Hand{f}{k}" for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")
                           for k in (1, 2, 3))
            mats = np.zeros((FRAMES, len(labels), 3, 3))
            for e, label in enumerate(labels):
                rest = rot(rng.normal(size=3) / np.sqrt(3), rng.uniform(10, 60))[0]
                angle = flexion(times, e) if "Thumb" not in label else np.zeros(FRAMES)
                if corrupt == label:
                    angle = angle[::-1].copy()
                mats[:, e] = rest @ rot([1., 0., 0.], angle)
            values = matrix_to_rotation_6d(torch.from_numpy(mats)).numpy()
            valid = np.ones((FRAMES, len(labels)), dtype=bool)
            channels[name] = MotionChannel(values, valid, np.zeros_like(valid), np.ones(valid.shape), labels,
                                           "f", units, "src", "a" * 64, "c", inference_method="solve")
        else:
            channels[name] = MotionChannel(np.zeros((FRAMES, 1, width)), np.zeros((FRAMES, 1), bool),
                                           np.zeros((FRAMES, 1), bool), np.zeros((FRAMES, 1)), (name,),
                                           "f", units, "src", "a" * 64, "c")
    return MultichannelMotion(times, "clock", "s", channels)


def glove_csv(side, *, labels_from_state, duplicates=()):
    """Glove export whose index_02 etc. z angles reproduce the skeleton's flexion."""
    labels = labels_from_state
    rows = []
    total = OFFSET + FRAMES * 5 + 40
    for g in range(total):
        t = (g - OFFSET) / 120
        angles = {}
        for finger, k in VALIDATED_JOINTS:
            e = labels.index(f"{side}Hand{finger}{k}")
            angles[f"{finger.lower()}_0{k}"] = (0., 0., flexion(t, e) + 11.)  # glove zero differs
        joints = ["hand"] + [f"{f}_0{k}" for f in ("index", "middle", "pinky", "ring") for k in range(4)] + \
                 ["thumb_01", "thumb_02", "thumb_03"]
        flat = [v for j in joints for v in angles.get(j, (0., 0., 0.))]
        frame = 5 * 120 + g  # starts at 00:00:05:000
        code = (frame // 120 // 60) * 10 ** 5 + (frame // 120 % 60) * 10 ** 3 + frame % 120
        line = ",".join(["200"] * 32 + ["0"] * 10 + [str(code), "1", "00:00:05:00"] + [repr(float(v)) for v in flat])
        rows.append(line)
        if g in duplicates:
            rows.append(line if duplicates[g] == "same" else line.rsplit(",", 1)[0] + ",1.5")
    return ("\n".join([",".join(GLOVE_COLUMNS), *rows]) + "\n").encode()


def gloves(s, **kwargs):
    labels = s.channels["left_hand"].labels
    out = {}
    for side in ("Left", "Right"):
        side_labels = tuple(l.replace("Left", side) for l in labels)
        out[side] = parse_glove_csv(glove_csv(side, labels_from_state=side_labels, **kwargs), side=side)
    return out


def test_twist_recovers_hinge_angle_exactly_through_any_rest_rotation():
    rest = rot([.3, -.5, .81], 47.)[0]
    phi = np.linspace(-170, 170, 41)
    mats = rest @ rot([1., 0., 0.], phi)
    twist = twist_degrees(mats, np.ones(41, bool))
    np.testing.assert_allclose(twist, phi - phi[0], atol=1e-9)  # relative to the first valid frame
    valid = np.ones(41, bool)
    valid[0] = False
    np.testing.assert_allclose(twist_degrees(mats, valid)[1:], phi[1:] - phi[1], atol=1e-9)
    assert np.isnan(twist_degrees(mats, valid)[0])


def test_twist_ignores_swing_about_orthogonal_axes():
    phi = np.linspace(0, 80, 9)
    swing = rot([0., 0., 1.], 20.)[0]  # constant swing before the hinge cancels in R0^T R(t)
    np.testing.assert_allclose(twist_degrees(swing @ rot([1, 0, 0], phi), np.ones(9, bool)), phi, atol=1e-9)


def test_device_timecode_is_an_exact_120_hz_grid_and_duplicates_are_resolved():
    s = state()
    labels = s.channels["left_hand"].labels
    rec = parse_glove_csv(glove_csv("Left", labels_from_state=labels, duplicates={3: "same", 9: "diff"}),
                          side="Left")
    assert np.all(np.diff(rec.frames) >= 1) and rec.frames[0] == 600
    assert rec.collapsed_duplicates == 1 and rec.ambiguous_frames == 2
    assert 600 + 9 not in set(rec.frames.tolist())  # conflicting samples are both dropped


def test_one_offset_recovers_the_true_alignment_and_validates_both_hands():
    s = state()
    result = validate_hands(s, gloves(s))
    for side in ("Left", "Right"):
        assert result[side].status == "validated" and result[side].offset_frames == OFFSET
        joint = result[side].joints[f"{side}HandIndex2"]
        assert joint["correlation"] > .999999 and joint["centred_mae_degrees"] < 1e-6
        assert len(result[side].joints) == 12  # thumbs are never compared


def test_a_single_disagreeing_joint_fails_only_its_hand():
    s = state(corrupt="RightHandMiddle2")
    result = validate_hands(s, gloves(s))
    assert result["Left"].status == "validated"
    assert result["Right"].status == "failed"
    assert not result["Right"].joints["RightHandMiddle2"]["passed"]


def test_missing_glove_is_reported_not_assumed():
    s = state()
    result = validate_hands(s, {"Left": gloves(s)["Left"]})
    assert result["Right"].status == "no_glove" and result["Left"].status == "validated"


@pytest.mark.parametrize("meta,message", [
    ({"side": "Up", "fpsRate": 120, "masterTimeCodeSource": "Vicon"}, "side"),
    ({"side": "Left", "fpsRate": 60, "masterTimeCodeSource": "Vicon"}, "rate"),
    ({"side": "Left", "fpsRate": 120, "masterTimeCodeSource": "Internal"}, "Vicon"),
])
def test_metadata_contract(meta, message):
    with pytest.raises(ValueError, match=message):
        parse_glove_metadata(json.dumps(meta).encode())


def test_malformed_exports_are_refused():
    s = state()
    good = glove_csv("Left", labels_from_state=s.channels["left_hand"].labels)
    with pytest.raises(ValueError, match="header"):
        parse_glove_csv(b"x" + good, side="Left")
    lines = good.decode().splitlines()
    lines[5] = lines[5].replace(lines[5].split(",")[42], "5120", 1)  # frame 120 is out of range
    with pytest.raises(ValueError, match="out of range"):
        parse_glove_csv("\n".join(lines).encode(), side="Left")
    disordered = good.decode().splitlines()
    disordered[1:] = disordered[1:][::-1]
    with pytest.raises(ValueError, match="disordered"):
        parse_glove_csv("\n".join(disordered).encode(), side="Left")


def test_counter_restart_overlap_is_dropped_symmetrically():
    s = state()
    lines = glove_csv("Left", labels_from_state=s.channels["left_hand"].labels).decode().splitlines()
    # Measured pattern: 000 001 002 000 002 -> rows labelled 000..002 on both sides are ambiguous.
    restart = [lines[1], lines[2], lines[3], lines[1], lines[3]]
    payload = "\n".join([lines[0], *restart, *lines[4:]]).encode()
    rec = parse_glove_csv(payload, side="Left")
    assert rec.frames[0] == 603 and rec.ambiguous_frames == 5
    assert np.all(np.diff(rec.frames) > 0)


def test_unavailable_frames_are_excluded_not_converted():
    from dataclasses import replace
    s = state()
    hand = s.channels["left_hand"]
    valid = hand.valid.copy()
    valid[-3:] = False  # e.g. a terminal reset block outside the plausible subsequence
    masked = replace(hand, valid=valid, values=np.where(valid[..., None], hand.values, 0.),
                     confidence=np.where(valid, 1., 0.))
    s = replace(s, channels={**s.channels, "left_hand": masked})
    result = validate_hands(s, gloves(s))
    assert result["Left"].status == "validated" and result["Left"].offset_frames == OFFSET
    assert result["Left"].joints["LeftHandIndex2"]["samples"] == FRAMES - 3
