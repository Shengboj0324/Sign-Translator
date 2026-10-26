"""Left-right mirror of canonical states: exact against mirrored landmarks and forward kinematics."""
import numpy as np

from signtranslator.data_engineering.video_landmarks import ClipLandmarks, canonical_from_landmarks
from signtranslator.generation.canonical_skeleton import skeleton_positions
from signtranslator.pose.mirror import M, mirror_state, swap_side
from signtranslator.pose.rotations import rotation_6d_to_matrix

import torch

POSE_PAIRS = [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28)]
BLEND = ("_neutral", "browDownLeft", "browDownRight", "jawOpen", "mouthLeft", "mouthRight", "eyeBlinkLeft", "eyeBlinkRight")


def _clip(frames, rng):
    return ClipLandmarks(times=np.arange(frames) / 30., pose_world=np.zeros((frames, 33, 3)),
                         pose_visibility=np.ones((frames, 33)), pose_image=np.zeros((frames, 33, 2)),
                         pose_found=np.ones(frames, bool), hands_world=np.zeros((frames, 0, 21, 3)),
                         hands_image=np.zeros((frames, 0, 21, 2)), hands_found=np.zeros((frames, 0), bool),
                         blendshapes=rng.uniform(0, 1, (frames, len(BLEND))), blendshape_names=BLEND,
                         face_matrix=np.tile(np.eye(4), (frames, 1, 1)), face_found=np.ones(frames, bool))


def _points(rng, frames=6):
    """A plausible random upper body and two random hands (y-up, x toward the signer's left)."""
    base = {11: (.18, .45, 0.), 12: (-.18, .45, 0.), 13: (.3, .25, .05), 14: (-.3, .25, .05), 15: (.3, .3, .3),
            16: (-.3, .3, .3), 23: (.1, 0., 0.), 24: (-.1, 0., 0.), 25: (.1, -.4, 0.), 26: (-.1, -.4, 0.),
            27: (.1, -.8, 0.), 28: (-.1, -.8, 0.)}
    world = np.zeros((frames, 33, 3))
    for i, p in base.items():
        world[:, i] = np.asarray(p) + rng.normal(0, .05, (frames, 3))
    visible = np.zeros((frames, 33), bool)
    visible[:, list(base)] = True
    hands = {}
    for side, wrist in (("Left", 15), ("Right", 16)):
        pts = np.zeros((frames, 21, 3))
        pts[:, 0] = world[:, wrist]
        for finger, (dx, start) in enumerate(((.035, 1), (.02, 5), (0., 9), (-.02, 13), (-.04, 17))):
            for k in range(4):
                pts[:, start + k] = pts[:, 0] + np.array([dx, .02 + .07 * (finger > 0) + .025 * k, .01 * k]) \
                    + rng.normal(0, .01, (frames, 3))
        found = np.ones(frames, bool)
        found[0] = side == "Left"          # one frame with only the left hand found
        hands[side] = (pts, found)
    return world, visible, hands


def _state(world, visible, hands, rng, frames=6):
    state, _ = canonical_from_landmarks(_clip(frames, rng), sample_id="s", source_id="src", source_sha256="a" * 64,
                                        clock_id="c", inference_method="synthetic landmarks",
                                        points=dict(world=world, visible=visible, hands=hands))
    return state


def _mirror_points(world, visible, hands):
    w, v = world @ M, visible.copy()
    for a, b in POSE_PAIRS:
        w[:, [a, b]], v[:, [a, b]] = w[:, [b, a]], v[:, [b, a]]
    return w, v, {"Left": (hands["Right"][0] @ M, hands["Right"][1]), "Right": (hands["Left"][0] @ M, hands["Left"][1])}


def _matrices(channel):
    """(n, 3, 3) rotations of the valid entries, in row-major order of the valid mask."""
    return rotation_6d_to_matrix(torch.from_numpy(channel.values[channel.valid].astype(np.float64))).numpy()


def test_mirroring_the_state_equals_solving_the_mirrored_landmarks():
    rng = np.random.default_rng(0)
    world, visible, hands = _points(rng)
    state = _state(world, visible, hands, np.random.default_rng(1))
    expected = _state(*_mirror_points(world, visible, hands), np.random.default_rng(1))
    mirrored = mirror_state(state)
    for name in ("body", "left_hand", "right_hand"):
        got, want = mirrored.channels[name], expected.channels[name]
        assert got.labels == want.labels, name
        np.testing.assert_array_equal(got.valid, want.valid, err_msg=name)
        np.testing.assert_allclose(_matrices(got), _matrices(want), atol=1e-9, err_msg=name)


def test_forward_kinematics_of_the_mirror_is_the_reflected_skeleton():
    rng = np.random.default_rng(2)
    state = _state(*_points(rng), rng)
    joints, positions, _ = skeleton_positions(state)
    mjoints, mpositions, _ = skeleton_positions(mirror_state(state))
    assert set(joints) == set(mjoints)
    for k, j in enumerate(joints):
        np.testing.assert_allclose(mpositions[:, mjoints.index(swap_side(j))], positions[:, k] @ M, atol=1e-9,
                                   err_msg=j)


def test_face_and_blink_swap_sides_and_mirroring_twice_is_identity():
    rng = np.random.default_rng(3)
    state = _state(*_points(rng), rng)
    mirrored = mirror_state(state)
    face, mface = state.channels["face"], mirrored.channels["face"]
    labels = list(face.labels)
    for a, b in (("browDownLeft", "browDownRight"), ("mouthLeft", "mouthRight")):
        np.testing.assert_array_equal(mface.values[:, labels.index(a)], face.values[:, labels.index(b)])
    np.testing.assert_array_equal(mface.values[:, labels.index("jawOpen")], face.values[:, labels.index("jawOpen")])
    np.testing.assert_array_equal(mirrored.channels["blink"].values[:, 0], state.channels["blink"].values[:, 1])
    twice = mirror_state(mirrored)
    for name, channel in state.channels.items():
        np.testing.assert_allclose(twice.channels[name].values, channel.values, atol=1e-9, err_msg=name)
        np.testing.assert_array_equal(twice.channels[name].valid, channel.valid, err_msg=name)
        assert twice.channels[name].labels == channel.labels, name
