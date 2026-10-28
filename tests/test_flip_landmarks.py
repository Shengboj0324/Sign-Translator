"""Horizontal flip of raw landmarks (Sem-Lex un-mirroring): an involution whose solve equals the mirrored state."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from handedness_check import wrist_motion  # noqa: E402

from signtranslator.data_engineering.video_landmarks import (  # noqa: E402
    POSE_MIRROR_PAIRS, Y_UP, ClipLandmarks, assign_hands, canonical_from_landmarks, flip_landmarks)
from signtranslator.generation.canonical_skeleton import skeleton_positions  # noqa: E402
from signtranslator.pose.mirror import M, mirror_state, swap_side  # noqa: E402
from signtranslator.pose.rotations import rotation_6d_to_matrix  # noqa: E402

BLEND = ("_neutral", "browDownLeft", "browDownRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookInLeft",
         "eyeLookInRight", "jawOpen", "mouthLeft", "mouthRight")
SCALE = .8       # normalised image units per metre (x to the image right, y down, as MediaPipe's raw axes)


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _clip(rng, frames=12, right_motion=0.):
    """A raw MediaPipe clip (y-down world axes): an upper body with hips, two hands in reversed slots (slot 0 is the
    right hand), blendshapes, a face matrix; the right wrist swings by ``right_motion`` metres."""
    yup = np.zeros((frames, 33, 3))
    base = {0: (0., .65, .1), 2: (.03, .68, .08), 5: (-.03, .68, .08), 11: (.18, .45, 0.), 12: (-.18, .45, 0.),
            13: (.3, .25, .05), 14: (-.3, .25, .05), 15: (.3, .3, .3), 16: (-.3, .3, .3), 23: (.1, 0., 0.),
            24: (-.1, 0., 0.), 25: (.1, -.4, 0.), 26: (-.1, -.4, 0.), 27: (.1, -.8, 0.), 28: (-.1, -.8, 0.)}
    for i, p in base.items():
        yup[:, i] = np.asarray(p) + rng.normal(0, .02, (frames, 3))
    t = np.arange(frames)
    yup[:, 16] += right_motion * np.stack([np.sin(t), np.cos(t), .3 * np.sin(2 * t)], 1)
    yup[:, 14] += .3 * right_motion * np.stack([np.sin(t), np.cos(t), 0 * t], 1)
    visibility = np.zeros((frames, 33))
    visibility[:, list(base)] = 1.
    world = yup @ Y_UP                                  # back to MediaPipe's raw (y-down) axes
    image = .5 + SCALE * world[..., :2]
    hands_world, hands_image = np.zeros((frames, 2, 21, 3)), np.zeros((frames, 2, 21, 2))
    for slot, wrist in ((0, 16), (1, 15)):
        pts = np.zeros((frames, 21, 3))
        for finger, (dx, start) in enumerate(((.035, 1), (.02, 5), (0., 9), (-.02, 13), (-.04, 17))):
            for k in range(4):
                pts[:, start + k] = np.array([dx, .02 + .07 * (finger > 0) + .025 * k, .01 * k]) \
                    + rng.normal(0, .01, (frames, 3))
        hands_world[:, slot] = pts @ Y_UP
        hands_image[:, slot] = image[:, wrist][:, None] + SCALE * hands_world[:, slot, :, :2]
    found = np.ones((frames, 2), bool)
    found[0, 0] = False                                  # one frame with only the left hand found
    hands_image[~found] = 0.
    hands_world[~found] = 0.
    face = np.tile(np.eye(4), (frames, 1, 1))
    for k in range(frames):
        face[k, :3, :3] = _rotation(rng)
        face[k, :3, 3] = rng.normal(0, 5., 3) + [0, 0, -50.]
    return ClipLandmarks(times=t / 30., pose_world=world, pose_visibility=visibility, pose_image=image,
                         pose_found=np.ones(frames, bool), hands_world=hands_world, hands_image=hands_image,
                         hands_found=found, blendshapes=rng.uniform(0, 1, (frames, len(BLEND))), blendshape_names=BLEND,
                         face_matrix=face, face_found=np.ones(frames, bool), image_size=(640, 480))


def _solve(clip):
    state, _ = canonical_from_landmarks(clip, sample_id="s", source_id="src", source_sha256="a" * 64, clock_id="c",
                                        inference_method="synthetic", upright_chest=True, upright_root=True)
    return state


def _matrices(channel):
    return rotation_6d_to_matrix(torch.from_numpy(channel.values[channel.valid].astype(np.float64))).numpy()


def test_flipping_twice_is_the_identity():
    clip = _clip(np.random.default_rng(0))
    twice = flip_landmarks(flip_landmarks(clip))
    for name in ("times", "pose_world", "pose_visibility", "pose_image", "pose_found", "hands_world", "hands_image",
                 "hands_found", "blendshapes", "face_matrix", "face_found"):
        np.testing.assert_allclose(getattr(twice, name), getattr(clip, name), atol=1e-12, err_msg=name)
    assert twice.blendshape_names == clip.blendshape_names and twice.image_size == clip.image_size
    once = flip_landmarks(clip)
    assert not np.allclose(once.pose_image, clip.pose_image)
    for a, b in POSE_MIRROR_PAIRS:                       # pairs swap, and x reflects
        np.testing.assert_allclose(once.pose_image[:, a, 0], 1 - clip.pose_image[:, b, 0])
        np.testing.assert_allclose(once.pose_world[:, a], clip.pose_world[:, b] * [-1, 1, 1])


def test_hands_are_reassigned_to_the_other_side():
    clip = _clip(np.random.default_rng(1))
    before, after = assign_hands(clip), assign_hands(flip_landmarks(clip))
    for side, other in (("Left", "Right"), ("Right", "Left")):
        np.testing.assert_array_equal(after[side][1], before[other][1])
        np.testing.assert_allclose(after[side][0], before[other][0] * [-1, 1, 1], atol=1e-12)
    assert before["Left"][1].all() and not before["Right"][1][0]           # the frame with only the left hand


def test_solving_the_flipped_landmarks_equals_the_mirrored_state():
    clip = _clip(np.random.default_rng(2))
    state, flipped = _solve(clip), _solve(flip_landmarks(clip))
    mirrored = mirror_state(state)
    for name in ("body", "left_hand", "right_hand", "head_rotation"):
        got, want = flipped.channels[name], mirrored.channels[name]
        assert got.labels == want.labels, name
        np.testing.assert_array_equal(got.valid, want.valid, err_msg=name)
        assert got.valid.any(), name
        np.testing.assert_allclose(_matrices(got), _matrices(want), atol=1e-9, err_msg=name)
    for name in ("head_translation", "face", "blink"):
        got, want = flipped.channels[name], mirrored.channels[name]
        np.testing.assert_array_equal(got.valid, want.valid, err_msg=name)
        np.testing.assert_allclose(got.values, want.values, atol=1e-12, err_msg=name)


def test_a_right_arm_motion_becomes_a_left_arm_motion():
    clip = _clip(np.random.default_rng(3), frames=30, right_motion=.12)
    state, flipped = _solve(clip), _solve(flip_landmarks(clip))
    before, after = wrist_motion(state), wrist_motion(flipped)
    assert before["side"] == "Right" and after["side"] == "Left"
    assert np.isclose(after["left_path"], before["right_path"]) and np.isclose(after["right_path"], before["left_path"])
    joints, p, _ = skeleton_positions(state)
    joints_f, q, _ = skeleton_positions(flipped)
    assert joints == joints_f
    for k, name in enumerate(joints):                    # every flipped joint is M times its counterpart
        np.testing.assert_allclose(q[:, k], p[:, joints.index(swap_side(name))] @ M, atol=1e-9, err_msg=name)


def test_blink_and_blendshape_scores_swap_sides():
    clip = _clip(np.random.default_rng(4))
    once = flip_landmarks(clip)
    names = list(BLEND)
    for n in names:
        np.testing.assert_array_equal(once.blendshapes[:, names.index(n)], clip.blendshapes[:, names.index(swap_side(n))])
    state, flipped = _solve(clip), _solve(once)
    np.testing.assert_array_equal(flipped.channels["blink"].values[:, 0], state.channels["blink"].values[:, 1])
    np.testing.assert_array_equal(flipped.channels["blink"].values[:, 1], state.channels["blink"].values[:, 0])
    jaw = names.index("jawOpen")
    np.testing.assert_array_equal(once.blendshapes[:, jaw], clip.blendshapes[:, jaw])
