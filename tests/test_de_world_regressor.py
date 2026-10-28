"""Image-to-world regressors: feature invariances, exact target inversion, bundle round trip, adapter integration."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semlex_synthetic import FakeBlendshapes, T, fake_resources, rotation, semlex_array  # noqa: E402

from signtranslator.data_engineering import semlex_pose as sp  # noqa: E402
from signtranslator.data_engineering import world_regressor as wr  # noqa: E402
from signtranslator.data_engineering.video_landmarks import Y_UP, canonical_from_landmarks  # noqa: E402


def hand_points(n=6, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=.02, size=(21, 3))
    base[9] += [0., -.06, 0.]
    return base[None] + rng.normal(scale=.002, size=(n, 21, 3)) + [.5, .5, 0.]


def test_hand_features_are_invariant_to_in_plane_rotation_translation_and_scale():
    image = hand_points()
    f0, _, _, ok = wr.hand_features(image, 4 / 3)
    assert ok.all()
    turned = image.copy()
    centre = image[:, :1, :2] * [1, .75]
    q = (image[..., :2] * [1, .75]) - centre
    turned[..., :2] = (q @ rotation((0, 0, 1), 37.)[:2, :2].T * 1.8 + centre + [.1, -.05]) / [1, .75]
    turned[..., 2] *= 1.8
    f1, *_ = wr.hand_features(turned, 4 / 3)
    np.testing.assert_allclose(f0, f1, atol=1e-5)


def test_hand_target_inverts_exactly():
    image, world = hand_points(seed=1), hand_points(seed=2) - .5
    _, rot, _, _ = wr.hand_features(image, 1.)
    back = wr.hand_output_to_world(wr.hand_target(world, rot), rot)
    np.testing.assert_allclose(back, world - world[:, :1], atol=1e-6)


def test_pose_features_are_relative_to_the_shoulders():
    array, *_ = semlex_array((640, 480))
    pose = array[:, 478:511]
    f0, ok = wr.pose_features(pose, 4 / 3)
    shifted = pose + [.05, .02, 0.]
    f1, _ = wr.pose_features(shifted, 4 / 3)
    assert ok.all()
    np.testing.assert_allclose(f0[:, :75], f1[:, :75], atol=1e-5)          # relative part unchanged, context moves
    assert not np.allclose(f0[:, 75:], f1[:, 75:])
    target = wr.pose_target(np.repeat(np.arange(99.).reshape(1, 33, 3), 2, 0))
    np.testing.assert_allclose(target[:, 11] + target[:, 12], 0., atol=1e-5)


def small_bundle(seed=0):
    torch.manual_seed(seed)
    hand = {s: (wr.MLP(60, 63, width=16, depth=1).eval(), np.zeros(60, np.float32), np.ones(60, np.float32))
            for s in ("Left", "Right")}
    pose = (wr.MLP(79, 75 + 33, width=16, depth=1).eval(), np.zeros(79, np.float32), np.ones(79, np.float32))
    return wr.WorldRegressors(hand=hand, pose=pose)


def test_bundle_round_trip_and_pinned_hash():
    bundle = small_bundle()
    payload = bundle.save()
    loaded = wr.WorldRegressors.load(payload)
    image = hand_points()
    np.testing.assert_array_equal(bundle.hands(image, 1., "Left")[0], loaded.hands(image, 1., "Left")[0])
    array, *_ = semlex_array((640, 480))
    a, b = bundle.pose_world(array[:, 478:511], 4 / 3), loaded.pose_world(array[:, 478:511], 4 / 3)
    np.testing.assert_array_equal(a[0], b[0]) and np.testing.assert_array_equal(a[1], b[1])
    assert ((a[1] >= 0) & (a[1] <= 1)).all() and not a[0][:, 25:].any()
    with pytest.raises(ValueError, match="SHA-256"):
        wr.WorldRegressors.load(payload, expected_sha256="0" * 64)


class TrueWorld:
    """A stand-in regressor that returns the synthetic clip's true 3D points (MediaPipe axes) and a set visibility."""
    sha256 = "f" * 64

    def __init__(self, body, hands, visibility):
        self.body, self.hand, self.visibility = body @ Y_UP, {s: h @ Y_UP for s, h in hands.items()}, visibility

    def pose_world(self, image, aspect):
        return self.body, self.visibility, np.ones(len(image), bool)

    def hands(self, image, aspect, side):
        return self.hand[side] - self.hand[side][:, :1], np.ones(len(image), bool)


def test_the_adapter_solves_the_regressed_world_points():
    array, body, hands = semlex_array((640, 640))
    visibility = np.ones((T, 33))
    visibility[:, 23:] = 0.
    visibility[:, 14] = 0.                                     # learned: the right elbow is not visible
    stub = TrueWorld(body, hands, visibility)
    kw = dict(blendshape_model=FakeBlendshapes(), image_size=(640, 640), video_id="v", source_sha256="a" * 64)
    state, report = sp.convert_array(array, 1000. * T / 30, fake_resources(), regressors=stub, learned_visibility=True,
                                     flags="upright-chest", **kw)
    reference, _ = canonical_from_landmarks(
        sp.holistic_clip(array, 1000. * T / 30, fake_resources(), image_size=(640, 640),
                         blendshape_model=FakeBlendshapes())[0],
        sample_id="r", source_id="r", source_sha256="a" * 64, clock_id="r", inference_method="r", upright_chest=True,
        points=dict(world=body, visible=visibility >= .5,
                    hands={s: (h - h[:, :1], np.ones(T, bool)) for s, h in hands.items()}))
    for name in ("body", "left_hand", "right_hand"):
        np.testing.assert_array_equal(state.channels[name].valid, reference.channels[name].valid, err_msg=name)
        np.testing.assert_allclose(state.channels[name].values, reference.channels[name].values, atol=1e-9)
    labels = state.channels["body"].labels
    assert not state.channels["body"].valid[:, labels.index("RightUpperArm")].any()
    assert not state.channels["body"].valid[:, labels.index("Pelvis")].any()        # no upright root
    assert report["world"] == "learned:" + "f" * 64 and report["flags"] == "upright-chest"
    assert "learned image-to-world" in state.channels["body"].inference_method
    with pytest.raises(ValueError):
        sp.convert_array(array, 1000. * T / 30, fake_resources(), flags="none", **kw)
