"""UWB clip adapter: the verified file format, clock, normalisation, missing parts, hand sides, refusals."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from semlex_synthetic import FakeBlendshapes, T, face_points, fake_resources, rotation  # noqa: E402
from test_de_world_regressor import small_bundle  # noqa: E402
from uwb_synthetic import TrueWorldXY, uwb_clip_json, uwb_payload  # noqa: E402

from signtranslator.data_engineering import uwb_keypoints as uk  # noqa: E402
from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks  # noqa: E402

NAME = "ab-c_D3.000100-000107.json"          # 8 frames: 107 - 100 + 1


def test_clip_names_are_parsed_and_malformed_ones_refused():
    assert uk.parse_clip_name(NAME) == ("ab-c_D3.000100-000107", "ab-c_D3", 100, 107)
    assert uk.parse_clip_name("x/y/v.1.000005-000009") == ("v.1.000005-000009", "v.1", 5, 9)
    for bad in ("nodots.json", "v.12-ab.json", "v.000009-000005.json"):
        with pytest.raises(uk.Refusal):
            uk.parse_clip_name(bad)


def test_size_is_height_first_and_points_are_source_pixels():
    data, *_ = uwb_clip_json(size=(1280, 720))
    clip = uk.parse_clip(json.dumps(data).encode(), NAME)
    assert (clip.width, clip.height) == (1280, 720) and clip.aspect == pytest.approx(16 / 9)
    assert clip.pose.shape == (T, 33, 2) and clip.hands.shape == (T, 2, 21, 2) and clip.face.shape == (T, 478, 2)
    assert clip.frame_count_difference == 0
    np.testing.assert_allclose(clip.pose[0, 11], data["keypoints"][0]["pose_landmarks"][11])


def test_missing_parts_are_nan_and_malformed_blocks_refused():
    data, *_ = uwb_clip_json(missing_left=(1, 2), missing_pose=(5,), missing_face=(0,))
    clip = uk.parse_clip(data, NAME)
    assert np.isnan(clip.hands[[1, 2], 0]).all() and np.isfinite(clip.hands[[0, 3], 0]).all()
    assert np.isnan(clip.pose[5]).all() and np.isnan(clip.face[0]).all()
    for edit, message in ((lambda d: d["keypoints"][3].update(pose_landmarks=[[1., 2.]] * 32), "expected 0 or 33"),
                          (lambda d: d["keypoints"][3].update(left_hand_landmarks=[[1., 2., 3.]] * 21), r"\[x, y\]"),
                          (lambda d: d["keypoints"][3].update(face_landmarks=[["a", 2.]] * 478), "numeric"),
                          (lambda d: d["keypoints"][3].pop("face_landmarks"), "no face_landmarks"),
                          (lambda d: d.update(size=[720]), "height, width"),
                          (lambda d: d.update(keypoints=d["keypoints"][:1]), "1 frames")):
        bad, *_ = uwb_clip_json()
        edit(bad)
        with pytest.raises(uk.Refusal, match=message):
            uk.parse_clip(json.dumps(bad), NAME)
    with pytest.raises(uk.Refusal, match="unreadable JSON"):
        uk.parse_clip(b"{not json", NAME)


def test_frame_count_may_be_one_short_of_the_range_but_not_more_than_the_tolerance():
    data, *_ = uwb_clip_json()
    assert uk.parse_clip(data, "v.000100-000108").frame_count_difference == -1          # 'last - first' clips
    assert uk.parse_clip(data, "v.000100-000109").frame_count_difference == -2
    with pytest.raises(uk.Refusal, match="tolerance"):
        uk.parse_clip(data, "v.000100-000110")


def test_clock_normalisation_and_missing_parts_in_the_landmarks():
    data, body, hands = uwb_clip_json(size=(1280, 720), missing_left=(1, 2), missing_pose=(5,))
    clip = uk.parse_clip(data, NAME)
    landmarks, info = uk.clip_landmarks(clip, 25., regressors=TrueWorldXY(body, hands))
    np.testing.assert_allclose(landmarks.times, (100 + np.arange(T)) / 25.)
    np.testing.assert_allclose(landmarks.pose_image[0], clip.pose[0] / [1280, 720])
    assert landmarks.image_size == (1280, 720) and info["aspect"] == pytest.approx(16 / 9, abs=1e-6)
    assert not landmarks.pose_found[5] and landmarks.pose_found.sum() == T - 1
    assert not landmarks.hands_found[[1, 2, 5], 0].any() and landmarks.hands_found[[0, 3, 4], 0].all()
    assert not landmarks.hands_found[5].any()                       # no pose, no hand (Holistic attaches hands)
    assert (landmarks.pose_visibility[:, 25:] == 0).all() and (landmarks.pose_visibility[5] == 0).all()
    assert info["holistic_hand_frames"] == {"Left": T - 3, "Right": T - 1}
    assert info["hand_assignment"]["Left"]["assigned_same_side"] == T - 3


def test_learned_conversion_solves_the_regressed_points_with_the_generator_flags():
    data, body, hands = uwb_clip_json(size=(1280, 720))
    clip = uk.parse_clip(data, NAME)
    stub = TrueWorldXY(body, hands)
    state, report = uk.convert_clip(clip, 30., source_sha256="a" * 64, fps_source="assumed", regressors=stub,
                                    blendshape_model=FakeBlendshapes())
    landmarks, _ = uk.clip_landmarks(clip, 30., regressors=stub, blendshape_model=FakeBlendshapes())
    reference, _ = canonical_from_landmarks(
        landmarks, sample_id="r", source_id="r", source_sha256="a" * 64, clock_id="r", inference_method="r",
        upright_chest=True, upright_root=True,
        points=dict(world=body, visible=stub.visibility >= .5,
                    hands={s: (h - h[:, :1], np.ones(T, bool)) for s, h in hands.items()}))
    for name in ("body", "left_hand", "right_hand"):
        np.testing.assert_array_equal(state.channels[name].valid, reference.channels[name].valid, err_msg=name)
        np.testing.assert_allclose(state.channels[name].values, reference.channels[name].values, atol=1e-9)
    labels = state.channels["body"].labels
    assert state.channels["body"].valid[:, labels.index("Pelvis")].all()        # upright root
    assert not state.channels["head_rotation"].valid.any() and state.channels["face"].valid.any()
    assert "head_rotation" in report["unavailable"] and report["fps_source"] == "assumed"
    assert state.sample_id == clip.clip_id and "30fps-assumed" in state.clock_id
    method = state.channels["body"].inference_method
    assert "e" * 64 in method and "z-free" in method and "30 fps (assumed)" in method
    assert state.channels["body"].source_sha256 == "a" * 64
    np.testing.assert_allclose(state.timestamps, (100 + np.arange(T)) / 30.)


def test_swapped_hand_blocks_are_reassigned_by_wrist_proximity():
    data, body, hands = uwb_clip_json()
    swapped, *_ = uwb_clip_json(swap_hand_blocks=True)
    stub = TrueWorldXY(body, {"Left": hands["Right"], "Right": hands["Left"]})   # regressor sees the slots swapped
    a, _ = uk.convert_clip(uk.parse_clip(data, NAME), 30., source_sha256="a" * 64, fps_source="assumed",
                           regressors=TrueWorldXY(body, hands))
    b, report = uk.convert_clip(uk.parse_clip(swapped, NAME), 30., source_sha256="a" * 64, fps_source="assumed",
                                regressors=stub)
    for name in ("left_hand", "right_hand"):
        np.testing.assert_array_equal(a.channels[name].valid, b.channels[name].valid)
        np.testing.assert_allclose(a.channels[name].values, b.channels[name].values, atol=1e-9)
    assert report["hand_assignment"]["Left"]["assigned_other_side"] == T


def test_a_z_trained_bundle_and_bad_options_are_refused():
    data, body, hands = uwb_clip_json()
    clip = uk.parse_clip(data, NAME)
    with pytest.raises(ValueError, match="read z"):
        uk.clip_landmarks(clip, 30., regressors=small_bundle())                     # a v1/v2-style bundle
    with pytest.raises(ValueError, match="needs regressors"):
        uk.clip_landmarks(clip, 30.)
    with pytest.raises(ValueError, match="visibility head"):
        uk.clip_landmarks(clip, 30., world="pseudo")
    with pytest.raises(uk.Refusal, match="frame rate"):
        uk.clip_landmarks(clip, 5., regressors=TrueWorldXY(body, hands))
    with pytest.raises(ValueError, match="flags"):
        uk.convert_clip(clip, 30., source_sha256="a" * 64, fps_source="assumed", regressors=TrueWorldXY(body, hands),
                        flags="adapter")
    empty, *_ = uwb_clip_json(missing_pose=tuple(range(T)))
    with pytest.raises(uk.Refusal, match="no frame has a pose"):
        uk.convert_clip(uk.parse_clip(empty, NAME), 30., source_sha256="a" * 64, fps_source="assumed",
                        regressors=TrueWorldXY(body, hands))


def test_pseudo_world_is_the_image_plane():
    data, body, hands = uwb_clip_json(size=(1280, 720))
    clip = uk.parse_clip(data, NAME)
    landmarks, info = uk.clip_landmarks(clip, 30., world="pseudo", visibility="inside")
    assert info["adapter"] == uk.PSEUDO_ADAPTER and info["world"] == "pseudo-world-z0"
    assert not landmarks.pose_world[..., 2].any() and not landmarks.hands_world[..., 2].any()
    np.testing.assert_allclose(landmarks.pose_world[..., 1], clip.pose[..., 1] / 720 / (16 / 9))
    state, report = uk.convert_clip(clip, 30., source_sha256="a" * 64, fps_source="assumed", world="pseudo",
                                    visibility="inside")
    assert "pseudo-3D (x, y/a, 0)" in state.channels["body"].inference_method


def test_affine_head_rotation_recovers_an_orthographic_face():
    resources = fake_resources()
    nose = np.tile([[640., 300.]], (T, 1))
    face = face_points(resources, nose, (1280, 720), yaw=25., pitch=-10.)
    rotations, ok = uk.affine_head_rotation(face[..., :2], 16 / 9, resources.canonical, resources.basis)
    assert ok.all()
    for t in range(T):
        truth = rotation((0, 1, 0), 25. + t) @ rotation((1, 0, 0), -10.)
        np.testing.assert_allclose(rotations[t], truth, atol=1e-6)
    flat = face[..., :2].copy()
    flat[:, :, 1] = flat[:, :1, 1]                                  # a degenerate (collinear) mesh is invalid
    assert not uk.affine_head_rotation(flat, 16 / 9, resources.canonical, resources.basis)[1].any()


def test_head_affine_marks_head_rotation_valid_only_where_the_fit_is():
    data, body, hands = uwb_clip_json(size=(1280, 720), missing_face=(2,))
    clip = uk.parse_clip(data, NAME)
    state, report = uk.convert_clip(clip, 30., source_sha256="a" * 64, fps_source="assumed",
                                    regressors=TrueWorldXY(body, hands), blendshape_model=FakeBlendshapes(),
                                    head="affine", face_resources=fake_resources())
    head = state.channels["head_rotation"]
    assert head.valid[:, 0].tolist() == [t != 2 for t in range(T)]
    assert "scaled-orthographic" in state.channels["body"].inference_method
    with pytest.raises(ValueError, match="face_resources"):
        uk.clip_landmarks(clip, 30., regressors=TrueWorldXY(body, hands), head="affine")
