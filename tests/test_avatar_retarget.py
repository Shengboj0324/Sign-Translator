"""Canonical-state retarget onto the pinned MakeHuman rig: directions, handedness, face, fills."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.avatar_render.makehuman_face import load_makehuman_face_units
from signtranslator.avatar_render.retarget import (
    BONE_DRIVERS, FACE_UNIT_SOURCES, UNREPRESENTED_BLENDSHAPES, build_retarget_binding, face_weights,
    rest_landmarks, retarget_state,
)
from signtranslator.avatar_render.retarget_check import (
    direction_errors, lateral_agreement, raised_arm_points, state_from_points, structural_handedness,
)
from signtranslator.data_engineering.video_landmarks import BODY_LABELS, CONVENTION, finger_joints
from signtranslator.generation.canonical_skeleton import skeleton_positions
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from signtranslator.pose.rotations import matrix_to_rotation_6d

INTAKE = (Path(__file__).resolve().parents[1] / "Sign Translator Stage Documentation"
          / "evidence/w1-w2-2026-09-26/source-intake")
BLENDSHAPES = ("_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
               "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
               "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
               "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
               "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
               "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
               "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
               "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
               "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
               "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight")


@pytest.fixture(scope="module")
def rig():
    return load_makehuman_rig(INTAKE / "makehuman")


@pytest.fixture(scope="module")
def binding(rig):
    return build_retarget_binding(rig)


@pytest.fixture(scope="module")
def face_units(rig):
    return load_makehuman_face_units(INTAKE / "makehuman-face", rig)


def _channel(values, valid, labels, *, frame="camera-aligned-y-up-metres", convention=CONVENTION, units="unitless"):
    valid = np.asarray(valid, dtype=bool)
    values = np.where(valid[..., None], values, 0.)
    return MotionChannel(values=values, valid=valid, observed=np.zeros_like(valid),
                         confidence=np.where(valid, 1., 0.), labels=tuple(labels), coordinate_frame=frame,
                         units=units, source_id="test", source_sha256="c" * 64, convention=convention,
                         inference_method="synthetic test values" if valid.any() else None)


def random_state(frames=6, seed=0, *, max_angle=1.2, hand_valid=None, face=None, gaze=None, blink=None,
                 head=None, convention=CONVENTION):
    """A canonical state with random local rotations on every body/hand joint and the head."""
    rng = np.random.default_rng(seed)

    def rotations(count):
        vectors = rng.normal(size=(frames, count, 3))
        vectors *= (rng.uniform(0, max_angle, size=(frames, count, 1)) / np.linalg.norm(vectors, axis=-1,
                                                                                       keepdims=True))
        return matrix_to_rotation_6d(torch.from_numpy(Rotation.from_rotvec(vectors.reshape(-1, 3)).as_matrix()
                                                      .reshape(frames, count, 3, 3))).numpy()

    channels = {"body": _channel(rotations(len(BODY_LABELS)), np.ones((frames, len(BODY_LABELS))), BODY_LABELS,
                                 convention=convention)}
    for side, name in (("Left", "left_hand"), ("Right", "right_hand")):
        labels = [j.name for j in finger_joints(side)]
        valid = np.ones((frames, 15), dtype=bool) if hand_valid is None else np.repeat(hand_valid[:, None], 15, 1)
        channels[name] = _channel(rotations(15), valid, labels, convention=convention)
    head_values = rotations(1) if head is None else head
    channels["head_rotation"] = _channel(head_values, np.ones((frames, 1)), ("Head",),
                                         frame="mediapipe-face-geometry-camera-v1")
    if face is not None:
        channels["face"] = _channel(face[..., None], np.ones(face.shape), BLENDSHAPES,
                                    frame="mediapipe-face-blendshape-basis")
    if blink is not None:
        channels["blink"] = _channel(blink[..., None], np.ones(blink.shape), ("eyeBlinkLeft", "eyeBlinkRight"),
                                     frame="mediapipe-eyeBlink-as-eyelid-closure")
    if gaze is not None:
        for side in ("left", "right"):
            channels[f"{side}_gaze"] = _channel(gaze[:, None], np.ones((frames, 1)), (f"{side.title()}Eye",),
                                                frame="mediapipe-face-geometry-camera-v1")
    for name, (_, width, units) in CHANNELS.items():
        if name not in channels:
            channels[name] = MotionChannel(
                values=np.zeros((frames, 1, width)), valid=np.zeros((frames, 1), dtype=bool),
                observed=np.zeros((frames, 1), dtype=bool), confidence=np.zeros((frames, 1)),
                labels=(f"{name}-unavailable",), coordinate_frame="not-estimated", units=units,
                source_id="test", source_sha256="c" * 64, convention=convention)
    clock = np.cumsum(rng.uniform(.02, .05, size=frames))
    return MultichannelMotion(clock.astype(np.float64), "test-clock", "retarget-test", channels)


def pose_all(rig, retargeted):
    return np.stack([rig.pose(torch.from_numpy(np.ascontiguousarray(local))).bone_globals.numpy()
                     for local in retargeted.local_rotations])


def test_binding_is_computed_from_the_rig_and_reproducible(rig, binding):
    assert binding.rig_axes["left_axis"][0] > .99 and binding.rig_axes["facing"][2] > .9
    assert build_retarget_binding(rig).sha256() == binding.sha256()
    assert set(BONE_DRIVERS) <= set(rig.bone_names)
    driven = set(binding.drivers)
    assert driven.isdisjoint(binding.face_bones) and driven.isdisjoint(binding.follower_bones)
    assert driven | set(binding.face_bones) | set(binding.follower_bones) == set(rig.bone_names)
    for side, s in (("Left", "L"), ("Right", "R")):
        for k, finger in enumerate(("Thumb", "Index", "Middle", "Ring", "Pinky"), 1):
            for segment in (1, 2, 3):
                assert binding.drivers[f"finger{k}-{segment}.{s}"][1] == f"{side}Hand{finger}{segment}"
        assert binding.drivers[f"wrist.{s}"][1] == f"{side}Hand"
    assert binding.drivers["head"] == ("Head", "Head", 1.)


@pytest.mark.parametrize("hinge", [True, False])
def test_the_rigs_own_rest_landmarks_pose_the_rig_at_rest(rig, binding, hinge):
    world, visible, hands = rest_landmarks(rig)
    state = state_from_points(np.array([0.]), world, visible, hands)
    retargeted = retarget_state(rig, binding, state, allow_inferred=True, hinge_twist=hinge)
    eye = np.eye(3)
    index = {n: i for i, n in enumerate(rig.bone_names)}
    for bone in binding.drivers:
        if bone in ("head", "neck01", "neck02", "neck03"):
            continue                      # head_rotation is unavailable here: rest relation to the chest
        np.testing.assert_allclose(retargeted.local_rotations[0, index[bone]], eye, atol=1e-9, err_msg=bone)


@pytest.mark.parametrize("hinge", [True, False])
def test_rig_forward_kinematics_reproduces_every_canonical_bone_direction(rig, binding, hinge):
    state = random_state(frames=8, seed=3)
    retargeted = retarget_state(rig, binding, state, allow_inferred=True, hinge_twist=hinge)
    errors = direction_errors(rig, state, retargeted, pose_all(rig, retargeted))
    for group in ("upper_arm", "forearm", "finger", "thumb", "head_rotation", "head_facing"):
        assert errors["groups"][group]["n"] > 0
        assert errors["groups"][group]["max_deg"] < 1e-5, group
    assert len(errors["per_bone"]) == 2 * (2 + 15)


def test_targets_are_what_rig_pose_realises(rig, binding):
    state = random_state(frames=3, seed=5)
    retargeted = retarget_state(rig, binding, state, allow_inferred=True)
    globals_ = pose_all(rig, retargeted)
    np.testing.assert_allclose(globals_[..., :3, :3], retargeted.target_globals, atol=1e-10)


def test_raised_left_arm_raises_the_avatars_left_arm_on_the_viewers_right(rig, binding):
    world, visible, hands = raised_arm_points(rig, "Left")
    state = state_from_points(np.array([0.]), world, visible, hands)
    joints, positions, _ = skeleton_positions(state)
    canonical = dict(zip(joints, positions[0]))
    assert canonical["LeftHand"][0] > 0 and canonical["LeftHand"][1] > canonical["Head"][1]
    retargeted = retarget_state(rig, binding, state, allow_inferred=True)
    globals_ = pose_all(rig, retargeted)[0]
    index = {n: i for i, n in enumerate(rig.bone_names)}
    up = np.array([0., 1., 0.])
    upper = globals_[index["lowerarm01.L"], :3, 3] - globals_[index["upperarm01.L"], :3, 3]
    assert np.degrees(np.arccos(upper @ up / np.linalg.norm(upper))) < 1.
    wrist = globals_[index["wrist.L"], :3, 3]
    head_top = globals_[index["head"], 1, 3]
    assert wrist[0] > 0 and wrist[1] > head_top          # +x is the signer's left = viewer's right
    # the other arm keeps its rest pose
    rest = rig.rest_globals.numpy()
    np.testing.assert_allclose(globals_[index["wrist.R"], :3, 3], rest[index["wrist.R"], :3, 3], atol=1e-9)
    assert structural_handedness(rig)["left_bones_at_positive_x"]


def test_raised_right_arm_is_the_mirror_image(rig, binding):
    world, visible, hands = raised_arm_points(rig, "Right")
    state = state_from_points(np.array([0.]), world, visible, hands)
    globals_ = pose_all(rig, retarget_state(rig, binding, state, allow_inferred=True))[0]
    index = {n: i for i, n in enumerate(rig.bone_names)}
    left = pose_all(rig, retarget_state(rig, binding, state_from_points(
        np.array([0.]), *raised_arm_points(rig, "Left")), allow_inferred=True))[0]
    right_wrist, left_wrist = globals_[index["wrist.R"], :3, 3], left[index["wrist.L"], :3, 3]
    assert right_wrist[0] < 0
    np.testing.assert_allclose(right_wrist * [-1, 1, 1], left_wrist, atol=1e-6)


def test_lateral_side_agrees_frame_by_frame(rig, binding):
    state = random_state(frames=12, seed=11, max_angle=.8)
    retargeted = retarget_state(rig, binding, state, allow_inferred=True)
    agreement = lateral_agreement(rig, state, retargeted, pose_all(rig, retargeted))
    for side in ("left", "right"):
        if agreement[side]["scored_frames"]:
            assert agreement[side]["agreement"] == 1.


def test_inferred_values_need_an_explicit_opt_in(rig, binding):
    state = random_state(frames=2)
    with pytest.raises(ValueError, match="allow_inferred"):
        retarget_state(rig, binding, state, allow_inferred=False)


def test_unsupported_conventions_are_refused(rig, binding):
    state = random_state(frames=2, convention="synthetic-demo-v1")
    with pytest.raises(ValueError, match="convention"):
        retarget_state(rig, binding, state, allow_inferred=True)


def test_missing_hand_frames_are_slerped_counted_and_never_scored(rig, binding):
    frames = 7
    valid = np.array([True, True, False, False, False, True, True])
    state = random_state(frames=frames, seed=2, hand_valid=valid)
    body = state.channels["body"]
    hand_valid = body.valid.copy()
    hand_valid[:, BODY_LABELS.index("LeftHand")] = valid
    state = replace(state, channels={**state.channels, "body": replace(
        body, valid=hand_valid, values=np.where(hand_valid[..., None], body.values, 0.),
        confidence=np.where(hand_valid, 1., 0.))})
    retargeted = retarget_state(rig, binding, state, allow_inferred=True)
    fill = retargeted.canonical.fill
    assert fill["LeftHand"]["filled_frames"] == 3 and fill["LeftHand"]["gaps"] == 1
    assert fill["LeftHandIndex1"]["filled_frames"] == 3
    local = retargeted.canonical.local["LeftHand"]
    t = state.timestamps
    fraction = (t[3] - t[1]) / (t[5] - t[1])
    expected = Rotation.from_matrix(local[1]) * Rotation.from_rotvec(
        fraction * (Rotation.from_matrix(local[1]).inv() * Rotation.from_matrix(local[5])).as_rotvec())
    np.testing.assert_allclose(local[3], expected.as_matrix(), atol=1e-9)
    errors = direction_errors(rig, state, retargeted, pose_all(rig, retargeted))
    assert errors["groups"]["finger_left"]["n"] == 12 * (frames - 3)
    assert errors["groups"]["finger_right"]["n"] == 12 * (frames - 3)   # finger channels share the gap
    assert errors["groups"]["forearm_left"]["n"] == frames


def test_a_never_valid_hand_keeps_the_rest_relation_to_its_forearm(rig, binding):
    state = random_state(frames=3, seed=4, hand_valid=np.zeros(3, dtype=bool))
    body = state.channels["body"]
    valid = body.valid.copy()
    valid[:, BODY_LABELS.index("RightHand")] = False
    state = replace(state, channels={**state.channels, "body": replace(
        body, valid=valid, values=np.where(valid[..., None], body.values, 0.), confidence=np.where(valid, 1., 0.))})
    retargeted = retarget_state(rig, binding, state, allow_inferred=True, hinge_twist=False)
    assert retargeted.canonical.fill["RightHand"]["fallback"].startswith("never valid")
    index = {n: i for i, n in enumerate(rig.bone_names)}
    rest = rig.rest_globals.numpy()[:, :3, :3]
    # no forearm twist share without a hand change: wrist and fingers rigid with the forearm at rest relation
    g = retargeted.target_globals
    expected = g[:, index["lowerarm01.R"]] @ rest[index["lowerarm01.R"]].T @ rest[index["wrist.R"]]
    np.testing.assert_allclose(g[:, index["wrist.R"]], expected, atol=1e-9)


def test_forearm_twist_is_shared_without_changing_any_direction(rig, binding):
    world, visible, hands = rest_landmarks(rig)
    state = state_from_points(np.array([0.]), world, visible, hands)
    body = state.channels["body"]
    k = BODY_LABELS.index("LeftHand")
    values = body.values.copy()
    forearm_axis = np.array([1., 0., 0.])        # the left forearm's rest direction in its own frame
    local = Rotation.from_matrix(_six_to_matrix(values[0, k]))
    pronation = Rotation.from_rotvec(forearm_axis * np.pi / 2) * local     # 90 degrees about the forearm
    values[0, k] = matrix_to_rotation_6d(torch.from_numpy(pronation.as_matrix())).numpy()
    state = replace(state, channels={**state.channels, "body": replace(body, values=values)})
    retargeted = retarget_state(rig, binding, state, allow_inferred=True)
    index = {n: i for i, n in enumerate(rig.bone_names)}
    twist = Rotation.from_matrix(retargeted.local_rotations[0, index["lowerarm02.L"]])
    rest = rig.rest_globals.numpy()
    relative = rest[index["lowerarm01.L"], :3, :3].T @ rest[index["lowerarm02.L"], :3, :3]
    assert np.allclose(relative, np.eye(3), atol=1e-6)
    angle = np.degrees(np.linalg.norm(twist.as_rotvec()))
    assert 40. < angle < 50.                                  # half of a 90 degree pronation
    assert abs(abs(twist.as_rotvec()[1]) - np.linalg.norm(twist.as_rotvec())) < 1e-9   # about the bone axis
    errors = direction_errors(rig, state, retargeted, pose_all(rig, retargeted))
    assert errors["groups"]["forearm"]["max_deg"] < 1e-6


def _six_to_matrix(six):
    from signtranslator.avatar_render.retarget import matrices_from_6d
    return matrices_from_6d(np.asarray(six))


def test_elbow_bends_about_the_rest_hinge_with_the_hinge_twist(rig, binding):
    state = random_state(frames=10, seed=8, max_angle=1.)
    retargeted = retarget_state(rig, binding, state, allow_inferred=True, hinge_twist=True)
    index = {n: i for i, n in enumerate(rig.bone_names)}
    local = Rotation.from_matrix(retargeted.local_rotations[:, index["lowerarm01.L"]]).as_rotvec()
    rest = rig.rest_globals.numpy()
    heads = rest[:, :3, 3]
    u = heads[index["lowerarm01.L"]] - heads[index["upperarm01.L"]]
    f = heads[index["wrist.L"]] - heads[index["lowerarm01.L"]]
    hinge_local = rest[index["lowerarm01.L"], :3, :3].T @ np.cross(u, f)
    hinge_local /= np.linalg.norm(hinge_local)
    angles = np.linalg.norm(local, axis=1)
    bent = angles > np.radians(5)
    off_axis = np.linalg.norm(np.cross(local[bent] / angles[bent, None], hinge_local), axis=1)
    assert bent.any() and np.median(off_axis) < .05


def test_face_coefficients_drive_the_mapped_pose_units(rig, binding, face_units):
    frames = 3
    face = np.zeros((frames, len(BLENDSHAPES)))
    face[1, BLENDSHAPES.index("jawOpen")] = 1.
    face[2, BLENDSHAPES.index("browInnerUp")] = .5
    state = random_state(frames=frames, face=face, head=np.tile([1., 0, 0, 0, 1, 0], (frames, 1, 1)))
    weights, record = face_weights(state, baseline="none")
    assert weights["JawDrop"].tolist() == [0., 1., 0.]
    assert weights["LeftInnerBrowUp"][2] == weights["RightInnerBrowUp"][2] == .5
    assert set(record["unrepresented_present"]) == set(UNREPRESENTED_BLENDSHAPES)
    assert not record["unmapped_present"]
    retargeted = retarget_state(rig, binding, state, allow_inferred=True, face_units=face_units, face_baseline="none")
    index = {n: i for i, n in enumerate(rig.bone_names)}
    jaw = index["jaw"]
    np.testing.assert_allclose(retargeted.local_rotations[1, jaw], face_units.rotations("JawDrop")[jaw].numpy(),
                               atol=1e-9)
    np.testing.assert_allclose(retargeted.local_rotations[0, jaw], np.eye(3), atol=1e-12)
    assert retargeted.face["rendered"] and retargeted.face["eye_direction_source"].startswith("eyeLook")


def test_every_mediapipe_blendshape_is_rendered_or_declared_unrepresented():
    used = {name for sources in FACE_UNIT_SOURCES.values() for name, _ in sources}
    assert used | set(UNREPRESENTED_BLENDSHAPES) == set(BLENDSHAPES)
    assert used.isdisjoint(UNREPRESENTED_BLENDSHAPES)


def test_clip_median_baseline_and_blink_channel(rig):
    frames = 5
    face = np.full((frames, len(BLENDSHAPES)), .0)
    face[:, BLENDSHAPES.index("browDownLeft")] = [.8, .8, .8, .9, 1.]
    blink = np.array([[.1, .1], [.1, .1], [.95, .2], [.1, .1], [.1, .1]])
    state = random_state(frames=frames, face=face, blink=blink)
    weights, record = face_weights(state, baseline="clip_median")
    np.testing.assert_allclose(weights["LeftBrowDown"], [0., 0., 0., .5, 1.], atol=1e-9)
    assert record["baseline_medians"]["eyeBlinkLeft"] == pytest.approx(.1)
    assert weights["LeftUpperLidClosed"].argmax() == 2 and weights["RightUpperLidClosed"][2] > 0
    raw, _ = face_weights(state, baseline="none")
    assert raw["LeftBrowDown"][0] == pytest.approx(.8)


def test_valid_gaze_aims_the_eyes_and_replaces_eyelook(rig, binding, face_units):
    frames = 2
    direction = np.array([.3, -.2, 1.])
    direction /= np.linalg.norm(direction)
    face = np.zeros((frames, len(BLENDSHAPES)))
    face[:, BLENDSHAPES.index("eyeLookUpLeft")] = 1.           # must be ignored where gaze is valid
    state = random_state(frames=frames, face=face, gaze=np.tile(direction, (frames, 1)))
    retargeted = retarget_state(rig, binding, state, allow_inferred=True, face_units=face_units,
                                face_baseline="none")
    globals_ = pose_all(rig, retargeted)
    index = {n: i for i, n in enumerate(rig.bone_names)}
    for eye in ("eye.L", "eye.R"):
        np.testing.assert_allclose(globals_[:, index[eye], :3, 1], np.tile(direction, (frames, 1)), atol=1e-9)
    assert retargeted.face["gaze"] == {"L": frames, "R": frames}


def test_a_long_gap_eases_to_the_rest_relation_instead_of_gliding(rig, binding):
    valid = np.array([True, True, False, False, False, True, True])
    state = random_state(frames=7, seed=6, hand_valid=valid)
    state = replace(state, timestamps=np.array([0., .04, .1, 1., 1.9, 2., 2.04]))
    retargeted = retarget_state(rig, binding, state, allow_inferred=True)
    rest = binding.rest_locals()["LeftHandIndex2"]
    local = retargeted.canonical.local["LeftHandIndex2"]
    np.testing.assert_allclose(local[3], rest, atol=1e-9)                  # 0.96 s from either neighbour
    angle = lambda a, b: np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec())
    # 0.06 s after the last valid frame: 24% of the way from it to rest
    assert angle(local[1], local[2]) == pytest.approx(.24 * angle(local[1], rest), rel=1e-6)
    assert retargeted.canonical.fill["LeftHandIndex2"]["frames_eased_toward_rest"] == 3


def test_a_left_blink_closes_the_signers_left_eye(rig, binding, face_units):
    frames = 2
    face = np.zeros((frames, len(BLENDSHAPES)))
    face[1, BLENDSHAPES.index("eyeBlinkLeft")] = 1.
    state = random_state(frames=frames, face=face, head=np.tile([1., 0, 0, 0, 1, 0], (frames, 1, 1)))
    retargeted = retarget_state(rig, binding, state, allow_inferred=True, face_units=face_units, face_baseline="none")
    open_, closed = (rig.pose(torch.from_numpy(np.ascontiguousarray(local))).vertices.numpy()
                     for local in retargeted.local_rotations)
    moved = np.linalg.norm(closed - open_, axis=1)
    index = {n: i for i, n in enumerate(rig.bone_names)}
    head = retargeted.target_globals[0, index["head"]]
    centre = rig.pose(torch.from_numpy(np.ascontiguousarray(retargeted.local_rotations[0]))).bone_globals[
        index["head"], :3, 3].numpy()
    lateral = ((open_[moved > .5 * moved.max()] - centre) @ head)[:, 0]   # head-frame x: +x = signer's left
    assert len(lateral) and (lateral > 0).all()
