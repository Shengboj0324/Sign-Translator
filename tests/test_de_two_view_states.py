"""Session pipeline end to end: lag recovery, calibration, triangulated IK, gate."""
import numpy as np
from scipy.spatial.transform import Rotation

from signtranslator.data_engineering.landmark_kinematics import unit
from signtranslator.data_engineering.sapiens_keypoints import HAND21
from signtranslator.data_engineering.two_view_geometry import intrinsics, project, triangulate
from signtranslator.data_engineering.two_view_states import (
    ClipKeypoints, aligned, calibrate_session, hand_skeleton, refine_hands, rigid_bones, rotation_inputs, session_gate,
    synchronise,
    triangulated_bone_cv, triangulated_points,
)
from signtranslator.data_engineering.video_landmarks import BODY_JOINTS, Y_UP, solve_rotations

SIZE = (1280, 720)


def goliath_motion(frames=120, seed=0):
    """Rigid articulated signer in Goliath indices (camera coords, metres), smooth motion."""
    rng = np.random.default_rng(seed)
    parents = {5: None, 6: 5, 9: 5, 10: 6, 7: 5, 62: 7, 8: 6, 41: 8, 11: 9, 13: 11, 12: 10, 14: 12, 0: 5}
    for side, wrist in (("Left", 62), ("Right", 41)):
        hand = HAND21[side]
        for f in range(5):
            chain = (wrist, *hand[1 + 4 * f: 5 + 4 * f])
            for a, b in zip(chain, chain[1:]):
                parents[b] = a
    order = [5, 6, 9, 10, 0, 7, 8, 11, 12, 13, 14, 62, 41] + [j for j in parents if j not in
                                                                (5, 6, 9, 10, 0, 7, 8, 11, 12, 13, 14, 62, 41)]
    lengths = {j: rng.uniform(.02, .05) if j not in (6, 9, 10, 7, 8, 62, 41, 11, 12, 13, 14, 0) else
               {6: .38, 9: .5, 10: .5, 7: .3, 8: .3, 62: .26, 41: .26, 11: .45, 12: .45, 13: .42, 14: .42, 0: .3}[j]
               for j in parents if parents[j] is not None}
    base_dirs = {j: unit(rng.normal(size=3))[0] for j in lengths}
    base_dirs.update({6: np.array([-1., 0, 0]), 9: np.array([0., 1, 0]), 10: np.array([0., 1, 0]),
                      0: np.array([0., -1, 0]), 11: np.array([0., 1, 0]), 12: np.array([0., 1, 0]),
                      13: np.array([0., 1, 0]), 14: np.array([0., 1, 0])})
    points = np.zeros((frames, 308, 3))
    for t in range(frames):
        points[t, 5] = [.19, -.1 + .01 * np.sin(t / 9), 3.]
        for j in order[1:]:
            wobble = Rotation.from_rotvec(.6 * np.sin(t / 11 + j) * unit(np.array([1., j % 3, 1.]))[0]).as_matrix()
            direction = base_dirs[j] if j in (6, 9, 10, 0) else wobble @ base_dirs[j]
            points[t, j] = points[t, parents[j]] + lengths[j] * direction
    return points


def cameras(f1=1150., f2=1400.):
    target = np.array([0., 0., 3.])
    rotation = Rotation.from_euler("y", -75, degrees=True).as_matrix()
    t2 = -rotation @ (target - rotation.T @ np.array([0., 0., 3.1]))
    return (intrinsics(f1, SIZE) @ np.c_[np.eye(3), np.zeros(3)], intrinsics(f2, SIZE) @ np.c_[rotation, t2])


def clips(lag=2, noise=.6):
    points = goliath_motion()
    p1, p2 = cameras()
    rng = np.random.default_rng(5)
    used = sorted({i for b in rigid_bones() for i in b} | {0})
    conf = np.zeros((len(points), 308))
    conf[:, used] = 1.
    def view(p):
        out = np.zeros((len(points), 308, 2))
        out[:, used] = project(p, points[:, used].reshape(-1, 3)).reshape(len(points), len(used), 2)
        return out + rng.normal(scale=noise, size=out.shape)
    front, side = view(p1), view(p2)
    times = np.arange(len(points)) / 30.
    # The lateral clip starts `lag` frames earlier: side[t + lag] shows front[t].
    side = np.concatenate((side[:lag] * 0 + side[:1], side))[:len(points)] if lag else side
    return ClipKeypoints("f", times, front, conf), ClipKeypoints("s", times, side, conf), points


def test_session_pipeline_recovers_lag_cameras_and_exact_rotations():
    front, side, truth = clips()
    sync = synchronise(front, side)
    assert sync["lag_frames"] == 2 and sync["passed"]
    times, fp, fc, sp, sc = aligned(front, side, sync["lag_frames"])
    calibration = calibrate_session([(fp, fc, sp, sc)])
    # Default session calibration also fits one radial coefficient per camera; on these
    # undistorted views it must stay near zero, at a small cost in focal precision.
    assert abs(calibration.focal1 / 1150 - 1) < .04 and abs(calibration.focal2 / 1400 - 1) < .04
    assert max(abs(c) for c in calibration.distortion) < .05
    points, usable = triangulated_points(calibration, fp, fc, sp, sc)
    cv = triangulated_bone_cv(points)
    gate = session_gate(calibration, [sync], cv, monocular_cv=.08)
    assert gate["passed"], gate["reasons"]
    inputs = rotation_inputs(points, usable)
    rotations, hands = solve_rotations(inputs["world"], inputs["visible"], inputs["hands"])
    # Compare the solved upper-arm direction with the true one (scale-free, so exact up to noise).
    true = truth[:len(points)] @ Y_UP.T
    joint = next(j for j in BODY_JOINTS if j.name == "LeftForearm")
    solved_chest = rotations["Pelvis"][0] @ rotations["Chest"][0]
    upper = solved_chest @ rotations["LeftUpperArm"][0]
    forearm = upper @ rotations["LeftForearm"][0]
    observed, _ = unit(true[:, 62] - true[:, 7])
    predicted = np.einsum("tij,j->ti", forearm, unit(np.asarray(joint.rest))[0])
    angles = np.degrees(np.arccos(np.clip((observed * predicted).sum(1), -1, 1)))
    assert np.median(angles) < 1.5
    assert len(hands["Left"]) == 15


def test_gate_refuses_failures_with_reasons():
    front, side, _ = clips()
    times, fp, fc, sp, sc = aligned(front, side, 2)
    calibration = calibrate_session([(fp, fc, sp, sc)])
    bad_sync = [dict(lag_frames=2, correlation=.3, passed=False)]
    gate = session_gate(calibration, bad_sync, .05, monocular_cv=.01)
    assert not gate["passed"] and len(gate["reasons"]) == 2


def test_merged_state_takes_body_from_triangulation_and_face_from_monocular():
    from tests.test_de_video_landmarks import clip as mp_clip, convert
    from signtranslator.data_engineering.two_view_states import triangulated_state
    front, side, _ = clips()
    times, fp, fc, sp, sc = aligned(front, side, 2)
    calibration = calibrate_session([(fp, fc, sp, sc)])
    points, usable = triangulated_points(calibration, fp, fc, sp, sc)
    inputs = rotation_inputs(points, usable)
    rotations, hands = solve_rotations(inputs["world"], inputs["visible"], inputs["hands"])
    import numpy as np
    from dataclasses import replace
    base = mp_clip()
    frames = len(front.times)
    tile = lambda a: np.repeat(a[:1], frames, axis=0)
    long_clip = replace(base, times=np.arange(frames) / 30., pose_world=tile(base.pose_world),
                        pose_visibility=tile(base.pose_visibility), pose_image=tile(base.pose_image),
                        pose_found=tile(base.pose_found), hands_world=tile(base.hands_world),
                        hands_image=tile(base.hands_image), hands_found=tile(base.hands_found),
                        blendshapes=tile(base.blendshapes), face_matrix=tile(base.face_matrix), face_found=tile(base.face_found))
    monocular, _ = convert(long_clip)
    merged = triangulated_state(monocular, slice(0, len(times)), rotations, hands, source_id="tv",
                                source_sha256="c" * 64, inference_method="two-view")
    merged.validate()
    assert merged.channels["body"].coordinate_frame == "two-view-triangulated-camera1-y-up"
    assert merged.channels["face"].valid.all() and merged.channels["face"].source_id == "src"
    assert not merged.channels["head_translation"].valid.any() and len(merged.timestamps) == len(times)


def test_hand_refinement_reduces_finger_direction_error_and_keeps_arms():
    truth = goliath_motion()
    p1, p2 = cameras()
    rng = np.random.default_rng(9)
    flat = truth.reshape(-1, 3)
    shape = truth.shape[:2] + (2,)
    v1 = project(p1, flat).reshape(shape) + rng.normal(scale=1.5, size=shape)
    v2 = project(p2, flat).reshape(shape) + rng.normal(scale=1.5, size=shape)
    used = sorted({i for b in rigid_bones() for i in b})
    conf = np.zeros(truth.shape[:2])
    conf[:, used] = 1.
    raw = np.full(truth.shape, np.nan)
    raw[:, used] = triangulate(p1, p2, v1[:, used].reshape(-1, 2), v2[:, used].reshape(-1, 2)).reshape(len(truth), -1, 3)
    refined = refine_hands(raw, v1, v2, conf, conf, p1, p2)
    def finger_error(points):
        errors = []
        for side in ("Left", "Right"):
            skeleton = hand_skeleton(side)
            for k in range(1, len(skeleton.joints)):
                a, b = skeleton.joints[skeleton.parents[k]], skeleton.joints[k]
                d, g = unit(points[:, b] - points[:, a])[0], unit(truth[:, b] - truth[:, a])[0]
                errors.append(np.degrees(np.arccos(np.clip((d * g).sum(-1), -1, 1))))
        return float(np.median(errors))
    assert finger_error(refined) < .7 * finger_error(raw)
    np.testing.assert_array_equal(refined[:, [5, 6, 7, 8, 62, 41]], raw[:, [5, 6, 7, 8, 62, 41]])
    assert hand_skeleton("Left").joints[0] == HAND21["Left"][0] and len(hand_skeleton("Right").joints) == 21
