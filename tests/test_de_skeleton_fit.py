"""Skeleton fitting: rigid lengths, lower direction error than raw triangulation."""
import numpy as np

from signtranslator.data_engineering.landmark_kinematics import unit
from signtranslator.data_engineering.sapiens_keypoints import HAND21
from signtranslator.data_engineering.skeleton_fit import Skeleton, fit_skeleton
from signtranslator.data_engineering.two_view_geometry import project, triangulate
from tests.test_de_two_view_states import cameras, goliath_motion


def goliath_skeleton():
    joints, parents = [5], [-1]
    def add(j, parent):
        joints.append(j)
        parents.append(joints.index(parent))
    for j, p in ((6, 5), (7, 5), (62, 7), (8, 6), (41, 8)):
        add(j, p)
    for side, wrist in (("Left", 62), ("Right", 41)):
        hand = HAND21[side]
        for f in range(5):
            chain = (wrist, *hand[1 + 4 * f: 5 + 4 * f])
            for a, b in zip(chain, chain[1:]):
                add(b, a)
    return Skeleton(tuple(joints), tuple(parents))


def bone_errors(points, truth, skeleton):
    errors = []
    for k, parent in enumerate(skeleton.parents):
        if parent < 0:
            continue
        a, b = skeleton.joints[parent], skeleton.joints[k]
        d, _ = unit(points[:, b] - points[:, a])
        g, _ = unit(truth[:, b] - truth[:, a])
        errors.append(np.degrees(np.arccos(np.clip((d * g).sum(1), -1, 1))))
    return np.concatenate(errors)


def test_fit_beats_triangulation_and_keeps_bones_rigid():
    truth = goliath_motion(frames=60)
    p1, p2 = cameras()
    skeleton = goliath_skeleton()
    rng = np.random.default_rng(9)
    used = list(skeleton.joints)
    v1 = np.zeros((60, 308, 2)); v2 = np.zeros((60, 308, 2))
    for view, p in ((v1, p1), (v2, p2)):
        view[:, used] = project(p, truth[:, used].reshape(-1, 3)).reshape(60, len(used), 2)
        view += rng.normal(scale=2., size=view.shape)
    conf = np.zeros((60, 308)); conf[:, used] = 1.
    raw = np.full_like(truth, np.nan)
    raw[:, used] = triangulate(p1, p2, v1[:, used].reshape(-1, 2), v2[:, used].reshape(-1, 2)).reshape(60, len(used), 3)
    fitted, report = fit_skeleton(raw, v1, v2, conf, conf, p1, p2, skeleton, smoothness=5.)
    raw_err, fit_err = np.median(bone_errors(raw, truth, skeleton)), np.median(bone_errors(fitted, truth, skeleton))
    assert fit_err < .8 * raw_err, (raw_err, fit_err)
    for k, parent in enumerate(skeleton.parents[1:], start=1):
        lengths = np.linalg.norm(fitted[:, skeleton.joints[k]] - fitted[:, skeleton.joints[parent]], axis=1)
        assert np.ptp(lengths) < 1e-9  # exactly rigid
    assert report["end_rms_px"] <= report["start_rms_px"] + 1e-6


def hinged_hand(frames=60, seed=3):
    """A hand whose fingers flex in fixed planes (1-DOF interphalangeal joints)."""
    from signtranslator.data_engineering.two_view_states import hand_skeleton
    rng = np.random.default_rng(seed)
    skeleton = hand_skeleton("Left")
    hand = HAND21["Left"]
    points = np.zeros((frames, 308, 3))
    normal = np.array([0., 0., -1.])
    for t in range(frames):
        wrist = np.array([.1 * np.sin(t / 15), .05 * np.cos(t / 11), 3.])
        points[t, hand[0]] = wrist
        for f in range(5):
            axis = unit(np.array([np.cos(.25 * (f - 2)), -1., .1 * f]))[0]
            chain = hand[1 + 4 * f: 5 + 4 * f]
            points[t, chain[0]] = wrist + .08 * axis
            angle, previous = 0., chain[0]
            for k, (joint, length) in enumerate(zip(chain[1:], (.045, .03, .022))):
                angle += (.5 + .4 * k) * (1 + np.sin(t / 7 + f + k)) / 2
                direction = np.cos(angle) * axis + np.sin(angle) * normal
                points[t, joint] = points[t, previous] + length * direction
                previous = joint
    return points, skeleton, rng


def test_hinge_prior_restores_coplanar_phalanges():
    from signtranslator.data_engineering.two_view_states import finger_hinges
    truth, skeleton, rng = hinged_hand()
    p1, p2 = cameras()
    used = list(skeleton.joints)
    v1 = np.zeros((60, 308, 2)); v2 = np.zeros((60, 308, 2))
    for view, p in ((v1, p1), (v2, p2)):
        view[:, used] = project(p, truth[:, used].reshape(-1, 3)).reshape(60, len(used), 2)
        view += rng.normal(scale=1.5, size=view.shape)
    conf = np.zeros((60, 308)); conf[:, used] = 1.
    raw = np.full_like(truth, np.nan)
    raw[:, used] = triangulate(p1, p2, v1[:, used].reshape(-1, 2), v2[:, used].reshape(-1, 2)).reshape(60, len(used), 3)
    plain, _ = fit_skeleton(raw, v1, v2, conf, conf, p1, p2, skeleton, smoothness=5.)
    hinged, _ = fit_skeleton(raw, v1, v2, conf, conf, p1, p2, skeleton, smoothness=5.,
                             hinges=finger_hinges("Left"), hinge_weight=20.)
    def coplanarity(points):
        values = []
        for a, b, c in finger_hinges("Left"):
            parent = {skeleton.joints[k]: skeleton.joints[p] for k, p in enumerate(skeleton.parents) if p >= 0}
            u = [unit(points[:, j] - points[:, parent[j]])[0] for j in (a, b, c)]
            values.append(np.abs(np.einsum("ti,ti->t", u[0], np.cross(u[1], u[2]))))
        return float(np.median(values))
    assert coplanarity(truth) < 1e-9
    assert coplanarity(hinged) < .2 * coplanarity(plain)
    assert np.median(bone_errors(hinged, truth, skeleton)) < np.median(bone_errors(plain, truth, skeleton))
