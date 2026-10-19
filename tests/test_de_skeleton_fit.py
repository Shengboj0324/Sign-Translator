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
