"""Swing IK: exact reproduction, minimality, invariance, honest degeneracy."""
import numpy as np
import pytest

from signtranslator.data_engineering.landmark_kinematics import (
    SwingJoint, frame_from_axes, solve_chain, swing, unit,
)


def random_units(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_swing_is_a_minimal_rotation_taking_rest_to_target():
    rng = np.random.default_rng(0)
    rest, target = random_units(rng, 500), random_units(rng, 500)
    rotation, valid = swing(rest, target)
    assert valid.all()
    np.testing.assert_allclose(np.einsum("tij,tj->ti", rotation, rest), target, atol=1e-12)
    np.testing.assert_allclose(rotation @ rotation.transpose(0, 2, 1), np.broadcast_to(np.eye(3), (500, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-12)
    angle = np.arccos(np.clip((np.trace(rotation, axis1=1, axis2=2) - 1) / 2, -1, 1))
    np.testing.assert_allclose(angle, np.arccos(np.clip((rest * target).sum(1), -1, 1)), atol=1e-7)
    axis = np.cross(rest, target)
    np.testing.assert_allclose(np.einsum("tij,tj->ti", rotation, axis), axis, atol=1e-12)  # axis fixed: no twist


def test_antiparallel_and_degenerate_cases_are_invalid_not_guessed():
    _, valid = swing(np.array([[0., 1., 0.]]), np.array([[0., -1., 0.]]))
    assert not valid[0]
    _, valid = frame_from_axes(np.array([[1., 0., 0.]]), np.array([[2., 0., 0.]]))
    assert not valid[0]
    _, ok = unit(np.zeros((1, 3)))
    assert not ok[0]


def test_chain_forward_kinematics_reproduces_every_bone_direction_and_is_frame_invariant():
    rng = np.random.default_rng(1)
    frames = 40
    points = rng.normal(size=(frames, 6, 3))
    root, root_ok = frame_from_axes(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    joints = (SwingJoint("a", "root", 0, 3, (0., 1., 0.)), SwingJoint("b", "a", 3, 4, (0., 1., 0.)),
              SwingJoint("c", "b", 4, 5, (.3, .9, .1)))
    solved = solve_chain(points, {"root": (root, root_ok)}, joints)
    for joint in joints:
        local, world, valid = solved[joint.name]
        assert valid.all()
        rest, _ = unit(np.asarray(joint.rest))
        bone, _ = unit(points[:, joint.end] - points[:, joint.start])
        np.testing.assert_allclose(np.einsum("tij,j->ti", world, rest), bone, atol=1e-10)
    # A global rotation of all points changes only the root; every local rotation is unchanged.
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    turned = points @ q.T
    root2, ok2 = frame_from_axes(turned[:, 1] - turned[:, 0], turned[:, 2] - turned[:, 0])
    again = solve_chain(turned, {"root": (root2, ok2)}, joints)
    for joint in joints:
        np.testing.assert_allclose(again[joint.name][0], solved[joint.name][0], atol=1e-9)


def test_invalid_points_and_order_are_enforced():
    points = np.random.default_rng(2).normal(size=(5, 3, 3))
    root, ok = frame_from_axes(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    mask = np.ones((5, 3), bool)
    mask[2, 1] = False
    solved = solve_chain(points, {"r": (root, ok)}, (SwingJoint("j", "r", 0, 1, (1., 0., 0.)),), mask)
    assert solved["j"][2].tolist() == [True, True, False, True, True]
    with pytest.raises(ValueError, match="not solved before"):
        solve_chain(points, {"r": (root, ok)}, (SwingJoint("k", "missing", 0, 1, (1., 0., 0.)),))
