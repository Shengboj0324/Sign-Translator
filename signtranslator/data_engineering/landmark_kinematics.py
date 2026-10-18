"""Joint rotations from 3D landmarks by hierarchical swing inverse kinematics.

Landmark estimators (for example MediaPipe pose and hand world landmarks) give joint
*positions*; the canonical state carries joint *rotations*. This module converts one
into the other under a declared kinematic tree, exactly and with stated limits.

**Root frames.** A root's full orientation is built from three landmarks: a primary
axis ``a`` and a secondary direction ``s`` give the right-handed orthonormal frame
``x = a/|a|``, ``z = (x × s)/|x × s|``, ``y = z × x``. A degenerate configuration
(``|x × s|`` below tolerance) is invalid, never guessed.

**Swing.** For a child joint, its bone direction ``b`` (unit, world) is expressed in
the parent's world frame ``W_p`` as ``u = W_pᵀ b``. The local rotation is the
minimal rotation taking the declared rest direction ``r`` to ``u``:
``R = I + [v]× + [v]×² / (1 + c)`` with ``v = r × u``, ``c = r · u``. It is exact,
has angle ``arccos c``, and is undefined only at ``c = -1`` (antiparallel), where the
cell is invalid. ``W_child = W_p R``, so forward kinematics reproduces ``b`` exactly:
``W_child r = W_p R r = W_p u = b``.

**Twist.** Rotation about a bone's own axis is not observable from its two
endpoints, so every swing joint has zero twist by declaration. Root frames carry
full orientation. This is a representation limit, recorded in the convention name,
not an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEGENERATE = 1e-9


def unit(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised vectors and a validity mask (norm above tolerance)."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    valid = norms[..., 0] > DEGENERATE
    return np.where(valid[..., None], vectors / np.where(valid[..., None], norms, 1.), 0.), valid


def frame_from_axes(primary: np.ndarray, secondary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(..., 3, 3) right-handed frames with columns (x, y, z), and validity."""
    x, ok_x = unit(primary)
    z, ok_z = unit(np.cross(x, secondary))
    y = np.cross(z, x)
    frames = np.stack((x, y, z), axis=-1)
    valid = ok_x & ok_z
    return np.where(valid[..., None, None], frames, np.eye(3)), valid


def swing(rest: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimal rotations taking unit ``rest`` to unit ``target``, and validity."""
    rest = np.broadcast_to(rest, target.shape)
    v = np.cross(rest, target)
    c = np.sum(rest * target, axis=-1)
    valid = 1. + c > 1e-9
    k = np.zeros(target.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2] = -v[..., 2], v[..., 1]
    k[..., 1, 0], k[..., 1, 2] = v[..., 2], -v[..., 0]
    k[..., 2, 0], k[..., 2, 1] = -v[..., 1], v[..., 0]
    denominator = np.where(valid, 1. + c, 1.)[..., None, None]
    rotation = np.eye(3) + k + (k @ k) / denominator
    return np.where(valid[..., None, None], rotation, np.eye(3)), valid


@dataclass(frozen=True)
class SwingJoint:
    """One joint: rotates bone ``start -> end``; ``parent`` is a joint name or a root."""

    name: str
    parent: str
    start: int
    end: int
    rest: tuple[float, float, float]


def solve_chain(points: np.ndarray, roots: dict[str, tuple[np.ndarray, np.ndarray]],
                joints: tuple[SwingJoint, ...], point_valid: np.ndarray | None = None):
    """Local rotations and world frames for joints listed parent-before-child.

    ``points`` is (T, N, 3); ``roots`` maps root names to (frames (T,3,3), valid (T,)).
    Returns ``{name: (local (T,3,3), world (T,3,3), valid (T,))}`` including roots
    (whose local rotation is their world frame).
    """
    frames = len(points)
    if point_valid is None:
        point_valid = np.ones(points.shape[:2], dtype=bool)
    solved = {name: (world, world, valid) for name, (world, valid) in roots.items()}
    for joint in joints:
        if joint.parent not in solved:
            raise ValueError(f"{joint.name}: parent {joint.parent} is not solved before it")
        if joint.name in solved:
            raise ValueError(f"{joint.name} is declared twice")
        rest, rest_ok = unit(np.asarray(joint.rest, dtype=np.float64))
        if not rest_ok:
            raise ValueError(f"{joint.name}: rest direction must be nonzero")
        _, parent_world, parent_valid = solved[joint.parent]
        bone, bone_ok = unit(points[:, joint.end] - points[:, joint.start])
        local_direction = np.einsum("tji,tj->ti", parent_world, bone)
        local, swing_ok = swing(rest, local_direction)
        valid = (parent_valid & bone_ok & swing_ok & point_valid[:, joint.start] & point_valid[:, joint.end])
        local = np.where(valid[:, None, None], local, np.eye(3))
        world = parent_world @ local
        solved[joint.name] = (local, world, valid)
    assert all(len(v[0]) == frames for v in solved.values())
    return solved


__all__ = ["unit", "frame_from_axes", "swing", "SwingJoint", "solve_chain"]
