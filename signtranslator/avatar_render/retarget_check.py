"""Quantitative checks of a retarget: bone directions, head orientation, left/right.

The reference for every direction is :func:`canonical_skeleton.skeleton_positions`,
the project's own reading of a state, evaluated on the *stored* state (no filling);
the rig side is the bone frames returned by :meth:`MakeHumanRig.pose` (the same
forward kinematics that skins the mesh), not the retarget's own targets. Only
frames where the joint and every ancestor are valid as stored are scored.

The canonical skeleton has no head orientation (its head point rides the chest),
so the head is scored against ``head_rotation`` itself: the geodesic angle
between the rig head bone's world change from rest and the stored rotation.

Left/right is checked three ways: structurally (the signer's left canonical arm
drives ``.L`` bones, which sit at ``+x`` at rest), per frame (the lateral side of
each wrist relative to the chest agrees between skeleton and rig), and in the
rendered image (:func:`image_mirror_check`; a raised left arm appears on the
viewer's right, as the project draws a signer facing the camera).
"""

from __future__ import annotations

import numpy as np

from ..data_engineering.video_landmarks import ClipLandmarks, canonical_from_landmarks
from ..generation.canonical_skeleton import skeleton_positions
from ..pose.multichannel import MultichannelMotion
from .retarget import FINGERS, SIDES, matrices_from_6d, rest_landmarks

LATERAL_MARGIN_METRES = .05


def bone_pairs():
    """(group, canonical start, canonical end, chain joint, rig head bone, rig end bone or None)."""
    pairs = []
    for side, s, _ in SIDES:
        pairs.append(("upper_arm", f"{side}Shoulder", f"{side}ForeArm", f"{side}UpperArm",
                      f"upperarm01.{s}", f"lowerarm01.{s}"))
        pairs.append(("forearm", f"{side}ForeArm", f"{side}Hand", f"{side}Forearm",
                      f"lowerarm01.{s}", f"wrist.{s}"))
        for k, finger in enumerate(FINGERS, 1):
            for segment in (1, 2, 3):
                end = f"finger{k}-{segment + 1}.{s}" if segment < 3 else None
                pairs.append(("thumb" if finger == "Thumb" else "finger",
                              f"{side}Hand{finger}{segment}", f"{side}Hand{finger}{segment + 1}",
                              f"{side}Hand{finger}{segment}", f"finger{k}-{segment}.{s}", end))
    return tuple(pairs)


def _angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-12)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(1e-12)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1., 1.)))


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(n=0, median_deg=None, p95_deg=None, max_deg=None)
    return dict(n=int(len(values)), median_deg=float(np.median(values)),
                p95_deg=float(np.percentile(values, 95)), max_deg=float(values.max()))


def direction_errors(rig, state: MultichannelMotion, retargeted, bone_globals: np.ndarray) -> dict:
    """Per-group and per-bone angular errors (degrees) on frames valid as stored.

    ``bone_globals`` is (T, B, 4, 4) from :meth:`MakeHumanRig.pose` for each frame.
    """
    joints, positions, _ = skeleton_positions(state)
    position = {name: positions[:, i] for i, name in enumerate(joints)}
    index = {name: i for i, name in enumerate(rig.bone_names)}
    chain = retargeted.canonical.chain_valid
    per_bone, groups = {}, {}
    for group, start, end, joint, head, tail in bone_pairs():
        valid = chain[joint]
        canonical = position[end] - position[start]
        if tail is None:
            rig_direction = bone_globals[:, index[head], :3, 1]
        else:
            rig_direction = bone_globals[:, index[tail], :3, 3] - bone_globals[:, index[head], :3, 3]
        errors = _angles(canonical[valid], rig_direction[valid])
        per_bone[f"{start}->{end}"] = _summary(errors)
        groups.setdefault(group, []).append(errors)
        side = "left" if start.startswith("Left") else "right"
        groups.setdefault(f"{group}_{side}", []).append(errors)
    result = {name: _summary(np.concatenate(values)) for name, values in groups.items()}
    # Head: rig head world change from rest vs the stored camera-frame head rotation.
    head = state.channels["head_rotation"]
    valid = head.valid[:, 0]
    rest_head = rig.rest_globals[index["head"], :3, :3].detach().cpu().numpy()
    change = bone_globals[:, index["head"], :3, :3] @ rest_head.T
    if valid.any():
        stored = matrices_from_6d(head.values[valid, 0])
        relative = np.swapaxes(change[valid], -1, -2) @ stored
        angle = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1., 1.)))
        facing = _angles(change[valid][..., 2], stored[..., 2])
    else:
        angle = facing = np.zeros(0)
    result["head_rotation"] = _summary(angle)
    result["head_facing"] = _summary(facing)
    # Informational: the torso and neck are deliberately distributed over several rig bones.
    torso_valid = chain["Chest"]
    torso = _angles((position["Spine2"] - position["Hips"])[torso_valid],
                    (bone_globals[:, index["neck01"], :3, 3] - bone_globals[:, index["spine05"], :3, 3])[torso_valid])
    result["torso_informational"] = _summary(torso)
    return dict(groups=result, per_bone=per_bone)


def lateral_agreement(rig, state: MultichannelMotion, retargeted, bone_globals: np.ndarray) -> dict:
    """Does each wrist sit on the same side of the chest (camera x) in skeleton and rig?

    Scored where the canonical wrist is more than ``LATERAL_MARGIN_METRES`` from the
    chest midline (nominal lengths), so near-midline frames cannot flip by proportion.
    """
    joints, positions, _ = skeleton_positions(state)
    position = {name: positions[:, i] for i, name in enumerate(joints)}
    index = {name: i for i, name in enumerate(rig.bone_names)}
    out = {}
    for side, s, _ in SIDES:
        valid = retargeted.canonical.chain_valid[f"{side}Forearm"]
        canonical = (position[f"{side}Hand"] - position["Spine2"])[:, 0]
        rig_x = bone_globals[:, index[f"wrist.{s}"], 0, 3] - bone_globals[:, index["spine01"], 0, 3]
        scored = valid & (np.abs(canonical) > LATERAL_MARGIN_METRES)
        agree = np.sign(canonical[scored]) == np.sign(rig_x[scored])
        out[side.lower()] = dict(scored_frames=int(scored.sum()),
                                 agreement=float(agree.mean()) if scored.any() else None,
                                 canonical_on_own_side=float((np.sign(canonical[scored]) == (1 if side == "Left" else -1)).mean())
                                 if scored.any() else None)
    return out


def structural_handedness(rig) -> dict:
    """``.L`` bones at +x, ``.R`` at -x in the rest rig (the signer's left is +x)."""
    index = {name: i for i, name in enumerate(rig.bone_names)}
    heads = rig.rest_globals[:, :3, 3].detach().cpu().numpy()
    left = [n for n in rig.bone_names if n.endswith(".L")]
    right = [n for n in rig.bone_names if n.endswith(".R")]
    return dict(left_bones_at_positive_x=bool(all(heads[index[n], 0] >= -1e-9 for n in left)),
                right_bones_at_negative_x=bool(all(heads[index[n], 0] <= 1e-9 for n in right)),
                arm_left=float(heads[index["upperarm01.L"], 0]), arm_right=float(heads[index["upperarm01.R"], 0]))


# --------------------------------------------------------------------------- synthetic states

def state_from_points(times: np.ndarray, world: np.ndarray, visible: np.ndarray, hands: dict, *,
                      sample_id: str = "synthetic-retarget-probe") -> MultichannelMotion:
    """A canonical state built by the project's own adapter from y-up 3D points."""
    frames = len(times)
    clip = ClipLandmarks(
        times=np.asarray(times, dtype=np.float64), pose_world=np.zeros((frames, 33, 3)),
        pose_visibility=np.ones((frames, 33)), pose_image=np.zeros((frames, 33, 2)),
        pose_found=np.ones(frames, dtype=bool), hands_world=np.zeros((frames, 0, 21, 3)),
        hands_image=np.zeros((frames, 0, 21, 2)), hands_found=np.zeros((frames, 0), dtype=bool),
        blendshapes=np.zeros((frames, 0)), blendshape_names=(), face_matrix=np.tile(np.eye(4), (frames, 1, 1)),
        face_found=np.zeros(frames, dtype=bool))
    state, _ = canonical_from_landmarks(
        clip, sample_id=sample_id, source_id="synthetic-retarget-probe", source_sha256="e" * 64,
        clock_id="synthetic-clock", inference_method="declared synthetic landmarks, not observed",
        points=dict(world=world, visible=visible, hands=hands))
    return state


def _rotate_about(points: np.ndarray, centre: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return (points - centre) @ rotation.T + centre


def raised_arm_points(rig, side: str = "Left", frames: int = 1):
    """Rest landmarks of the rig with one whole arm (and hand) swung to point straight up."""
    from scipy.spatial.transform import Rotation
    from ..data_engineering.video_landmarks import (L_ELBOW, L_SHOULDER, L_WRIST, R_ELBOW, R_SHOULDER,
                                                    R_WRIST)
    world, visible, hands = rest_landmarks(rig)
    shoulder, elbow, wrist = (L_SHOULDER, L_ELBOW, L_WRIST) if side == "Left" else (R_SHOULDER, R_ELBOW, R_WRIST)
    centre = world[0, shoulder]
    arm = world[0, elbow] - centre
    arm = arm / np.linalg.norm(arm)
    up = np.array([0., 1., 0.])
    axis = np.cross(arm, up)
    rotation = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.arccos(arm @ up)).as_matrix()
    world = world.copy()
    for point in (elbow, wrist):
        world[0, point] = _rotate_about(world[0, point], centre, rotation)
    points, found = hands[side]
    hands = dict(hands)
    hands[side] = (_rotate_about(points, centre, rotation), found)
    tile = lambda a: np.repeat(a, frames, axis=0)
    return tile(world), tile(visible), {k: (tile(p), np.repeat(f, frames)) for k, (p, f) in hands.items()}


def image_mirror_check(image: np.ndarray, background: np.ndarray, *, head_top_row: int) -> dict:
    """In a front render of a raised-left-arm pose: foreground above the head lies right of centre.

    ``image`` is H×W×3 uint8; ``background`` the world colour (uint8 RGB); rows above
    ``head_top_row`` contain only the raised arm.
    """
    foreground = np.any(np.abs(image.astype(int) - background.astype(int)) > 12, axis=-1)
    band = foreground[:max(0, head_top_row)]
    columns = np.nonzero(band)[1]
    width = image.shape[1]
    if not len(columns):
        return dict(pixels_above_head=0, fraction_on_viewer_right=None, passed=False)
    right = float((columns > width / 2).mean())
    return dict(pixels_above_head=int(len(columns)), fraction_on_viewer_right=right,
                mean_column=float(columns.mean()), width=width, passed=bool(right > .95))


__all__ = ["bone_pairs", "direction_errors", "lateral_agreement", "structural_handedness",
           "state_from_points", "raised_arm_points", "image_mirror_check"]
