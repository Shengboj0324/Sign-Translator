"""Hands and forearms inside the avatar's own torso and head: a measure, and a depth-only correction.

The state's directions are copied onto the rig's proportions, and estimated depth is noisy, so a hand can land inside the
body. Location in ASL is linguistic, so the correction moves a hand only along the camera axis: the front view keeps
exactly where the wrist is drawn against the body, and a hand that was inside the chest ends at the chest's surface, in front of it.

**Field.** Signed distance grids of the rest body (the watertight ``body`` surface), split by dominant bone into head,
neck, chest and belly. Each part follows one bone, so a point is read in that bone's rest frame. Arms and legs are not in the
field. The grids are built from the pinned rig on load; nothing is tuned to a clip.

**Correction.** Per side and frame: how far along +z the wrist must go for the hand and the wrist half of the forearm to
leave the body. That need is dilated and smoothed in time, so the arm eases out rather than pops, then solved with fixed bone
lengths by two-bone IK from the shoulder, ``ROUNDS`` times on what remains. A forearm still inside is then cleared by
swivelling the elbow outward about the shoulder-wrist axis, which leaves the wrist where it is. The hand's world rotation is
kept exactly throughout, so handshape and palm orientation do not change.

It uses no arm label, text or source, so it treats real and generated clips alike. Not handled: hand against hand, hand
against the other arm or its own upper arm, and anything a hand does behind the head's silhouette beyond the depth move.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, map_coordinates, maximum_filter1d
from scipy.spatial import cKDTree

from .makehuman import SOURCE_METERS_PER_UNIT
from .motion_quality import forward_kinematics

VOXEL = .05                                 # source units (5 mm)
BOUNDS = ((-2.8, 2.8), (-1.2, 8.8), (-1.8, 2.0))
CLEARANCE = .03                             # source units (3 mm) past the surface
MARCH_STEP = .04
MARCH_REACH = 3.2                           # source units (32 cm): deeper than the torso
ROUNDS = 3
DILATE_FRAMES = 5
SMOOTH_SIGMA_FRAMES = 1.5
WRIST_HALF = .5                             # forearm vertices past this fraction count towards the wrist's move
SWIVEL_STEP_DEG = 5.
SWIVEL_MAX_DEG = 70.
REACH_MARGIN = .02                          # source units: a straight arm's hinge is undefined, so the arm stays short of it
SHALLOW_MM = (0., 5., 10., 20.)
PART_BONES = {"head": "head", "neck": "neck02", "chest": "spine01", "belly": "spine04"}
SPINE_CHEST = ("spine01", "spine02", "breast", "clavicle", "shoulder01")
SPINE_BELLY = ("spine03", "spine04", "spine05", "pelvis", "root")
HAND_PREFIXES = ("wrist", "metacarpal", "finger")
FOREARM_PREFIXES = ("lowerarm",)
SIDES = ("L", "R")


@dataclass(frozen=True)
class BodyField:
    origin: np.ndarray                      # (3,) rest coordinate of voxel (0, 0, 0)'s centre
    voxel: float
    sdf: dict                               # part -> (X, Y, Z) signed distance in source units (negative inside)
    part_bone: dict                         # part -> bone index
    inverse_rest: dict                      # part -> (4, 4)
    hand_vertices: dict                     # side -> vertex indices of the body surface
    forearm_vertices: dict
    weights: dict                           # kind.side -> ((V, K) weights, (K,) bone indices)
    rest_vertices: np.ndarray


def _body_triangles(rig) -> tuple[np.ndarray, np.ndarray]:
    quads = [rig.faces[i] for i, groups in enumerate(rig.face_groups) if "body" in groups]
    triangles = []
    for face in quads:
        triangles.extend((face[0], face[k], face[k + 1]) for k in range(1, len(face) - 1))
    triangles = np.asarray(triangles, dtype=np.int64)
    return triangles, np.unique(triangles)


def _solid(vertices: np.ndarray, triangles: np.ndarray, origin: np.ndarray, shape: tuple[int, int, int], voxel: float) -> np.ndarray:
    """Inside mask of a closed surface by parity of crossings along z through every voxel column."""
    crossings: dict[tuple[int, int], list[float]] = {}
    corners = vertices[triangles]
    low = np.floor((corners[..., :2].min(1) - origin[:2]) / voxel).astype(int)
    high = np.ceil((corners[..., :2].max(1) - origin[:2]) / voxel).astype(int)
    for (a, b, c), lo, hi in zip(corners, low, high):
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array(shape[:2]) - 1)
        if (lo > hi).any():
            continue
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-14:
            continue
        xs = origin[0] + voxel * np.arange(lo[0], hi[0] + 1) + 1.1e-5              # off the grid, so no ray meets an edge
        ys = origin[1] + voxel * np.arange(lo[1], hi[1] + 1) + 1.3e-5
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        u = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / det
        v = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / det
        w = 1. - u - v
        hit = (u >= 0) & (v >= 0) & (w >= 0)
        if not hit.any():
            continue
        z = u[hit] * a[2] + v[hit] * b[2] + w[hit] * c[2]
        for i, j, value in zip(*(np.nonzero(hit)), z):
            crossings.setdefault((int(i + lo[0]), int(j + lo[1])), []).append(float(value))
    inside = np.zeros(shape, dtype=bool)
    zs = origin[2] + voxel * np.arange(shape[2])
    for (i, j), values in crossings.items():
        values = np.sort(values)
        for k in range(0, len(values) - 1, 2):
            inside[i, j] |= (zs >= values[k]) & (zs <= values[k + 1])
    return inside


def _descendants(rig, root: str) -> set[int]:
    index = rig.bone_names.index(root)
    out = {index}
    for i, parent in enumerate(rig.parents):
        if parent in out:
            out.add(i)
    return out


def _part_of_bones(rig) -> np.ndarray:
    """(B,) part name per bone: head, neck, chest, belly, or '' for arms and legs."""
    head = _descendants(rig, "head")
    parts = []
    for i, name in enumerate(rig.bone_names):
        if i in head:
            parts.append("head")
        elif name.startswith("neck"):
            parts.append("neck")
        elif name.startswith(SPINE_CHEST):
            parts.append("chest")
        elif name.startswith(SPINE_BELLY):
            parts.append("belly")
        else:
            parts.append("")
    return np.array(parts)


def _vertices_of(rig, surface: np.ndarray, prefixes: tuple[str, ...], side: str) -> np.ndarray:
    dominant = rig.weights.detach().cpu().numpy().argmax(1)
    names = rig.bone_names
    selected = [i for i, n in enumerate(names) if n.endswith(f".{side}") and n.startswith(prefixes)]
    mask = np.isin(dominant, selected)
    keep = np.zeros(len(mask), dtype=bool)
    keep[surface] = True
    return np.flatnonzero(mask & keep)


def build_field(rig, *, voxel: float = VOXEL) -> BodyField:
    """The signed distance grids and vertex sets of the rig's rest body."""
    vertices = rig.rest_vertices.detach().cpu().numpy()
    triangles, surface = _body_triangles(rig)
    origin = np.array([b[0] for b in BOUNDS])
    shape = tuple(int(np.ceil((b[1] - b[0]) / voxel)) + 1 for b in BOUNDS)
    solid = _solid(vertices, triangles, origin, shape, voxel)
    dominant = rig.weights.detach().cpu().numpy().argmax(1)
    parts_of_bones = _part_of_bones(rig)
    surface_part = parts_of_bones[dominant[surface]]
    tree = cKDTree(vertices[surface])
    centres = np.stack(np.nonzero(solid), axis=1) * voxel + origin
    _, nearest = tree.query(centres)
    labels = surface_part[nearest]
    occupied = np.nonzero(solid)
    rest = rig.rest_globals.detach().cpu().numpy()
    index = {name: i for i, name in enumerate(rig.bone_names)}
    sdf, part_bone, inverse = {}, {}, {}
    for part, bone in PART_BONES.items():
        inside = np.zeros(shape, dtype=bool)
        chosen = labels == part
        inside[occupied[0][chosen], occupied[1][chosen], occupied[2][chosen]] = True
        sdf[part] = ((distance_transform_edt(~inside) - distance_transform_edt(inside)) * voxel).astype(np.float32)
        part_bone[part] = index[bone]
        inverse[part] = np.linalg.inv(rest[index[bone]])
    hand = {s: _vertices_of(rig, surface, HAND_PREFIXES, s) for s in SIDES}
    forearm = {s: _vertices_of(rig, surface, FOREARM_PREFIXES, s) for s in SIDES}
    weights_all = rig.weights.detach().cpu().numpy()
    weights = {}
    for kind, sets in (("hand", hand), ("forearm", forearm)):
        for side, indices in sets.items():
            used = np.flatnonzero(weights_all[indices].sum(0) > 0)
            weights[f"{kind}.{side}"] = (weights_all[indices][:, used], used)
    return BodyField(origin, voxel, sdf, part_bone, inverse, hand, forearm, weights, vertices)


def _signed_distance(field: BodyField, bone_globals: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,) smallest signed distance over the parts, and its part number, for (N, 3) world points at one frame."""
    best = np.full(len(points), np.inf)
    which = np.zeros(len(points), dtype=np.int64)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    for k, part in enumerate(PART_BONES):
        skin = bone_globals[field.part_bone[part]] @ field.inverse_rest[part]        # rest coordinates -> this frame's world
        rest_points = np.linalg.solve(skin, homogeneous.T)[:3]
        coordinates = (rest_points - field.origin[:, None]) / field.voxel
        distance = map_coordinates(field.sdf[part], coordinates, order=1, mode="constant", cval=1.)
        closer = distance < best
        best[closer], which[closer] = distance[closer], k
    return best, which


def _skin(field: BodyField, kind: str, side: str, bone_globals: np.ndarray, inverse_rest: np.ndarray) -> np.ndarray:
    """(T, V, 3) posed vertices of one vertex set."""
    indices = (field.hand_vertices if kind == "hand" else field.forearm_vertices)[side]
    weights, bones = field.weights[f"{kind}.{side}"]
    rest = np.concatenate((field.rest_vertices[indices], np.ones((len(indices), 1))), axis=1)
    skin = bone_globals[:, bones] @ inverse_rest[bones]                     # (T, K, 4, 4)
    posed = np.einsum("tkab,vb,vk->tva", skin[..., :3, :], rest, weights, optimize=True)
    return posed


def penetration(rig, field: BodyField, local_rotations: np.ndarray) -> dict:
    """How often and how deep hand and forearm vertices sit inside the torso and head, in mm; per side and pooled."""
    bone_globals = forward_kinematics(rig, local_rotations)
    return _penetration_from_globals(rig, field, bone_globals)


def _penetration_from_globals(rig, field: BodyField, bone_globals: np.ndarray) -> dict:
    inverse_rest = np.linalg.inv(rig.rest_globals.detach().cpu().numpy())
    names = list(PART_BONES)
    frames = len(bone_globals)
    out = {}
    for kind in ("hand", "forearm"):
        deepest = {"torso": np.zeros((frames, 2)), "head": np.zeros((frames, 2))}
        for s, side in enumerate(SIDES):
            posed = _skin(field, kind, side, bone_globals, inverse_rest)
            for t in range(frames):
                distance, part = _signed_distance(field, bone_globals[t], posed[t])
                for target, members in (("torso", (names.index("chest"), names.index("belly"), names.index("neck"))),
                                        ("head", (names.index("head"),))):
                    here = distance[np.isin(part, members)]
                    if here.size:
                        deepest[target][t, s] = max(0., -float(here.min()))
        for target, values in deepest.items():
            millimetres = values.max(1) * SOURCE_METERS_PER_UNIT * 1000.
            row = {f"frames_over_{int(limit)}mm": float((millimetres > limit).mean()) for limit in SHALLOW_MM}
            hit = millimetres[millimetres > 0.]
            row.update(depth_p50_mm=float(np.median(hit)) if hit.size else 0.,
                       depth_p95_mm=float(np.percentile(hit, 95)) if hit.size else 0.,
                       depth_max_mm=float(hit.max()) if hit.size else 0.)
            out[f"{kind}.{target}"] = row
    return out


def _exit_distance(field: BodyField, bone_globals: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(N,) distance along world +z until each point is ``CLEARANCE`` outside the field; 0 if it already is."""
    distance, _ = _signed_distance(field, bone_globals, points)
    need = np.zeros(len(points))
    inside = np.flatnonzero(distance < CLEARANCE)
    if not inside.size:
        return need
    moved = points[inside].copy()
    pending = np.arange(len(inside))
    for step in range(1, int(MARCH_REACH / MARCH_STEP) + 1):
        moved[pending, 2] += MARCH_STEP
        here, _ = _signed_distance(field, bone_globals, moved[pending])
        out = here >= CLEARANCE
        need[inside[pending[out]]] = step * MARCH_STEP
        pending = pending[~out]
        if not pending.size:
            break
    if pending.size:
        need[inside[pending]] = MARCH_REACH
    return need


def _wrist_need(field: BodyField, bone_globals: np.ndarray, inverse_rest: np.ndarray, side: str,
                forearm_along: np.ndarray) -> np.ndarray:
    """(T,) wrist displacement along +z that clears this side's hand and the wrist half of its forearm."""
    frames = len(bone_globals)
    hand = _skin(field, "hand", side, bone_globals, inverse_rest)
    forearm = _skin(field, "forearm", side, bone_globals, inverse_rest)
    near_wrist = forearm_along > WRIST_HALF
    need = np.zeros(frames)
    for t in range(frames):
        need[t] = _exit_distance(field, bone_globals[t], hand[t]).max(initial=0.)
        if near_wrist.any():
            share = _exit_distance(field, bone_globals[t], forearm[t][near_wrist]) / forearm_along[near_wrist]
            need[t] = max(need[t], float(share.max(initial=0.)))
    return need


def _rotation_about(axis: np.ndarray, angle: float | np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotations about unit axes by ``angle`` radians (Rodrigues)."""
    k = axis
    K = np.zeros((*k.shape[:-1], 3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    angle = np.asarray(angle)[..., None, None]
    return np.eye(3) + np.sin(angle) * K + (1. - np.cos(angle)) * (K @ K)


def _swivel_angles(field: BodyField, bone_globals: np.ndarray, forearm: np.ndarray, shoulder: np.ndarray, wrist: np.ndarray,
                   elbow: np.ndarray) -> np.ndarray:
    """(T,) smallest outward swivel of the elbow about the shoulder-wrist axis that clears the forearm, or best reached.

    The hand does not move: the axis runs through the wrist. Outward means away from the body's midline.
    """
    axis = _unit(wrist - shoulder)
    frames = len(bone_globals)
    angles = np.zeros(frames)
    for t in range(frames):
        distance, _ = _signed_distance(field, bone_globals[t], forearm[t])
        if distance.min() >= 0.:
            continue
        plus = _rotation_about(axis[t], .1) @ (elbow[t] - shoulder[t]) + shoulder[t]
        sign = 1. if abs(plus[0]) >= abs(elbow[t, 0]) else -1.
        best, chosen = distance.min(), 0.
        for angle in np.radians(np.arange(SWIVEL_STEP_DEG, SWIVEL_MAX_DEG + SWIVEL_STEP_DEG, SWIVEL_STEP_DEG)):
            turned = (_rotation_about(axis[t], sign * angle) @ (forearm[t] - shoulder[t]).T).T + shoulder[t]
            here, _ = _signed_distance(field, bone_globals[t], turned)
            if here.min() > best + 1e-3:
                best, chosen = here.min(), sign * angle
            if best >= 0.:
                break
        angles[t] = chosen
    return angles


def _smoothed(need: np.ndarray) -> np.ndarray:
    return gaussian_filter1d(maximum_filter1d(need, size=DILATE_FRAMES, mode="nearest"), SMOOTH_SIGMA_FRAMES, mode="nearest")


def _unit(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(1e-12)


def _swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(T, 3, 3) minimal rotations taking unit ``a`` onto unit ``b``."""
    axis = np.cross(a, b)
    s = np.linalg.norm(axis, axis=-1)
    c = np.clip((a * b).sum(-1), -1., 1.)
    k = axis / s[..., None].clip(1e-12)
    K = np.zeros((*axis.shape[:-1], 3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    out = np.eye(3) + s[..., None, None] * K + (1. - c)[..., None, None] * (K @ K)
    return np.where((s < 1e-9)[..., None, None], np.eye(3), out)


def _solve_arm(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray, wrist_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-bone IK with fixed lengths, per frame: the wrist moves by at most ``wrist_z`` along +z and never in x or y.

    A depth the arm cannot reach is cut to what it can, so the hand may stay inside rather than move across the image.
    Returns (upper swing, forearm swing).
    """
    a = np.linalg.norm(elbow - shoulder, axis=-1)
    b = np.linalg.norm(wrist - elbow, axis=-1)
    reach = a + b - REACH_MARGIN
    lateral = ((wrist - shoulder)[:, :2] ** 2).sum(-1)
    furthest = shoulder[:, 2] + np.sqrt(np.clip(reach ** 2 - lateral, 0., None))
    target = wrist.copy()
    target[:, 2] = np.where(wrist_z > 0., np.minimum(wrist[:, 2] + wrist_z, np.maximum(furthest, wrist[:, 2])), wrist[:, 2])
    offset = target - shoulder
    distance = np.linalg.norm(offset, axis=-1).clip(1e-9)
    axis = offset / distance[:, None]
    along = (a ** 2 - b ** 2 + distance ** 2) / (2. * distance)
    height = np.sqrt(np.clip(a ** 2 - along ** 2, 0., None))
    towards = elbow - shoulder
    towards = towards - (towards * axis).sum(-1, keepdims=True) * axis
    solved_elbow = shoulder + axis * along[:, None] + _unit(towards) * height[:, None]
    solved_wrist = shoulder + offset
    return _swing(_unit(elbow - shoulder), _unit(solved_elbow - shoulder)), _swing(_unit(wrist - elbow), _unit(solved_wrist - solved_elbow))


def correct(rig, field: BodyField, local_rotations: np.ndarray, target_globals: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Depth-only wrist correction, then an elbow swivel (module docstring). Returns locals, world rotations and a record."""
    index = {name: i for i, name in enumerate(rig.bone_names)}
    inverse_rest = np.linalg.inv(rig.rest_globals.detach().cpu().numpy())
    rest = rig.rest_globals.detach().cpu().numpy()[:, :3, :3]
    heads = rig.rest_globals.detach().cpu().numpy()[:, :3, 3]
    local = local_rotations.copy()
    worlds = target_globals.copy()
    frames = len(local)
    record = {}
    original = forward_kinematics(rig, local)

    def recompose(chain: list[int]) -> None:
        for bone in chain:
            parent = rig.parents[bone]
            preceding = worlds[:, parent] @ (rest[parent].T @ rest[bone])
            local[:, bone] = np.swapaxes(preceding, -1, -2) @ worlds[:, bone]

    for side in SIDES:
        upper = [index[f"upperarm01.{side}"], index[f"upperarm02.{side}"]]
        fore = [index[f"lowerarm01.{side}"], index[f"lowerarm02.{side}"]]
        wrist_bone = index[f"wrist.{side}"]
        chain = [*upper, *fore, wrist_bone]
        direction = heads[wrist_bone] - heads[fore[0]]
        points = field.rest_vertices[field.forearm_vertices[side]]
        along = np.clip((points - heads[fore[0]]) @ direction / (direction @ direction), 0., 1.)
        moved_frames = np.zeros(frames, dtype=bool)
        for _ in range(ROUNDS):
            globals_ = forward_kinematics(rig, local)
            need = _wrist_need(field, globals_, inverse_rest, side, along)
            if need.max(initial=0.) <= 0.:
                break
            depth = _smoothed(need)
            moved_frames |= depth > 1e-6
            swing_upper, swing_fore = _solve_arm(globals_[:, upper[0], :3, 3], globals_[:, fore[0], :3, 3],
                                                 globals_[:, wrist_bone, :3, 3], depth)
            for bone in upper:
                worlds[:, bone] = swing_upper @ worlds[:, bone]
            for bone in fore:
                worlds[:, bone] = swing_fore @ worlds[:, bone]
            recompose(chain)
        globals_ = forward_kinematics(rig, local)
        shoulder, elbow, wrist = (globals_[:, b, :3, 3] for b in (upper[0], fore[0], wrist_bone))
        forearm = _skin(field, "forearm", side, globals_, inverse_rest)
        angles = _swivel_angles(field, globals_, forearm, shoulder, wrist, elbow)
        signs = np.sign(angles)
        angles = np.abs(angles)
        smooth = gaussian_filter1d(maximum_filter1d(angles, size=DILATE_FRAMES, mode="nearest"), SMOOTH_SIGMA_FRAMES, mode="nearest")
        sign = 1. if signs.sum() >= 0 else -1.             # one direction per side and clip: a swivel that flips would flap
        turn = _rotation_about(_unit(wrist - shoulder), sign * smooth)
        for bone in (*upper, *fore):
            worlds[:, bone] = turn @ worlds[:, bone]
        recompose(chain)
        moved = forward_kinematics(rig, local)
        before, after = original[:, wrist_bone, :3, 3], moved[:, wrist_bone, :3, 3]
        shift = (after - before) * SOURCE_METERS_PER_UNIT * 100.
        record[side] = dict(frames_depth_moved=int(moved_frames.sum()), frames_swivelled=int((smooth > 1e-4).sum()),
                            swivel_deg_max=float(np.degrees(smooth).max()), swivel_deg_p95=float(np.percentile(np.degrees(smooth), 95)),
                            wrist_depth_move_cm_max=float(np.abs(shift[:, 2]).max()),
                            wrist_depth_move_cm_p95=float(np.percentile(np.abs(shift[:, 2]), 95)),
                            wrist_image_shift_cm_max=float(np.linalg.norm(shift[:, :2], axis=-1).max()),
                            wrist_image_shift_cm_p95=float(np.percentile(np.linalg.norm(shift[:, :2], axis=-1), 95)))
    return local, worlds, record


def apply_correction(rig, field: BodyField, retargeted) -> tuple[object, dict]:
    """A retargeted motion with its hands and forearms out of the body; the record of before, after and movement."""
    before = penetration(rig, field, retargeted.local_rotations)
    local, worlds, moved = correct(rig, field, retargeted.local_rotations, retargeted.target_globals)
    after = penetration(rig, field, local)
    return replace(retargeted, local_rotations=local, target_globals=worlds), dict(before=before, after=after, moved=moved)


__all__ = ["BodyField", "build_field", "penetration", "correct", "apply_correction"]
