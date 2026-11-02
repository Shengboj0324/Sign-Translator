"""Hands covered by the avatar's own body: measures, and a depth-only correction.

The state's directions are copied onto the rig's proportions, and estimated depth is noisy, so a hand can land inside the
body or behind it. Location in ASL is linguistic, so the correction moves a hand only along the camera axis: the front view keeps
exactly where the wrist is drawn against the body, and a covered hand ends just in front of what covered it.
An arm too straight to reach that depth keeps the hand's place in the image and stays partly covered; that is counted in the record.

**What covers a hand** (:func:`front_maps`). Per frame, the z of the frontmost drawn surface in each 1 cm cell of the front view: the torso and
head of the body, the top without its sleeves, and the hair, all as skinned. A hand vertex with that surface in front of it is covered, whether it is inside
the body or behind it, so there is no jump between the two. :func:`hidden_from_front` counts these.

**Correction** (:data:`CORRECTION_V2`). Per side and frame: how far along +z the wrist must go for no hand vertex to be covered. That need is
dilated and smoothed in time, so the arm eases out rather than pops, then solved with fixed bone lengths by two-bone IK from the shoulder.
The elbow goes to the point of its circle drawn nearest where it was, so the arm in the image stays the state's. The hand's world rotation is kept
exactly, so handshape and palm orientation do not change. **A forearm doesn't drive anything:** moving a clear hand or swinging an elbow to clear a
dark sleeve from a dark top cost far more of the arm raters see than it saved (docs/AVATAR_RENDER.md section 13), so a forearm may stay inside the
torso, and :func:`penetration` still counts that against the rest body's signed distance grids.

It uses no arm label, text or source, so it treats real and generated clips alike. Not handled: hand against hand, hand
against the other arm or its own upper arm, and a hand behind its own sleeve.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, map_coordinates, maximum_filter, maximum_filter1d
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

from .makehuman import SOURCE_METERS_PER_UNIT
from .motion_quality import forward_kinematics

VOXEL = .05                                 # source units (5 mm)
BOUNDS = ((-2.8, 2.8), (-1.2, 8.8), (-1.8, 2.0))
CLEARANCE = .03                             # source units (3 mm) past the surface
MARCH_STEP = .04
MARCH_REACH = 3.2                           # source units (32 cm): deeper than the torso
ROUNDS = 3
FRONT_CELL = .1                             # source units (1 cm): the drawn body's front surface, per frame
FRONT_BOUNDS = ((-3.4, 3.4), (-1.4, 9.8))
FRONT_SAMPLES = 3                           # per face edge
DILATE_FRAMES = 5
SMOOTH_SIGMA_FRAMES = 1.5
WRIST_HALF = .5                             # forearm vertices past this fraction count towards the wrist's move
SWIVEL_STEP_DEG = 5.
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
    front_rest: np.ndarray | None = None    # (V, 4) rest vertices of what is drawn in front of the camera's +z ray: torso, head, top, hair
    front_weights: object = None            # (V, B) sparse skin weights of those vertices
    front_samples: object = None            # (S, V) sparse: points on their faces


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


def _front_surface(rig, drawn: dict | None):
    """Rest vertices, skin weights and face sample points of the surfaces a hand can hide behind: the rig's torso and head, and if given
    the drawn ``top`` without its sleeves and the ``hair``."""
    weights_all = rig.weights.detach().cpu().numpy()
    rest = rig.rest_vertices.detach().cpu().numpy()
    dominant = weights_all.argmax(1)
    parts_of_bones = _part_of_bones(rig)
    faces = [f for i, f in enumerate(rig.faces) if "body" in rig.face_groups[i] and all(parts_of_bones[dominant[v]] != "" for v in f)]
    sources = [(rest, weights_all, faces)]
    for name, attachment in (drawn or {}).items():
        if name not in ("top", "hair"):
            continue
        vertices = attachment.rest_vertices.detach().cpu().numpy()
        weights = attachment.weights.detach().cpu().numpy()
        keep = parts_of_bones[weights.argmax(1)] != ""
        sources.append((vertices, weights, [f for f in attachment.faces if all(keep[v] for v in f)]))
    grid = (np.arange(FRONT_SAMPLES) + .5) / FRONT_SAMPLES
    rows, cols, values = [], [], []
    offset, row = 0, 0
    all_rest, all_weights = [], []
    for vertices, weights, polygons in sources:
        for face in polygons:
            a, b, c, d = (face[0], face[1], face[2], face[3] if len(face) > 3 else face[2])
            for u in grid:
                for v in grid:
                    rows += [row] * 4
                    cols += [offset + a, offset + b, offset + c, offset + d]
                    values += [(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v]
                    row += 1
        all_rest.append(vertices)
        all_weights.append(weights)
        offset += len(vertices)
    front_rest = np.concatenate(all_rest)
    samples = csr_matrix((values, (rows, cols)), shape=(row, offset))
    used = np.unique(samples.indices)
    samples = samples[:, used]
    return (np.concatenate((front_rest[used], np.ones((len(used), 1))), axis=1), csr_matrix(np.concatenate(all_weights)[used]), samples)


def build_field(rig, *, voxel: float = VOXEL, drawn: dict | None = None) -> BodyField:
    """The signed distance grids and vertex sets of the rig's rest body; ``drawn``: the attachments in the picture (the top, the hair)."""
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
    front_rest, front_weights, front_samples = _front_surface(rig, drawn)
    return BodyField(origin, voxel, sdf, part_bone, inverse, hand, forearm, weights, vertices, front_rest, front_weights, front_samples)


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


def front_maps(rig, field: BodyField, bone_globals: np.ndarray) -> np.ndarray:
    """(T, X, Y) float32: the z of the drawn body's frontmost surface in each 1 cm cell of the front view, -inf where none is drawn."""
    inverse_rest = np.linalg.inv(rig.rest_globals.detach().cpu().numpy())
    x0, y0 = FRONT_BOUNDS[0][0], FRONT_BOUNDS[1][0]
    shape = tuple(int(round((hi - lo) / FRONT_CELL)) for lo, hi in FRONT_BOUNDS)
    out = np.full((len(bone_globals), *shape), -np.inf, dtype=np.float32)
    weights = field.front_weights.tocoo()
    for t, globals_ in enumerate(bone_globals):
        skin = (globals_ @ inverse_rest)[:, :3, :]                                            # (B, 3, 4)
        contribution = np.einsum("nab,nb->na", skin[weights.col], field.front_rest[weights.row]) * weights.data[:, None]
        posed = np.zeros((len(field.front_rest), 3))
        np.add.at(posed, weights.row, contribution)
        points = field.front_samples @ posed
        i = np.floor((points[:, 0] - x0) / FRONT_CELL).astype(np.int64)
        j = np.floor((points[:, 1] - y0) / FRONT_CELL).astype(np.int64)
        keep = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
        np.maximum.at(out[t], (i[keep], j[keep]), points[keep, 2].astype(np.float32))
        holes = ~np.isfinite(out[t])
        if holes.any():                                                                        # a cell no sample fell in, inside the surface
            near = maximum_filter(out[t], size=3)
            enclosed = maximum_filter(np.isfinite(out[t]).astype(np.int8), footprint=np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])) \
                & maximum_filter(np.isfinite(out[t]).astype(np.int8), footprint=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]))
            out[t][holes & (enclosed > 0)] = near[holes & (enclosed > 0)]
    return out


def _read_front(maps: np.ndarray, t: int, points: np.ndarray) -> np.ndarray:
    """(N,) the front surface's z at each point of (N, 3); -inf where nothing is drawn."""
    i = np.floor((points[:, 0] - FRONT_BOUNDS[0][0]) / FRONT_CELL).astype(np.int64)
    j = np.floor((points[:, 1] - FRONT_BOUNDS[1][0]) / FRONT_CELL).astype(np.int64)
    inside = (i >= 0) & (i < maps.shape[1]) & (j >= 0) & (j < maps.shape[2])
    out = np.full(len(points), -np.inf)
    out[inside] = maps[t, i[inside], j[inside]]
    return out


def hidden_from_front(rig, field: BodyField, local_rotations: np.ndarray, maps: np.ndarray | None = None) -> dict:
    """How much of each hand the body covers in the front view: a vertex is hidden when drawn body lies in front of it, inside the body or behind it."""
    globals_ = forward_kinematics(rig, local_rotations)
    inverse_rest = np.linalg.inv(rig.rest_globals.detach().cpu().numpy())
    maps = front_maps(rig, field, globals_) if maps is None else maps
    frames = len(globals_)
    share = np.zeros((frames, 2))
    behind = np.zeros((frames, 2))
    for s, side in enumerate(SIDES):
        posed = _skin(field, "hand", side, globals_, inverse_rest)
        for t in range(frames):
            gap = _read_front(maps, t, posed[t]) - posed[t][:, 2]
            hidden = gap > 0.
            share[t, s] = hidden.mean()
            behind[t, s] = gap[hidden].max() if hidden.any() else 0.
    any_vertex, half, whole = share.max(1) > 0., share.max(1) >= .5, share.max(1) >= .95
    return dict(frames_any_vertex=float(any_vertex.mean()), frames_half_of_a_hand=float(half.mean()), frames_whole_hand=float(whole.mean()),
                vertex_share=float(share.mean()), deepest_mm=float(behind.max() * SOURCE_METERS_PER_UNIT * 1000.),
                frames_any_by_side={side: float((share[:, s] > 0.).mean()) for s, side in enumerate(SIDES)})


def _wrist_need(field: BodyField, maps_clear: np.ndarray, bone_globals: np.ndarray, inverse_rest: np.ndarray, side: str, forearm_along: np.ndarray,
                forearm_drive: bool) -> np.ndarray:
    """(T,) how far along +z the wrist must go so that no hand vertex has drawn body in front of it; the hand keeps its rotation, so
    the move is exact. Whether a vertex is inside the body or behind it makes no difference."""
    frames = len(bone_globals)
    hand = _skin(field, "hand", side, bone_globals, inverse_rest)
    forearm = _skin(field, "forearm", side, bone_globals, inverse_rest) if forearm_drive else None
    near_wrist = forearm_along > WRIST_HALF
    need = np.zeros(frames)
    for t in range(frames):
        gap = _read_front(maps_clear, t, hand[t]) + CLEARANCE - hand[t][:, 2]
        need[t] = max(0., float(gap.max(initial=-1.)))
        if forearm_drive and near_wrist.any():
            gap = (_read_front(maps_clear, t, forearm[t][near_wrist]) + CLEARANCE - forearm[t][near_wrist][:, 2]) / forearm_along[near_wrist]
            need[t] = max(need[t], float(gap.max(initial=-1.)))
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
                   elbow: np.ndarray, cap_deg: float) -> np.ndarray:
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
        for angle in np.radians(np.arange(SWIVEL_STEP_DEG, cap_deg + SWIVEL_STEP_DEG / 2., SWIVEL_STEP_DEG)):
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


def _solve_arm(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray, wrist_z: np.ndarray, *,
               elbow_nearest_in_image: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Two-bone IK with fixed lengths, per frame: the wrist moves by at most ``wrist_z`` along +z and never in x or y.

    A depth the arm cannot reach is cut to what it can, so the hand may stay covered rather than move across the image. The elbow stays
    in the arm's plane, or with ``elbow_nearest_in_image`` goes to the point of its circle drawn nearest where it was.
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
    towards = _unit(towards - (towards * axis).sum(-1, keepdims=True) * axis)
    if elbow_nearest_in_image:
        side = np.cross(axis, towards)                                                  # towards, side: the circle's plane
        centre = shoulder + axis * along[:, None]
        angles = np.radians(np.arange(-90., 90.5, 1.))
        ring = (towards[:, None] * np.cos(angles)[None, :, None] + side[:, None] * np.sin(angles)[None, :, None]) * height[:, None, None]
        miss = (((centre[:, None] + ring) - elbow[:, None])[..., :2] ** 2).sum(-1)      # (T, A): how far from the elbow as drawn
        chosen = gaussian_filter1d(angles[miss.argmin(1)], SMOOTH_SIGMA_FRAMES, mode="nearest")     # a grid step must not read as a tremble
        towards = towards * np.cos(chosen)[:, None] + side * np.sin(chosen)[:, None]
    solved_elbow = shoulder + axis * along[:, None] + towards * height[:, None]
    solved_wrist = shoulder + offset
    return _swing(_unit(elbow - shoulder), _unit(solved_elbow - shoulder)), _swing(_unit(wrist - elbow), _unit(solved_wrist - solved_elbow))


@dataclass(frozen=True)
class CorrectionSettings:
    forearm_drives_wrist: bool
    swivel_cap_deg: float
    elbow_nearest_in_image: bool


#: What ships. The other values exist for the comparison that chose these (reports/avatar_render_fixes.json).
CORRECTION_V2 = CorrectionSettings(forearm_drives_wrist=False, swivel_cap_deg=0., elbow_nearest_in_image=True)


def correct(rig, field: BodyField, local_rotations: np.ndarray, target_globals: np.ndarray,
            settings: CorrectionSettings) -> tuple[np.ndarray, np.ndarray, dict]:
    """Depth-only wrist correction, then any elbow swivel (module docstring). Returns locals, world rotations and a record."""
    index = {name: i for i, name in enumerate(rig.bone_names)}
    inverse_rest = np.linalg.inv(rig.rest_globals.detach().cpu().numpy())
    rest = rig.rest_globals.detach().cpu().numpy()[:, :3, :3]
    heads = rig.rest_globals.detach().cpu().numpy()[:, :3, 3]
    local = local_rotations.copy()
    worlds = target_globals.copy()
    frames = len(local)
    record = {}
    original = forward_kinematics(rig, local)
    maps = front_maps(rig, field, original)
    maps_clear = np.stack([maximum_filter(m, size=3) for m in maps])          # a hand vertex near a cell's edge reads the higher neighbour

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
            need = _wrist_need(field, maps_clear, globals_, inverse_rest, side, along, settings.forearm_drives_wrist)
            if need.max(initial=0.) <= 0.:
                break
            depth = _smoothed(need)
            moved_frames |= depth > 1e-6
            swing_upper, swing_fore = _solve_arm(globals_[:, upper[0], :3, 3], globals_[:, fore[0], :3, 3], globals_[:, wrist_bone, :3, 3], depth,
                                                 elbow_nearest_in_image=settings.elbow_nearest_in_image)
            for bone in upper:
                worlds[:, bone] = swing_upper @ worlds[:, bone]
            for bone in fore:
                worlds[:, bone] = swing_fore @ worlds[:, bone]
            recompose(chain)
        smooth = np.zeros(frames)
        if settings.swivel_cap_deg > 0.:
            globals_ = forward_kinematics(rig, local)
            shoulder, elbow, wrist = (globals_[:, b, :3, 3] for b in (upper[0], fore[0], wrist_bone))
            forearm = _skin(field, "forearm", side, globals_, inverse_rest)
            angles = _swivel_angles(field, globals_, forearm, shoulder, wrist, elbow, settings.swivel_cap_deg)
            signs = np.sign(angles)
            smooth = gaussian_filter1d(maximum_filter1d(np.abs(angles), size=DILATE_FRAMES, mode="nearest"), SMOOTH_SIGMA_FRAMES, mode="nearest")
            sign = 1. if signs.sum() >= 0 else -1.         # one direction per side and clip: a swivel that flips would flap
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


def arm_image_changes(rig, before: np.ndarray, after: np.ndarray, pixels_per_unit: float) -> dict:
    """What the correction does to the arm as drawn, per side, against the pose before it: elbow shift in pixels, forearm direction
    and length in the image, and each bone's 3D direction."""
    index = {name: i for i, name in enumerate(rig.bone_names)}
    out = {}
    for side in SIDES:
        s, e, w = (index[f"upperarm01.{side}"], index[f"lowerarm01.{side}"], index[f"wrist.{side}"])
        positions = lambda g, b: g[:, b, :3, 3]                                                                           # noqa: E731
        elbow_px = np.linalg.norm((positions(after, e) - positions(before, e))[:, :2], axis=-1) * pixels_per_unit
        fore0, fore1 = positions(before, w) - positions(before, e), positions(after, w) - positions(after, e)
        upper0, upper1 = positions(before, e) - positions(before, s), positions(after, e) - positions(after, s)
        length0, length1 = np.linalg.norm(fore0[:, :2], axis=-1), np.linalg.norm(fore1[:, :2], axis=-1)
        cosine = (fore0[:, :2] * fore1[:, :2]).sum(-1) / np.maximum(length0 * length1, 1e-9)
        image_angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
        direction = lambda a, b: np.degrees(np.arccos(np.clip((_unit(a) * _unit(b)).sum(-1), -1., 1.)))                    # noqa: E731
        forearm_3d, upper_3d = direction(fore0, fore1), direction(upper0, upper1)
        moved = (elbow_px > .5) | (forearm_3d > .5)
        pick = lambda v: dict(all_p50=float(np.median(v)), all_p95=float(np.percentile(v, 95)), max=float(v.max()),               # noqa: E731
                              moved_p50=float(np.median(v[moved])) if moved.any() else 0., moved_p95=float(np.percentile(v[moved], 95)) if moved.any() else 0.)
        ratio = length1 / np.maximum(length0, 1e-9)
        out[side] = dict(frames_moved=float(moved.mean()), elbow_image_shift_px=pick(elbow_px), forearm_image_direction_deg=pick(image_angle),
                         forearm_3d_direction_deg=pick(forearm_3d), upper_arm_3d_direction_deg=pick(upper_3d),
                         forearm_image_length_ratio=dict(p05=float(np.percentile(ratio, 5)), median=float(np.median(ratio)), minimum=float(ratio.min())),
                         frames_elbow_over_10px=float((elbow_px > 10.).mean()), frames_elbow_over_30px=float((elbow_px > 30.).mean()))
    return out


def apply_correction(rig, field: BodyField, retargeted, settings: CorrectionSettings | None = None) -> tuple[object, dict]:
    """A retargeted motion with its hands clear of the drawn body; the record of before, after and movement."""
    settings = CORRECTION_V2 if settings is None else settings
    before = penetration(rig, field, retargeted.local_rotations)
    hidden_before = hidden_from_front(rig, field, retargeted.local_rotations)
    local, worlds, moved = correct(rig, field, retargeted.local_rotations, retargeted.target_globals, settings)
    after = penetration(rig, field, local)
    hidden_after = hidden_from_front(rig, field, local)
    return replace(retargeted, local_rotations=local, target_globals=worlds), dict(before=before, after=after, hidden_before=hidden_before,
                                                                                    hidden_after=hidden_after, moved=moved)


__all__ = ["BodyField", "CorrectionSettings", "CORRECTION_V2", "build_field", "penetration", "hidden_from_front", "front_maps", "correct",
           "apply_correction", "arm_image_changes"]
