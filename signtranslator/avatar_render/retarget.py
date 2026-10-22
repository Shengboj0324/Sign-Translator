"""Retarget canonical swing-IK states onto the audited MakeHuman default rig.

The canonical state (``video_landmarks``/``two_view_states``) stores, per joint, a
local rotation ``L_j``; world frames compose as ``W_j = W_parent(j) L_j`` and the
bone of ``j`` points along ``W_j r_j`` with ``r_j`` the declared rest direction
(:mod:`signtranslator.generation.canonical_skeleton`). The MakeHuman rig has its
own rest frames ``Rest_b`` (bone ``y`` = head-to-tail). Nothing in this module is
hand-tuned; the connection is computed:

**Axes.** Both frames are right-handed with ``x`` the signer's left, ``y`` up and
``z`` the signer's facing direction (the camera sees the signer from ``+z``). For
the canonical frame this is declared by the estimators (MediaPipe image ``x``
right → signer's left for a signer facing the camera; ``z`` toward the camera).
For the rig it is *checked* from the rest geometry (``.L`` shoulders at ``+x``,
eyes in front of the head joint, the neck above the hips), so the axis map is the
identity and no reflection can occur.

**Reference.** The project's own landmark IK (:func:`solve_rotations`) is run on the
MakeHuman rest skeleton itself — shoulders, elbows, wrists, hips, knees, ankles
and the 21 hand points read from rest bone heads — giving each canonical joint's
world frame ``W_ref_j`` for the rig's rest pose. A state equal to that reference
therefore poses the rig exactly at rest.

**Retarget.** Per frame, the world change of each canonical joint from that
reference is ``Δ_j = W_j W_ref_jᵀ``. Every MakeHuman bone ``b`` driven by ``j``
gets the world rotation ``G_b = Δ_j Rest_b`` (bones such as ``upperarm01`` and
``upperarm02`` that together span one canonical bone share ``Δ_j``, so the whole
segment turns rigidly). Because the IK reproduces bone directions exactly
(``W_ref r = unit(tail - head)`` on the rest rig), ``G_b`` maps the rig's rest
bone onto ``W_j r_j``: MakeHuman forward kinematics reproduces the canonical bone
*directions* exactly, whatever the lengths. Local rotations for
:meth:`MakeHumanRig.pose` follow as ``L_b = (G_parent Rest_parentᵀ Rest_b)ᵀ G_b``.

Declared distributions that do not change any measured direction: the spine
blends pelvis to chest over ``spine05``–``spine02``; the neck blends chest to head
over ``neck01``–``neck03``; ``lowerarm02`` takes half of the forearm-to-hand twist
about the forearm axis (so pronation does not collapse at the wrist); the upper
arm's twist about its own axis is chosen so the elbow bends about the rig's
rest elbow hinge (see :func:`_hinge_twist`). Unmapped bones (clavicles, shoulders,
metacarpals, feet, breasts) keep their rest pose relative to their parent.

**Head.** ``head_rotation`` is a camera-frame rotation of the MediaPipe face
geometry (identity = face looking along ``+z``, upright). ``G_head = H Rest_head``.

**Face.** MediaPipe/ARKit-style blendshape coefficients drive the audited
MakeHuman face pose units (bone rotations) by log-linear blending; see
``FACE_UNIT_SOURCES`` and ``UNREPRESENTED_BLENDSHAPES``. Valid gaze vectors (camera
frame) aim the eye bones; otherwise ``eyeLook*`` coefficients drive the eye units.

**Missing values.** A joint unavailable at a frame is filled by slerp between its
nearest valid frames on the state's clock when they are at most 0.5 s apart; in a
longer gap (or at the clip's ends) it eases from the nearer valid value to the rig's
rest relation to its parent over 0.25 s and holds there; a joint never valid keeps
that rest relation. Every fill is counted and reported; fills are excluded from
every error statistic.

This is geometry: it does not establish signing fidelity, contacts (proportions
differ, so hand-to-body contacts are not preserved), or anatomical joint limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from ..data_engineering.video_landmarks import (
    BODY_JOINTS, BODY_LABELS, L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST, finger_joints, solve_rotations,
)
from ..pose.multichannel import CHANNELS, MultichannelMotion
from .makehuman import MakeHumanRig
from .makehuman_attachment import _rig_identity

SUPPORTED_CONVENTIONS = ("mediapipe-tasks-swing-ik-y-up-v1", "sapiens-two-view-swing-ik-y-up-v1")
SIDES = (("Left", "L", 1.), ("Right", "R", -1.))
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
RETARGET_VERSION = "makehuman-default-delta-retarget-v1"
NECK_FRACTIONS = {"neck01": .25, "neck02": .5, "neck03": .75}
SPINE_FRACTIONS = {"spine05": .25, "spine04": .5, "spine03": .75}
FOREARM_TWIST_SHARE = .5
HINGE_FULL_WEIGHT_DEGREES = 25.
MAX_SLERP_GAP_SECONDS = .5
REST_EASE_SECONDS = .25
LEG_JOINTS = ("LeftThigh", "LeftShin", "RightThigh", "RightShin")
#: Canonical channels the retarget does not render, and why.
UNMAPPED_CHANNELS = {
    "root_translation": "the avatar's root stays fixed in front of a fixed camera (source frames differ: "
                        "session world for two-view states, unavailable for monocular ones)",
    "head_translation": "the head follows the skeleton; its translation is not imposed on the neck",
    "contact": "no contact or collision model",
}

# Canonical joints in parent-before-child order, their parents, channels and rest directions.
def _canonical_tree():
    parents = {"Pelvis": None, "Chest": "Pelvis"}
    rests = {}
    for joint in BODY_JOINTS:
        parents[joint.name] = joint.parent
        rests[joint.name] = np.asarray(joint.rest, dtype=np.float64)
    parents["LeftHand"], parents["RightHand"] = "LeftForearm", "RightForearm"
    channel = {name: "body" for name in BODY_LABELS}
    for side, _, _ in SIDES:
        for joint in finger_joints(side):
            parents[joint.name] = f"{side}Hand" if joint.parent == "Hand" else joint.parent
            rests[joint.name] = np.asarray(joint.rest, dtype=np.float64) / np.linalg.norm(joint.rest)
            channel[joint.name] = f"{side.lower()}_hand"
    parents["Head"] = "Chest"        # only for the never-valid fallback; the head is a world rotation
    channel["Head"] = "head_rotation"
    return parents, rests, channel


CANONICAL_PARENTS, CANONICAL_REST, CANONICAL_CHANNEL = _canonical_tree()
WORLD_JOINTS = ("Pelvis", "Head")    # their stored rotation is already a world rotation


def _drivers() -> dict[str, tuple[str, str, float]]:
    """bone -> (from joint, to joint, fraction): G_b = slerp(Δ_from, Δ_to, fraction) Rest_b."""
    drivers = {"root": ("Pelvis", "Pelvis", 1.), "spine02": ("Chest", "Chest", 1.),
               "spine01": ("Chest", "Chest", 1.), "head": ("Head", "Head", 1.)}
    drivers.update({bone: ("Pelvis", "Chest", f) for bone, f in SPINE_FRACTIONS.items()})
    drivers.update({bone: ("Chest", "Head", f) for bone, f in NECK_FRACTIONS.items()})
    for side, s, _ in SIDES:
        for bone, joint in ((f"upperarm01.{s}", "UpperArm"), (f"upperarm02.{s}", "UpperArm"),
                            (f"lowerarm01.{s}", "Forearm"), (f"lowerarm02.{s}", "Forearm"),
                            (f"wrist.{s}", "Hand"), (f"upperleg01.{s}", "Thigh"),
                            (f"upperleg02.{s}", "Thigh"), (f"lowerleg01.{s}", "Shin"),
                            (f"lowerleg02.{s}", "Shin")):
            drivers[bone] = (f"{side}{joint}", f"{side}{joint}", 1.)
        for k, finger in enumerate(FINGERS, 1):
            for segment in (1, 2, 3):
                joint = f"{side}Hand{finger}{segment}"
                drivers[f"finger{k}-{segment}.{s}"] = (joint, joint, 1.)
    return drivers


BONE_DRIVERS = _drivers()

# MakeHuman face pose unit <- ((state coefficient name, weight), ...). Names follow the
# signer's own side on both sides (MakeHuman "Left" units move the rig's +x side,
# checked on the mesh; MediaPipe/ARKit "Left" is assumed to name the signer's left).
FACE_UNIT_SOURCES = {
    "LeftBrowDown": (("browDownLeft", 1.),), "RightBrowDown": (("browDownRight", 1.),),
    "LeftInnerBrowUp": (("browInnerUp", 1.),), "RightInnerBrowUp": (("browInnerUp", 1.),),
    "LeftOuterBrowUp": (("browOuterUpLeft", 1.),), "RightOuterBrowUp": (("browOuterUpRight", 1.),),
    "NoseWrinkler": (("noseSneerLeft", .5), ("noseSneerRight", .5)),
    "LeftUpperLidClosed": (("eyeBlinkLeft", 1.),), "RightUpperLidClosed": (("eyeBlinkRight", 1.),),
    "LeftUpperLidOpen": (("eyeWideLeft", 1.),), "RightUpperLidOpen": (("eyeWideRight", 1.),),
    "LeftLowerLidUp": (("eyeSquintLeft", 1.),), "RightLowerLidUp": (("eyeSquintRight", 1.),),
    "LeftEyeDown": (("eyeLookDownLeft", 1.),), "RightEyeDown": (("eyeLookDownRight", 1.),),
    "LeftEyeUp": (("eyeLookUpLeft", 1.),), "RightEyeUp": (("eyeLookUpRight", 1.),),
    "LeftEyeturnLeft": (("eyeLookOutLeft", 1.),), "LeftEyeturnRight": (("eyeLookInLeft", 1.),),
    "RightEyeturnRight": (("eyeLookOutRight", 1.),), "RightEyeturnLeft": (("eyeLookInRight", 1.),),
    "LeftCheekUp": (("cheekSquintLeft", 1.),), "RightCheekUp": (("cheekSquintRight", 1.),),
    "CheeksPump": (("cheekPuff", 1.),),
    "JawDrop": (("jawOpen", 1.),), "ChinForward": (("jawForward", 1.),),
    "ChinLeft": (("jawLeft", 1.),), "ChinRight": (("jawRight", 1.),),
    "lowerLipDown": (("mouthLowerDownLeft", .5), ("mouthLowerDownRight", .5)),
    "UpperLipUp": (("mouthUpperUpLeft", .5), ("mouthUpperUpRight", .5)),
    "lowerLipBackward": (("mouthRollLower", 1.),), "UpperLipBackward": (("mouthRollUpper", 1.),),
    "lowerLipUp": (("mouthShrugLower", 1.),),
    "LipsKiss": (("mouthPucker", 1.),),
    "UpperLipForward": (("mouthFunnel", .5),), "lowerLipForward": (("mouthFunnel", .5),),
    "MouthMoveLeft": (("mouthLeft", 1.),), "MouthMoveRight": (("mouthRight", 1.),),
    "MouthLeftPullUp": (("mouthSmileLeft", 1.),), "MouthRightPullUp": (("mouthSmileRight", 1.),),
    "MouthLeftPullDown": (("mouthFrownLeft", 1.),), "MouthRightPullDown": (("mouthFrownRight", 1.),),
    "MouthLeftPullSide": (("mouthStretchLeft", 1.),), "MouthRightPullSide": (("mouthStretchRight", 1.),),
}
EYE_LOOK_UNITS = ("LeftEyeDown", "RightEyeDown", "LeftEyeUp", "RightEyeUp", "LeftEyeturnLeft",
                  "LeftEyeturnRight", "RightEyeturnRight", "RightEyeturnLeft")
#: MediaPipe blendshapes with no MakeHuman pose unit (or no faithful one); never rendered.
UNREPRESENTED_BLENDSHAPES = {
    "_neutral": "not an expression; MediaPipe's residual class",
    "mouthClose": "lips sealed over an open jaw: no MakeHuman unit separates lips from jaw",
    "mouthPressLeft": "lip press: no MakeHuman unit", "mouthPressRight": "lip press: no MakeHuman unit",
    "mouthDimpleLeft": "dimple: no MakeHuman unit", "mouthDimpleRight": "dimple: no MakeHuman unit",
    "mouthShrugUpper": "upper-lip shrug: no MakeHuman unit distinct from UpperLipUp",
}
UNUSED_UNITS = ("JawDropStretched", "CheeksSuck", "NasolabialDeepener", "ChinDown", "UpperLipStretched",
                "MouthLeftPlatysma", "MouthRightPlatysma", "TongueOut", "TongueUshape", "TongueUp",
                "TongueDown", "TongueLeft", "TongueRight", "TonguePointUp", "TonguePointDown")


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def matrices_from_6d(values: np.ndarray) -> np.ndarray:
    """Gram–Schmidt columns, identical to ``canonical_skeleton``'s interpretation."""
    a, b = values[..., :3], values[..., 3:]
    r1 = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9)
    b = b - (r1 * b).sum(-1, keepdims=True) * r1
    r2 = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(1e-9)
    return np.stack((r1, r2, np.cross(r1, r2)), axis=-1)


def _rotvec(matrices: np.ndarray) -> np.ndarray:
    shape = matrices.shape[:-2]
    return Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3)


def _expmap(rotvecs: np.ndarray) -> np.ndarray:
    shape = rotvecs.shape[:-1]
    return Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_matrix().reshape(*shape, 3, 3)


def _slerp(a: np.ndarray, b: np.ndarray, fraction) -> np.ndarray:
    """Geodesic interpolation a -> b, batched; ``fraction`` broadcasts over the batch."""
    relative = np.swapaxes(a, -1, -2) @ b
    return a @ _expmap(_rotvec(relative) * np.asarray(fraction, dtype=np.float64)[..., None])


def _swing(rest: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal rotations taking unit ``rest`` to unit ``target`` (batched; antiparallel → π about ⊥)."""
    rest, target = np.broadcast_arrays(rest, target)
    axis = np.cross(rest, target)
    sine = np.linalg.norm(axis, axis=-1)
    cosine = np.clip((rest * target).sum(-1), -1., 1.)
    angle = np.arctan2(sine, cosine)
    fallback = np.cross(rest, np.where(np.abs(rest[..., :1]) < .9, [[1., 0., 0.]], [[0., 1., 0.]]))
    axis = np.where(sine[..., None] > 1e-12, axis, fallback)
    return _expmap(_unit(axis) * angle[..., None])


def _rig_axes_check(rig: MakeHumanRig) -> dict:
    """Verify the rig's rest geometry has x = signer's left, y = up, z = facing."""
    index = {name: i for i, name in enumerate(rig.bone_names)}
    head = rig.rest_globals[:, :3, 3].detach().cpu().numpy()
    left = head[index["upperarm01.L"]] - head[index["upperarm01.R"]]
    up = (head[index["neck01"]] - head[index["spine05"]])
    facing = (head[index["eye.L"]] + head[index["eye.R"]]) / 2 - head[index["head"]]
    checks = dict(left_axis=_unit(left).tolist(), up_axis=_unit(up).tolist(), facing=_unit(facing).tolist())
    if not (_unit(left)[0] > .99 and _unit(up)[1] > .95 and _unit(facing)[2] > .9):
        raise ValueError(f"rig rest geometry does not have x=left, y=up, z=facing: {checks}")
    return checks


def rest_landmarks(rig: MakeHumanRig):
    """MediaPipe-indexed rest landmarks of the rig (pose 33 and both 21-point hands)."""
    index = {name: i for i, name in enumerate(rig.bone_names)}
    rest = rig.rest_globals.detach().cpu().numpy()
    head = lambda bone: rest[index[bone], :3, 3]
    tip = lambda bone: rest[index[bone], :3, 3] + .2 * rest[index[bone], :3, 1]
    world = np.zeros((1, 33, 3))
    visible = np.zeros((1, 33), dtype=bool)
    for point, bone in ((L_SHOULDER, "upperarm01.L"), (R_SHOULDER, "upperarm01.R"),
                        (L_ELBOW, "lowerarm01.L"), (R_ELBOW, "lowerarm01.R"),
                        (L_WRIST, "wrist.L"), (R_WRIST, "wrist.R"),
                        (L_HIP, "upperleg01.L"), (R_HIP, "upperleg01.R"),
                        (L_KNEE, "lowerleg01.L"), (R_KNEE, "lowerleg01.R"),
                        (L_ANKLE, "foot.L"), (R_ANKLE, "foot.R")):
        world[0, point], visible[0, point] = head(bone), True
    hands = {}
    for side, s, _ in SIDES:
        points = np.zeros((1, 21, 3))
        points[0, 0] = head(f"wrist.{s}")
        for k in range(5):
            chain = [f"finger{k + 1}-{segment}.{s}" for segment in (1, 2, 3)]
            for j, bone in enumerate(chain):
                points[0, 1 + 4 * k + j] = head(bone)
            points[0, 4 + 4 * k] = tip(chain[-1])
        hands[side] = (points, np.ones(1, dtype=bool))
    return world, visible, hands


def compose_worlds(local: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """World frames from canonical local rotations, exactly as ``skeleton_positions`` composes them."""
    world = {}
    for joint, parent in CANONICAL_PARENTS.items():
        if joint not in local:
            continue
        world[joint] = local[joint] if joint in WORLD_JOINTS else world[parent] @ local[joint]
    return world


@dataclass(frozen=True)
class RetargetBinding:
    """Computed, reproducible connection from canonical states to the pinned rig."""

    asset_commit: str
    rig_identity: str
    version: str
    rig_axes: dict
    reference_worlds: dict          # canonical joint -> 3x3 world frame of the rig's rest pose
    drivers: dict                   # bone -> (from joint, to joint, fraction)
    face_unit_sources: dict
    face_bones: tuple
    follower_bones: tuple           # held at their rest relation to the parent

    def to_dict(self) -> dict:
        return dict(asset_commit=self.asset_commit, rig_identity=self.rig_identity, version=self.version,
                    rig_axes=self.rig_axes,
                    reference_worlds={k: np.asarray(v).round(12).tolist() for k, v in self.reference_worlds.items()},
                    drivers={k: list(v) for k, v in self.drivers.items()},
                    face_unit_sources={k: [list(p) for p in v] for k, v in self.face_unit_sources.items()},
                    face_bones=list(self.face_bones), follower_bones=list(self.follower_bones),
                    supported_conventions=list(SUPPORTED_CONVENTIONS),
                    spine_fractions=SPINE_FRACTIONS, neck_fractions=NECK_FRACTIONS,
                    forearm_twist_share=FOREARM_TWIST_SHARE,
                    hinge_full_weight_degrees=HINGE_FULL_WEIGHT_DEGREES, unmapped_channels=UNMAPPED_CHANNELS,
                    max_slerp_gap_seconds=MAX_SLERP_GAP_SECONDS, rest_ease_seconds=REST_EASE_SECONDS)

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def rest_locals(self) -> dict:
        """Per joint, the canonical local rotation that reproduces the rig's rest relation to its parent."""
        out = {}
        for joint, parent in CANONICAL_PARENTS.items():
            reference = np.asarray(self.reference_worlds[joint])
            out[joint] = reference if joint in WORLD_JOINTS else np.asarray(self.reference_worlds[parent]).T @ reference
        return out

    def check_rig(self, rig: MakeHumanRig) -> None:
        if rig.asset_commit != self.asset_commit or _rig_identity(rig) != self.rig_identity:
            raise ValueError("retarget binding was built for a different rig")


def build_retarget_binding(rig: MakeHumanRig) -> RetargetBinding:
    """Compute the binding from the rig's own rest geometry and the project's landmark IK."""
    rig.validate()
    axes = _rig_axes_check(rig)
    world, visible, hands = rest_landmarks(rig)
    rotations, hand_rotations = solve_rotations(world, visible, hands)
    local = {name: value[0][0] for name, value in rotations.items()}
    if not all(bool(value[1][0]) for name, value in rotations.items() if name in CANONICAL_PARENTS):
        raise ValueError("the rig's rest skeleton does not solve to a valid reference")
    for side, joints in hand_rotations.items():
        for name, (matrices, valid) in joints.items():
            if not bool(valid[0]):
                raise ValueError(f"reference IK failed for {name}")
            local[name] = matrices[0]
    local["Head"] = np.eye(3)
    reference = compose_worlds(local)
    names = set(rig.bone_names)
    missing = sorted(set(BONE_DRIVERS) - names)
    if missing:
        raise ValueError(f"rig lacks driven bones {missing}")
    index = {name: i for i, name in enumerate(rig.bone_names)}
    head_index = index["head"]
    descendants = set()
    for i, parent in enumerate(rig.parents):
        if parent == head_index or parent in descendants:
            descendants.add(i)
    face_bones = tuple(rig.bone_names[i] for i in sorted(descendants))
    followers = tuple(name for name in rig.bone_names if name not in BONE_DRIVERS and name not in face_bones)
    return RetargetBinding(asset_commit=rig.asset_commit, rig_identity=_rig_identity(rig),
                           version=RETARGET_VERSION, rig_axes=axes, reference_worlds=reference,
                           drivers=dict(BONE_DRIVERS), face_unit_sources=dict(FACE_UNIT_SOURCES),
                           face_bones=face_bones, follower_bones=followers)


# --------------------------------------------------------------------------- state side

def _check_channel_conventions(state: MultichannelMotion) -> None:
    for name in ("body", "left_hand", "right_hand"):
        channel = state.channels[name]
        if channel.valid.any() and channel.convention not in SUPPORTED_CONVENTIONS:
            raise ValueError(f"{name}: convention {channel.convention!r} is not a supported swing-IK convention")
    head = state.channels["head_rotation"]
    if head.valid.any() and not head.coordinate_frame.startswith("mediapipe-face-geometry"):
        raise ValueError(f"head_rotation frame {head.coordinate_frame!r} is not a MediaPipe face-geometry frame")


def _fill_rotations(clock: np.ndarray, matrices: np.ndarray, valid: np.ndarray, rest: np.ndarray | None = None):
    """Fill invalid frames; returns (filled, gaps, longest gap seconds, frames eased toward rest).

    A gap whose valid neighbours are at most ``MAX_SLERP_GAP_SECONDS`` apart is slerped
    between them on the clock. In a longer gap (or one at either end of the clip) each
    frame eases from its nearer valid neighbour to ``rest`` (the rig's rest relation to the
    parent) over ``REST_EASE_SECONDS`` and holds it, so a hand that leaves the frame for
    seconds settles instead of gliding; without ``rest`` it holds the nearer value.
    """
    frames = len(clock)
    if valid.all():
        return matrices.copy(), 0, 0., 0
    indices = np.arange(frames)
    previous = np.maximum.accumulate(np.where(valid, indices, -1))
    following = np.minimum.accumulate(np.where(valid, indices, frames)[::-1])[::-1]
    out = matrices.copy()
    missing = ~valid
    has_p, has_n = previous >= 0, following < frames
    p = np.where(has_p, previous, following)
    n = np.where(has_n, following, previous)
    span = np.where(has_p & has_n, clock[np.clip(n, 0, frames - 1)] - clock[np.clip(p, 0, frames - 1)], np.inf)
    short = missing & (span <= MAX_SLERP_GAP_SECONDS)
    fraction = np.where(np.isfinite(span) & (span > 0), (clock - clock[p]) / np.where(np.isfinite(span) & (span > 0), span, 1.), 0.)
    if short.any():
        out[short] = _slerp(matrices[p[short]], matrices[n[short]], fraction[short])
    long = missing & ~short
    eased = 0
    if long.any():
        since = np.where(has_p, clock - clock[np.clip(previous, 0, frames - 1)], np.inf)
        until = np.where(has_n, clock[np.clip(following, 0, frames - 1)] - clock, np.inf)
        nearer = np.where(since <= until, p, n)
        base = matrices[nearer[long]]
        if rest is None:
            out[long] = base
        else:
            weight = np.clip(np.minimum(since, until)[long] / REST_EASE_SECONDS, 0., 1.)
            out[long] = _slerp(base, np.broadcast_to(rest, base.shape), weight)
            eased = int(long.sum())
    gaps, longest, start = 0, 0., None
    for t in range(frames + 1):
        if t < frames and missing[t]:
            start = t if start is None else start
        elif start is not None:
            gaps += 1
            lo = clock[start - 1] if start > 0 else clock[start]
            hi = clock[t] if t < frames else clock[t - 1]
            longest = max(longest, float(hi - lo))
            start = None
    return out, gaps, longest, eased


@dataclass(frozen=True)
class CanonicalMotion:
    """Filled canonical locals and worlds, with the raw validity used for statistics."""

    local: dict
    world: dict
    raw_valid: dict          # joint -> (T,) validity as stored (before filling)
    chain_valid: dict        # joint -> (T,) joint and all its ancestors valid as stored
    fill: dict               # joint -> fill record
    posed_inferred_values: bool


def canonical_motion(state: MultichannelMotion, *, allow_inferred: bool,
                     rest_locals: dict | None = None) -> CanonicalMotion:
    """Read, validate and fill every canonical joint of a state (see :func:`_fill_rotations`).

    ``rest_locals`` maps joints to the local rotation of the rig's rest relation to the parent,
    the target of long gaps.
    """
    if type(allow_inferred) is not bool:
        raise ValueError("allow_inferred must be an explicit boolean")
    state.validate()
    _check_channel_conventions(state)
    clock = state.timestamps
    frames = len(clock)
    local, raw_valid, fill = {}, {}, {}
    inferred = False
    for joint in CANONICAL_PARENTS:
        channel = state.channels[CANONICAL_CHANNEL[joint]]
        if joint not in channel.labels:
            raw_valid[joint] = np.zeros(frames, dtype=bool)
            fill[joint] = dict(valid_frames=0, filled_frames=frames, gaps=0, longest_gap_seconds=None,
                               fallback="absent from the state: rest relation to the parent")
            continue
        k = channel.labels.index(joint)
        valid = channel.valid[:, k].copy()
        if (valid & ~channel.observed[:, k]).any():
            if not allow_inferred:
                raise ValueError(f"{joint}: valid but unobserved values require allow_inferred=True")
            inferred = True
        raw_valid[joint] = valid
        if not valid.any():
            fill[joint] = dict(valid_frames=0, filled_frames=frames, gaps=0, longest_gap_seconds=None,
                               fallback="never valid: rest relation to the parent")
            continue
        matrices = np.tile(np.eye(3), (frames, 1, 1))
        matrices[valid] = matrices_from_6d(channel.values[valid, k])
        filled, gaps, longest, eased = _fill_rotations(clock, matrices, valid,
                                                       None if rest_locals is None else rest_locals.get(joint))
        local[joint] = filled
        fill[joint] = dict(valid_frames=int(valid.sum()), filled_frames=int((~valid).sum()), gaps=gaps,
                           frames_eased_toward_rest=eased,
                           longest_gap_seconds=longest if gaps else 0., fallback=None)
    chain_valid = {}
    for joint, parent in CANONICAL_PARENTS.items():
        own = raw_valid[joint]
        chain_valid[joint] = own if (parent is None or joint in WORLD_JOINTS) else own & chain_valid[parent]
    world = {}
    for joint, parent in CANONICAL_PARENTS.items():
        if joint not in local:
            continue
        if joint in WORLD_JOINTS:
            world[joint] = local[joint]
        elif parent in world:
            world[joint] = world[parent] @ local[joint]
    return CanonicalMotion(local, world, raw_valid, chain_valid, fill, inferred)


def _deltas(binding: RetargetBinding, motion: CanonicalMotion, frames: int) -> dict[str, np.ndarray]:
    """Δ_j = W_j W_ref_jᵀ; a joint without data (or with an ancestor without data) inherits its parent's Δ."""
    deltas = {}
    for joint, parent in CANONICAL_PARENTS.items():
        reference = np.asarray(binding.reference_worlds[joint])
        if joint in motion.world:
            deltas[joint] = motion.world[joint] @ reference.T
        elif joint in motion.local:
            # The joint has data but an ancestor has none: compose on the ancestor's rig-rest stand-in.
            stand_in = deltas[parent] @ np.asarray(binding.reference_worlds[parent])
            deltas[joint] = stand_in @ motion.local[joint] @ reference.T
        else:
            deltas[joint] = deltas[parent] if parent is not None else np.tile(np.eye(3), (frames, 1, 1))
    return deltas


def _hinge_twist(direction_upper: np.ndarray, direction_fore: np.ndarray, hinge_rest_world: np.ndarray,
                 g_upper: np.ndarray) -> np.ndarray:
    """Twist about the upper-arm axis aligning the rig's rest elbow hinge with the observed bend.

    ``hinge_rest_world`` is the rig's rest bend normal (upper × forearm); carried by ``g_upper``
    it should equal the observed normal ``u × f``. The twist is weighted by the bend angle
    (full above ``HINGE_FULL_WEIGHT_DEGREES``, zero for a straight arm, where the hinge is undefined).
    """
    normal = np.cross(direction_upper, direction_fore)
    bend = np.degrees(np.arctan2(np.linalg.norm(normal, axis=-1), (direction_upper * direction_fore).sum(-1)))
    carried = g_upper @ hinge_rest_world               # the rig's rest hinge, carried by the upper arm
    # remove components along the bone axis, measure the signed angle about it
    axis = direction_upper
    a = carried - (carried * axis).sum(-1, keepdims=True) * axis
    b = normal - (normal * axis).sum(-1, keepdims=True) * axis
    ok = (np.linalg.norm(a, axis=-1) > 1e-9) & (np.linalg.norm(b, axis=-1) > 1e-9)
    b = np.where(ok[:, None], b, a)
    angle = np.arctan2((np.cross(a, b) * axis).sum(-1), (a * b).sum(-1))
    weight = np.clip(bend / HINGE_FULL_WEIGHT_DEGREES, 0., 1.) * ok
    return _expmap(axis * (angle * weight)[:, None])


def _twist_about(rotation: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Signed twist angle of ``rotation`` about unit ``axis`` (swing–twist decomposition)."""
    quaternion = Rotation.from_matrix(rotation).as_quat()           # x, y, z, w
    projection = (quaternion[:, :3] * axis).sum(-1)
    return 2 * np.arctan2(projection, quaternion[:, 3])


@dataclass(frozen=True)
class RetargetedMotion:
    """Per-frame local rotations for :meth:`MakeHumanRig.pose` with full provenance."""

    local_rotations: np.ndarray      # (T, B, 3, 3) float64
    target_globals: np.ndarray       # (T, B, 3, 3) world rotations the locals realise
    canonical: CanonicalMotion
    face: dict
    binding_sha256: str
    sample_id: str
    clock: np.ndarray
    options: dict

    def provenance(self) -> dict:
        return dict(binding_sha256=self.binding_sha256, sample_id=self.sample_id,
                    frames=len(self.clock), posed_inferred_values=self.canonical.posed_inferred_values,
                    fill=self.canonical.fill, face=self.face, options=self.options, accepted_provenance=False)


def face_weights(state: MultichannelMotion, *, baseline: str = "clip_median"):
    """Per-unit (T,) weights in [0, 1], and a record of what was used.

    ``baseline='clip_median'`` subtracts each coefficient's median over the clip's valid
    frames and rescales the remainder to [0, 1] (MediaPipe scores on signing video carry
    large constant offsets, e.g. browDown ≈ 0.8 on neutral faces); ``'none'`` uses raw scores.
    The blink channel, when valid, replaces the eyeBlink* face coefficients.
    """
    if baseline not in ("clip_median", "none"):
        raise ValueError("baseline must be 'clip_median' or 'none'")
    frames = len(state.timestamps)
    face, blink = state.channels["face"], state.channels["blink"]
    scores, valid = {}, {}
    for k, name in enumerate(face.labels):
        scores[name], valid[name] = face.values[:, k, 0].astype(np.float64), face.valid[:, k].copy()
    for k, name in enumerate(blink.labels):
        if name in ("eyeBlinkLeft", "eyeBlinkRight") and blink.valid[:, k].any():
            scores[name], valid[name] = blink.values[:, k, 0].astype(np.float64), blink.valid[:, k].copy()
    medians = {}
    for name in scores:
        if not valid[name].any():
            continue
        values = np.clip(scores[name], 0., 1.)
        if baseline == "clip_median":
            medians[name] = float(np.median(values[valid[name]]))
            values = np.clip((values - medians[name]) / max(1e-6, 1. - medians[name]), 0., 1.)
        # invalid frames are filled by linear interpolation in time, held at the ends
        t = state.timestamps
        values = np.interp(t, t[valid[name]], values[valid[name]])
        scores[name] = values
    weights, used = {}, set()
    for unit, sources in FACE_UNIT_SOURCES.items():
        total = np.zeros(frames)
        for name, coefficient in sources:
            if name in scores and valid[name].any():
                total = total + coefficient * scores[name]
                used.add(name)
        weights[unit] = np.clip(total, 0., 1.)
    present = sorted(n for n in scores if valid[n].any())
    record = dict(baseline=baseline, baseline_medians=medians, rendered_coefficients=sorted(used),
                  unrepresented_present=sorted(n for n in present if n in UNREPRESENTED_BLENDSHAPES),
                  unmapped_present=sorted(n for n in present if n not in used and n not in UNREPRESENTED_BLENDSHAPES),
                  face_valid_frames=int(face.valid.all(axis=1).sum()))
    return weights, record


def retarget_state(rig: MakeHumanRig, binding: RetargetBinding, state: MultichannelMotion, *,
                   allow_inferred: bool, face_units=None, face_baseline: str = "clip_median",
                   hinge_twist: bool = True, drive_legs: bool = False) -> RetargetedMotion:
    """Local rotations posing the rig like the state, frame by frame on the state's clock.

    Legs are held at their rest relation to the pelvis unless ``drive_legs``: signing
    video frames the signer from the waist up, so thigh/shin values are unobserved in
    monocular states and unconstrained in generated ones, and a waist-up render does
    not need them.
    """
    binding.check_rig(rig)
    motion = canonical_motion(state, allow_inferred=allow_inferred, rest_locals=binding.rest_locals())
    frames = len(state.timestamps)
    deltas = _deltas(binding, motion, frames)
    names = rig.bone_names
    index = {name: i for i, name in enumerate(names)}
    rest = rig.rest_globals.detach().cpu().numpy()[:, :3, :3]
    heads = rig.rest_globals.detach().cpu().numpy()[:, :3, 3]
    targets = np.zeros((frames, len(names), 3, 3))
    driven = np.zeros(len(names), dtype=bool)
    for bone, (start, end, fraction) in binding.drivers.items():
        if not drive_legs and end in LEG_JOINTS:
            continue
        delta = deltas[end] if start == end or fraction == 1. else _slerp(deltas[start], deltas[end], fraction)
        targets[:, index[bone]] = delta @ rest[index[bone]]
        driven[index[bone]] = True

    for side, s, _ in SIDES:
        upper = [index[f"upperarm01.{s}"], index[f"upperarm02.{s}"]]
        fore, fore2, wrist = index[f"lowerarm01.{s}"], index[f"lowerarm02.{s}"], index[f"wrist.{s}"]
        if hinge_twist:
            # The rig's rest bend normal, from the segment directions shoulder->elbow->wrist
            # (upperarm01/02 are not collinear, so the twist axis is the whole segment).
            u_rest = _unit(heads[index[f"lowerarm01.{s}"]] - heads[upper[0]])
            f_rest = _unit(heads[wrist] - heads[fore])
            hinge = _unit(np.cross(u_rest, f_rest))
            g_upper = targets[:, upper[0]] @ rest[upper[0]].T          # = Δ_upper
            u = g_upper @ u_rest
            f = (targets[:, fore] @ rest[fore].T) @ f_rest
            twist = _hinge_twist(u, f, hinge, g_upper)
            for bone in upper:
                targets[:, bone] = twist @ targets[:, bone]
            # forearm: keep its rest relation to the (twisted) upper arm, then swing onto its direction
            for bone in (fore, fore2):
                aim = targets[:, bone][:, :, 1]
                base = targets[:, upper[1]] @ (rest[upper[1]].T @ rest[bone])
                targets[:, bone] = _swing(base[:, :, 1], aim) @ base
        # half of the forearm-to-hand twist about the forearm axis goes to lowerarm02
        relative = np.swapaxes(targets[:, fore], -1, -2) @ targets[:, wrist] @ rest[wrist].T @ rest[fore]
        angle = _twist_about(relative, np.array([0., 1., 0.]))
        share = _expmap(np.array([0., 1., 0.]) * (FOREARM_TWIST_SHARE * angle)[:, None])
        targets[:, fore2] = targets[:, fore2] @ share

    face_local = np.tile(np.eye(3), (frames, len(names), 1, 1))
    face_record = dict(rendered=False, reason="no face pose units supplied")
    if face_units is not None:
        if face_units.rig_identity != binding.rig_identity:
            raise ValueError("face pose units were bound to a different rig")
        weights, face_record = face_weights(state, baseline=face_baseline)
        gaze_valid = {s: state.channels[f"{side.lower()}_gaze"].valid[:, 0] for side, s, _ in SIDES}
        face_index = [index[b] for b in binding.face_bones]
        logs = []
        for unit in FACE_UNIT_SOURCES:
            pose = face_units.rotations(unit).detach().cpu().numpy()[face_index]
            logs.append(_rotvec(pose))
        logs = np.stack(logs)                                         # (U, F, 3)
        matrix = np.stack([weights[unit] for unit in FACE_UNIT_SOURCES], axis=1)  # (T, U)
        eye_units = np.array([unit in EYE_LOOK_UNITS for unit in FACE_UNIT_SOURCES])
        any_gaze = gaze_valid["L"] | gaze_valid["R"]
        matrix = np.where(any_gaze[:, None] & eye_units[None], 0., matrix)
        face_local[:, face_index] = _expmap(np.einsum("tu,ufk->tfk", matrix, logs))
        face_record["rendered"] = True
        face_record["gaze"] = {s: int(v.sum()) for s, v in gaze_valid.items()}
        face_record["eye_direction_source"] = ("gaze channels where valid, eyeLook* elsewhere"
                                               if any_gaze.any() else "eyeLook* coefficients (gaze unavailable)")

    # Compose top-down: driven bones take their target, others their face/rest local.
    parents = rig.parents
    local = np.empty((frames, len(names), 3, 3))
    globals_ = np.empty_like(local)
    relative = [rest[i] if p < 0 else rest[p].T @ rest[i] for i, p in enumerate(parents)]
    gaze = {s: state.channels[f"{side.lower()}_gaze"] for side, s, _ in SIDES}
    for i, parent in enumerate(parents):
        preceding = relative[i] if parent < 0 else globals_[:, parent] @ relative[i]
        if driven[i]:
            local[:, i] = np.swapaxes(preceding, -1, -2) @ targets[:, i]
            globals_[:, i] = targets[:, i]
        else:
            local[:, i] = face_local[:, i]
            globals_[:, i] = preceding @ local[:, i]
            for side, s, _ in SIDES:
                if names[i] == f"eye.{s}" and face_units is not None and gaze[s].valid[:, 0].any():
                    ok = gaze[s].valid[:, 0]
                    direction = _unit(np.where(ok[:, None], gaze[s].values[:, 0], [[0., 0., 1.]]))
                    aimed = _swing(globals_[:, i][:, :, 1], direction) @ globals_[:, i]
                    globals_[ok, i] = aimed[ok]
                    local[ok, i] = np.swapaxes(preceding[ok], -1, -2) @ aimed[ok]
    options = dict(hinge_twist=hinge_twist, face_baseline=face_baseline if face_units is not None else None,
                   legs="driven by the state" if drive_legs else "held at their rest relation to the pelvis",
                   unmapped_channels=UNMAPPED_CHANNELS)
    return RetargetedMotion(local, globals_, motion, face_record, binding.sha256(), state.sample_id,
                            np.asarray(state.timestamps, dtype=np.float64), options)


__all__ = ["RetargetBinding", "RetargetedMotion", "CanonicalMotion", "build_retarget_binding", "rest_landmarks",
           "canonical_motion", "retarget_state", "face_weights", "compose_worlds", "matrices_from_6d",
           "BONE_DRIVERS", "FACE_UNIT_SOURCES", "UNMAPPED_CHANNELS", "LEG_JOINTS", "UNREPRESENTED_BLENDSHAPES", "UNUSED_UNITS",
           "SUPPORTED_CONVENTIONS", "CANONICAL_PARENTS", "CANONICAL_REST"]
