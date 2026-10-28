"""Co-observed canonical states estimated from continuous signing video.

No continuous ASL 3D capture can be acquired without a request, so this adapter
estimates one from video the project may use (How2Sign, CC BY-NC 4.0): body, both
hands, head and face are estimated **from the same frames**, so they share one
clock by construction — the property no isolated or multi-device source supplies.
Every value is a monocular estimate: ``observed=False`` with an inference method
naming the exact model hashes.

**Estimators.** MediaPipe Tasks (Apache-2.0 code; Google model cards): pose world
landmarks (33, metres, hip-centred), hand world landmarks (21 per hand, metres,
hand-centred), and the face landmarker's 52 blendshape scores and facial
transformation matrix.

**Axes.** MediaPipe world landmarks are image-aligned (x right, y down, z away from
the camera). They are mapped to y-up by ``(x, y, z) -> (x, -y, -z)``, a proper
rotation (determinant +1), so handedness of every frame is preserved.

**Kinematics.** Rotations come from :mod:`landmark_kinematics`: pelvis and chest are
full root frames built from hips/shoulders and the spine; limbs and fingers are
minimal swings from declared rest directions, with zero twist by declaration. The
wrist is the hand's full frame relative to the forearm's world frame; both are
camera-aligned, which is the stated assumption that makes them composable.

**Hands.** MediaPipe's handedness label assumes a mirrored (selfie) image. It is
not used: each detected hand is assigned to the pose's anatomical left or right
wrist by image distance, within a declared radius, at most one hand per side.

**Unavailable by declaration.** ``root_translation`` (world landmarks are
hip-centred), ``left_gaze``/``right_gaze`` (no calibrated mapping from the
landmarks to a direction is established; the ``eyeLook*`` blendshapes stay in
``face``), and ``contact``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from ..pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from ..pose.rotations import matrix_to_rotation_6d
from .landmark_kinematics import SwingJoint, frame_from_axes, solve_chain

MODEL_SHA256 = {
    "pose_landmarker_heavy.task": "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
    "hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
}
MODEL_URLS = {
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}
VISIBILITY_THRESHOLD = .5
HAND_ASSIGNMENT_RADIUS = .12  # normalised image units between hand wrist and pose wrist
Y_UP = np.diag([1., -1., -1.])
CONVENTION = "mediapipe-tasks-swing-ik-y-up-v1"
FRAME = "camera-aligned-y-up-metres"

# Pose landmark indices (MediaPipe BlazePose 33).
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28
MID_HIP, MID_SHOULDER = 33, 34  # appended

BODY_JOINTS = (
    SwingJoint("LeftUpperArm", "Chest", L_SHOULDER, L_ELBOW, (1., 0., 0.)),
    SwingJoint("LeftForearm", "LeftUpperArm", L_ELBOW, L_WRIST, (1., 0., 0.)),
    SwingJoint("RightUpperArm", "Chest", R_SHOULDER, R_ELBOW, (-1., 0., 0.)),
    SwingJoint("RightForearm", "RightUpperArm", R_ELBOW, R_WRIST, (-1., 0., 0.)),
    SwingJoint("LeftThigh", "Pelvis", L_HIP, L_KNEE, (0., -1., 0.)),
    SwingJoint("LeftShin", "LeftThigh", L_KNEE, L_ANKLE, (0., -1., 0.)),
    SwingJoint("RightThigh", "Pelvis", R_HIP, R_KNEE, (0., -1., 0.)),
    SwingJoint("RightShin", "RightThigh", R_KNEE, R_ANKLE, (0., -1., 0.)),
)
BODY_LABELS = ("Pelvis", "Chest", *(j.name for j in BODY_JOINTS), "LeftHand", "RightHand")
_FINGER_POINTS = {"Thumb": (1, 2, 3, 4), "Index": (5, 6, 7, 8), "Middle": (9, 10, 11, 12),
                  "Ring": (13, 14, 15, 16), "Pinky": (17, 18, 19, 20)}
_THUMB_REST = (2 ** -.5, 2 ** -.5, 0.)


def finger_joints(side: str) -> tuple[SwingJoint, ...]:
    joints = []
    for finger, points in _FINGER_POINTS.items():
        rest = _THUMB_REST if finger == "Thumb" else (0., 1., 0.)
        parent = "Hand"
        for k in (1, 2, 3):
            name = f"{side}Hand{finger}{k}"
            joints.append(SwingJoint(name, parent, points[k - 1], points[k], rest))
            parent = name
    return tuple(joints)


@dataclass(frozen=True)
class ClipLandmarks:
    """Per-frame estimator outputs for one clip, before any kinematics."""

    times: np.ndarray            # (T,) seconds from the video's presentation timestamps
    pose_world: np.ndarray       # (T, 33, 3) MediaPipe world axes, metres
    pose_visibility: np.ndarray  # (T, 33)
    pose_image: np.ndarray       # (T, 33, 2) normalised image coordinates
    pose_found: np.ndarray       # (T,)
    hands_world: np.ndarray      # (T, H, 21, 3) detected hands, zero-padded
    hands_image: np.ndarray      # (T, H, 21, 2)
    hands_found: np.ndarray      # (T, H)
    blendshapes: np.ndarray      # (T, 52)
    blendshape_names: tuple[str, ...]
    face_matrix: np.ndarray      # (T, 4, 4)
    face_found: np.ndarray       # (T,)
    roi: bool = False            # face/hands came from crops; head translation is then unavailable
    other_person: np.ndarray | None = None  # (T,) largest other person's shoulder width / the widest person's (0 when
                                            # alone), held between counted frames; None when people were not counted


def assign_hands(clip: ClipLandmarks) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per side: (T, 21, 3) world landmarks and (T,) validity, by wrist proximity."""
    frames = len(clip.times)
    out = {side: (np.zeros((frames, 21, 3)), np.zeros(frames, dtype=bool)) for side in ("Left", "Right")}
    for t in range(frames):
        if not clip.pose_found[t]:
            continue
        candidates = []
        for side, wrist in (("Left", L_WRIST), ("Right", R_WRIST)):
            for h in np.flatnonzero(clip.hands_found[t]):
                distance = float(np.linalg.norm(clip.hands_image[t, h, 0] - clip.pose_image[t, wrist]))
                if distance <= HAND_ASSIGNMENT_RADIUS:
                    candidates.append((distance, side, h))
        used_sides, used_hands = set(), set()
        for distance, side, h in sorted(candidates):
            if side in used_sides or h in used_hands:
                continue
            used_sides.add(side)
            used_hands.add(h)
            out[side][0][t] = clip.hands_world[t, h]
            out[side][1][t] = True
    return out


def _polar_rotation(matrices: np.ndarray) -> np.ndarray:
    """Nearest rotation (SVD polar factor) of each 3x3, determinant forced to +1."""
    u, _, vt = np.linalg.svd(matrices)
    d = np.sign(np.linalg.det(u @ vt))
    u[..., :, -1] *= d[..., None]
    return u @ vt


def elbows_in_image(pose_image: np.ndarray) -> dict[str, np.ndarray]:
    """Per side, (T,) whether the pose's elbow image landmark lies inside the frame: ``0 <= x <= 1`` and
    ``0 <= y <= 1`` in MediaPipe's normalised image coordinates (bounds included; non-finite is outside)."""
    out = {}
    for side, elbow in (("Left", L_ELBOW), ("Right", R_ELBOW)):
        xy = np.asarray(pose_image)[:, elbow, :2]
        out[side] = np.isfinite(xy).all(1) & (xy >= 0.).all(1) & (xy <= 1.).all(1)
    return out


def solve_rotations(world: np.ndarray, visible_pose: np.ndarray,
                    hands: dict[str, tuple[np.ndarray, np.ndarray]], *,
                    upright_chest: bool = False, elbow_in_image: bool = False,
                    pose_image: np.ndarray | None = None) -> tuple[dict, dict]:
    """Body and finger rotations from y-up 3D points in the MediaPipe pose-33 index layout.

    ``world`` is (T, 33, 3) with only the body indices used here required; ``visible_pose``
    is (T, 33); ``hands`` maps side -> ((T, 21, 3) y-up points in base-to-tip finger
    order, (T,) validity). Any estimator or triangulation can feed this.

    ``upright_chest`` is for framings that cut the body above the hips (webcam recordings
    such as ASL Citizen): the chest frame takes its up axis from the camera vertical, made
    orthogonal to the shoulder line, instead of from hips to shoulders. It needs only the
    shoulders, keeps the chest's yaw and roll, and drops its lean; the arms (children of the
    chest) and the hands then solve wherever their own points are seen. The pelvis, the
    chest's rotation relative to it, and the legs stay invalid without the hips.

    ``elbow_in_image`` (needs ``pose_image``, (T, 33, 2) normalised image coordinates) is for
    framings that cut the arms: MediaPipe still places an elbow that is outside the image, often
    with visibility >= 0.5, and that elbow is a guess. With it, a side's elbow counts as visible
    only on frames where its image landmark lies inside the frame (``elbows_in_image``), on top
    of the visibility rule. That gates exactly three joints per side, and nothing else:

    - ``{Side}UpperArm`` (shoulder -> elbow bone),
    - ``{Side}Forearm`` (elbow -> wrist bone; also a child of the upper arm),
    - ``{Side}Hand`` (the wrist: the hand frame relative to the forearm's world frame).

    Fingers (rooted at the palm frame), chest, pelvis and legs are untouched; the wrist and
    shoulder landmarks are not position-gated. The default (False) leaves every output
    bit-identical to the solver without the option.
    """
    if elbow_in_image:
        if pose_image is None or np.shape(pose_image)[:2] != np.shape(visible_pose)[:2]:
            raise ValueError("elbow_in_image needs pose_image with the pose's (T, 33) layout")
        inside = elbows_in_image(pose_image)
        visible_pose = np.array(visible_pose, dtype=bool, copy=True)
        visible_pose[:, L_ELBOW] &= inside["Left"]
        visible_pose[:, R_ELBOW] &= inside["Right"]
    mid_hip = (world[:, L_HIP] + world[:, R_HIP]) / 2
    mid_shoulder = (world[:, L_SHOULDER] + world[:, R_SHOULDER]) / 2
    points = np.concatenate((world, mid_hip[:, None], mid_shoulder[:, None]), axis=1)
    visible = np.concatenate((visible_pose, visible_pose[:, [L_HIP, R_HIP]].all(1, keepdims=True),
                              visible_pose[:, [L_SHOULDER, R_SHOULDER]].all(1, keepdims=True)), axis=1)
    up = mid_shoulder - mid_hip
    pelvis, pelvis_ok = frame_from_axes(world[:, L_HIP] - world[:, R_HIP], up)
    chest_up = np.broadcast_to(np.array([0., 1., 0.]), up.shape) if upright_chest else up
    chest, chest_ok = frame_from_axes(world[:, L_SHOULDER] - world[:, R_SHOULDER], chest_up)
    pelvis_ok &= visible[:, MID_HIP] & visible[:, MID_SHOULDER]
    chest_ok &= visible[:, MID_SHOULDER] & (True if upright_chest else visible[:, MID_HIP])
    body = solve_chain(points, {"Pelvis": (pelvis, pelvis_ok), "Chest": (chest, chest_ok)}, BODY_JOINTS, visible)
    chest_local = np.einsum("tji,tjk->tik", pelvis, chest)
    rotations = {"Pelvis": (pelvis, pelvis_ok), "Chest": (chest_local, pelvis_ok & chest_ok)}
    rotations.update({j.name: (body[j.name][0], body[j.name][2]) for j in BODY_JOINTS})
    hand_rotations = {}
    for side, (hand_points, found) in hands.items():
        frame, ok = frame_from_axes(hand_points[:, 5] - hand_points[:, 17], hand_points[:, 9] - hand_points[:, 0])
        ok &= found
        _, forearm_world, forearm_ok = body[f"{side}Forearm"]
        rotations[f"{side}Hand"] = (np.einsum("tji,tjk->tik", forearm_world, frame), ok & forearm_ok)
        solved = solve_chain(hand_points, {"Hand": (frame, ok)}, finger_joints(side))
        hand_rotations[side] = {j.name: (solved[j.name][0], solved[j.name][2]) for j in finger_joints(side)}
    return rotations, hand_rotations


def canonical_from_landmarks(clip: ClipLandmarks, *, sample_id: str, source_id: str, source_sha256: str,
                             clock_id: str, inference_method: str, points: dict | None = None,
                             frame: str = FRAME, upright_chest: bool = False,
                             elbow_in_image: bool = False) -> tuple[MultichannelMotion, dict]:
    """Canonical state from a clip's landmarks.

    By default body and hands come from the clip's monocular world landmarks. Pass
    ``points`` = {"world": (T,33,3) y-up, "visible": (T,33), "hands": {side: ((T,21,3), (T,))}}
    to use other 3D points (for example two-view triangulation) while the face,
    head and blink channels still come from the clip.

    ``upright_chest`` and ``elbow_in_image`` are passed to :func:`solve_rotations`; the elbow
    gate reads the clip's own ``pose_image``. With ``elbow_in_image`` the report also carries
    ``elbow_in_image_frames`` (per side, pose frames whose elbow lies inside the image); without
    it the report is unchanged.
    """
    frames = len(clip.times)
    if points is None:
        world = clip.pose_world @ Y_UP.T
        visible_pose = (clip.pose_visibility >= VISIBILITY_THRESHOLD) & clip.pose_found[:, None]
        assigned = assign_hands(clip)
        hands = {side: (landmarks @ Y_UP.T, found) for side, (landmarks, found) in assigned.items()}
    else:
        world, visible_pose, hands = points["world"], points["visible"], points["hands"]
        if world.shape[0] != frames:
            raise ValueError("3D points must share the clip's frames")
    rotations, hand_rotations = solve_rotations(world, visible_pose, hands, upright_chest=upright_chest,
                                                elbow_in_image=elbow_in_image, pose_image=clip.pose_image)
    hands = {side: (None, hands[side][1]) for side in hands}
    def rotation_channel(names, table):
        mats = np.stack([table[n][0] for n in names], axis=1)
        valid = np.stack([table[n][1] for n in names], axis=1)
        values = matrix_to_rotation_6d(torch.from_numpy(np.ascontiguousarray(mats))).numpy()
        return values, valid

    def channel(values, valid, labels, units, frame):
        values = np.where(valid[..., None], values, 0.)
        return MotionChannel(values=values, valid=valid, observed=np.zeros_like(valid),
                             confidence=np.where(valid, 1., 0.), labels=tuple(labels), coordinate_frame=frame,
                             units=units, source_id=source_id, source_sha256=source_sha256, convention=CONVENTION,
                             inference_method=inference_method if valid.any() else None)

    channels = {"body": channel(*rotation_channel(BODY_LABELS, rotations), BODY_LABELS, "unitless", frame)}
    for side, name in (("Left", "left_hand"), ("Right", "right_hand")):
        labels = tuple(j.name for j in finger_joints(side))
        channels[name] = channel(*rotation_channel(labels, hand_rotations[side]), labels, "unitless", frame)
    face_ok = clip.face_found & bool(clip.blendshape_names)
    if face_ok.any():
        rotation = _polar_rotation(clip.face_matrix[:, :3, :3])
        rotation = np.where(face_ok[:, None, None], rotation, np.eye(3))
        head = matrix_to_rotation_6d(torch.from_numpy(np.ascontiguousarray(rotation))).numpy()[:, None]
        head_frame = "mediapipe-face-geometry-roi-camera-v1" if clip.roi else "mediapipe-face-geometry-camera-v1"
        channels["head_rotation"] = channel(head, face_ok[:, None], ("Head",), "unitless", head_frame)
        if not clip.roi:
            channels["head_translation"] = channel((clip.face_matrix[:, :3, 3] * .01)[:, None], face_ok[:, None],
                                                   ("Head",), "m", head_frame)
        scores = np.clip(clip.blendshapes, 0., 1.)
        channels["face"] = channel(scores[..., None], np.broadcast_to(face_ok[:, None], scores.shape).copy(),
                                   clip.blendshape_names, "unitless", "mediapipe-face-blendshape-basis")
        blink = [clip.blendshape_names.index(n) for n in ("eyeBlinkLeft", "eyeBlinkRight")]
        channels["blink"] = channel(scores[:, blink, None], np.broadcast_to(face_ok[:, None], (frames, 2)).copy(),
                                    ("eyeBlinkLeft", "eyeBlinkRight"), "unitless", "mediapipe-eyeBlink-as-eyelid-closure")
    for name in CHANNELS:
        if name not in channels:
            _, width, units = CHANNELS[name]
            channels[name] = MotionChannel(
                values=np.zeros((frames, 1, width)), valid=np.zeros((frames, 1), dtype=bool),
                observed=np.zeros((frames, 1), dtype=bool), confidence=np.zeros((frames, 1)),
                labels=(f"{name}-unavailable",), coordinate_frame="not-estimated", units=units,
                source_id=source_id, source_sha256=source_sha256, convention=CONVENTION)
    state = MultichannelMotion(np.asarray(clip.times, dtype=np.float64), clock_id, sample_id, channels)
    report = dict(frames=frames, pose_frames=int(clip.pose_found.sum()), face_frames=int(face_ok.sum()),
                  left_hand_frames=int(hands["Left"][1].sum()), right_hand_frames=int(hands["Right"][1].sum()),
                  arm_frames={j: int(rotations[j][1].sum()) for j in ("LeftForearm", "RightForearm")},
                  upright_chest=upright_chest,
                  unavailable=sorted(n for n, c in channels.items() if not c.valid.any()))
    if elbow_in_image:
        inside = elbows_in_image(clip.pose_image)
        report["elbow_in_image"] = True
        report["elbow_in_image_frames"] = {side: int((inside[side] & clip.pose_found).sum()) for side in inside}
    return state, report


RAW_LANDMARKS_SCHEMA = "mediapipe-raw-landmarks-v1"
# float16 only where its resolution is far below the estimator's noise: metric world landmarks (|x| < 2 m: < 1 mm),
# visibility and blendshape scores (in [0, 1]: < 5e-4). Image coordinates (a pixel at 1080p is ~5e-4 and the elbow
# gate tests the exact 0/1 bounds) and the face matrix (centimetre translations) stay float32; times stay float64.
_RAW_DTYPES = dict(times=np.float64, pose_world=np.float16, pose_visibility=np.float16, pose_image=np.float32,
                   pose_found=bool, hands_world=np.float16, hands_image=np.float32, hands_found=bool,
                   blendshapes=np.float16, face_matrix=np.float32, face_found=bool, other_person=np.float16)


def save_raw_landmarks(clip: ClipLandmarks) -> bytes:
    """The clip's per-frame MediaPipe outputs as compact ``.npz`` bytes: pose world landmarks, pose image landmarks
    with visibility, both hand slots (world and image, with detection flags; the side assignment is redone from them
    by :func:`assign_hands`), face blendshapes (with names) and transformation matrix, the people count when present.
    :func:`load_raw_landmarks` returns a ``ClipLandmarks`` every solver accepts, so a later solver change is a
    re-solve, not a re-extraction. Solve from the loaded copy to make the state reproducible from these bytes."""
    import io
    import json
    arrays = {name: np.asarray(getattr(clip, name)).astype(dtype) for name, dtype in _RAW_DTYPES.items()
              if getattr(clip, name) is not None}
    arrays["blendshape_names"] = np.array(clip.blendshape_names, dtype=str)
    arrays["metadata"] = np.array(json.dumps(dict(schema=RAW_LANDMARKS_SCHEMA, roi=bool(clip.roi),
                                                  frames=int(len(clip.times))), sort_keys=True))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def load_raw_landmarks(payload: bytes, *, expected_sha256: str | None = None) -> ClipLandmarks:
    """A ``ClipLandmarks`` from :func:`save_raw_landmarks` bytes (no pickle), floats widened to float64."""
    import io
    import json
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("raw landmarks: content SHA-256 differs from the recorded one")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("schema") != RAW_LANDMARKS_SCHEMA:
            raise ValueError(f"raw landmarks: unknown schema {metadata.get('schema')!r}")
        values = {name: (archive[name].astype(bool) if dtype is bool else archive[name].astype(np.float64))
                  for name, dtype in _RAW_DTYPES.items() if name in archive.files}
        names = tuple(str(n) for n in archive["blendshape_names"])
    return ClipLandmarks(blendshape_names=names, roi=bool(metadata["roi"]),
                         other_person=values.pop("other_person", None), **values)


def verify_models(directory: Path) -> dict[str, Path]:
    """Model paths, each checked against its pinned SHA-256."""
    paths = {}
    for name, digest in MODEL_SHA256.items():
        path = Path(directory) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{name}: model bytes differ from the pinned hash")
        paths[name] = path
    return paths


def probe_runtime(models: dict[str, Path]) -> None:
    """Create and close each landmarker once, so a broken runtime (a missing system library such as
    libEGL) fails the job at start instead of being recorded as a refusal of every clip."""
    import mediapipe as mp
    vision, base = mp.tasks.vision, mp.tasks.BaseOptions
    try:
        for make, options, name in (
                (vision.PoseLandmarker, vision.PoseLandmarkerOptions, "pose_landmarker_heavy.task"),
                (vision.HandLandmarker, vision.HandLandmarkerOptions, "hand_landmarker.task"),
                (vision.FaceLandmarker, vision.FaceLandmarkerOptions, "face_landmarker.task")):
            make.create_from_options(options(base_options=base(model_asset_path=str(models[name])))).close()
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"the MediaPipe runtime cannot create landmarkers: {error}") from error


def square_crop(image: np.ndarray, centre_xy: np.ndarray, side_px: float):
    """Square crop around a pixel centre, clipped to the image; returns crop and its offset."""
    height, width = image.shape[:2]
    half = max(int(round(side_px / 2)), 8)
    x0, y0 = int(round(centre_xy[0])) - half, int(round(centre_xy[1])) - half
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + 2 * half, width), min(y0 + 2 * half, height)
    if x1c - x0c < 16 or y1c - y0c < 16:
        return None, None
    return np.ascontiguousarray(image[y0c:y1c, x0c:x1c]), np.array([x0c, y0c], dtype=np.float64)


def extract_clip(video: Path, models: dict[str, Path], *, start_seconds: float = 0.,
                 stop_seconds: float | None = None, roi: bool = True,
                 face_scale: float = .9, hand_scale: float = .9, count_people_every: int = 0) -> ClipLandmarks:
    """Run the landmarkers over every decoded frame in [start, stop).

    With ``roi`` (default), face and hand landmarkers run on square crops centred on
    the pose's nose and wrists, sized by shoulder width, each in its own tracker.
    This keeps them in range on full-body shots; landmarks are mapped back to
    full-image coordinates. In ROI mode the face matrix's translation refers to the
    crop's camera, so the adapter reports head translation as unavailable.

    With ``count_people_every`` = n > 0, every n-th frame also goes through a separate pose landmarker in image mode
    that looks for up to three people. ``other_person`` records the second widest person's shoulder width relative
    to the widest's, which lets a single-signer filter drop interviews and group shots. The signer's own tracker
    stays single-person: asking it for two people changed a lone signer's landmarks (median 2%, 90th percentile 13%
    of the image width on a CMU Panoptic clip).
    """
    import av
    with av.open(str(video)) as probe:                  # refuse unreadable clips before loading models
        if not probe.streams.video:
            raise ValueError("the file has no video stream")
    import mediapipe as mp
    vision = mp.tasks.vision
    base = mp.tasks.BaseOptions
    mode = vision.RunningMode.VIDEO

    def hand_tracker(count):
        return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(models["hand_landmarker.task"])), running_mode=mode, num_hands=count))

    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(models["pose_landmarker_heavy.task"])), running_mode=mode))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=base(model_asset_path=str(models["face_landmarker.task"])), running_mode=mode,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1))
    hands_trackers = [hand_tracker(1), hand_tracker(1)] if roi else [hand_tracker(2)]
    counter = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(models["pose_landmarker_heavy.task"])),
        running_mode=vision.RunningMode.IMAGE, num_poses=3)) if count_people_every > 0 else None
    other = 0.
    rows, names = [], None

    def to_image(array):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(array))

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if start_seconds > 0:
            container.seek(int(start_seconds / stream.time_base), stream=stream, backward=True)
        last_ms, last_t = -1, -float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * frame.time_base)
            if t < start_seconds:
                continue
            if t <= last_t:        # a duplicate or backward timestamp (some downloaded streams): that frame is dropped
                continue
            last_t = t
            if stop_seconds is not None and t >= stop_seconds:
                break
            ms = max(int(round(t * 1000)), last_ms + 1)  # landmarkers need strictly increasing ms
            last_ms = ms
            rgb = frame.to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            size = np.array([width, height], dtype=np.float64)
            p = pose.detect_for_video(to_image(rgb), ms)
            row = dict(t=t, pose_found=bool(p.pose_world_landmarks), hw=[], hi=[])
            if counter is not None:
                if len(rows) % count_people_every == 0:
                    widths = sorted((float(np.linalg.norm(np.array([lm[L_SHOULDER].x - lm[R_SHOULDER].x,
                                                                    lm[L_SHOULDER].y - lm[R_SHOULDER].y]) * size))
                                     for lm in counter.detect(to_image(rgb)).pose_landmarks), reverse=True)
                    other = widths[1] / max(widths[0], 1e-6) if len(widths) > 1 else 0.
                row["other"] = other
            row["pw"] = np.array([[l.x, l.y, l.z] for l in p.pose_world_landmarks[0]]) if row["pose_found"] else np.zeros((33, 3))
            row["pv"] = np.array([l.visibility for l in p.pose_landmarks[0]]) if row["pose_found"] else np.zeros(33)
            row["pi"] = np.array([[l.x, l.y] for l in p.pose_landmarks[0]]) if row["pose_found"] else np.zeros((33, 2))
            face_input, hand_inputs = None, []
            if roi and row["pose_found"]:
                pixels = row["pi"] * size
                shoulder = float(np.linalg.norm(pixels[L_SHOULDER] - pixels[R_SHOULDER]))
                face_input = square_crop(rgb, pixels[0], face_scale * shoulder)
                hand_inputs = [square_crop(rgb, pixels[w], hand_scale * shoulder) for w in (L_WRIST, R_WRIST)]
            elif not roi:
                face_input = (np.ascontiguousarray(rgb), np.zeros(2))
                hand_inputs = [(np.ascontiguousarray(rgb), np.zeros(2))]
            # Every tracker sees one image per frame (a blank 16x16 when no crop exists),
            # so each video-mode stream keeps strictly increasing timestamps.
            blank = np.zeros((16, 16, 3), np.uint8)
            for tracker, item in zip(hands_trackers, hand_inputs or [(None, None)] * len(hands_trackers)):
                crop, offset = item if item is not None else (None, None)
                h = tracker.detect_for_video(to_image(crop if crop is not None else blank), ms)
                if crop is None:
                    continue
                scale = np.array(crop.shape[1::-1], dtype=np.float64)
                for world, image_lm in zip(h.hand_world_landmarks, h.hand_landmarks):
                    row["hw"].append(np.array([[l.x, l.y, l.z] for l in world]))
                    row["hi"].append((np.array([[l.x, l.y] for l in image_lm]) * scale + offset) / size)
            crop, offset = face_input if face_input is not None else (None, None)
            f = face.detect_for_video(to_image(crop if crop is not None else blank), ms)
            row["face_found"] = crop is not None and bool(f.face_blendshapes)
            if row["face_found"]:
                categories = f.face_blendshapes[0]
                current = tuple(c.category_name for c in categories)
                if names is None:
                    names = current
                elif current != names:
                    raise ValueError("blendshape categories changed between frames")
                row["bs"] = np.array([c.score for c in categories])
                row["fm"] = np.asarray(f.facial_transformation_matrixes[0], dtype=np.float64)
            rows.append(row)
    for detector in (pose, face, *hands_trackers, *([counter] if counter is not None else [])):
        detector.close()
    if not rows:
        raise ValueError("no frames in the requested window")
    times = np.array([r["t"] for r in rows])
    if np.any(np.diff(times) <= 0):
        raise ValueError("video presentation timestamps do not strictly increase")
    width_bs = len(names) if names is not None else 0
    slots = 2
    hw = np.zeros((len(rows), slots, 21, 3))
    hi = np.zeros((len(rows), slots, 21, 2))
    hf = np.zeros((len(rows), slots), dtype=bool)
    for t, r in enumerate(rows):
        for k, (w, i) in enumerate(zip(r["hw"][:slots], r["hi"][:slots])):
            hw[t, k], hi[t, k], hf[t, k] = w, i, True
    return ClipLandmarks(
        times=times, pose_world=np.stack([r["pw"] for r in rows]), pose_visibility=np.stack([r["pv"] for r in rows]),
        pose_image=np.stack([r["pi"] for r in rows]), pose_found=np.array([r["pose_found"] for r in rows]),
        hands_world=hw, hands_image=hi, hands_found=hf,
        blendshapes=np.stack([r.get("bs", np.zeros(width_bs)) for r in rows]),
        blendshape_names=names or (), face_matrix=np.stack([r.get("fm", np.eye(4)) for r in rows]),
        face_found=np.array([r["face_found"] for r in rows]), roi=roi,
        other_person=np.array([r["other"] for r in rows]) if counter is not None else None)


def inference_method() -> str:
    import mediapipe as mp
    return ("monocular estimate: mediapipe " + mp.__version__ + " tasks; "
            + ", ".join(f"{n}={d[:16]}" for n, d in sorted(MODEL_SHA256.items()))
            + f"; {CONVENTION}; values are fitted, not observed")


__all__ = ["MODEL_SHA256", "MODEL_URLS", "ClipLandmarks", "square_crop", "probe_runtime", "solve_rotations", "BODY_LABELS", "BODY_JOINTS", "finger_joints",
           "assign_hands", "canonical_from_landmarks", "verify_models", "extract_clip", "inference_method",
           "elbows_in_image", "save_raw_landmarks", "load_raw_landmarks", "RAW_LANDMARKS_SCHEMA"]
