"""UWB YouTube-ASL clip keypoints -> ``ClipLandmarks`` -> canonical states (adapter ``uwb-holistic-xy-learned-world-v1``).

The YouTube-ASL Clip Keypoint Dataset (Zelezny, Hruz, Straka, Gueuwou; UWB / JSALT 2024; LINDAT 11234/1-5898;
**CC BY 4.0**) publishes MediaPipe Holistic output for 390,547 caption clips of YouTube-ASL, in 10 zips. It stores no
video, no relative z, no visibility, no world landmarks and no frame rate. This module turns one clip into the
:class:`~.video_landmarks.ClipLandmarks` that :func:`~.video_landmarks.canonical_from_landmarks` solves, with the world
points from the z-free learned image-to-world regressors (:data:`~.world_regressor.FEATURES_XY`).

**Clip file** (verified 2026-10-29 on 120 random clips, 12 from each zip, 15,944 frames; the tests pin it). Member
``<video_id>.<first>-<last>.json`` (no directories; 390,547 members), a JSON object with:

- ``keypoints``: one object per frame with ``pose_landmarks`` (33), ``left_hand_landmarks`` (21),
  ``right_hand_landmarks`` (21), ``face_landmarks`` (478 = mesh + irises); each point ``[x, y]`` in **pixels of the
  source video frame**; a part that was not detected is an empty list (never NaN, never partial). Pose missing on
  75 of 15,944 frames, face on 732;
- ``size``: ``[height, width]`` of the source frame (**height first**: on 119 of 120 clips the face lies inside
  ``x <= size[1], y <= size[0]``, on 34 inside the swapped bounds; ``yolo_sign_space`` ``[x0, y0, x1, y1]`` always
  has ``x1 <= size[1]`` and ``y1 <= size[0]``);
- ``cropped_keypoints`` / ``cropped_size``: the same points mapped linearly into a 512 x 512 crop of ``sign_space``
  (``full = cropped * side / 512 + offset``, checked to 1e-5): redundant, not used; ``bbox_*``, ``sign_space``,
  ``yolo_sign_space``: crop bookkeeping, not used.
- Frame count: ``last - first + 1`` on 95 of 120 clips, ``last - first`` on 23, other on 2. Frame ``j`` of the clip is
  taken as source frame ``first + j``; the +-1 frame ambiguity is what the paired gate's lag search absorbs.
- Hands: Holistic's ``left_hand_landmarks`` wrist lies nearer the pose's left wrist (15) on 99.3% of 10,886 frames
  and ``right_hand_landmarks`` nearer the right wrist (16) on 99.8% of 13,843: the subject's own sides in the
  unmirrored upload, as in Sem-Lex. Pose 11 (left shoulder) lies image-right in 120 of 120 clips.

**Clock.** Clip names are source frame ranges at the video's native rate, which the dataset does not publish. Against
our own caption cues of the same videos (``scripts/uwb_lifter_gate.py``) ``first = floor(start * fps)`` with the
video's true rate (23.976, 25, 29.97 on the first three videos checked; residual median 0.25 frames). The state's
clock is ``t_j = (first + j) / fps`` seconds from the start of the source video (the same origin as our YouTube
states' presentation timestamps); the rate and where it came from (``fps_source``: a per-video estimate, or the
assumed default) are recorded in the clock id and the report.

**Normalisation.** ``x / W``, ``y / H`` with ``(H, W) = size``; aspect ``a = W / H``. The regressors read ``(x, y / a)``.

**World points.** ``world="learned"``: pose and hands from :class:`~.world_regressor.WorldRegressors` whose feature
version must be ``FEATURES_XY`` (a z-trained bundle fed zero depth is refused: it would be silently wrong).
``world="pseudo"``: the geometric baseline ``(x, y / a, 0)`` (``semlex_pose.pseudo_world`` with zero depth gain).

**Visibility** (a declared rule; leg points 25..32 never visible under the learned world, which places none):

- ``"learned"``: the pose regressor's visibility head, as the Sem-Lex gate v2 decided;
- ``"inside"``: finite and inside the image (``"margin:0"``);
- ``"margin:<m>"``: finite and inside the frame extended by ``m`` of its width and height on every side (elbows
  just below a chest-up frame survive);
- ``"all"``: every finite point (diagnostic: the ungated arm, to measure what a rule drops);
- ``"head"`` (UWB lifter v2): shoulders, elbows and wrists (points 11..16) by the bundle's validity head
  (``WorldRegressors.validity``: a point is kept when its probability reaches its side's threshold), every other
  point by ``inside``. It needs the learned world and a bundle with a validity head.

YouTube framings cut the arms: on 120 sampled clips the elbow lies below the frame on 32% of pose frames, so the rule
decides how many arm frames survive (``scripts/uwb_lifter_gate.py``, ``scripts/uwb_outside_frame_accuracy.py``).

**Face.** Blendshapes from MediaPipe's own ``face_blendshapes.tflite`` on the stored mesh in pixels
(``semlex_pose.BlendshapeModel``; it reads only x, y), when a model is given. Head rotation needs depth that is not
stored: ``head="none"`` (default) marks it unavailable; ``head="affine"`` fits a scaled-orthographic camera to
MediaPipe's canonical face (its 33 Procrustes-basis vertices) and takes the nearest rotation, an estimate that the
paired gate measures before any use. Head translation is never available (as in our ROI extraction).

Nothing is invented: a frame or part the stored data cannot support is invalid in the state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re

import numpy as np

from .video_landmarks import CONVENTION, ClipLandmarks, assign_hands, canonical_from_landmarks

ADAPTER = "uwb-holistic-xy-learned-world-v1"
PSEUDO_ADAPTER = "uwb-holistic-xy-pseudo-world-v1"
SOURCE_ID = "youtube_asl_uwb_keypoints"
LICENCE = ("CC BY 4.0 (YouTube-ASL Clip Keypoint Dataset, Zelezny, Hruz, Straka, Gueuwou; UWB / JSALT 2024; "
           "LINDAT 11234/1-5898); derived from YouTube videos owned by their creators: research use, private store, "
           "no redistribution")
CLIP_NAME = re.compile(r"^(?P<video>.+)\.(?P<first>\d+)-(?P<last>\d+)$")
DEFAULT_FPS = 30.
MIN_FPS, MAX_FPS = 10., 70.
FRAME_COUNT_TOLERANCE = 2      # |frames - (last - first + 1)|: 0 or 1 on 118 of 120 sampled clips
POINTS = {"pose_landmarks": 33, "left_hand_landmarks": 21, "right_hand_landmarks": 21, "face_landmarks": 478}
MESH_POINTS = 468
SOLVER_FLAGS = {"generator": dict(upright_chest=True, upright_root=True, elbow_in_image=False),
                "upright-chest": dict(upright_chest=True, upright_root=False, elbow_in_image=False)}


class Refusal(ValueError):
    """The clip cannot be converted without inventing data; the message is the reason."""


@dataclass(frozen=True)
class UwbClip:
    """One UWB clip's points in source-frame pixels (NaN where a part was not detected)."""

    clip_id: str
    video_id: str
    first: int
    last: int
    width: int
    height: int
    pose: np.ndarray       # (T, 33, 2)
    hands: np.ndarray      # (T, 2, 21, 2): slot 0 Holistic left_hand_landmarks, slot 1 right_hand_landmarks
    face: np.ndarray       # (T, 478, 2)

    @property
    def frames(self) -> int:
        return len(self.pose)

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def frame_count_difference(self) -> int:
        """frames - (last - first + 1)."""
        return self.frames - (self.last - self.first + 1)


def parse_clip_name(name: str) -> tuple[str, str, int, int]:
    """(clip id, video id, first frame, last frame) of a member name such as ``abc.000313-000409.json``."""
    stem = name.rsplit("/", 1)[-1]
    stem = stem[:-5] if stem.endswith(".json") else stem
    match = CLIP_NAME.match(stem)
    if match is None:
        raise Refusal(f"clip name {name!r} is not <video>.<first>-<last>")
    first, last = int(match["first"]), int(match["last"])
    if last < first:
        raise Refusal(f"clip {stem}: last frame {last} before first {first}")
    return stem, match["video"], first, last


def _block(frames: list, key: str, clip_id: str) -> np.ndarray:
    count = POINTS[key]
    out = np.full((len(frames), count, 2), np.nan)
    for t, frame in enumerate(frames):
        points = frame.get(key)
        if points is None:
            raise Refusal(f"{clip_id}: frame {t} has no {key}")
        if len(points) == 0:
            continue
        if len(points) != count:
            raise Refusal(f"{clip_id}: frame {t} {key} has {len(points)} points, expected 0 or {count}")
        try:
            array = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise Refusal(f"{clip_id}: frame {t} {key} is not numeric") from error
        if array.shape != (count, 2):
            raise Refusal(f"{clip_id}: frame {t} {key} points are not [x, y]")
        if not np.isfinite(array).all():
            raise Refusal(f"{clip_id}: frame {t} {key} holds non-finite values")
        out[t] = array
    return out


def parse_clip(payload, name: str, *, frame_count_tolerance: int = FRAME_COUNT_TOLERANCE) -> UwbClip:
    """A :class:`UwbClip` from a member's bytes (or its decoded JSON object); ``Refusal`` when malformed."""
    clip_id, video_id, first, last = parse_clip_name(name)
    if isinstance(payload, (bytes, bytearray, str)):
        try:
            data = json.loads(payload)
        except ValueError as error:
            raise Refusal(f"{clip_id}: unreadable JSON: {error}"[:300]) from error
    else:
        data = payload
    if not isinstance(data, dict) or not isinstance(data.get("keypoints"), list):
        raise Refusal(f"{clip_id}: no keypoints list")
    size = data.get("size")
    if (not isinstance(size, list) or len(size) != 2 or not all(isinstance(v, (int, float)) for v in size)
            or min(size) <= 0):
        raise Refusal(f"{clip_id}: size {size!r} is not [height, width]")
    height, width = int(size[0]), int(size[1])
    frames = data["keypoints"]
    if len(frames) < 2:
        raise Refusal(f"{clip_id}: {len(frames)} frames")
    if not all(isinstance(f, dict) for f in frames):
        raise Refusal(f"{clip_id}: a frame is not an object")
    clip = UwbClip(clip_id=clip_id, video_id=video_id, first=first, last=last, width=width, height=height,
                   pose=_block(frames, "pose_landmarks", clip_id),
                   hands=np.stack((_block(frames, "left_hand_landmarks", clip_id),
                                   _block(frames, "right_hand_landmarks", clip_id)), axis=1),
                   face=_block(frames, "face_landmarks", clip_id))
    if abs(clip.frame_count_difference) > frame_count_tolerance:
        raise Refusal(f"{clip_id}: {clip.frames} frames for the range {first}-{last} "
                      f"({clip.frame_count_difference:+d}; tolerance {frame_count_tolerance})")
    return clip


def _found(block: np.ndarray) -> np.ndarray:
    """(T,) detected: every value finite and the points not all equal (an all-zero block is not a detection)."""
    finite = np.isfinite(block).all(axis=(-2, -1))
    spread = np.zeros(len(block))
    if finite.any():
        spread[finite] = np.ptp(block[finite], axis=-2).max(axis=-1)
    return finite & (spread > 0)


def affine_head_rotation(face_xy: np.ndarray, aspect: float, canonical: np.ndarray, basis: np.ndarray):
    """(F, 3, 3) head rotations and (F,) validity from normalised face meshes (F, >= 468, 2), by a least-squares
    scaled-orthographic camera ``(x, -y / a) ~ s R[:2] m + t`` fitted to MediaPipe's canonical face ``m`` on the basis
    vertices; ``R[:2]`` is the nearest pair of orthonormal rows (polar factor) and ``R[2] = R[0] x R[1]``. Axes: x
    right, y up, z toward the viewer, as MediaPipe's face geometry pipeline. Depth is not observed: this is an
    estimate, valid only where the fitted rows are near-orthogonal and of similar length."""
    design = np.c_[canonical[basis], np.ones(len(basis))]
    pinv = np.linalg.pinv(design)                                          # (4, B)
    target = np.stack((face_xy[:, basis, 0], -face_xy[:, basis, 1] / aspect), axis=-1)
    coefficients = np.einsum("kb,fbc->fkc", pinv, target)                  # (F, 4, 2)
    rows = np.swapaxes(coefficients[:, :3], 1, 2)                          # (F, 2, 3)
    u, singular, vt = np.linalg.svd(rows, full_matrices=False)
    top = u @ vt                                                           # (F, 2, 3) orthonormal rows
    rotation = np.concatenate((top, np.cross(top[:, 0], top[:, 1])[:, None]), axis=1)
    ratio = singular[:, 1] / np.maximum(singular[:, 0], 1e-12)
    n1, n2 = np.linalg.norm(rows[:, 0], axis=-1), np.linalg.norm(rows[:, 1], axis=-1)
    cosine = np.abs((rows[:, 0] * rows[:, 1]).sum(-1)) / np.maximum(n1 * n2, 1e-12)
    valid = np.isfinite(rotation).all((1, 2)) & (ratio > .7) & (cosine < .3)
    return np.where(valid[:, None, None], rotation, np.eye(3)), valid


def _unavailable(name: str, frames: int, source_sha256: str):
    from ..pose.multichannel import CHANNELS, MotionChannel
    _, width, units = CHANNELS[name]
    return MotionChannel(values=np.zeros((frames, 1, width)), valid=np.zeros((frames, 1), dtype=bool),
                         observed=np.zeros((frames, 1), dtype=bool), confidence=np.zeros((frames, 1)),
                         labels=(f"{name}-unavailable",), coordinate_frame="not-estimated", units=units,
                         source_id=SOURCE_ID, source_sha256=source_sha256, convention=CONVENTION)


def visibility_margin(rule: str) -> float | None:
    """The frame margin of a visibility rule: None for ``learned``, 0 for ``inside`` and for ``head`` (its points
    other than 11..16), ``m`` for ``margin:<m>`` (0 <= m <= 2), infinity for ``all``; ``ValueError`` otherwise."""
    if rule == "learned":
        return None
    if rule in ("inside", "head"):
        return 0.
    if rule == "all":
        return float("inf")
    if isinstance(rule, str) and rule.startswith("margin:"):
        try:
            m = float(rule[7:])
        except ValueError:
            m = -1.
        if 0. <= m <= 2.:
            return m
    raise ValueError(f"visibility rule {rule!r}: 'learned', 'inside', 'head', 'margin:<m>' (0 <= m <= 2) or 'all'")


def clip_landmarks(clip: UwbClip, fps: float, *, regressors=None, world: str = "learned", visibility: str = "learned",
                   blendshape_model=None, head: str = "none", face_resources=None) -> tuple[ClipLandmarks, dict]:
    """``ClipLandmarks`` (and a diagnostic dict) of one clip at ``fps`` source frames per second."""
    if not np.isfinite(fps) or not MIN_FPS <= fps <= MAX_FPS:
        raise Refusal(f"frame rate {fps} outside [{MIN_FPS}, {MAX_FPS}]")
    if world not in ("learned", "pseudo"):
        raise ValueError("world must be 'learned' or 'pseudo'")
    margin = visibility_margin(visibility)
    if head not in ("none", "affine"):
        raise ValueError("head must be 'none' or 'affine'")
    if visibility == "learned" and world != "learned":
        raise ValueError("learned visibility needs the learned world (the pose regressor's visibility head)")
    if visibility == "head" and (world != "learned" or getattr(regressors, "validity", None) is None):
        raise ValueError("visibility 'head' needs the learned world and a bundle with a validity head")
    if world == "learned":
        from .world_regressor import Z_FREE_FEATURES
        if regressors is None:
            raise ValueError("world='learned' needs regressors")
        if getattr(regressors, "features", None) not in Z_FREE_FEATURES:
            raise ValueError(f"regressors trained on {getattr(regressors, 'features', None)!r} read z, which UWB does "
                             f"not store; a z-free bundle {sorted(Z_FREE_FEATURES)} is required")
    if head == "affine" and face_resources is None:
        raise ValueError("head='affine' needs face_resources (the canonical face)")
    frames, a = clip.frames, clip.aspect
    scale = np.array([clip.width, clip.height], dtype=np.float64)
    times = (clip.first + np.arange(frames, dtype=np.float64)) / fps
    pose_found = _found(clip.pose)
    hands_found = np.stack([_found(clip.hands[:, k]) for k in range(2)], axis=1) & pose_found[:, None]
    face_detected = _found(clip.face)
    pose = np.where(pose_found[:, None, None], np.nan_to_num(clip.pose) / scale, 0.)
    hands = np.where(hands_found[:, :, None, None], np.nan_to_num(clip.hands) / scale, 0.)
    info = dict(adapter=ADAPTER if world == "learned" else PSEUDO_ADAPTER, clip_id=clip.clip_id, video_id=clip.video_id,
                first_frame=clip.first, last_frame=clip.last, frames=frames,
                frame_count_difference=clip.frame_count_difference, fps=float(fps),
                image_size=[clip.width, clip.height], aspect=round(a, 6), pose_frames=int(pose_found.sum()),
                face_detected_frames=int(face_detected.sum()),
                holistic_hand_frames=dict(Left=int(hands_found[:, 0].sum()), Right=int(hands_found[:, 1].sum())),
                visibility=visibility, head=head)
    pose3, hands3 = np.concatenate((pose, np.zeros(pose.shape[:-1] + (1,))), -1), \
        np.concatenate((hands, np.zeros(hands.shape[:-1] + (1,))), -1)
    if margin is None:
        rule = None                             # the visibility head, below
    elif np.isinf(margin):
        rule = np.broadcast_to(pose_found[:, None], pose_found.shape + (33,)).copy()
    else:
        rule = ((pose >= -margin) & (pose <= 1. + margin)).all(-1) & pose_found[:, None]
    if world == "pseudo":
        pose_world, hands_world = pose3.copy(), hands3.copy()
        pose_world[..., 1] /= a
        hands_world[..., 1] /= a
        pose_visibility = rule.astype(np.float64)
        info["world"] = "pseudo-world-z0"
    else:
        pose_world, learned, ok = regressors.pose_world(pose3, a)
        pose_found = pose_found & ok
        hands_found = hands_found & pose_found[:, None]
        pose_visibility = (learned if rule is None else rule) * pose_found[:, None]
        pose_visibility = np.array(pose_visibility, dtype=np.float64)
        if visibility == "head":                 # points 11..16 by the validity head, the rest by `inside`
            from .world_regressor import VALIDITY_POINTS
            kept, _ = regressors.validity_visible(pose3, a)
            pose_visibility[:, list(VALIDITY_POINTS)] = kept & pose_found[:, None]
        pose_visibility[:, 25:] = 0.             # the regressors place no leg points (25..32 are zero): never visible
        hands_world = np.zeros_like(hands3)
        for k, side in enumerate(("Left", "Right")):
            w, ok = regressors.hands(hands3[:, k], a, side)
            hands_found[:, k] &= ok
            hands_world[:, k] = np.where(hands_found[:, k, None, None], w, 0.)
        info["world"] = f"learned:{regressors.sha256}"
        info["regressors_sha256"] = regressors.sha256
    face_found = face_detected & pose_found
    face_matrix = np.tile(np.eye(4), (frames, 1, 1))
    names: tuple = ()
    scores = np.zeros((frames, 0))
    head_ok = np.zeros(frames, dtype=bool)
    if blendshape_model is not None and face_found.any():
        from .semlex_pose import BLENDSHAPE_NAMES
        names = BLENDSHAPE_NAMES
        scores = np.zeros((frames, len(names)))
        scores[face_found] = blendshape_model(np.nan_to_num(clip.face[face_found]))
        if head == "affine":
            face_n = np.nan_to_num(clip.face) / scale
            rotation, ok = affine_head_rotation(face_n[face_found], a, face_resources.canonical, face_resources.basis)
            face_matrix[np.flatnonzero(face_found), :3, :3] = rotation
            head_ok[np.flatnonzero(face_found)] = ok
    else:
        face_found = np.zeros(frames, dtype=bool)
    info.update(face_frames=int(face_found.sum()), head_frames=int(head_ok.sum()))
    landmarks = ClipLandmarks(
        times=times, pose_world=pose_world, pose_visibility=pose_visibility, pose_image=pose.copy(),
        pose_found=pose_found, hands_world=hands_world, hands_image=hands.copy(), hands_found=hands_found,
        blendshapes=scores, blendshape_names=names, face_matrix=face_matrix, face_found=face_found, roi=True,
        other_person=None, image_size=(clip.width, clip.height))
    assigned = assign_hands(landmarks)         # canonical_from_landmarks repeats this exact call; here only to report it
    agreement = {}
    for k, side in enumerate(("Left", "Right")):
        held = hands_found[:, k]
        other = "Right" if side == "Left" else "Left"

        def took(s):
            return held & assigned[s][1] & (assigned[s][0] == landmarks.hands_world[:, k]).all((1, 2))
        same, swapped = took(side), took(other)
        agreement[side] = dict(holistic=int(held.sum()), assigned_same_side=int(same.sum()),
                               assigned_other_side=int(swapped.sum()), dropped=int((held & ~same & ~swapped).sum()))
    info["hand_assignment"] = agreement
    info["_head_ok"] = head_ok
    return landmarks, info


def inference_method(info: dict, flags: str, fps_source: str) -> str:
    learned = info["world"].startswith("learned:")
    world = (f"world landmarks from z-free learned image-to-world regressors (bundle sha256 {info['world'][8:]})"
             if learned else "pseudo-3D (x, y/a, 0): image plane only")
    margin = visibility_margin(info["visibility"])
    visibility = ("the validity head (per-side thresholds) for shoulders, elbows and wrists, finite and inside the "
                  "image elsewhere" if info["visibility"] == "head"
                  else "the pose regressor's visibility head" if margin is None
                  else "finite and inside the image" if margin == 0.
                  else "every finite point" if np.isinf(margin)
                  else f"finite and inside the frame extended by {margin:g} of its size on every side")
    visibility += " (leg points never visible)" * learned
    head = ("head rotation by a scaled-orthographic fit to MediaPipe's canonical face (unvalidated estimate)"
            if info["head"] == "affine" else "head rotation unavailable (no depth stored)")
    return (f"UWB YouTube-ASL Clip Keypoint Dataset MediaPipe Holistic 2D pixel landmarks (publisher's files; "
            f"Holistic version not stated; licence: {LICENCE}); {info['adapter']}: {world}; aspect a={info['aspect']:.4f} from the stored "
            f"frame size {info['image_size'][0]}x{info['image_size'][1]}; clock: source frame index / {info['fps']:g} fps "
            f"({fps_source}); visibility = {visibility}; blendshapes by face_blendshapes.tflite on the stored mesh; "
            f"{head}; {CONVENTION}; solver flags {flags}: chest up axis from the camera vertical (upright prior)"
            + ("; pelvis the camera-upright identity (upright root)" if SOLVER_FLAGS[flags]["upright_root"] else "")
            + "; no elbow gate; values are fitted, not observed")


def convert_clip(clip: UwbClip, fps: float, *, source_sha256: str, fps_source: str, regressors=None,
                 world: str = "learned", visibility: str = "learned", blendshape_model=None, head: str = "none",
                 face_resources=None, flags: str = "generator"):
    """Canonical state and report for one parsed clip (``Refusal`` when it cannot be converted)."""
    if flags not in SOLVER_FLAGS:
        raise ValueError(f"flags must be one of {sorted(SOLVER_FLAGS)}")
    landmarks, info = clip_landmarks(clip, fps, regressors=regressors, world=world, visibility=visibility,
                                     blendshape_model=blendshape_model, head=head, face_resources=face_resources)
    head_ok = info.pop("_head_ok")
    if not landmarks.pose_found.any():
        raise Refusal("no frame has a pose")
    state, report = canonical_from_landmarks(
        landmarks, sample_id=clip.clip_id, source_id=SOURCE_ID, source_sha256=source_sha256,
        clock_id=f"{SOURCE_ID}:{clip.video_id}:frame-index-at-{fps:g}fps-{fps_source}",
        inference_method=inference_method(info, flags, fps_source), **SOLVER_FLAGS[flags])
    channels = dict(state.channels)
    if "head_rotation" in channels and channels["head_rotation"].valid.any():
        if head == "none":
            channels["head_rotation"] = _unavailable("head_rotation", clip.frames, source_sha256)
        else:
            ch = channels["head_rotation"]
            valid = ch.valid & head_ok[:, None]
            channels["head_rotation"] = replace(ch, valid=valid, values=np.where(valid[..., None], ch.values, 0.),
                                                confidence=np.where(valid, 1., 0.))
        state = replace(state, channels=channels)
    report = dict(report, **info, flags=flags, fps_source=fps_source,
                  unavailable=sorted(n for n, c in state.channels.items() if not c.valid.any()))
    return state, report


__all__ = ["ADAPTER", "PSEUDO_ADAPTER", "SOURCE_ID", "LICENCE", "DEFAULT_FPS", "SOLVER_FLAGS", "Refusal", "UwbClip",
           "parse_clip_name", "parse_clip", "affine_head_rotation", "visibility_margin", "clip_landmarks", "inference_method",
           "convert_clip"]
