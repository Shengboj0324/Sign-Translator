"""Sem-Lex Holistic pose files -> ``ClipLandmarks`` -> canonical states (adapter ``semlex-holistic-pseudo-world-v1``).

The Sem-Lex Benchmark (Kezar et al. 2023; CC BY-NC-SA 4.0) publishes no video, only MediaPipe Holistic output: one
``(T, 553, 3)`` float16 array per clip, normalised image ``x`` and ``y`` plus Holistic's relative ``z``, NaN where a part
was not detected. No world landmarks, visibility, blendshapes, face matrix or frame size are stored. This module turns
one array into the :class:`~.video_landmarks.ClipLandmarks` that :func:`~.video_landmarks.canonical_from_landmarks`
solves, reproducing what our own extraction (``extract_clip``, ROI mode) computes wherever the stored data allow it,
and marking everything else invalid.

**Layout** (verified on 400 test clips, 22,649 frames; ``tests/test_de_semlex_pose.py`` pins it):

- ``0..467`` face mesh, ``468..472`` the subject's right iris (centre 468 beside the eye ring 33/133),
  ``473..477`` the left iris (centre 473 beside 362/263);
- ``478..510`` pose (BlazePose 33; pose 2, the left eye, lies at the mesh's left eye on 99.96% of frames);
- ``511..531`` Holistic's *left* hand: its wrist is nearer the pose's **left** wrist (15) on 96.7% of 4,908 frames
  (median 0.039 against 0.452 image widths), so hand A at mean x 0.60 is the subject's left hand in an unmirrored image;
- ``532..552`` the right hand (nearer the pose's right wrist on 98.9% of 5,153 frames).

Each block is entirely finite or entirely NaN in every frame of the sample (no partial blocks). A handful of frames
(9 of 22,570 face frames) hold an all-zero face; any all-zero or zero-spread block is treated as not detected.

**Clock.** Frames are assumed uniform over the clip's duration ``D`` (ms, from the metadata): ``t_k = k D / (1000 T)``
seconds, ``k = 0..T-1``. The rate ``T / D`` must lie in ``[MIN_FPS, MAX_FPS]``, otherwise the clip is refused.

**Aspect.** Normalised image coordinates divide x by the width ``W`` and y by the height ``H``; directions need
``a = W / H``, which is not stored. :func:`estimate_aspect` fits, per face frame, the least-squares affine camera
``(x, y) ~ B [M; 1]`` from MediaPipe's canonical face model ``M`` (the 33 Procrustes-basis vertices of
``geometry_pipeline_metadata_landmarks.binarypb`` inside the pinned ``face_landmarker.task``) to the stored mesh. For a
scaled-orthographic camera ``(xW, yH) = s R2 M + t`` the rows of ``B`` are ``s r1 / W`` and ``s r2 / H`` with ``|r1| =
|r2| = 1``, so ``|B_2| / |B_1| = W / H``, whatever the head's rotation. The clip's estimate is the median over face
frames; it is snapped to the nearest common frame shape (16:9, 4:3, 1:1, 3:4, 9:16) when within ``ASPECT_TOLERANCE`` in
log ratio, and the clip is refused otherwise (``aspect`` can be passed when known).

**Pseudo-world** (MediaPipe world axes: x right, y down, z away from the camera). For a stored point ``(x, y, z)``::

    P = (x,  y / a,  k z)        in image widths

``x`` is in image widths already; ``y H / W = y / a``; MediaPipe states that ``z`` uses "roughly the same scale as x"
(image widths), and ``k`` is a measured depth gain: ``POSE_Z_SCALE = 0.3`` for pose points and ``HAND_Z_SCALE = 1.5``
for hand points (the pose model's image z is several times deeper than the body; see the constants).
``canonical_from_landmarks`` then applies its y-up map ``(x, y, z) -> (x, -y, -z)``. The swing IK uses only bone
directions and frames built from difference vectors, so the unknown global scale and translation of ``P`` do not
matter; perspective does (``P`` is the image, not metric 3D, so a bone that points at the camera away from the image
centre gains a lateral component), and so does the quality of the relative depth. Both are measured by
``scripts/validate_semlex_adapter.py`` against How2Sign two-view states and our own extraction.

**Visibility.** Holistic stored none. A pose point is visible when it is finite and inside the image (``0 <= x, y <=
1``); for this tight webcam framing the hips are never visible, so the chest needs ``upright_chest`` and the stored
root ``upright_root``; elbows also pass ``elbow_in_image`` (which this rule already implies). Legs are therefore
invalid on every frame.

**Hands.** Detected where the block is finite. Both Holistic blocks are put in hand slots and assigned to a side by
the unchanged :func:`~.video_landmarks.assign_hands` (wrist proximity); the report counts how often that agrees with
Holistic's own labels.

**Face.** Our extraction runs the face landmarker on a square crop centred on the pose nose (side ``0.9`` shoulder
widths), so its face matrix is relative to that crop's camera. The adapter rebuilds the same crop from the stored pose
(at the given frame size, or at a nominal ``NOMINAL_HEIGHT``-pixel frame of the estimated aspect), maps the mesh into
it, and runs an exact port of MediaPipe's face geometry pipeline (``face_geometry/libs/geometry_pipeline.cc``: 63 deg
vertical field of view, near plane 1, two-pass scale estimate, weighted orthogonal Procrustes): ``head_rotation`` is
the polar factor of its 3x3, as in our extraction. Blendshapes come from MediaPipe's own ``face_blendshapes.tflite``
(inside the pinned task bundle), run by OpenCV's TFLite importer on the same 146-landmark subset in pixel
coordinates that MediaPipe's ``FaceBlendshapesGraph`` feeds it. On FaceLandmarker output, both reproduce MediaPipe
(face matrix within 2e-4, rotation within 0.04 deg; blendshapes within 5e-6). Head translation is unavailable (as in
our ROI extraction). Holistic's legacy face mesh is a different model version from the Tasks face landmarker that the
blendshape model was trained behind; the validation measures only the conversion, not that estimator difference.

Nothing is invented: a frame or part that the stored data cannot support is invalid in the state.

**Validation** (``scripts/validate_semlex_adapter.py``, 2026-10-28; 40 How2Sign val clips cropped to Sem-Lex framing at
4:3, 16:9 and 1:1, Sem-Lex-format arrays emulated with the Tasks models; held-out half, estimated aspect; median / p90
degrees against How2Sign two-view states on frames every method holds):

- arms 13.2 / 28.0 (our extraction on the same crop 18.7 / 38.8, on the full frame 19.6 / 28.8); after one global
  rotation per method 15.4 / 23.1 (18.4 / 33.9; full frame 12.2 / 30.8);
- wrists 18.3 / 35.8 (20.7 / 168.7: our extraction's hand frames flip on > 10% of these frames);
- fingers 21.8 / 60.7 (25.1 / 59.9); shoulder line 7.6 / 16.4 (12.3 / 18.3);
- head rotation against our extraction 0.05 / 0.08; blendshapes MAE 0.001, median r 0.9998;
- aspect: 90 of 90 crops snapped to the true shape (raw estimate within -5.3% .. +5.3%, p10..p90).
- Frames only the adapter marks valid (elbow inside the image where the pose model's visibility is < 0.5) are less
  accurate: arms 19.8 against 13.2 on common frames. This failed the declared bar (within 3 deg).

On 1,410 real Sem-Lex test clips (``scripts/check_semlex_sample.py``): 1,392 converted (18 refused: aspect estimates
1.19-1.21, between 1:1 and 4:3); forearm on >= 30% of frames in 668, a hand in 773, both in 383; recogniser v2 top-5
0.063 (95% CI 0.048-0.080) on 939 in-vocabulary clips against 0.004 blank (0.138 where arm and hand are tracked),
far below its 0.73 on ASL Citizen: the states carry sign information, but v2 does not read them well.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import struct
import zipfile

import numpy as np

from .video_landmarks import (CONVENTION, L_SHOULDER, MODEL_SHA256, R_SHOULDER, ClipLandmarks, assign_hands,
                              canonical_from_landmarks)

ADAPTER = "semlex-holistic-pseudo-world-v1"
SOURCE_ID = "semlex_holistic_pose"
POINTS = 553
FACE = slice(0, 478)
MESH_POINTS = 468
POSE = slice(478, 511)
LEFT_HAND = slice(511, 532)    # Holistic left_hand_landmarks: the pose's left wrist (subject's left)
RIGHT_HAND = slice(532, 553)
MIN_FPS, MAX_FPS = 10., 70.
# Depth gains, chosen on the calibration half of scripts/validate_semlex_adapter.py (median direction error against
# How2Sign two-view states; grids 0.1..1 and 0.5..3) and reported on the held-out half. MediaPipe's pose image z is
# about 4-5x deeper than its metric shape (z range / x range 2.6 on the upper body); the hand image z is shallower.
POSE_Z_SCALE = .3
HAND_Z_SCALE = 1.5
VISIBILITY_MARGIN = 0.         # a pose point is visible inside [m, 1 - m] in both normalised image coordinates
STANDARD_ASPECTS = {"16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1., "3:4": 3 / 4, "9:16": 9 / 16}
ASPECT_TOLERANCE = .10         # |log(estimate / standard)| <= log(1.10)
MIN_ASPECT_FRAMES = 5
NOMINAL_HEIGHT = 480           # pixel height used to rebuild the face crop when the frame size is unknown
FACE_SCALE = .9                # extract_clip's default face crop side, in shoulder widths
FOV_DEGREES, NEAR = 63., 1.    # MediaPipe face geometry environment (perspective camera)
BLENDSHAPE_NAMES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight")
# MediaPipe FaceBlendshapesGraph's landmark subset (kLandmarksSubsetIdxs), in its order.
BLENDSHAPE_SUBSET = (
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153,
    154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251,
    263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318,
    321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386,
    387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477)


class Refusal(ValueError):
    """The clip cannot be converted without inventing data; the message is the reason."""


# ----------------------------------------------------------------------------------------------------------------------
# MediaPipe face resources (from the pinned face_landmarker.task)

def _varint(buffer: bytes, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        if i >= len(buffer):
            raise ValueError("truncated protobuf varint")
        byte = buffer[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7
        if shift > 63:
            raise ValueError("protobuf varint too long")


def _protobuf_fields(buffer: bytes):
    """(field number, wire type, value) of one protobuf message; nested messages stay bytes."""
    i = 0
    while i < len(buffer):
        tag, i = _varint(buffer, i)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, i = _varint(buffer, i)
        elif wire == 1:
            value, i = buffer[i:i + 8], i + 8
        elif wire == 2:
            size, i = _varint(buffer, i)
            value, i = buffer[i:i + size], i + size
        elif wire == 5:
            value, i = buffer[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if i > len(buffer):
            raise ValueError("truncated protobuf field")
        yield number, wire, value


def parse_geometry_metadata(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Canonical face vertices (468, 3) and Procrustes weights (468,) from ``GeometryPipelineMetadata``.

    Fields (geometry_pipeline_metadata.proto): 1 ``canonical_mesh`` (Mesh3d: 1 vertex_type, 2 primitive_type,
    3 vertex_buffer floats ``x y z u v`` per vertex, 4 index_buffer), 2 ``procrustes_landmark_basis`` (1 landmark_id,
    2 weight), 3 ``input_source``."""
    floats, basis = [], []
    for number, wire, value in _protobuf_fields(payload):
        if number == 1 and wire == 2:
            for n, w, v in _protobuf_fields(value):
                if n == 1 and v != 0 or n == 2 and v != 0:
                    raise ValueError("unexpected canonical mesh vertex or primitive type")
                if n == 3:
                    floats += list(struct.unpack(f"<{len(v) // 4}f", v)) if w == 2 else [struct.unpack("<f", v)[0]]
        elif number == 2 and wire == 2:
            ref = {n: v for n, _, v in _protobuf_fields(value)}
            basis.append((int(ref.get(1, 0)), struct.unpack("<f", ref[2])[0] if 2 in ref else 0.))
    if len(floats) != MESH_POINTS * 5 or not basis:
        raise ValueError(f"face geometry metadata: {len(floats)} vertex floats, {len(basis)} basis landmarks")
    vertices = np.asarray(floats, dtype=np.float64).reshape(MESH_POINTS, 5)[:, :3]
    weights = np.zeros(MESH_POINTS)
    for landmark, weight in basis:
        if not 0 <= landmark < MESH_POINTS:
            raise ValueError("Procrustes basis landmark out of range")
        weights[landmark] = weight
    return vertices, weights


@dataclass(frozen=True)
class FaceResources:
    canonical: np.ndarray      # (468, 3) centimetres, MediaPipe's canonical face model
    weights: np.ndarray        # (468,) Procrustes weights (33 nonzero)
    blendshape_model: bytes    # face_blendshapes.tflite
    task_sha256: str

    @property
    def basis(self) -> np.ndarray:
        return np.flatnonzero(self.weights > 0)


def load_face_resources(face_task: Path) -> FaceResources:
    """The canonical face and the blendshape model from ``face_landmarker.task``, checked against its pinned hash."""
    payload = Path(face_task).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != MODEL_SHA256["face_landmarker.task"]:
        raise ValueError("face_landmarker.task: model bytes differ from the pinned hash")
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        canonical, weights = parse_geometry_metadata(bundle.read("geometry_pipeline_metadata_landmarks.binarypb"))
        blendshapes = bundle.read("face_blendshapes.tflite")
    return FaceResources(canonical, weights, blendshapes, digest)


class BlendshapeModel:
    """MediaPipe's blendshape network, run by OpenCV's TFLite importer (one network per process)."""

    def __init__(self, resources: FaceResources):
        import cv2
        self._net = cv2.dnn.readNetFromTFLite(np.frombuffer(resources.blendshape_model, np.uint8))

    def __call__(self, pixels: np.ndarray) -> np.ndarray:
        """(T, 478, 2) landmarks in pixels -> (T, 52) scores, as MediaPipe's graph computes them."""
        out = np.zeros((len(pixels), len(BLENDSHAPE_NAMES)))
        for t, frame in enumerate(pixels):
            self._net.setInput(np.ascontiguousarray(frame[list(BLENDSHAPE_SUBSET), :2], dtype=np.float32)[None])
            scores = np.asarray(self._net.forward(), dtype=np.float64).ravel()
            if scores.shape != (len(BLENDSHAPE_NAMES),):
                raise RuntimeError(f"blendshape model returned shape {scores.shape}")
            out[t] = scores
        return out


# ----------------------------------------------------------------------------------------------------------------------
# Face geometry (port of MediaPipe's geometry pipeline)

def weighted_procrustes(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """4x4 similarity ``s R x + t`` minimising ``sum_i w_i |s R a_i + t - b_i|^2`` (``source``, ``target``: (3, N));
    MediaPipe's ``ProcrustesSolver::SolveWeightedOrthogonalProblem`` with its sign handling."""
    sqrt_w = np.sqrt(weights)
    ws, wt = source * sqrt_w, target * sqrt_w
    total = float((sqrt_w * sqrt_w).sum())
    centre = (ws * sqrt_w).sum(1) / total
    centred = ws - centre[:, None] * sqrt_w[None]
    design = wt @ centred.T
    if np.linalg.norm(design) <= 1e-9:
        raise ValueError("Procrustes design matrix norm is too small")
    u, _, vt = np.linalg.svd(design)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        u[:, 2] *= -1
    rotation = u @ vt
    denominator = float((centred * ws).sum())
    scale = float(((rotation @ centred) * wt).sum()) / denominator if denominator > 1e-9 else 0.
    if scale <= 1e-9:
        raise ValueError("Procrustes scale is too small")
    rs = scale * rotation
    translation = ((wt - rs @ ws) * sqrt_w).sum(1) / total
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rs, translation
    return out


def face_transform(landmarks: np.ndarray, width: float, height: float, resources: FaceResources) -> np.ndarray:
    """MediaPipe's facial transformation matrix for one face: ``landmarks`` (>= 468, 3) normalised to an image of
    ``width`` x ``height`` pixels (only their ratio matters)."""
    h_near = 2 * NEAR * np.tan(.5 * np.radians(FOV_DEGREES))
    w_near = width * h_near / height
    left, bottom = -.5 * w_near, -.5 * h_near
    p = np.array(landmarks[:MESH_POINTS], dtype=np.float64)
    p[:, 1] = 1. - p[:, 1]                                   # origin at the top-left corner
    p[:, 0] = p[:, 0] * w_near + left
    p[:, 1] = p[:, 1] * h_near + bottom
    p[:, 2] = p[:, 2] * w_near
    depth = p[:, 2].mean()

    def move(q, scale):
        q = q.copy()
        q[:, 2] = (q[:, 2] - depth + NEAR) / scale
        return q

    def unproject(q):
        q = q.copy()
        q[:, :2] *= q[:, 2:3] / NEAR
        return q

    def handed(q):
        q = q.copy()
        q[:, 2] *= -1
        return q

    def scale_of(q):
        return float(np.linalg.norm(weighted_procrustes(resources.canonical.T, q.T, resources.weights)[:3, 0]))
    first = scale_of(handed(p))
    second = scale_of(handed(unproject(move(p, first))))
    metric = handed(unproject(move(p, first * second)))
    return weighted_procrustes(resources.canonical.T, metric.T, resources.weights)


def face_crop(nose_xy: np.ndarray, shoulders_xy: np.ndarray, size: tuple[int, int], scale: float = FACE_SCALE):
    """``extract_clip``'s face crop: ``video_landmarks.square_crop`` around the pose nose, side ``scale`` shoulder widths,
    clipped to the image. Inputs are normalised; returns (x0, y0, width, height) in pixels, or None."""
    width, height = size
    pixel = np.array([width, height], dtype=np.float64)
    shoulder = float(np.linalg.norm((shoulders_xy[0] - shoulders_xy[1]) * pixel))
    centre = nose_xy * pixel
    half = max(int(round(scale * shoulder / 2)), 8)
    x0, y0 = int(round(centre[0])) - half, int(round(centre[1])) - half
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + 2 * half, width), min(y0 + 2 * half, height)
    if x1c - x0c < 16 or y1c - y0c < 16:
        return None
    return x0c, y0c, x1c - x0c, y1c - y0c


# ----------------------------------------------------------------------------------------------------------------------
# Aspect

def aspect_per_frame(face_xy: np.ndarray, resources: FaceResources) -> np.ndarray:
    """(F,) ``|B_2| / |B_1|`` of the affine camera fitted to each frame's basis landmarks; NaN when degenerate."""
    idx = resources.basis
    design = np.c_[resources.canonical[idx], np.ones(len(idx))]
    pinv = np.linalg.pinv(design)                                      # (4, N)
    coefficients = np.einsum("kn,fnc->fkc", pinv, face_xy[:, idx, :2])  # (F, 4, 2)
    b1, b2 = coefficients[:, :3, 0], coefficients[:, :3, 1]
    n1, n2 = np.linalg.norm(b1, axis=1), np.linalg.norm(b2, axis=1)
    cosine = np.abs((b1 * b2).sum(1)) / np.maximum(n1 * n2, 1e-12)
    ok = (n1 > 1e-9) & (n2 > 1e-9) & (cosine < .3)                     # a scaled-orthographic camera has cosine 0
    return np.where(ok, n2 / np.where(ok, n1, 1.), np.nan)


def estimate_aspect(face_xy: np.ndarray, found: np.ndarray, resources: FaceResources) -> dict:
    """The clip's W / H from its face mesh: median of the per-frame affine estimates, snapped to a common frame shape.
    ``aspect`` is None (with ``reason``) when it cannot be identified."""
    frames = face_xy[found]
    per_frame = aspect_per_frame(frames, resources) if len(frames) else np.zeros(0)
    per_frame = per_frame[np.isfinite(per_frame)]
    out = dict(method="face-affine-median-snapped", frames=int(len(per_frame)), raw=None, aspect=None, snapped_to=None)
    if len(per_frame) < MIN_ASPECT_FRAMES:
        return dict(out, reason=f"aspect needs >= {MIN_ASPECT_FRAMES} face frames, has {len(per_frame)}")
    raw = float(np.median(per_frame))
    out.update(raw=round(raw, 4), spread_p10_p90=[round(float(v), 4) for v in np.percentile(per_frame, [10, 90])])
    name, value = min(STANDARD_ASPECTS.items(), key=lambda item: abs(np.log(raw / item[1])))
    if abs(np.log(raw / value)) > np.log(1 + ASPECT_TOLERANCE):
        return dict(out, reason=f"aspect estimate {raw:.3f} is not within {ASPECT_TOLERANCE:.0%} of a common frame shape")
    return dict(out, aspect=value, snapped_to=name)


# ----------------------------------------------------------------------------------------------------------------------
# Parsing and conversion

def pseudo_world(xyz: np.ndarray, aspect: float, z_scale: float) -> np.ndarray:
    """``(x, y / a, k z)``: normalised image points in MediaPipe world axes, in image widths."""
    out = np.array(xyz, dtype=np.float64)
    out[..., 1] = out[..., 1] / aspect
    out[..., 2] = out[..., 2] * z_scale
    return out


def _block_found(block: np.ndarray) -> np.ndarray:
    """(T,) a block is detected when all its values are finite and its points are not all equal (all-zero rows)."""
    finite = np.isfinite(block).all(axis=(1, 2))
    spread = np.zeros(len(block))
    if finite.any():
        spread[finite] = np.ptp(block[finite][..., :2], axis=1).max(axis=1)
    return finite & (spread > 0)


def check_array(array: np.ndarray) -> np.ndarray:
    """The array as float64 after shape and value checks (``Refusal`` on failure)."""
    if not isinstance(array, np.ndarray) or array.dtype.kind != "f":
        raise Refusal("pose file is not a floating-point array")
    if array.ndim != 3 or array.shape[1:] != (POINTS, 3):
        raise Refusal(f"pose array shape {array.shape} is not (T, {POINTS}, 3)")
    if len(array) < 2:
        raise Refusal("pose array has fewer than 2 frames")
    values = array.astype(np.float64)
    if np.isinf(values).any():
        raise Refusal("pose array holds infinite values")
    return values


def holistic_clip(array: np.ndarray, duration_ms: float, resources: FaceResources, *,
                  aspect: float | None = None, image_size: tuple[int, int] | None = None,
                  blendshape_model: BlendshapeModel | None = None, pose_z_scale: float = POSE_Z_SCALE,
                  hand_z_scale: float = HAND_Z_SCALE,
                  visibility_margin: float = VISIBILITY_MARGIN) -> tuple[ClipLandmarks, dict]:
    """``ClipLandmarks`` (and a diagnostic dict) from one Holistic array of a clip lasting ``duration_ms``.

    ``image_size`` (pixels) fixes the aspect and the face crop exactly; otherwise ``aspect`` (W / H) or the face-mesh
    estimate sets the aspect and the crop is rebuilt at a nominal height of ``NOMINAL_HEIGHT`` pixels."""
    values = check_array(array)
    frames = len(values)
    if not np.isfinite(duration_ms) or duration_ms <= 0:
        raise Refusal("clip duration is not positive")
    fps = frames / (duration_ms / 1000.)
    if not MIN_FPS <= fps <= MAX_FPS:
        raise Refusal(f"{frames} frames over {duration_ms:.0f} ms is {fps:.1f} fps, outside [{MIN_FPS}, {MAX_FPS}]")
    times = np.arange(frames, dtype=np.float64) * (duration_ms / 1000.) / frames
    face, pose = values[:, FACE], values[:, POSE]
    hands = np.stack((values[:, LEFT_HAND], values[:, RIGHT_HAND]), axis=1)          # (T, 2, 21, 3)
    pose_found = _block_found(pose)
    hands_found = np.stack([_block_found(hands[:, k]) for k in range(2)], axis=1) & pose_found[:, None]
    face_detected = _block_found(face)
    info = dict(adapter=ADAPTER, frames=frames, fps=round(fps, 3), pose_frames=int(pose_found.sum()),
                face_detected_frames=int(face_detected.sum()),
                holistic_hand_frames=dict(Left=int(hands_found[:, 0].sum()), Right=int(hands_found[:, 1].sum())),
                partial_blocks=int(sum(int((np.isfinite(b).any(axis=(1, 2)) & ~np.isfinite(b).all(axis=(1, 2))).sum())
                                       for b in (face, pose, hands[:, 0], hands[:, 1]))),
                pose_z_scale=pose_z_scale, hand_z_scale=hand_z_scale, visibility_margin=visibility_margin)
    if image_size is not None:
        width, height = (int(v) for v in image_size)
        if width <= 0 or height <= 0:
            raise Refusal("image size must be positive")
        info["aspect"] = dict(method="given-image-size", aspect=width / height, image_size=[width, height])
    elif aspect is not None:
        if not np.isfinite(aspect) or aspect <= 0:
            raise Refusal("aspect must be positive")
        info["aspect"] = dict(method="given", aspect=float(aspect))
    else:
        info["aspect"] = estimate_aspect(np.nan_to_num(face[..., :2]), face_detected, resources)
        if info["aspect"]["aspect"] is None:
            raise Refusal(info["aspect"]["reason"])
    a = float(info["aspect"]["aspect"])
    if image_size is None:
        width, height = int(round(a * NOMINAL_HEIGHT)), NOMINAL_HEIGHT

    pose = np.where(pose_found[:, None, None], pose, 0.)
    hands = np.where(hands_found[:, :, None, None], hands, 0.)
    if not 0. <= visibility_margin < .5:
        raise ValueError("visibility margin must lie in [0, 0.5)")
    inside = ((pose[..., :2] >= visibility_margin) & (pose[..., :2] <= 1. - visibility_margin)).all(-1)
    visibility = (inside & pose_found[:, None]).astype(np.float64)

    # Face: rebuilt ROI crop, face matrix and blendshapes on frames with a face, a pose and a usable crop.
    face_found = face_detected & pose_found
    matrices = np.tile(np.eye(4), (frames, 1, 1))
    scores = np.zeros((frames, len(BLENDSHAPE_NAMES)))
    no_crop = 0
    for t in np.flatnonzero(face_found):
        crop = face_crop(pose[t, 0, :2], pose[t, [L_SHOULDER, R_SHOULDER], :2], (width, height))
        if crop is None:
            face_found[t] = False
            no_crop += 1
            continue
        x0, y0, cw, ch = crop
        local = np.empty((MESH_POINTS, 3))
        local[:, 0] = (face[t, :MESH_POINTS, 0] * width - x0) / cw
        local[:, 1] = (face[t, :MESH_POINTS, 1] * height - y0) / ch
        local[:, 2] = face[t, :MESH_POINTS, 2] * width / cw
        try:
            matrices[t] = face_transform(local, cw, ch, resources)
        except ValueError:
            face_found[t] = False
            no_crop += 1
    if face_found.any():
        model = blendshape_model or BlendshapeModel(resources)
        pixels = face[face_found][..., :2] * np.array([width, height], dtype=np.float64)
        scores[face_found] = model(pixels)
    info.update(face_frames=int(face_found.sum()), face_without_crop=no_crop)

    clip = ClipLandmarks(
        times=times, pose_world=pseudo_world(pose, a, pose_z_scale), pose_visibility=visibility,
        pose_image=pose[..., :2].copy(), pose_found=pose_found,
        hands_world=pseudo_world(hands, a, hand_z_scale), hands_image=hands[..., :2].copy(), hands_found=hands_found,
        blendshapes=scores, blendshape_names=BLENDSHAPE_NAMES if face_found.any() else (), face_matrix=matrices,
        face_found=face_found, roi=True, other_person=None, image_size=None)
    assigned = assign_hands(clip)          # canonical_from_landmarks repeats this exact call; here only to report it
    agreement = {}
    for k, side in enumerate(("Left", "Right")):
        held = hands_found[:, k]
        other = "Right" if side == "Left" else "Left"

        def took(s):
            return held & assigned[s][1] & (assigned[s][0] == clip.hands_world[:, k]).all((1, 2))
        same, swapped = took(side), took(other)
        agreement[side] = dict(holistic=int(held.sum()), assigned_same_side=int(same.sum()),
                               assigned_other_side=int(swapped.sum()), dropped=int((held & ~same & ~swapped).sum()))
    info["hand_assignment"] = agreement
    return clip, info


def inference_method(info: dict) -> str:
    aspect = info["aspect"]
    return (f"Sem-Lex Benchmark MediaPipe Holistic landmarks (publisher's pose files; Holistic version not stated); "
            f"{ADAPTER}: pseudo-3D (x, y/a, k z) with a={aspect['aspect']:.4f} ({aspect['method']}"
            + (f", snapped to {aspect['snapped_to']} from {aspect['raw']}" if aspect.get("snapped_to") else "")
            + f"), pose k={info['pose_z_scale']}, hand k={info['hand_z_scale']}; visibility = inside the image"
            + f" by a margin of {info['visibility_margin']}; "
            "face matrix by a port of MediaPipe's face geometry pipeline on the rebuilt ROI crop; blendshapes by "
            f"face_blendshapes.tflite of face_landmarker.task={MODEL_SHA256['face_landmarker.task'][:16]}; {CONVENTION}; "
            "chest up axis from the camera vertical (upright prior); pelvis the camera-upright identity on frames "
            "without hips (upright root); arms only on frames whose elbow lies inside the image; "
            "values are fitted, not observed")


def convert_array(array: np.ndarray, duration_ms: float, resources: FaceResources, *, video_id: str, source_sha256: str,
                  aspect: float | None = None, image_size: tuple[int, int] | None = None,
                  blendshape_model: BlendshapeModel | None = None, pose_z_scale: float = POSE_Z_SCALE,
                  hand_z_scale: float = HAND_Z_SCALE, visibility_margin: float = VISIBILITY_MARGIN):
    """Canonical state and report for one Sem-Lex clip (``Refusal`` when it cannot be converted)."""
    clip, info = holistic_clip(array, duration_ms, resources, aspect=aspect, image_size=image_size,
                               blendshape_model=blendshape_model, pose_z_scale=pose_z_scale, hand_z_scale=hand_z_scale,
                               visibility_margin=visibility_margin)
    if not clip.pose_found.any():
        raise Refusal("no frame has a pose")
    state, report = canonical_from_landmarks(
        clip, sample_id=video_id, source_id=SOURCE_ID, source_sha256=source_sha256,
        clock_id=f"{SOURCE_ID}:{video_id}:uniform-over-duration", inference_method=inference_method(info),
        upright_chest=True, elbow_in_image=True, upright_root=True)
    return state, dict(report, **info)


def load_pose_file(payload: bytes) -> np.ndarray:
    """A Sem-Lex ``.npy`` member's array (no pickle)."""
    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except ValueError as error:
        raise Refusal(f"unreadable .npy: {error}") from error
    if not isinstance(array, np.ndarray):
        raise Refusal("the .npy holds no array")
    return array


__all__ = ["ADAPTER", "SOURCE_ID", "FACE", "POSE", "LEFT_HAND", "RIGHT_HAND", "POINTS", "BLENDSHAPE_NAMES",
           "BLENDSHAPE_SUBSET", "STANDARD_ASPECTS", "Refusal", "FaceResources", "BlendshapeModel",
           "load_face_resources", "parse_geometry_metadata", "weighted_procrustes", "face_transform", "face_crop",
           "aspect_per_frame", "estimate_aspect", "pseudo_world", "check_array", "holistic_clip", "convert_array",
           "inference_method", "load_pose_file"]
