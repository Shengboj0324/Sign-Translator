"""Eye gaze from MediaPipe eye blendshapes and head rotation, calibrated on MPIIFaceGaze.

The monocular states of every How2Sign clip carry MediaPipe's 52 face blendshapes,
eight of which (``eyeLook{In,Out,Up,Down}{Left,Right}``) are iris displacements in
the eye, and the head rotation of the face crop's camera. They are not gaze yet:
their scale, offset and coupling to eyelid state (blink, squint, wide) are unknown.
This module maps them to a direction per eye,

    yaw_e   = a_eᵀ [1, z_e],    pitch_e = b_eᵀ [1, z_e],
    u_e     = (sin yaw cos pitch, sin pitch, cos yaw cos pitch)     (in the head),
    gaze    = normalise(R_head u_L + R_head u_R)                      (camera),

with ``z_e`` the design of the model's ``variant`` (default ``monocular``, chosen on
leave-one-participant-out error among the variants without a head term; the
cyclopean mean already pools the two eyes): ``monocular`` — the eye's seven
eye blendshapes (look in/out/up/down, blink, squint, wide); ``binocular`` — both
eyes' fourteen, since the eyes move conjugately and the fellow eye's iris is a second
noisy measurement of the same rotation; ``binocular+head`` — also the head's yaw and
pitch, absorbing any pose dependence of the blendshape regressor, clipped to the
calibration set's 1st–99th percentile range (``head_range``) so that the term never
extrapolates — How2Sign heads turn further than MPIIFaceGaze's. The camera axes x right, y up, z toward the viewer (the head-rotation frame of
the states), and ``a_e``, ``b_e`` fitted on MPIIFaceGaze by robust least squares on
the 3D unit-vector residual, which equals 2 sin(error/2) and so ranks errors by angle
(``scripts/fit_gaze_model.py``). Left and right are the signer's anatomical sides;
which MediaPipe suffix is which is settled by the data at fit time, not assumed.

**One direction for both eyes.** Vergence — the angle between the eyes' lines of
sight, 7° at the calibration screen's 0.55 m, under 2° at an interlocutor's 2 m —
moves each iris by well under a pixel at How2Sign's face resolution, so it is not
measured: a per-eye fit would only reproduce the calibration set's viewing distance.
Both eye models are fitted to the cyclopean direction (the bisector of the two eyes'
rays to the target, the direction from their midpoint to within 0.1°), and both
channels carry their mean: parallel gaze, which errs by under 1° per eye at 2 m.

Gaze is withheld (invalid, zero payload) where the face is not found or the eye is
closing (blink ≥ ``MAX_BLINK``: the iris is occluded). Confidence falls linearly
with blink to zero at that threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

FEATURES = ("eyeLookIn", "eyeLookOut", "eyeLookUp", "eyeLookDown", "eyeBlink", "eyeSquint", "eyeWide")
VARIANTS = ("monocular", "binocular", "binocular+head")
SIDES = ("Left", "Right")
MAX_BLINK = .5


def eye_features(names, scores: np.ndarray, suffix: str) -> np.ndarray:
    """(T, 7) blendshape features of one MediaPipe eye suffix ('Left' or 'Right')."""
    return np.stack([scores[..., list(names).index(f + suffix)] for f in FEATURES], axis=-1)


def head_angles(head: np.ndarray) -> np.ndarray:
    """(T, 2) yaw and pitch of the head's forward axis (third column) in the camera."""
    forward = head[..., :, 2]
    return np.stack((np.arctan2(forward[..., 0], forward[..., 2]), np.arcsin(np.clip(forward[..., 1], -1, 1))), axis=-1)


def design(names, scores: np.ndarray, head: np.ndarray, own: str, other: str, variant: str,
           head_range=None) -> np.ndarray:
    """(T, 1 + k) regressors of one eye: bias, then the variant's features."""
    parts = [np.ones(scores.shape[:-1] + (1,)), eye_features(names, scores, own)]
    if variant in ("binocular", "binocular+head"):
        parts.append(eye_features(names, scores, other))
    if variant == "binocular+head":
        angles = head_angles(head)
        if head_range is not None:
            limits = np.asarray(head_range)
            angles = np.clip(angles, limits[:, 0], limits[:, 1])
        parts.append(angles)
    if variant not in VARIANTS:
        raise ValueError(f"unknown gaze variant {variant}")
    return np.concatenate(parts, axis=-1)


def direction(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    return np.stack((np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)), axis=-1)


def angle_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(-1)))


@dataclass(frozen=True)
class GazeModel:
    """Per anatomical eye: the MediaPipe suffix it reads and its yaw/pitch coefficients."""
    suffix: dict                     # anatomical side -> MediaPipe blendshape suffix
    yaw: dict                        # side -> coefficients on design(...)
    pitch: dict
    evaluation: dict                 # leave-one-subject-out angular errors etc.
    variant: str = "monocular"
    head_range: tuple | None = None     # ((yaw min, max), (pitch min, max)) radians the head term saw

    def design(self, side: str, names, scores: np.ndarray, head: np.ndarray) -> np.ndarray:
        other = "Right" if side == "Left" else "Left"
        return design(names, scores, head, self.suffix[side], self.suffix[other], self.variant, self.head_range)

    def predict(self, names, scores: np.ndarray, head: np.ndarray) -> dict:
        """side -> (T, 3) unit gaze in the head-rotation camera frame, from (T, 52) scores and (T, 3, 3) head rotations."""
        total = 0.
        for side in SIDES:
            z = self.design(side, names, scores, head)
            total = total + np.einsum("tij,tj->ti", head, direction(z @ np.asarray(self.yaw[side]),
                                                                     z @ np.asarray(self.pitch[side])))
        cyclopean = total / np.linalg.norm(total, axis=-1, keepdims=True)
        return {side: cyclopean for side in SIDES}

    def to_json(self) -> str:
        return json.dumps(dict(schema_version=1, features=FEATURES, max_blink=MAX_BLINK, suffix=self.suffix,
                               variant=self.variant,
                               head_range=None if self.head_range is None else np.asarray(self.head_range).tolist(),
                               yaw={k: list(map(float, v)) for k, v in self.yaw.items()},
                               pitch={k: list(map(float, v)) for k, v in self.pitch.items()},
                               evaluation=self.evaluation), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "GazeModel":
        data = json.loads(text)
        if tuple(data["features"]) != FEATURES or data["schema_version"] != 1:
            raise ValueError("gaze model features differ from this code's")
        return cls(data["suffix"], {k: np.array(v) for k, v in data["yaw"].items()},
                   {k: np.array(v) for k, v in data["pitch"].items()}, data.get("evaluation", {}),
                   data.get("variant", "monocular"),
                   None if data.get("head_range") is None else tuple(map(tuple, data["head_range"])))


def gaze_channels(face, head_rotation, model: GazeModel, *, source_id: str, source_sha256: str,
                  inference_method: str) -> dict:
    """``left_gaze``/``right_gaze`` channels from a state's ``face`` and ``head_rotation`` channels.

    Valid where both are valid and the eye's blink is below ``MAX_BLINK``; confidence
    1 − blink / ``MAX_BLINK``. The frame is the head rotation's (camera axes).
    """
    import torch
    from ..pose.multichannel import MotionChannel
    from ..pose.rotations import rotation_6d_to_matrix
    frames = face.values.shape[0]
    head_ok = head_rotation.valid[:, 0]
    identity = np.array([1., 0., 0., 0., 1., 0.])
    head = rotation_6d_to_matrix(torch.from_numpy(np.where(head_ok[:, None], head_rotation.values[:, 0], identity))).numpy()
    names = tuple(face.labels)
    scores = face.values[..., 0]
    face_ok = face.valid.all(axis=1)
    predicted = model.predict(names, scores, head)
    out = {}
    for side in SIDES:
        blink = scores[:, names.index("eyeBlink" + model.suffix[side])]
        valid = face_ok & head_ok & (blink < MAX_BLINK) & np.isfinite(predicted[side]).all(axis=1)
        confidence = np.where(valid, np.clip(1 - blink / MAX_BLINK, 0., 1.), 0.)
        valid = valid & (confidence > 0)
        out[f"{side.lower()}_gaze"] = MotionChannel(
            values=np.where(valid[:, None], predicted[side], 0.)[:, None], valid=valid[:, None],
            observed=np.zeros((frames, 1), dtype=bool), confidence=np.where(valid, confidence, 0.)[:, None],
            labels=(f"{side}Eye",), coordinate_frame=head_rotation.coordinate_frame, units="unitless",
            source_id=source_id, source_sha256=source_sha256, convention="gaze-blendshape-calibrated-mpiifacegaze-v1",
            inference_method=inference_method if valid.any() else None)
    return out


def fit_eye(design: np.ndarray, head: np.ndarray, target: np.ndarray, *, ridge: float = 1e-3):
    """(yaw coefficients, pitch coefficients) minimising the robust unit-vector residual of one eye.

    ``design`` is (T, 1 + k) with the bias first (see :func:`design`).
    """
    from scipy.optimize import least_squares
    local = np.einsum("tji,tj->ti", head, target)                 # target in the head frame
    yaw0 = np.arctan2(local[:, 0], local[:, 2])
    pitch0 = np.arcsin(np.clip(local[:, 1], -1, 1))
    reg = ridge * np.eye(design.shape[1])
    start = np.concatenate([np.linalg.solve(design.T @ design + reg, design.T @ y) for y in (yaw0, pitch0)])
    k = design.shape[1]

    def residual(theta):
        u = direction(design @ theta[:k], design @ theta[k:])
        return np.concatenate(((u - local).ravel(), np.sqrt(ridge) * theta[1:]))
    result = least_squares(residual, start, loss="soft_l1", f_scale=.1)
    return result.x[:k], result.x[k:]


__all__ = ["gaze_channels", "design", "head_angles", "VARIANTS", "FEATURES", "SIDES", "MAX_BLINK", "eye_features", "direction", "angle_degrees", "GazeModel", "fit_eye"]
