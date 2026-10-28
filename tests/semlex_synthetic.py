"""Synthetic Sem-Lex Holistic arrays with known 3D geometry (shared by the adapter and converter tests)."""
from __future__ import annotations

import numpy as np

from signtranslator.data_engineering.semlex_pose import FaceResources

T = 8


def fake_resources(seed: int = 0) -> FaceResources:
    """A face-like canonical model: 468 points on an ellipsoid (cm), 33 basis landmarks with positive weights."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(468, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    canonical = u * [7., 9., 5.]
    weights = np.zeros(468)
    weights[rng.choice(468, 33, replace=False)] = rng.uniform(.01, .06, 33)
    return FaceResources(canonical, weights, b"", "0" * 64)


class FakeBlendshapes:
    """Deterministic stand-in for the TFLite model: scores from the mouth opening in pixels, in [0, 1]."""

    def __call__(self, pixels: np.ndarray) -> np.ndarray:
        out = np.zeros((len(pixels), 52))
        out[:, 9] = out[:, 10] = .25
        out[:, 25] = np.clip(np.ptp(pixels[:, :, 1], axis=1) / 1000., 0., 1.)
        return out


def rotation(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = np.radians(degrees)
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def body_y_up(frames: int = T, seed: int = 1) -> np.ndarray:
    """(T, 33, 3) y-up points in metres: shoulders, arms and hips (hips below the image), head above."""
    rng = np.random.default_rng(seed)
    p = rng.normal(scale=.02, size=(frames, 33, 3))
    anchors = {0: [0., .62, .08], 2: [.03, .66, .06], 5: [-.03, .66, .06], 7: [.07, .64, 0.], 8: [-.07, .64, 0.],
               11: [.19, .45, 0.], 12: [-.19, .45, 0.], 13: [.30, .22, .08], 14: [-.31, .20, .12],
               15: [.20, .30, .30], 16: [-.18, .32, .28], 23: [.11, -.10, 0.], 24: [-.11, -.10, 0.]}
    for k, v in anchors.items():
        p[:, k] = np.asarray(v) + .015 * np.arange(frames)[:, None] * rng.normal(size=3)
    return p


def hand_y_up(wrist: np.ndarray, seed: int) -> np.ndarray:
    """(T, 21, 3) y-up hand points: palm upward from the wrist, fingers in a fan, a little depth."""
    rng = np.random.default_rng(seed)
    frames = len(wrist)
    base = np.zeros((21, 3))
    for f, x in enumerate((-.03, -.015, 0., .015, .03)):
        for j in range(4):
            base[1 + 4 * f + j] = [x * (1 + .3 * j), .03 + .025 * j, .01 * j * (f - 2) / 2]
    base[1:5, 0] -= .02
    out = np.repeat(base[None], frames, 0) + rng.normal(scale=.002, size=(frames, 21, 3))
    return out + wrist[:, None]


def project(points_y_up: np.ndarray, size: tuple[int, int], *, scale_px: float, centre_px) -> np.ndarray:
    """Orthographic image of y-up metres: normalised x, y and MediaPipe-style z (image widths, away positive)."""
    width, height = size
    x = (centre_px[0] + scale_px * points_y_up[..., 0]) / width
    y = (centre_px[1] - scale_px * points_y_up[..., 1]) / height
    z = -scale_px * points_y_up[..., 2] / width
    return np.stack((x, y, z), axis=-1)


def face_points(resources: FaceResources, nose_px, size, *, frames: int = T, yaw: float = 10., pitch: float = -5.,
                scale_px: float = 6.) -> np.ndarray:
    """(T, 478, 3) normalised face mesh: the canonical model turned by yaw/pitch, orthographic, centred on the nose;
    the 10 iris points copy two eye-region vertices."""
    width, height = size
    out = np.zeros((frames, 478, 3))
    for t in range(frames):
        r = rotation((0, 1, 0), yaw + t) @ rotation((1, 0, 0), pitch)
        p = resources.canonical @ r.T                                 # y up, z toward the viewer
        x = (nose_px[t][0] + scale_px * p[:, 0]) / width
        y = (nose_px[t][1] - scale_px * p[:, 1]) / height
        z = -scale_px * p[:, 2] / width
        out[t, :468] = np.c_[x, y, z]
        out[t, 468:473] = out[t, 33]
        out[t, 473:478] = out[t, 263]
    return out


def semlex_array(size=(640, 480), *, frames: int = T, resources: FaceResources | None = None, seed: int = 1):
    """(T, 553, 3) array in Sem-Lex's layout and the y-up 3D points it was projected from."""
    resources = resources or fake_resources()
    width, height = size
    scale_px = .43 * width / .38                      # shoulder width 0.43 of the image width, as in Sem-Lex
    centre = (width / 2, .67 * height + .45 * scale_px)  # shoulders at y = 0.67
    body = body_y_up(frames, seed)
    hands = {"Left": hand_y_up(body[:, 15], seed + 1), "Right": hand_y_up(body[:, 16], seed + 2)}
    out = np.full((frames, 553, 3), np.nan)
    pose = project(body, size, scale_px=scale_px, centre_px=centre)
    out[:, 478:511] = pose
    out[:, 511:532] = project(hands["Left"], size, scale_px=scale_px, centre_px=centre)
    out[:, 532:553] = project(hands["Right"], size, scale_px=scale_px, centre_px=centre)
    nose_px = pose[:, 0, :2] * [width, height]
    out[:, :478] = face_points(resources, nose_px, size, frames=frames)
    return out, body, hands
