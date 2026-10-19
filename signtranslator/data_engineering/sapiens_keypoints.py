"""Sapiens (Goliath 308) 2D keypoints for both How2Sign views: crop, decode, map.

Sapiens (Meta, CC BY-NC 4.0) is a top-down keypoint model with 1024×768 input and
heatmap output; in a 2026 study of estimators for sign language translation it was
the most occlusion-robust and near the top on BLEU. This module holds the parts that
do not need the model: the person box, the crop, heatmap decoding and the index maps.

**Person box.** How2Sign was recorded in front of a green screen, so the signer is the
largest connected non-green region: a pixel is background when ``G - max(R, B)`` exceeds
a declared margin. No detector is needed and none can mislabel a second person.

**Decoding.** Each heatmap's integer argmax is refined by an independent parabola fit
along each axis through the argmax and its two neighbours, ``δ = (l − r) / (2(l − 2c + r))``,
exact for a separable quadratic peak and bounded to ±0.5 px. Heatmap coordinates map
back to the image through the crop box.

**Layout.** Goliath indices are anatomical (left = the subject's left). The maps below
place them in the MediaPipe pose-33 body layout and in 21-point hands ordered wrist,
then each finger base to tip, so :func:`video_landmarks.solve_rotations` applies.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

INPUT_SIZE = (1024, 768)          # (height, width)
MEAN = (123.5, 116.5, 103.5)
STD = (58.5, 57.0, 57.5)
GREEN_MARGIN = 40

BODY_TO_POSE33 = {0: 0, 5: 11, 6: 12, 7: 13, 8: 14, 62: 15, 41: 16, 9: 23, 10: 24, 11: 25, 12: 26, 13: 27, 14: 28}
HAND21 = {
    "Left": (62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58),
    "Right": (41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37),
}


def green_screen_box(rgb: np.ndarray, *, margin: int = GREEN_MARGIN) -> tuple[float, float, float, float] | None:
    """(x0, y0, x1, y1) of the largest non-green connected region, or None."""
    from scipy import ndimage
    image = rgb.astype(np.int16)
    background = image[..., 1] - np.maximum(image[..., 0], image[..., 2]) > margin
    labels, count = ndimage.label(~background)
    if count == 0:
        return None
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
    largest = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(labels == largest)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def crop_box(box, *, margin: float = .1) -> tuple[float, float, float, float]:
    """Expand by ``margin`` and to the model's 4:3 (height:width) aspect, keeping the centre."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    width, height = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    target = INPUT_SIZE[0] / INPUT_SIZE[1]
    if height / width > target:
        width = height / target
    else:
        height = width * target
    return cx - width / 2, cy - height / 2, width, height


def preprocess(rgb: np.ndarray, box) -> torch.Tensor:
    """(1, 3, 1024, 768) normalised crop; regions outside the image are zero-padded."""
    x, y, width, height = box
    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()[None]
    # Affine sampling grid in normalised image coordinates, so out-of-image is padded.
    rows = torch.linspace(y, y + height, INPUT_SIZE[0])
    cols = torch.linspace(x, x + width, INPUT_SIZE[1])
    gy = rows[:, None].expand(INPUT_SIZE) / (rgb.shape[0] - 1) * 2 - 1
    gx = cols[None, :].expand(INPUT_SIZE) / (rgb.shape[1] - 1) * 2 - 1
    crop = F.grid_sample(image, torch.stack((gx, gy), -1)[None], mode="bilinear", padding_mode="zeros",
                         align_corners=True)
    mean = torch.tensor(MEAN)[None, :, None, None]
    std = torch.tensor(STD)[None, :, None, None]
    return (crop - mean) / std


def decode_heatmaps(heatmaps: np.ndarray, box) -> tuple[np.ndarray, np.ndarray]:
    """(K, H, W) heatmaps -> (K, 2) image coordinates and (K,) peak confidence."""
    k, h, w = heatmaps.shape
    flat = heatmaps.reshape(k, -1)
    index = flat.argmax(axis=1)
    py, px = np.divmod(index, w)
    confidence = flat[np.arange(k), index]
    def refine(values, centre, size):
        offset = np.zeros(len(values[0]))
        inside = (centre > 0) & (centre < size - 1)
        left, middle, right = values
        denominator = left - 2 * middle + right
        ok = inside & (denominator < 0)
        offset[ok] = (left[ok] - right[ok]) / (2 * denominator[ok])
        return np.clip(offset, -.5, .5)
    ks = np.arange(k)
    dx = refine((heatmaps[ks, py, np.clip(px - 1, 0, w - 1)], heatmaps[ks, py, px],
                 heatmaps[ks, py, np.clip(px + 1, 0, w - 1)]), px, w)
    dy = refine((heatmaps[ks, np.clip(py - 1, 0, h - 1), px], heatmaps[ks, py, px],
                 heatmaps[ks, np.clip(py + 1, 0, h - 1), px]), py, h)
    x, y, width, height = box
    # Heatmap cell centres span the crop; input pixel = heatmap * (input / heatmap) size.
    u = (px + dx + .5) / w * width + x - .5 * width / INPUT_SIZE[1]
    v = (py + dy + .5) / h * height + y - .5 * height / INPUT_SIZE[0]
    return np.stack((u, v), axis=1), confidence


# Left/right channel pairs for the keypoints this project uses (body and both hands).
FLIP_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (62, 41), (9, 10), (11, 12), (13, 14),
              *zip(HAND21["Left"][1:], HAND21["Right"][1:]))


def flip_average(heatmaps: np.ndarray, flipped: np.ndarray, pairs=FLIP_PAIRS) -> np.ndarray:
    """Average heatmaps with those of the horizontally mirrored crop.

    The mirrored prediction is flipped back along width and its left/right channels
    swapped, so channel k of both terms describes the same anatomical keypoint.
    Channels outside ``pairs`` are self-symmetric (e.g. the nose) and simply averaged.
    """
    back = flipped[:, :, ::-1].copy()
    swapped = back.copy()
    for a, b in pairs:
        swapped[a], swapped[b] = back[b], back[a]
    return (heatmaps + swapped) / 2


def hand_box(points: np.ndarray, confidence: np.ndarray, side: str, shoulder_width: float,
             *, scale: float = 1.6, threshold: float = .3):
    """Square-ish box around one hand's confident keypoints, at least a fraction of the shoulders."""
    index = list(HAND21[side])
    ok = confidence[index] >= threshold
    if ok.sum() < 3:
        return None
    xy = points[index][ok]
    centre = (xy.min(0) + xy.max(0)) / 2
    extent = max(float(np.ptp(xy, axis=0).max()), .25 * shoulder_width) * scale
    return (centre[0] - extent / 2, centre[1] - extent / 2, centre[0] + extent / 2, centre[1] + extent / 2)


def to_pose33(points: np.ndarray, confidence: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Goliath (T, 308, D) -> MediaPipe pose-33 layout (T, 33, D) and visibility (T, 33)."""
    frames, _, dims = points.shape
    out = np.zeros((frames, 33, dims))
    visible = np.zeros((frames, 33), dtype=bool)
    for goliath, pose in BODY_TO_POSE33.items():
        out[:, pose] = points[:, goliath]
        visible[:, pose] = confidence[:, goliath] >= threshold
    return out, visible


def to_hand21(points: np.ndarray, confidence: np.ndarray, side: str, threshold: float):
    """Goliath (T, 308, D) -> (T, 21, D) hand points and per-frame validity (all 21 confident)."""
    index = list(HAND21[side])
    return points[:, index], (confidence[:, index] >= threshold).all(axis=1)


__all__ = ["INPUT_SIZE", "BODY_TO_POSE33", "HAND21", "FLIP_PAIRS", "green_screen_box", "crop_box", "preprocess",
           "decode_heatmaps", "flip_average", "hand_box", "to_pose33", "to_hand21"]
