"""Batched Sapiens pre- and post-processing on the inference device.

The per-frame CPU path of :mod:`sapiens_keypoints` (``preprocess``, ``decode_heatmaps``,
``flip_average``) is exact but slow at scale: on an L40S pod it spent 133 ms of CPU per
frame against 84 ms of GPU, and moved 60 MB of float32 heatmaps per frame to the host.
These functions compute the same quantities for a whole batch on the device:

* :func:`preprocess_batch` — the same bilinear affine crop (``grid_sample`` with
  ``align_corners=True`` over ``linspace`` rows and columns, zero padding) and
  normalisation, from uint8 frames;
* :func:`flip_average_batch` — the same left/right-swapped mirror average;
* :func:`decode_batch` — the same arg-max and per-axis parabola refinement, so only
  (K, 2) points and (K,) confidences leave the device.

The green-screen box stays on the CPU (connected components), computed by
:func:`green_screen_box_fast` on a 2× decimated mask (box edges within 2 px of the
full-resolution box, inside the crop's 10% margin) in worker processes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .sapiens_keypoints import FLIP_PAIRS, GREEN_MARGIN, INPUT_SIZE, MEAN, STD


def green_screen_box_fast(rgb: np.ndarray, *, margin: int = GREEN_MARGIN, step: int = 2):
    """:func:`sapiens_keypoints.green_screen_box` on every ``step``-th pixel; edges rounded outward."""
    from scipy import ndimage
    image = rgb[::step, ::step].astype(np.int16)
    background = image[..., 1] - np.maximum(image[..., 0], image[..., 2]) > margin
    labels, count = ndimage.label(~background)
    if count == 0:
        return None
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(labels == largest)
    h, w = rgb.shape[:2]
    return (float(xs.min() * step), float(ys.min() * step),
            float(min((xs.max() + 1) * step, w)), float(min((ys.max() + 1) * step, h)))


def preprocess_batch(frames: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 3) uint8 frames and (B, 4) (x, y, width, height) crops -> (B, 3, 1024, 768) normalised."""
    b, h, w, _ = frames.shape
    image = frames.permute(0, 3, 1, 2).float()
    steps = lambda n: torch.linspace(0., 1., n, device=frames.device, dtype=torch.float64)
    x, y, bw, bh = (boxes[:, i].to(torch.float64)[:, None] for i in range(4))
    rows = y + bh * steps(INPUT_SIZE[0])[None]                       # (B, 1024)
    cols = x + bw * steps(INPUT_SIZE[1])[None]                       # (B, 768)
    gy = (rows / (h - 1) * 2 - 1)[:, :, None].expand(b, *INPUT_SIZE)
    gx = (cols / (w - 1) * 2 - 1)[:, None, :].expand(b, *INPUT_SIZE)
    grid = torch.stack((gx, gy), -1).to(image.dtype)
    crop = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    mean = torch.tensor(MEAN, device=frames.device)[None, :, None, None]
    std = torch.tensor(STD, device=frames.device)[None, :, None, None]
    return (crop - mean) / std


def flip_average_batch(heatmaps: torch.Tensor, flipped: torch.Tensor, pairs=FLIP_PAIRS) -> torch.Tensor:
    """(B, K, h, w) heatmaps averaged with the mirrored prediction, flipped back with left/right swapped."""
    back = torch.flip(flipped, dims=[3])
    order = torch.arange(heatmaps.shape[1], device=heatmaps.device)
    for a, b in pairs:
        order[a], order[b] = b, a
    return (heatmaps + back[:, order]) / 2


def decode_batch(heatmaps: torch.Tensor, boxes: torch.Tensor):
    """(B, K, h, w) heatmaps and (B, 4) boxes -> (B, K, 2) image points (float64) and (B, K) peak values."""
    b, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, k, -1)
    confidence, index = flat.max(dim=2)
    py, px = index // w, index % w

    def at(yy, xx):
        return heatmaps.gather(2, yy.clamp(0, h - 1)[..., None, None].expand(b, k, 1, w)).squeeze(2) \
                       .gather(2, xx.clamp(0, w - 1)[..., None]).squeeze(2)

    def refine(left, middle, right, centre, size):
        denominator = left - 2 * middle + right
        ok = (centre > 0) & (centre < size - 1) & (denominator < 0)
        offset = torch.where(ok, (left - right) / torch.where(ok, 2 * denominator, torch.ones_like(denominator)),
                             torch.zeros_like(denominator))
        return offset.clamp(-.5, .5)

    middle = at(py, px)
    dx = refine(at(py, px - 1), middle, at(py, px + 1), px, w).double()
    dy = refine(at(py - 1, px), middle, at(py + 1, px), py, h).double()
    x, y, width, height = (boxes[:, i].double()[:, None] for i in range(4))
    u = (px.double() + dx + .5) / w * width + x - .5 * width / INPUT_SIZE[1]
    v = (py.double() + dy + .5) / h * height + y - .5 * height / INPUT_SIZE[0]
    return torch.stack((u, v), dim=-1), confidence


__all__ = ["green_screen_box_fast", "preprocess_batch", "flip_average_batch", "decode_batch"]
