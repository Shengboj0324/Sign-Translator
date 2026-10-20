"""Batched device-side Sapiens processing equals the per-frame CPU reference."""
import numpy as np
import torch

from signtranslator.data_engineering.sapiens_batched import (
    decode_batch, flip_average_batch, green_screen_box_fast, preprocess_batch,
)
from signtranslator.data_engineering.sapiens_keypoints import (
    crop_box, decode_heatmaps, flip_average, green_screen_box, preprocess,
)


def green_frame(seed=0, h=180, w=320):
    rng = np.random.default_rng(seed)
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[...] = (30, 200, 40)
    y0, x0 = rng.integers(10, 60), rng.integers(40, 120)
    rgb[y0:y0 + 90, x0:x0 + 110] = rng.integers(0, 255, size=(90, 110, 3), dtype=np.uint8) // 2 + 60
    return rgb


def test_preprocess_batch_equals_the_per_frame_crop():
    frames = [green_frame(s) for s in range(3)]
    boxes = [crop_box(green_screen_box(f)) for f in frames]
    boxes[2] = (-30., -20., 400., 533.3)                         # partly outside: zero padding
    ref = torch.cat([preprocess(f, b) for f, b in zip(frames, boxes)])
    out = preprocess_batch(torch.from_numpy(np.stack(frames)), torch.tensor(boxes, dtype=torch.float64))
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=0)


def test_flip_average_batch_equals_reference():
    rng = np.random.default_rng(1)
    a, b = rng.random((2, 308, 8, 6)).astype(np.float32), rng.random((2, 308, 8, 6)).astype(np.float32)
    out = flip_average_batch(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    for i in range(2):
        np.testing.assert_allclose(out[i], flip_average(a[i], b[i]), atol=1e-7)


def test_decode_batch_equals_reference_including_edges():
    rng = np.random.default_rng(2)
    heat = rng.random((3, 308, 64, 48)).astype(np.float32) * .1
    for i in range(3):                                         # smooth peaks, some on the border
        for k in range(308):
            cy, cx = rng.uniform(-1, 64), rng.uniform(-1, 48)
            yy, xx = np.mgrid[:64, :48]
            heat[i, k] += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4)
    boxes = [(10., 20., 300., 400.), (-5., 0., 640., 853.3), (100., 50., 150., 200.)]
    pts, conf = decode_batch(torch.from_numpy(heat), torch.tensor(boxes, dtype=torch.float64))
    for i in range(3):
        rp, rc = decode_heatmaps(heat[i], boxes[i])
        np.testing.assert_allclose(pts[i].numpy(), rp, atol=1e-6)
        np.testing.assert_allclose(conf[i].numpy(), rc, atol=1e-7)


def test_fast_green_screen_box_is_within_two_pixels():
    for s in range(5):
        f = green_frame(s)
        assert np.abs(np.array(green_screen_box_fast(f)) - np.array(green_screen_box(f))).max() <= 2
    assert green_screen_box_fast(np.zeros((10, 10, 3), np.uint8) + np.array([0, 255, 0], np.uint8)) is None
