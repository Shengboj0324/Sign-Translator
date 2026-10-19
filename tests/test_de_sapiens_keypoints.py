"""Sapiens helpers: chroma box, aspect crop, exact sub-pixel decoding, layout maps."""
import numpy as np

from signtranslator.data_engineering.sapiens_keypoints import (
    HAND21, INPUT_SIZE, crop_box, decode_heatmaps, flip_average, green_screen_box, hand_box, preprocess,
    to_hand21, to_pose33,
)


def test_green_screen_box_finds_the_signer_not_noise():
    image = np.zeros((720, 1280, 3), np.uint8)
    image[..., 1] = 200  # green screen
    image[100:600, 500:780] = (180, 150, 140)  # signer
    image[10:14, 10:14] = (180, 150, 140)      # small speck
    assert green_screen_box(image) == (500., 100., 780., 600.)


def test_crop_has_the_model_aspect_and_keeps_the_centre():
    x, y, w, h = crop_box((500., 100., 780., 600.))
    assert np.isclose(h / w, INPUT_SIZE[0] / INPUT_SIZE[1])
    assert np.isclose(x + w / 2, 640.) and np.isclose(y + h / 2, 350.)
    tensor = preprocess(np.zeros((720, 1280, 3), np.uint8), (x, y, w, h))
    assert tensor.shape == (1, 3, *INPUT_SIZE)


def test_decoding_recovers_a_subpixel_peak_through_the_crop():
    h, w = 256, 192
    box = (100., 50., 600., 800.)  # (x, y, width, height)
    true = np.array([[73.3, 140.8], [12.6, 30.1]])  # heatmap (x, y)
    yy, xx = np.mgrid[0:h, 0:w]
    maps = np.stack([np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.5 ** 2)) for cx, cy in true])
    points, confidence = decode_heatmaps(maps, box)
    scale = np.array([box[2] / w, box[3] / h])
    expected = (true + .5) * scale + np.array(box[:2]) - .5 * np.array([box[2] / INPUT_SIZE[1], box[3] / INPUT_SIZE[0]])
    np.testing.assert_allclose(points, expected, atol=.1 * scale.max())
    assert (confidence > .8).all()


def test_layout_maps_are_complete_and_disjoint():
    assert len(set(HAND21["Left"]) | set(HAND21["Right"])) == 42
    points = np.random.default_rng(0).normal(size=(3, 308, 3))
    pose, visible = to_pose33(points, np.ones((3, 308)), .3)
    np.testing.assert_array_equal(pose[:, 15], points[:, 62])  # subject's left wrist
    hand, ok = to_hand21(points, np.ones((3, 308)), "Right", .3)
    np.testing.assert_array_equal(hand[:, 0], points[:, 41]) and ok.all()


def test_flip_average_maps_mirrored_channels_back_exactly():
    rng = np.random.default_rng(1)
    heat = rng.random((308, 8, 6))
    mirrored = heat[:, :, ::-1].copy()          # what a perfectly equivariant model returns
    mirrored[[5, 6]] = mirrored[[6, 5]]          # with left/right channels exchanged
    mirrored[[62, 41]] = mirrored[[41, 62]]
    for a, b in zip(HAND21["Left"][1:], HAND21["Right"][1:]):
        mirrored[[a, b]] = mirrored[[b, a]]
    out = flip_average(heat, mirrored)
    for k in (5, 6, 62, 41, 0, HAND21["Left"][3]):
        np.testing.assert_allclose(out[k], heat[k])


def test_hand_box_covers_the_hand_with_a_floor_size():
    points = np.zeros((308, 2))
    points[list(HAND21["Left"])] = np.random.default_rng(2).uniform([100, 200], [130, 240], (21, 2))
    box = hand_box(points, np.ones(308), "Left", shoulder_width=400.)
    assert box[2] - box[0] >= 1.6 * 100 - 1e-9 and box[0] < 100 and box[3] > 240
    assert hand_box(points, np.zeros(308), "Left", 400.) is None
