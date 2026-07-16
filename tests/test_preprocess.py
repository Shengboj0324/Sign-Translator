"""Property-based tests for keypoint preprocessing and augmentation.

Each transform is asserted against its defining mathematical property, not a
golden output, so an incorrect implementation cannot silently pass.
"""

import torch
import pytest

from signtranslator.data.preprocess import (
    root_center, scale_normalize, temporal_resample, rotate_y, mirror,
    add_jitter, PoseNormalizer, RandomAugment,
)


def _pose(n=None, c=3, t=16, v=27, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (c, t, v) if n is None else (n, c, t, v)
    return torch.randn(shape, generator=g)


def test_root_center_zeros_root_and_is_translation_invariant():
    pose = _pose()
    centred = root_center(pose, root_index=1)
    assert torch.allclose(centred[:, :, 1], torch.zeros(3, 16), atol=1e-6)
    offset = torch.tensor([2.0, -3.0, 0.5]).view(3, 1, 1)
    assert torch.allclose(root_center(pose + offset, 1), centred, atol=1e-5)


def test_scale_normalize_is_scale_invariant():
    pose = root_center(_pose(), 1)
    for s in (0.5, 2.0, 10.0):
        assert torch.allclose(scale_normalize(s * pose), scale_normalize(pose), atol=1e-5)


def test_scale_normalize_unit_rms():
    pose = _pose()
    out = scale_normalize(pose)
    rms = out.pow(2).mean().sqrt()
    assert torch.allclose(rms, torch.tensor(1.0), atol=1e-5)


def test_temporal_resample_identity_and_shape():
    pose = _pose()
    assert torch.allclose(temporal_resample(pose, 16), pose, atol=1e-5)
    up = temporal_resample(pose, 32)
    assert up.shape == (3, 32, 27)
    batched = temporal_resample(_pose(n=4), 8)
    assert batched.shape == (4, 3, 8, 27)


def test_temporal_resample_linear_midpoint():
    # A single joint moving linearly in time; upsampling must stay on the line.
    t = torch.linspace(0, 1, 5).view(1, 5, 1)
    pose = torch.cat([t, 2 * t, -t], dim=0)  # (C=3, T=5, V=1)
    up = temporal_resample(pose, 9)
    expected = torch.linspace(0, 1, 9)
    assert torch.allclose(up[0, :, 0], expected, atol=1e-5)
    assert torch.allclose(up[1, :, 0], 2 * expected, atol=1e-5)


def test_rotation_is_isometry():
    pose = _pose()
    rot = rotate_y(pose, 0.7)

    def pairwise(p):  # distances between joints at frame 0 (double precision)
        pts = p[:, 0, :].t().double()  # (V, C)
        diff = pts.unsqueeze(0) - pts.unsqueeze(1)
        return diff.pow(2).sum(-1).sqrt()

    assert torch.allclose(pairwise(pose), pairwise(rot), atol=1e-5)


def test_rotation_zero_angle_identity():
    pose = _pose()
    assert torch.allclose(rotate_y(pose, 0.0), pose, atol=1e-6)


def test_rotation_requires_three_channels():
    with pytest.raises(ValueError):
        rotate_y(_pose(c=2), 0.3)


def test_mirror_is_involution():
    pose = _pose()
    swap = [(3, 6), (4, 7), (5, 8)]  # right/left arm pairs
    twice = mirror(mirror(pose, swap), swap)
    assert torch.allclose(twice, pose, atol=1e-6)


def test_mirror_flips_x_sign_without_swap():
    pose = _pose()
    m = mirror(pose)
    assert torch.allclose(m[0], -pose[0], atol=1e-6)  # x negated
    assert torch.allclose(m[1], pose[1], atol=1e-6)   # y unchanged


def test_jitter_preserves_shape_and_is_zero_mean():
    pose = torch.zeros(3, 64, 27)
    g = torch.Generator().manual_seed(0)
    out = add_jitter(pose, sigma=0.1, generator=g)
    assert out.shape == pose.shape
    assert abs(float(out.mean())) < 0.02  # ~zero mean over many samples


def test_normalizer_combines_translation_and_scale_invariance():
    pose = _pose()
    norm = PoseNormalizer(root_index=1)
    offset = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
    a = norm(pose)
    b = norm(3.0 * pose + offset)
    # scale + translation should be removed => identical up to the (fixed) root.
    assert torch.allclose(a, b, atol=1e-4)


def test_random_augment_preserves_shape_and_is_deterministic_with_seed():
    pose = _pose()
    aug1 = RandomAugment(left_right_swap=[(3, 6)], seed=42)(pose)
    aug2 = RandomAugment(left_right_swap=[(3, 6)], seed=42)(pose)
    assert aug1.shape == pose.shape
    assert torch.allclose(aug1, aug2, atol=1e-6)  # reproducible
