"""Keypoint preprocessing and augmentation for 3D sign-language motion.

All operations act on a pose tensor with the joint layout ``(..., C, T, V)``:
the last three axes are channels (x, y, z), frames, and joints. A leading batch
axis is optional, so both ``(C, T, V)`` and ``(N, C, T, V)`` are accepted.

Design principle: every transform here has a *checkable mathematical property*
(translation/scale invariance, isometry, involution) that the test-suite
asserts, rather than trusting the implementation by inspection.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _check_layout(pose: torch.Tensor) -> None:
    if pose.dim() < 3:
        raise ValueError("pose must have at least 3 dims (C, T, V)")


def root_center(pose: torch.Tensor, root_index: int = 1) -> torch.Tensor:
    """Subtract the root joint's position from every joint, per frame.

    Removes absolute position, giving **translation invariance**: for any
    constant offset ``d``, ``root_center(pose + d) == root_center(pose)``.
    """
    _check_layout(pose)
    root = pose[..., root_index:root_index + 1]  # (..., C, T, 1)
    return pose - root


def scale_normalize(pose: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Divide by the per-sample RMS joint displacement.

    Gives **scale invariance** for already root-centred input: for any positive
    scalar ``s``, ``scale_normalize(s * pose) == scale_normalize(pose)``. The
    scale is computed over the (C, T, V) axes so it is a single positive scalar
    per sample.
    """
    _check_layout(pose)
    dims = (-3, -2, -1)
    scale = pose.pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(eps)
    return pose / scale


def temporal_resample(pose: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Linearly resample the temporal axis to ``target_frames``.

    Resampling to the current length is the identity (up to float error).
    """
    _check_layout(pose)
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    squeeze = pose.dim() == 3
    x = pose.unsqueeze(0) if squeeze else pose
    n, c, t, v = x.shape
    if t == target_frames:
        out = x
    else:
        x = x.permute(0, 1, 3, 2).reshape(n, c * v, t)          # (N, C*V, T)
        x = F.interpolate(x, size=target_frames, mode="linear", align_corners=True)
        out = x.reshape(n, c, v, target_frames).permute(0, 1, 3, 2).contiguous()
    return out.squeeze(0) if squeeze else out


def rotate_y(pose: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate the skeleton about the vertical (y) axis by ``angle`` radians.

    A rotation is an **isometry**: pairwise joint distances (hence bone lengths)
    are preserved exactly. Requires 3 spatial channels ordered (x, y, z).
    """
    _check_layout(pose)
    if pose.shape[-3] != 3:
        raise ValueError("rotate_y requires 3 channels ordered (x, y, z)")
    cos, sin = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    x = pose[..., 0, :, :]
    y = pose[..., 1, :, :]
    z = pose[..., 2, :, :]
    x_new = cos * x + sin * z
    z_new = -sin * x + cos * z
    return torch.stack([x_new, y, z_new], dim=-3)


def mirror(pose: torch.Tensor, left_right_swap: Sequence[tuple] = ()) -> torch.Tensor:
    """Left-right mirror: negate the x channel and swap paired joints.

    Mirroring is an **involution**: applying it twice with the same swap map
    returns the original. ``left_right_swap`` lists ``(left_idx, right_idx)``
    joint pairs to exchange.
    """
    _check_layout(pose)
    out = pose.clone()
    out[..., 0, :, :] = -out[..., 0, :, :]  # flip x
    for a, b in left_right_swap:
        tmp = out[..., :, :, a].clone()
        out[..., :, :, a] = out[..., :, :, b]
        out[..., :, :, b] = tmp
    return out


def add_jitter(pose: torch.Tensor, sigma: float = 0.01,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """Add zero-mean Gaussian noise (shape preserved, expected value unchanged)."""
    _check_layout(pose)
    noise = torch.randn(pose.shape, generator=generator, dtype=pose.dtype,
                        device=pose.device)
    return pose + sigma * noise


class PoseNormalizer:
    """Compose root-centering and scale-normalization (the standard front-end)."""

    def __init__(self, root_index: int = 1, do_scale: bool = True) -> None:
        self.root_index = root_index
        self.do_scale = do_scale

    def __call__(self, pose: torch.Tensor) -> torch.Tensor:
        pose = root_center(pose, self.root_index)
        if self.do_scale:
            pose = scale_normalize(pose)
        return pose


class RandomAugment:
    """Stochastic training-time augmentation (rotation + mirror + jitter)."""

    def __init__(self, max_rot: float = 0.3, mirror_prob: float = 0.5,
                 jitter_sigma: float = 0.01,
                 left_right_swap: Sequence[tuple] = (), seed: int | None = None) -> None:
        self.max_rot = max_rot
        self.mirror_prob = mirror_prob
        self.jitter_sigma = jitter_sigma
        self.left_right_swap = tuple(left_right_swap)
        self.gen = None
        if seed is not None:
            self.gen = torch.Generator().manual_seed(seed)

    def _rand(self) -> float:
        if self.gen is not None:
            return float(torch.rand(1, generator=self.gen))
        return float(torch.rand(1))

    def __call__(self, pose: torch.Tensor) -> torch.Tensor:
        angle = (self._rand() * 2 - 1) * self.max_rot
        if pose.shape[-3] == 3:
            pose = rotate_y(pose, angle)
        if self._rand() < self.mirror_prob:
            pose = mirror(pose, self.left_right_swap)
        if self.jitter_sigma > 0:
            pose = add_jitter(pose, self.jitter_sigma, self.gen)
        return pose
