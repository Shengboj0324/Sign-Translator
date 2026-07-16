"""Synthetic paired (motion, gloss) dataset.

This is a *deterministic, structured* toy dataset used to (a) let the whole
system build and train without any external data, and (b) provide a
non-trivial signal so tests can assert the model actually *learns* (loss
decreases / overfits a batch). It is NOT a substitute for a real sign-language
corpus (e.g. How2Sign, PHOENIX-2014T); the ``README`` documents that clearly.

Each "class" ``k`` has:
    * a distinct token id sequence (its gloss), and
    * a distinct smooth 3D joint trajectory (a class-specific mixture of sines),
so that motion and language are genuinely correlated and contrastive alignment
is meaningful.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticSignDataset(Dataset):
    def __init__(self, num_classes: int = 8, samples_per_class: int = 32,
                 num_joints: int = 27, in_channels: int = 3, num_frames: int = 64,
                 seq_len: int = 6, vocab_size: int = 4096, noise: float = 0.02,
                 seed: int = 0) -> None:
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.noise = noise
        rng = np.random.default_rng(seed)

        # Per-class latent trajectory parameters.
        self.freqs = rng.uniform(0.5, 3.0, size=(num_classes, num_joints, in_channels))
        self.phases = rng.uniform(0, 2 * np.pi, size=(num_classes, num_joints, in_channels))
        self.amps = rng.uniform(0.5, 1.5, size=(num_classes, num_joints, in_channels))
        # Per-class gloss token ids (reserve 0 as PAD).
        self.glosses = rng.integers(1, vocab_size, size=(num_classes, seq_len))
        self._rng = rng

    def __len__(self) -> int:
        return self.num_classes * self.samples_per_class

    def _trajectory(self, k: int, jitter: np.ndarray) -> np.ndarray:
        t = np.linspace(0, 1, self.num_frames)[None, None, :]  # (1,1,T)
        f = self.freqs[k][..., None]      # (V, C, 1)
        p = self.phases[k][..., None]
        a = self.amps[k][..., None]
        traj = a * np.sin(2 * np.pi * f * t + p)  # (V, C, T)
        traj = traj + jitter[..., None]
        return np.transpose(traj, (1, 2, 0))       # (C, T, V)

    def __getitem__(self, idx: int):
        k = idx // self.samples_per_class
        jitter = self._rng.normal(0, self.noise, size=(self.num_joints, self.in_channels))
        pose = self._trajectory(k, jitter).astype(np.float32)
        tokens = self.glosses[k].astype(np.int64)
        return {
            "pose": torch.from_numpy(pose),          # (C, T, V)
            "tokens": torch.from_numpy(tokens),      # (L,)
            "label": torch.tensor(k, dtype=torch.long),
        }


def collate_batch(batch):
    pose = torch.stack([b["pose"] for b in batch], dim=0)
    tokens = torch.stack([b["tokens"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    mask = (tokens != 0)
    return {"pose": pose, "tokens": tokens, "text_mask": mask, "label": labels}
