"""Training data for the 2D-keypoint back-translator: shards written by ``build_uwb_keypoint_shards.py`` (video
landmarks) and ``build_fk_keypoint_shards.py`` (canonical states through forward kinematics), one format.

An index (``index.json``) lists every shard's key and SHA-256, and the shards' source frame rate. Shards are
SHA-checked on load and kept in memory, at about 25 MB per 1,000 clips. ``__getitem__`` does the following:
- resamples a clip from its source rate to ``target_fps``;
- with ``augment``, stretches time by a random factor in [0.8, 1.25] (the UWB rate is assumed, so this also covers
  a 24–37.5 fps source), and applies a small random similarity transform (±8°, ×0.9–1.1, ±0.1 shift) to the
  arm and wrist points. Hand-local points only rotate with it;
- truncates to ``max_seconds``;
- returns ``(features (T, 144) float32, text)``.
"""
from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .keypoints2d import features, resample

_GLOBAL = np.r_[np.arange(6), 6, 27]          # arm points and both hand wrists (signing-space coordinates)
_LOCAL = np.setdiff1d(np.arange(48), _GLOBAL)  # hand points relative to their wrist


def load_index(store, key: str) -> dict:
    return json.loads(store.get(key))


class KeypointShardDataset(Dataset):
    def __init__(self, store, index_keys, *, partition: str, target_fps: float = 24., augment: bool = False,
                 max_seconds: float = 20., limit: int = 0, seed: int = 0) -> None:
        self.target_fps, self.augment, self.max_frames = target_fps, augment, int(max_seconds * target_fps)
        self.clips, self.texts, self.ids, self.sources = [], [], [], []
        self.rng = np.random.default_rng(seed)
        for key in index_keys:
            index = load_index(store, key)
            fps = float(index.get("source_fps") or index["source_fps_assumed"])
            for part in index["parts"]:
                for shard in part["shards"]:
                    if not shard["partitions"].get(partition):
                        continue
                    payload = store.get(shard["key"])
                    if hashlib.sha256(payload).hexdigest() != shard["sha256"]:
                        raise ValueError(f"{shard['key']}: SHA-256 differs from the index")
                    meta_payload = store.get(shard["key"][:-4] + ".json")
                    if hashlib.sha256(meta_payload).hexdigest() != shard["meta_sha256"]:
                        raise ValueError(f"{shard['key']}: clip list SHA-256 differs from the index")
                    data, meta = np.load(io.BytesIO(payload)), json.loads(meta_payload)
                    points, valid, offsets = data["points"], data["valid"], data["offsets"]
                    for k, m in enumerate(meta):
                        if m["partition"] != partition:
                            continue
                        a, b = int(offsets[k]), int(offsets[k + 1])
                        self.clips.append((points[a:b], valid[a:b], fps))
                        self.texts.append(m["text"]); self.ids.append(m["clip_id"]); self.sources.append(key)
                        if limit and len(self.clips) >= limit:
                            return

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        points, valid, fps = self.clips[i]
        points = points.astype(np.float32)
        rng = np.random.default_rng() if self.augment else None
        if rng is not None:
            fps = fps * rng.uniform(.8, 1.25)
        points, valid = resample(points, valid, fps=fps, target_fps=self.target_fps)
        if rng is not None:
            angle = np.radians(rng.uniform(-8, 8))
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
            points = points.copy()
            points[:, _GLOBAL] = points[:, _GLOBAL] @ rotation.T * rng.uniform(.9, 1.1) + rng.uniform(-.1, .1, 2)
            points[:, _LOCAL] = points[:, _LOCAL] @ rotation.T
            points[~valid] = 0.
        points, valid = points[:self.max_frames], valid[:self.max_frames]
        return torch.from_numpy(features(points, valid)), self.texts[i]


def collate_keypoints(items):
    """((features (B, T, 144), valid (B, T)), texts)."""
    T = max(len(f) for f, _ in items)
    x = torch.zeros(len(items), T, items[0][0].shape[-1])
    valid = torch.zeros(len(items), T, dtype=torch.bool)
    for k, (f, _) in enumerate(items):
        x[k, :len(f)] = f
        valid[k, :len(f)] = True
    return (x, valid), [t for _, t in items]


__all__ = ["KeypointShardDataset", "collate_keypoints", "load_index"]
