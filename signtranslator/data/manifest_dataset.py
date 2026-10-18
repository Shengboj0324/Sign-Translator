"""PyTorch dataset over a frozen corpus manifest, with verified bounded caching.

Each item is one admitted sample, fetched through :class:`ContentCache` (hash
checked on every read) and masked to the channels its evidence admits. Partition
membership comes from the manifest's own performer split, so a test performer's
sample can never be served as training data by construction. Collate with
:func:`collate_multichannel`; skeleton and face samples have different layouts
and are rejected if mixed in one batch.
"""
from __future__ import annotations

from typing import Callable, Mapping

from torch.utils.data import Dataset

from ..data_engineering.corpus_manifest import ContentCache, load_sample, usable_channels
from ..pose.multichannel import MultichannelMotion

PARTITIONS = ("train", "test", "all")


class ManifestMotionDataset(Dataset):
    def __init__(self, manifest: Mapping, kind: str, *, partition: str, cache: ContentCache,
                 fetch: Callable[[str], bytes], require_channels=()) -> None:
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}")
        if kind not in ("skeleton", "face"):
            raise ValueError("kind must be 'skeleton' or 'face'")
        split = manifest["split"]
        train = split["train_performer_sha256"]
        test = set(split["test_performer_sha256"])
        if train in test:
            raise ValueError("a performer cannot be in both partitions")
        required = set(require_channels)

        def member(record) -> bool:
            if partition == "all":
                return True
            performer = record.get("performer_sha256")
            return performer == train if partition == "train" else performer in test

        self.records = [r for r in manifest["records"]
                        if member(r) and usable_channels(r, kind)
                        and required <= set(usable_channels(r, kind))]
        self.kind, self.cache, self.fetch = kind, cache, fetch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MultichannelMotion:
        return load_sample(self.records[index], self.kind, cache=self.cache, fetch=self.fetch)


__all__ = ["ManifestMotionDataset", "PARTITIONS"]
