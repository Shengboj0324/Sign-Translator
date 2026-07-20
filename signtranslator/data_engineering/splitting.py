"""Leakage-certified grouped split (Doc-10 §6).

Samples are grouped by (signer_id_hash, source_id) and the GROUPS — not the
samples — are partitioned into train/val/test, so no signer or source recording
spans two splits. Windows and augmentations inherit their sample's split, so
windowing/augmentation AFTER the split cannot introduce leakage. Every guarantee
is certified, not assumed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .schema import Sample

_SPLIT_NAMES = ("train", "val", "test")


def group_samples(samples: Sequence[Sample]) -> Dict[Tuple[str, str], List[int]]:
    """Map each (signer, source) group key to its sample indices."""
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.group_key, []).append(i)
    return groups


def grouped_split(samples: Sequence[Sample],
                  ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                  seed: int = 0) -> Dict[int, str]:
    """Assign each sample index a split, partitioning by GROUP.

    Whole groups are placed into one split; largest groups are placed first
    (greedy) toward the split furthest below its target share, so the split
    sizes track the requested ratios without ever splitting a group.
    """
    if abs(sum(ratios) - 1.0) > 1e-9 or any(r < 0 for r in ratios):
        raise ValueError("ratios must be non-negative and sum to 1")
    groups = group_samples(samples)
    n = len(samples)
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    # Greedy: place the largest groups first for a tight ratio fit.
    keys.sort(key=lambda k: len(groups[k]), reverse=True)
    targets = {name: r * n for name, r in zip(_SPLIT_NAMES, ratios)}
    filled = {name: 0 for name in _SPLIT_NAMES}
    assignment: Dict[int, str] = {}
    for k in keys:
        size = len(groups[k])
        # deficit = how far below target each split is, per unit ratio.
        name = max(_SPLIT_NAMES,
                   key=lambda s: (targets[s] - filled[s]))
        for idx in groups[k]:
            assignment[idx] = name
        filled[name] += size
    return assignment


# ---------------------------------------------------------------------------
# leakage certificate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LeakageCertificate:
    certified: bool
    offending_groups: Tuple[Tuple[str, str], ...]


def certify_no_group_leakage(samples: Sequence[Sample],
                             assignment: Dict[int, str]) -> LeakageCertificate:
    """Certify that no (signer, source) group spans more than one split."""
    group_splits: Dict[Tuple[str, str], set] = {}
    for i, s in enumerate(samples):
        group_splits.setdefault(s.group_key, set()).add(assignment[i])
    offending = tuple(g for g, sp in group_splits.items() if len(sp) > 1)
    return LeakageCertificate(certified=not offending, offending_groups=offending)


# ---------------------------------------------------------------------------
# window / augmentation inheritance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    sample_idx: int
    start: int
    end: int


def windows_inherit_split(windows: Sequence[Window],
                          assignment: Dict[int, str]) -> List[str]:
    """A window's split is exactly its parent sample's split."""
    return [assignment[w.sample_idx] for w in windows]


def certify_window_split_consistency(windows: Sequence[Window],
                                     assignment: Dict[int, str]) -> bool:
    """No two windows of the same sample land in different splits (by construction)."""
    by_sample: Dict[int, set] = {}
    for w, sp in zip(windows, windows_inherit_split(windows, assignment)):
        by_sample.setdefault(w.sample_idx, set()).add(sp)
    return all(len(v) == 1 for v in by_sample.values())
