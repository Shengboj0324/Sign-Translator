"""Leakage-certified grouped split (Doc-10 §6).

The leakage constraint is *not* satisfied by grouping the pair ``(signer,
source)``: two pairs may share a signer (or a source) and still be assigned to
different splits.  We instead build connected components in the bipartite graph
whose vertices are signer ids and source ids and whose samples are edges.  A
whole connected component is assigned to one split.  Consequently neither a
signer nor a source can cross a split, including transitive cases.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .schema import Sample

_SPLIT_NAMES = ("train", "val", "test")


ComponentKey = Tuple[Tuple[str, ...], Tuple[str, ...]]


def group_samples(samples: Sequence[Sample]) -> Dict[ComponentKey, List[int]]:
    """Return maximal signer/source connected components.

    A component key is ``(sorted_signers, sorted_sources)`` and is therefore
    stable under input ordering.  This stronger grouping closes both signer and
    source leakage rather than merely keeping identical signer/source pairs
    together.
    """
    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    signer_owner: Dict[str, int] = {}
    source_owner: Dict[str, int] = {}
    for index, sample in enumerate(samples):
        for value, owners in ((sample.signer_id_hash, signer_owner),
                              (sample.source_id, source_owner)):
            previous = owners.setdefault(value, index)
            union(index, previous)

    members: Dict[int, List[int]] = {}
    for index in range(len(samples)):
        members.setdefault(find(index), []).append(index)

    groups: Dict[ComponentKey, List[int]] = {}
    for indices in members.values():
        signers = tuple(sorted({samples[index].signer_id_hash for index in indices}))
        sources = tuple(sorted({samples[index].source_id for index in indices}))
        groups[(signers, sources)] = indices
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
    active_splits = tuple(name for name, ratio in zip(_SPLIT_NAMES, ratios) if ratio > 0)
    for position, k in enumerate(keys):
        size = len(groups[k])
        empty = [name for name in active_splits if filled[name] == 0]
        remaining_components = len(keys) - position
        # If every still-empty requested split needs one of the remaining
        # components, reserve those components now.  This gives non-empty
        # requested splits whenever the component count makes that possible.
        candidates = empty if empty and remaining_components <= len(empty) else active_splits
        name = max(candidates, key=lambda split: targets[split] - filled[split])
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
    offending_signers: Tuple[str, ...] = ()
    offending_sources: Tuple[str, ...] = ()

    @property
    def offending_groups(self) -> Tuple[Tuple[str, str], ...]:
        """Backwards-compatible tagged list of violated atomic constraints."""
        return (tuple(("signer", value) for value in self.offending_signers)
                + tuple(("source", value) for value in self.offending_sources))


def certify_no_group_leakage(samples: Sequence[Sample],
                             assignment: Dict[int, str]) -> LeakageCertificate:
    """Certify total assignment plus independent signer/source separation."""
    expected = set(range(len(samples)))
    if set(assignment) != expected:
        missing = sorted(expected - set(assignment))
        extra = sorted(set(assignment) - expected)
        raise ValueError(f"assignment keys mismatch: missing={missing}, extra={extra}")
    invalid = sorted({split for split in assignment.values() if split not in _SPLIT_NAMES})
    if invalid:
        raise ValueError(f"invalid split names: {invalid}")

    signer_splits: Dict[str, set] = {}
    source_splits: Dict[str, set] = {}
    for index, sample in enumerate(samples):
        split = assignment[index]
        signer_splits.setdefault(sample.signer_id_hash, set()).add(split)
        source_splits.setdefault(sample.source_id, set()).add(split)
    offending_signers = tuple(sorted(key for key, value in signer_splits.items()
                                     if len(value) > 1))
    offending_sources = tuple(sorted(key for key, value in source_splits.items()
                                     if len(value) > 1))
    return LeakageCertificate(
        certified=not offending_signers and not offending_sources,
        offending_signers=offending_signers,
        offending_sources=offending_sources,
    )


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
