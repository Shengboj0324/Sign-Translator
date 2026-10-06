"""Corpus-wide admission and lazy, revalidated loading of governed motion pairs.

Admission covers the submitted population only. It is not evidence of population
completeness, statistical power, human qualifications, or phase acceptance.
Split views share one admitted corpus; constructing unrelated split corpora does
not establish isolation between them.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import hashlib
from numbers import Integral
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from ..data_engineering.phase2_policy import AuthorizationEvidence, POLICY_VERSION
from ..data_engineering.schema import Sample
from ..data_engineering.source_portfolio import SourceCandidate
from ..planning.supervision import GovernedSIRAnnotation, certify_supervision_batch
from ..planning.tensors import SIRTargets, tensorize_sir_annotations
from ..reproducibility import canonical_json_bytes
from .governed_motion import GovernedMotionPair, load_governed_motion_pair
from .multichannel import MotionBatch, collate_multichannel


@dataclass(frozen=True)
class GovernedMotionRecord:
    motion_path: Path
    motion_sha256: str
    annotation: GovernedSIRAnnotation
    sample: Sample
    video_path: Path
    transcript_path: Path
    annotation_authorization_path: Path
    alignment_path: Path
    alignment_sha256: str
    source_files: dict[str, Path]

    def load(self, sources, authorizations) -> GovernedMotionPair:
        # Do not use asdict: it would turn the typed annotation/sample into dicts.
        arguments = {name: getattr(self, name) for name in self.__dataclass_fields__}
        return load_governed_motion_pair(**arguments, sources=sources,
                                        authorizations=authorizations)


def _layout(pair: GovernedMotionPair) -> dict:
    return {name: (channel.labels, channel.coordinate_frame, channel.units,
                   channel.convention, channel.values.dtype.str, channel.confidence.dtype.str)
            for name, channel in pair.motion.channels.items()}


class GovernedMotionCorpus:
    """Admit the full declared population before exposing train/val/test views.

    Inputs are privately snapshotted. Changed source/evidence files are rejected
    when read; a changed consent decision requires a new admission snapshot.
    Records are loaded one at a time during admission, not padded into one tensor.
    """

    def __init__(self, records: Sequence[GovernedMotionRecord], *,
                 sources: tuple[SourceCandidate, ...],
                 authorizations: dict[str, AuthorizationEvidence]):
        records = tuple(records)
        if not records or any(not isinstance(r, GovernedMotionRecord) for r in records):
            raise ValueError("a nonempty population of governed records is required")
        self._records = deepcopy(records)
        self._sources = deepcopy(sources)
        self._authorizations = deepcopy(authorizations)
        sample_ids = [record.sample.sample_id for record in self._records]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("duplicate sample identities in corpus")
        annotations = [record.annotation for record in self._records]
        # Individual loading below checks these declared identities against bytes.
        certificate = certify_supervision_batch(
            annotations, {r.sample.sample_id: r.sample for r in self._records},
            {a.source.sample_id: a.source.video_sha256 for a in annotations},
            {a.source.sample_id: a.source.transcript_sha256 for a in annotations})
        if not certificate.approved_for_research_training:
            raise PermissionError(f"corpus supervision rejected: {certificate.violations}")
        expected_layout = None
        content_splits: dict[tuple[str, str], str] = {}
        identity_records = []
        for record in self._records:
            pair = record.load(self._sources, self._authorizations)
            layout = _layout(pair)
            if expected_layout is None:
                expected_layout = layout
            elif layout != expected_layout:
                raise ValueError("corpus channel layouts/dtypes require explicit adaptation")
            # Renaming a signer/recording cannot separate identical input bytes.
            # Native archives shared by records are conservatively one group;
            # finer independence requires a reviewed recording-level source map.
            identities = {("video", pair.annotation.source.video_sha256),
                          ("motion", record.motion_sha256)}
            identities.update(("native_motion", ch.source_sha256)
                              for ch in pair.motion.channels.values() if ch.valid.any())
            for identity in identities:
                previous = content_splits.setdefault(identity, record.sample.split)
                if previous != record.sample.split:
                    raise PermissionError(f"content crosses splits: {identity[0]}")
            identity_records.append({
                "sample_id": record.sample.sample_id, "split": record.sample.split,
                "annotation_sha256": pair.annotation.content_sha256(),
                "motion_sha256": record.motion_sha256,
                "alignment_sha256": record.alignment_sha256,
                "consent": record.sample.consent.name,
                "authorization": record.sample.authorization.to_manifest(),
            })
        self._manifest = canonical_json_bytes({
            "schema_version": 1, "policy_version": POLICY_VERSION,
            "records": identity_records,
            "sources": [asdict(source) for source in self._sources],
            "authorizations": {key: {
                "source_id": value.source_id, "consent": value.consent.name,
                "authorization": value.authorization.to_manifest(),
            } for key, value in sorted(self._authorizations.items())},
            "phase_exit_approved": False,
        })

    @property
    def manifest_bytes(self) -> bytes:
        return self._manifest

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self._manifest).hexdigest()

    def split(self, name: str) -> "GovernedMotionDataset":
        if name not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        indices = tuple(i for i, r in enumerate(self._records) if r.sample.split == name)
        if not indices:
            raise ValueError(f"no admitted samples in split {name}")
        return GovernedMotionDataset(self, indices, name)


@dataclass(frozen=True)
class GovernedItem:
    corpus: GovernedMotionCorpus
    index: int
    split: str


class GovernedMotionDataset(Dataset):
    """A split view yielding references, loaded and checked by the collator."""

    def __init__(self, corpus: GovernedMotionCorpus, indices: tuple[int, ...], split: str):
        if not isinstance(corpus, GovernedMotionCorpus) or split not in ('train', 'val', 'test'):
            raise ValueError("dataset requires an admitted corpus and named split")
        indices = tuple(indices)
        if (not indices or any(type(i) is not int or not 0 <= i < len(corpus._records)
                               for i in indices) or len(set(indices)) != len(indices)):
            raise ValueError("dataset requires unique admitted record indices")
        if any(corpus._records[i].sample.split != split for i in indices):
            raise ValueError("dataset indices do not match admission split")
        self._corpus, self._indices, self._split = corpus, indices, split

    @property
    def training_contract(self) -> dict:
        """Fresh JSON metadata binding this exact ordered view, not just its size."""
        return {"schema_version": 1, "corpus_sha256": self._corpus.content_sha256,
                "split": self._split, "record_indices": list(self._indices)}

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> GovernedItem:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("dataset index must be an integer")
        return GovernedItem(self._corpus, self._indices[int(index)], self._split)


@dataclass(frozen=True)
class GovernedMotionBatch:
    motion: MotionBatch
    annotations: tuple[GovernedSIRAnnotation, ...]
    annotation_times: torch.Tensor       # B,T; use motion.frame_valid for padding
    event_ids: tuple[tuple[int, ...], ...]
    event_frame_membership: tuple[torch.Tensor, ...]  # each E,T, padded in time only
    corpus_sha256: str
    split: str
    sir_targets: SIRTargets
    transcript_payloads: tuple[bytes, ...]  # original source bytes, not gloss or tokens
    annotation_extents: tuple[tuple[float, float], ...]  # immutable declared clock extents


def move_governed_batch(batch: GovernedMotionBatch,
                        device: str | torch.device) -> GovernedMotionBatch:
    """Move all tensors without casting, flattening or changing immutable bindings.

    This is transport, not admission or mutable-tensor validation. Unsupported
    device dtypes fail explicitly; clocks must never be narrowed to float32.
    """
    if not isinstance(batch, GovernedMotionBatch):
        raise TypeError("expected a governed motion batch")

    def move_fields(value):
        return replace(value, **{field.name: tensor.to(device)
                                 for field in fields(value)
                                 if torch.is_tensor(tensor := getattr(value, field.name))})

    motion = replace(move_fields(batch.motion), channels={
        name: move_fields(channel) for name, channel in batch.motion.channels.items()})
    return replace(batch, motion=motion, sir_targets=move_fields(batch.sir_targets),
                   annotation_times=batch.annotation_times.to(device),
                   event_frame_membership=tuple(mask.to(device)
                                                for mask in batch.event_frame_membership))


def collate_governed_motion(items: Sequence[GovernedItem]) -> GovernedMotionBatch:
    """Re-read admitted bytes at batching; never consume a cached mutable pair."""
    items = tuple(items)
    if not items or any(not isinstance(item, GovernedItem) for item in items):
        raise ValueError("nonempty governed dataset references are required")
    corpus, split = items[0].corpus, items[0].split
    if not isinstance(corpus, GovernedMotionCorpus):
        raise ValueError("references must belong to an admitted corpus")
    if any(item.corpus is not corpus or item.split != split for item in items):
        raise ValueError("cannot mix corpus admissions or splits within a batch")
    pairs = []
    for item in items:
        if (isinstance(item.index, bool) or not isinstance(item.index, Integral)
                or not 0 <= item.index < len(corpus._records)):
            raise ValueError("invalid admitted record index")
        record = corpus._records[item.index]
        if record.sample.split != split:
            raise ValueError("reference split does not match admission")
        pairs.append(record.load(corpus._sources, corpus._authorizations))
    motion = collate_multichannel([pair.motion for pair in pairs])
    times = torch.zeros_like(motion.timestamps)
    membership = []
    for row, pair in enumerate(pairs):
        count = len(pair.annotation_times)
        times[row, :count] = torch.from_numpy(pair.annotation_times.copy())
        mask = torch.zeros((len(pair.event_ids), times.shape[1]), dtype=torch.bool)
        mask[:, :count] = torch.from_numpy(pair.event_frame_membership.copy())
        membership.append(mask)
    annotations = tuple(pair.annotation for pair in pairs)
    targets = tensorize_sir_annotations(
        annotations, expected_lexicon=corpus._records[0].annotation.lexicon,
        expected_convention=corpus._records[0].annotation.convention)
    return GovernedMotionBatch(motion, annotations, times,
                               tuple(pair.event_ids for pair in pairs), tuple(membership),
                               corpus.content_sha256, split, targets,
                               tuple(pair.transcript_payload for pair in pairs),
                               tuple(pair.annotation_extent for pair in pairs))
