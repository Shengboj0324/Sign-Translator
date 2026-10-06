"""Explicit, byte-bound SIR label coverage and aligned lexical supervision.

The vocabulary is a governed project artifact, not an inferred vocabulary from
training or test labels. Entry order defines classifier columns. Event kinds
form distinct namespaces, including nonmanual markers and fingerspelling units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from ..grammar.sir import EventKind
from .supervision import ArtifactKind, GovernedArtifact, GovernedSIRAnnotation
from .tensors import EVENT_KINDS, tensorize_sir_annotations

_MAX_BYTES = 4 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate vocabulary field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite vocabulary number: {value}")


@dataclass(frozen=True)
class SIRLabelEntry:
    kind: EventKind
    label_id: int
    identity: str


@dataclass(frozen=True)
class GovernedLabelVocabulary:
    lexicon: GovernedArtifact
    convention: GovernedArtifact
    payload: bytes
    entries: tuple[SIRLabelEntry, ...] = field(init=False)

    def __post_init__(self):
        if (not isinstance(self.lexicon, GovernedArtifact)
                or self.lexicon.kind is not ArtifactKind.SIR_LEXICON
                or not isinstance(self.convention, GovernedArtifact)
                or self.convention.kind is not ArtifactKind.ASL_CONVENTION):
            raise ValueError("typed lexicon and convention artifacts are required")
        if not isinstance(self.payload, bytes) or not 0 < len(self.payload) <= _MAX_BYTES:
            raise ValueError("vocabulary requires bounded immutable bytes")
        if hashlib.sha256(self.payload).hexdigest() != self.lexicon.sha256:
            raise ValueError("vocabulary bytes differ from governed lexicon identity")
        value = json.loads(self.payload.decode('utf-8'), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
        fields = {'schema_version', 'artifact_id', 'version', 'convention_sha256', 'entries'}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("vocabulary fields must match schema version 1")
        if type(value['schema_version']) is not int or value['schema_version'] != 1:
            raise ValueError("unsupported vocabulary schema version")
        if (value['artifact_id'] != self.lexicon.artifact_id
                or value['version'] != self.lexicon.version
                or value['convention_sha256'] != self.convention.sha256):
            raise ValueError("vocabulary artifact/convention binding mismatch")
        rows = value['entries']
        if not isinstance(rows, list) or not 0 < len(rows) <= 100_000:
            raise ValueError("vocabulary requires a nonempty bounded entry list")
        entries, keys, identities = [], set(), set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'kind', 'label_id', 'identity'}:
                raise ValueError("unexpected vocabulary entry fields")
            kind = EventKind(row['kind'])
            label = row['label_id']
            if type(label) is not int or not 0 <= label <= torch.iinfo(torch.int64).max:
                raise ValueError("vocabulary label must be a nonnegative int64")
            identity = row['identity']
            if not isinstance(identity, str) or not identity.strip() or identity != identity.strip():
                raise ValueError("each label needs an explicit lexical or marker identity")
            key, semantic_key = (kind, label), (kind, identity)
            if key in keys or semantic_key in identities:
                raise ValueError("duplicate or ambiguous vocabulary entry")
            keys.add(key)
            identities.add(semantic_key)
            entries.append(SIRLabelEntry(kind, label, identity))
        object.__setattr__(self, 'entries', tuple(entries))

    @classmethod
    def load(cls, path: Path, *, lexicon: GovernedArtifact,
             convention: GovernedArtifact) -> "GovernedLabelVocabulary":
        if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
            raise ValueError("vocabulary must be a regular local file")
        with path.open('rb') as stream:
            payload = stream.read(_MAX_BYTES + 1)
        return cls(lexicon, convention, payload)


def encode_governed_labels(annotations: Sequence[GovernedSIRAnnotation],
                           vocabulary: GovernedLabelVocabulary) -> torch.Tensor:
    """Return B,E classifier indices; -1 means padding only, never unknown.

    Reconstruct from immutable reviewed annotation payloads rather than trusting
    mutable target tensors. Every active event must have exact typed coverage.
    """
    if not isinstance(vocabulary, GovernedLabelVocabulary):
        raise ValueError("a byte-bound governed vocabulary is required")
    targets = tensorize_sir_annotations(annotations, expected_lexicon=vocabulary.lexicon,
                                       expected_convention=vocabulary.convention)
    lookup = {(entry.kind, entry.label_id): index for index, entry in enumerate(vocabulary.entries)}
    encoded = torch.full_like(targets.label_ids, -1)
    for row, index in torch.nonzero(targets.event_valid).tolist():
        kind = EVENT_KINDS[int(targets.event_kinds[row, index])]
        label = int(targets.label_ids[row, index])
        try:
            encoded[row, index] = lookup[(kind, label)]
        except KeyError as error:
            raise ValueError(f"uncovered governed label: {kind.value}:{label}") from error
    return encoded


@dataclass(frozen=True)
class AlignedSIRLabelLogits:
    values: torch.Tensor                # B,E,C, canonical event order
    vocabulary_sha256: str
    annotation_sha256: tuple[str, ...]  # binds row order and event scaffolding


def aligned_sir_label_loss(prediction: AlignedSIRLabelLogits,
                           annotations: Sequence[GovernedSIRAnnotation],
                           vocabulary: GovernedLabelVocabulary) -> torch.Tensor:
    """Mean per-example categorical NLL on supplied, aligned event slots.

    Each example first averages over its real events; longer annotations do not
    silently gain weight. This diagnostic/training component assumes the event
    scaffold is supplied. It is not a free-generation planner loss or a score of
    comprehension, timing, edge prediction, motion, or unseen-event recovery.
    """
    if not isinstance(prediction, AlignedSIRLabelLogits):
        raise ValueError("typed aligned SIR logits are required")
    annotations = tuple(annotations)
    labels = encode_governed_labels(annotations, vocabulary)
    if (prediction.vocabulary_sha256 != vocabulary.lexicon.sha256
            or prediction.annotation_sha256 != tuple(a.content_sha256() for a in annotations)):
        raise ValueError("prediction vocabulary or annotation order binding mismatch")
    scores = prediction.values
    if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float32, torch.float64)
            or scores.shape != (*labels.shape, len(vocabulary.entries))
            or not bool(torch.isfinite(scores).all())):
        raise ValueError("logits must be finite float32/float64 with exact B,E,C shape")
    labels = labels.to(scores.device)
    per_event = F.cross_entropy(scores.transpose(1, 2), labels, ignore_index=-1, reduction='none')
    counts = (labels != -1).sum(dim=1)
    result = (per_event.sum(dim=1) / counts.to(per_event.dtype)).mean()
    if not bool(torch.isfinite(per_event).all()) or not bool(torch.isfinite(result)):
        raise ValueError("categorical objective overflowed its numeric dtype")
    return result
