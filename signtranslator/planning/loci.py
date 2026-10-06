"""Locus classes declared by the exact reviewed convention bytes.

The convention document must expose locus_schema_version=1 and a loci list of
{id, identity} rows in contiguous ID order. Other convention content is preserved
by its byte identity. Symbolic identities do not prove physical calibration.
Missing annotation loci remain unknown; no absent-locus class is inferred.
"""
from dataclasses import dataclass, field
import hashlib
import json

import torch
from torch.nn import functional as F

from .label_vocabulary import GovernedLabelVocabulary
from .supervision import ArtifactKind, GovernedArtifact
from .tensors import tensorize_sir_annotations


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate convention field')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite convention constant')


@dataclass(frozen=True)
class LocusAlphabet:
    convention: GovernedArtifact
    payload: bytes
    identities: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        if (not isinstance(self.convention, GovernedArtifact)
                or self.convention.kind is not ArtifactKind.ASL_CONVENTION):
            raise ValueError('typed ASL convention required')
        if not isinstance(self.payload, bytes) or not 0 < len(self.payload) <= 4 * 1024 * 1024:
            raise ValueError('bounded immutable convention bytes required')
        if hashlib.sha256(self.payload).hexdigest() != self.convention.sha256:
            raise ValueError('locus alphabet must come from exact convention bytes')
        value = json.loads(self.payload.decode('utf-8'), object_pairs_hook=_unique,
                           parse_constant=_reject_constant)
        if (not isinstance(value, dict) or type(value.get('locus_schema_version')) is not int
                or value['locus_schema_version'] != 1):
            raise ValueError('explicit locus schema version 1 required')
        rows = value.get('loci')
        if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
            raise ValueError('nonempty bounded locus alphabet required')
        identities = []
        for i, row in enumerate(rows):
            if (not isinstance(row, dict) or set(row) != {'id', 'identity'}
                    or type(row['id']) is not int or row['id'] != i):
                raise ValueError('locus IDs must be contiguous in classifier order from zero')
            identity = row['identity']
            if (not isinstance(identity, str) or not identity.strip()
                    or identity != identity.strip() or identity in identities):
                raise ValueError('unique explicit locus identities required')
            identities.append(identity)
        object.__setattr__(self, 'identities', tuple(identities))


@dataclass(frozen=True)
class LocusTargets:
    classes: torch.Tensor  # B,E; -1 for unknown or padding
    known: torch.Tensor
    annotation_sha256: tuple[str, ...]
    convention_sha256: str
    vocabulary_sha256: str


def locus_targets(annotations, vocabulary: GovernedLabelVocabulary, alphabet: LocusAlphabet) -> LocusTargets:
    if not isinstance(vocabulary, GovernedLabelVocabulary) or not isinstance(alphabet, LocusAlphabet):
        raise ValueError('typed vocabulary and locus alphabet required')
    if alphabet.convention != vocabulary.convention:
        raise ValueError('locus alphabet convention differs from label vocabulary')
    target = tensorize_sir_annotations(annotations, expected_lexicon=vocabulary.lexicon,
                                      expected_convention=alphabet.convention)
    known = target.event_valid & target.locus_present
    if bool((target.locus_ids[known] >= len(alphabet.identities)).any()):
        raise ValueError('annotated locus outside declared alphabet')
    return LocusTargets(torch.where(known, target.locus_ids, -1), known,
                        target.annotation_sha256, alphabet.convention.sha256, vocabulary.lexicon.sha256)


@dataclass(frozen=True)
class LocusLogits:
    values: torch.Tensor
    annotation_sha256: tuple[str, ...]
    convention_sha256: str
    vocabulary_sha256: str


def locus_loss_per_example(prediction: LocusLogits, annotations, vocabulary, alphabet):
    if not isinstance(prediction, LocusLogits):
        raise ValueError('typed locus logits required')
    target = locus_targets(annotations, vocabulary, alphabet)
    if (prediction.annotation_sha256 != target.annotation_sha256
            or prediction.convention_sha256 != target.convention_sha256
            or prediction.vocabulary_sha256 != target.vocabulary_sha256):
        raise ValueError('locus prediction identity mismatch')
    scores = prediction.values
    if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float32, torch.float64)
            or scores.shape != (*target.known.shape, len(alphabet.identities))
            or not bool(torch.isfinite(scores).all())):
        raise ValueError('finite B,E,L locus logits of exact alphabet shape required')
    known = target.known.to(scores.device)
    classes = target.classes.to(scores.device)
    losses = F.cross_entropy(scores.transpose(1, 2), classes, ignore_index=-1, reduction='none')
    counts = known.sum(1)
    means = losses.sum(1) / counts.clamp_min(1).to(scores.dtype)
    if not bool(torch.isfinite(means).all()):
        raise ValueError('locus objective overflow')
    return means, counts > 0
