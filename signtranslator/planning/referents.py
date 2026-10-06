"""Referent equality supervision invariant to annotation-local ID renaming.

This is a symmetric pair property, not the presence of a directed COREF edge.
Missing IDs are unknown. Each unordered known pair contributes once; diagonal,
padding and unknown endpoints are excluded. No new/absent identity is invented.
"""
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from .label_vocabulary import GovernedLabelVocabulary
from .supervision import GovernedSIRAnnotation
from .tensors import tensorize_sir_annotations


@dataclass(frozen=True)
class ReferentTargets:
    equal: torch.Tensor  # B,E,E bool; targets populated only on known upper triangle
    known: torch.Tensor
    annotation_sha256: tuple[str, ...]
    vocabulary_sha256: str


def referent_targets(annotations: Sequence[GovernedSIRAnnotation],
                     vocabulary: GovernedLabelVocabulary) -> ReferentTargets:
    if not isinstance(vocabulary, GovernedLabelVocabulary):
        raise ValueError('governed vocabulary required')
    target = tensorize_sir_annotations(annotations, expected_lexicon=vocabulary.lexicon,
                                      expected_convention=vocabulary.convention)
    present = target.referent_present & target.event_valid
    count = present.shape[1]
    known = present[:, :, None] & present[:, None, :]
    known &= torch.ones(count, count, dtype=torch.bool).triu(diagonal=1)
    equal = (target.referent_ids[:, :, None] == target.referent_ids[:, None, :]) & known
    return ReferentTargets(equal, known, target.annotation_sha256, vocabulary.lexicon.sha256)


@dataclass(frozen=True)
class ReferentLogits:
    values: torch.Tensor  # B,E,E, exactly symmetric
    annotation_sha256: tuple[str, ...]
    vocabulary_sha256: str


def referent_loss_per_example(prediction: ReferentLogits,
                              annotations: Sequence[GovernedSIRAnnotation],
                              vocabulary: GovernedLabelVocabulary) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean known-pair NLL and support; unsupported zero is bookkeeping."""
    if not isinstance(prediction, ReferentLogits):
        raise ValueError('typed referent logits required')
    target = referent_targets(annotations, vocabulary)
    if (prediction.annotation_sha256 != target.annotation_sha256
            or prediction.vocabulary_sha256 != target.vocabulary_sha256):
        raise ValueError('referent prediction identity mismatch')
    scores = prediction.values
    if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float32, torch.float64)
            or scores.shape != target.known.shape or not bool(torch.isfinite(scores).all())
            or not torch.equal(scores, scores.transpose(1, 2))):
        raise ValueError('finite exactly symmetric B,E,E referent logits required')
    known = target.known.to(scores.device)
    loss = F.binary_cross_entropy_with_logits(scores, target.equal.to(scores.device, scores.dtype), reduction='none')
    masked = torch.where(known, loss, torch.zeros_like(loss))
    counts = known.flatten(1).sum(1)
    means = masked.flatten(1).sum(1) / counts.clamp_min(1).to(scores.dtype)
    if not bool(torch.isfinite(means).all()):
        raise ValueError('referent objective overflow')
    return means, counts > 0
