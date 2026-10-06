"""Governed variable-length kind/label sequence targets, including a stop event.

Canonical event-row order is serialization order, not temporal precedence. The
sequence is a partial linguistic target; it is never a complete temporal SIR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from .label_vocabulary import GovernedLabelVocabulary, encode_governed_labels
from .supervision import GovernedSIRAnnotation


@dataclass(frozen=True)
class SIRLabelSequenceTargets:
    inputs: torch.Tensor              # B,S: PAD=0, START=1, class i -> i+2
    outputs: torch.Tensor             # B,S: STOP=0, class i -> i+1, padding=-1
    lengths: torch.Tensor             # B: real labels plus exactly one STOP
    vocabulary_sha256: str
    annotation_sha256: tuple[str, ...]


def label_sequence_targets(annotations: Sequence[GovernedSIRAnnotation],
                           vocabulary: GovernedLabelVocabulary) -> SIRLabelSequenceTargets:
    annotations = tuple(annotations)
    labels = encode_governed_labels(annotations, vocabulary)
    lengths = (labels != -1).sum(dim=1) + 1
    inputs = torch.zeros((labels.shape[0], labels.shape[1] + 1), dtype=torch.int64)
    outputs = torch.full_like(inputs, -1)
    for row, length in enumerate(lengths.tolist()):
        count = length - 1
        inputs[row, 0] = 1
        inputs[row, 1:length] = labels[row, :count] + 2
        outputs[row, :count] = labels[row, :count] + 1
        outputs[row, count] = 0
    return SIRLabelSequenceTargets(inputs, outputs, lengths, vocabulary.lexicon.sha256,
                                   tuple(a.content_sha256() for a in annotations))


@dataclass(frozen=True)
class SIRLabelSequenceLogits:
    values: torch.Tensor              # B,S,C+1; output column zero is STOP
    vocabulary_sha256: str
    annotation_sha256: tuple[str, ...]


def label_sequence_loss(prediction: SIRLabelSequenceLogits,
                        annotations: Sequence[GovernedSIRAnnotation],
                        vocabulary: GovernedLabelVocabulary) -> torch.Tensor:
    """Mean of per-example mean NLL over all labels AND the single STOP.

    Padding is excluded. Including STOP gives length decisions their own observed
    target instead of rewarding an externally supplied event count. This is
    teacher-forced training likelihood, not free-running sequence accuracy.
    """
    if not isinstance(prediction, SIRLabelSequenceLogits):
        raise ValueError('typed sequence logits required')
    target = label_sequence_targets(annotations, vocabulary)
    if (prediction.vocabulary_sha256 != target.vocabulary_sha256
            or prediction.annotation_sha256 != target.annotation_sha256):
        raise ValueError('sequence vocabulary or annotation order mismatch')
    scores = prediction.values
    if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float32, torch.float64)
            or scores.shape != (*target.outputs.shape, len(vocabulary.entries) + 1)
            or not bool(torch.isfinite(scores).all())):
        raise ValueError('sequence scores must be finite float32/64 with exact B,S,C+1 shape')
    losses = F.cross_entropy(scores.transpose(1, 2), target.outputs.to(scores.device),
                             ignore_index=-1, reduction='none')
    result = (losses.sum(dim=1) / target.lengths.to(device=scores.device, dtype=scores.dtype)).mean()
    if not bool(torch.isfinite(losses).all()) or not bool(torch.isfinite(result)):
        raise ValueError('sequence objective overflowed its numeric dtype')
    return result
