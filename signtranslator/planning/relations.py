"""Partially observed directed SIR relations with conservative negative support.

Missing edges are unknown. Negative targets require a contradiction established
by recorded endpoints, kinds or two explicit unequal referents. No transitive or
symmetric closure, closed-world annotation assumption, or missing-reference
substitution is introduced. Relations are multilabel, not exclusive classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from ..training.target_cells import selected_target_cells

from ..grammar.sir import EdgeType
from .label_vocabulary import GovernedLabelVocabulary
from .supervision import GovernedSIRAnnotation
from .tensors import EDGE_TYPES, tensorize_sir_annotations


@dataclass(frozen=True)
class SIRRelationTargets:
    positive: torch.Tensor            # B,E,E,R bool
    known: torch.Tensor               # positives or logically contradicted relations
    annotation_sha256: tuple[str, ...]
    vocabulary_sha256: str
    relation_types: tuple[EdgeType, ...] = EDGE_TYPES


def relation_targets(annotations: Sequence[GovernedSIRAnnotation],
                      vocabulary: GovernedLabelVocabulary) -> SIRRelationTargets:
    if not isinstance(vocabulary, GovernedLabelVocabulary):
        raise ValueError('governed vocabulary required')
    annotations = tuple(annotations)
    targets = tensorize_sir_annotations(annotations, expected_lexicon=vocabulary.lexicon,
                                        expected_convention=vocabulary.convention)
    count, events = targets.event_valid.shape
    positive = torch.zeros((count, events, events, len(EDGE_TYPES)), dtype=torch.bool)
    known = torch.zeros_like(positive)
    for row, annotation in enumerate(annotations):
        graph = annotation.graph()
        positions = {event.id: i for i, event in enumerate(graph.events)}
        for i, a in enumerate(graph.events):
            for j, b in enumerate(graph.events):
                if i == j:
                    continue  # self edges are outside the model's candidate domain
                impossible = {
                    EdgeType.PRECEDENCE: a.t_end > b.t_start,
                    EdgeType.OVERLAP: not a.overlaps_time(b),
                    EdgeType.SCOPE: (a.kind.is_manual or not b.kind.is_manual
                                     or not (a.t_start <= b.t_start and a.t_end >= b.t_end)),
                    EdgeType.COREF: (a.referent is not None and b.referent is not None
                                     and a.referent != b.referent),
                    EdgeType.LOCUS: False,  # absence of a recorded locus is unknown
                }
                for k, relation in enumerate(EDGE_TYPES):
                    known[row, i, j, k] = impossible[relation]
        for edge in graph.edges:
            i, j, k = positions[edge.source], positions[edge.target], EDGE_TYPES.index(edge.type)
            if known[row, i, j, k]:
                raise ValueError('recorded relation contradicts its annotation fields')
            positive[row, i, j, k] = known[row, i, j, k] = True
    return SIRRelationTargets(positive, known, targets.annotation_sha256, vocabulary.lexicon.sha256)


@dataclass(frozen=True)
class SIRRelationLogits:
    values: torch.Tensor              # B,E,E,R; independent Bernoulli logits
    annotation_sha256: tuple[str, ...]
    vocabulary_sha256: str
    relation_types: tuple[EdgeType, ...] = EDGE_TYPES


def relation_loss(prediction: SIRRelationLogits, annotations: Sequence[GovernedSIRAnnotation],
                   vocabulary: GovernedLabelVocabulary) -> torch.Tensor:
    """Mean per-example Bernoulli NLL over explicitly supported relation cells.

    Every example must have support; otherwise this required objective is
    unavailable and fails. Unknowns, self edges and padding get zero gradient.
    Unequal known-cell counts do not increase an example's weight. This loss
    cannot estimate complete edge accuracy or calibrate unreviewed negatives.
    """
    values, supported_examples = relation_loss_per_example(prediction, annotations, vocabulary)
    if not bool(supported_examples.all()):
        raise ValueError('required relation objective unavailable for an example without known cells')
    return values.mean()


def relation_loss_per_example(prediction: SIRRelationLogits,
                               annotations: Sequence[GovernedSIRAnnotation],
                               vocabulary: GovernedLabelVocabulary, *, with_target_cells=False):
    """Return B mean losses and explicit availability; unsupported zeros are bookkeeping."""
    if not isinstance(prediction, SIRRelationLogits):
        raise ValueError('typed relation logits required')
    target = relation_targets(annotations, vocabulary)
    if (prediction.annotation_sha256 != target.annotation_sha256
            or prediction.vocabulary_sha256 != target.vocabulary_sha256
            or prediction.relation_types != EDGE_TYPES):
        raise ValueError('relation prediction identity or codebook mismatch')
    scores = prediction.values
    if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float32, torch.float64)
            or scores.shape != target.known.shape or not bool(torch.isfinite(scores).all())):
        raise ValueError('relation logits must be finite float32/64 of exact B,E,E,R shape')
    known = target.known.to(scores.device)
    counts = known.flatten(1).sum(dim=1)
    labels = target.positive.to(device=scores.device, dtype=scores.dtype)
    per_cell = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
    supported = torch.where(known, per_cell, torch.zeros_like(per_cell))
    result = supported.flatten(1).sum(dim=1) / counts.clamp_min(1).to(scores.dtype)
    if not bool(torch.isfinite(supported).all()) or not bool(torch.isfinite(result).all()):
        raise ValueError('relation objective overflow')
    if with_target_cells:
        cells = selected_target_cells(target.known, target.positive,
                                      axes=('source_event', 'target_event', 'relation_type'), class_count=2)
        return result, counts > 0, cells
    return result, counts > 0
