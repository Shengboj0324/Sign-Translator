"""End-to-end diagnostic graph decoding, never an empirical acceptance gate.

Caller-supplied relation thresholds are explicitly uncalibrated. Every directed
pair/type score is positive, negative, or undecided; any undecided cell prevents a
complete graph. Positive decisions are never deleted to repair contradictions.
This interface does not certify training support for any model branch or class.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
import math

import torch

from .candidate_graph import assemble_candidate_graph
from .joint_spatial import JointSpatialResult, decode_joint_spatial
from .label_vocabulary import GovernedLabelVocabulary
from .loci import LocusAlphabet
from .tensors import EDGE_TYPES
from .text_loci import LocusSequenceCandidate


@dataclass(frozen=True)
class DiagnosticRelationThresholds:
    negative_below: tuple[float, ...]
    positive_above: tuple[float, ...]
    relation_types: tuple = EDGE_TYPES

    def __post_init__(self):
        if self.relation_types != EDGE_TYPES:
            raise ValueError('exact relation codebook required')
        for bounds in (self.negative_below, self.positive_above):
            if (not isinstance(bounds, tuple) or len(bounds) != len(EDGE_TYPES)
                    or any(type(x) is not float or not math.isfinite(x) for x in bounds)):
                raise ValueError('one finite float threshold per relation type required')
        if any(lo >= hi for lo, hi in zip(self.negative_below, self.positive_above)):
            raise ValueError('each negative threshold must be strictly below its positive threshold')


@dataclass(frozen=True)
class DiagnosticGraphResult:
    status: str
    graph_json: bytes | None
    violations: tuple[str, ...]
    ambiguous_relations: int
    spatial: JointSpatialResult | None
    relation_thresholds: DiagnosticRelationThresholds


def decode_diagnostic_graph(candidate: LocusSequenceCandidate,
                            vocabulary: GovernedLabelVocabulary, alphabet: LocusAlphabet, *,
                            relation_thresholds: DiagnosticRelationThresholds,
                            place: tuple[bool, ...], source_extent: tuple[float, float],
                            referent_weight: Fraction, locus_weight: Fraction,
                            max_events: int, max_work: int) -> DiagnosticGraphResult:
    """Compose joint spatial decisions, relation decisions and structural checks.

    No default thresholds/weights/placement mask are supplied. Even a successful
    result is named uncalibrated_structural_candidate. It is not a reviewed SIR
    annotation, deployment authorization, branch-support claim or ASL acceptance.
    """
    if (not isinstance(candidate, LocusSequenceCandidate)
            or not isinstance(vocabulary, GovernedLabelVocabulary)
            or not isinstance(alphabet, LocusAlphabet)
            or not isinstance(relation_thresholds, DiagnosticRelationThresholds)):
        raise ValueError('typed candidate, vocabularies and diagnostic thresholds required')
    relational = candidate.referential.relational
    temporal = relational.temporal
    if (alphabet.convention != vocabulary.convention
            or temporal.vocabulary_sha256 != vocabulary.lexicon.sha256
            or candidate.convention_sha256 != vocabulary.convention.sha256):
        raise ValueError('candidate lexical and spatial conventions must match')
    if temporal.status != 'terminated':
        return DiagnosticGraphResult('label_decoding_not_terminated', None, (), 0, None, relation_thresholds)

    def snapshot(value):
        if not isinstance(value, torch.Tensor):
            raise ValueError('all terminated candidate score and domain tensors required')
        return value.detach().to(device='cpu', copy=True)

    relational = replace(relational, relation_logits=snapshot(relational.relation_logits),
                         relation_valid=snapshot(relational.relation_valid))
    reference = replace(candidate.referential, relational=relational,
                        equality_logits=snapshot(candidate.referential.equality_logits),
                        pair_valid=snapshot(candidate.referential.pair_valid))
    candidate = replace(candidate, referential=reference, locus_logits=snapshot(candidate.locus_logits))
    n = len(temporal.events)
    scores = relational.relation_logits
    domain = ~torch.eye(n, dtype=torch.bool)
    if (relational.relation_types != EDGE_TYPES or scores.dtype not in (torch.float32, torch.float64)
            or scores.shape != (n, n, len(EDGE_TYPES)) or not bool(torch.isfinite(scores).all())
            or relational.relation_valid.dtype != torch.bool
            or not torch.equal(relational.relation_valid, domain)):
        raise ValueError('finite relation scores and exact self-excluding domain required')
    spatial = decode_joint_spatial(candidate, alphabet, place=place, referent_weight=referent_weight,
                                   locus_weight=locus_weight, max_events=max_events, max_work=max_work)
    if spatial.status != 'unique_optimum_candidate':
        return DiagnosticGraphResult('spatial_' + spatial.status, None, (), 0, spatial, relation_thresholds)
    # float32 values embed exactly in float64; do not round supplied thresholds
    # down to the model dtype at a decision boundary.
    values = scores.to(torch.float64)
    low = torch.tensor(relation_thresholds.negative_below, dtype=torch.float64)
    high = torch.tensor(relation_thresholds.positive_above, dtype=torch.float64)
    selected = (values > high) & domain[..., None]
    negative = (values < low) & domain[..., None]
    undecided = domain[..., None] & ~(selected | negative)
    ambiguous = int(undecided.sum())
    if ambiguous:
        return DiagnosticGraphResult('relation_ambiguous', None, (), ambiguous, spatial, relation_thresholds)
    assembled = assemble_candidate_graph(relational, vocabulary, selected_edges=selected,
                                          referents=spatial.referents, loci=spatial.loci,
                                          num_loci=len(alphabet.identities), source_extent=source_extent)
    status = 'structure_rejected' if assembled.violations else 'uncalibrated_structural_candidate'
    return DiagnosticGraphResult(status, assembled.graph_json, assembled.violations, 0, spatial, relation_thresholds)
