"""Structural assembly of explicit decisions, never calibrated ASL acceptance.

No thresholds, reference IDs or repairs are inferred. Event IDs are local decoded
positions, not identities copied from an annotation. Successful output is frozen
canonical JSON so later mutation of score/decision tensors cannot change it.
"""
from dataclasses import dataclass
import json
import math

import torch

from ..grammar.sir import SIREdge, SIREvent, SIRGraph, sir_to_dict, validate_sir
from .label_vocabulary import GovernedLabelVocabulary
from .tensors import EDGE_TYPES
from .text_relations import RelationalSequenceCandidate


@dataclass(frozen=True)
class StructuralCandidateResult:
    graph_json: bytes | None
    violations: tuple[str, ...]


def assemble_candidate_graph(candidate: RelationalSequenceCandidate,
                             vocabulary: GovernedLabelVocabulary, *,
                             selected_edges: torch.Tensor,
                             referents: tuple[int | None, ...],
                             loci: tuple[int | None, ...],
                             num_loci: int,
                             source_extent: tuple[float, float]) -> StructuralCandidateResult:
    """Check caller-supplied discrete decisions against generated events.

    Malformed contracts raise ValueError. Well-formed but inconsistent graph
    proposals return violations and no graph. Unknown references remain None;
    they are neither predicted absence nor invented identity classes.
    """
    if not isinstance(candidate, RelationalSequenceCandidate) or not isinstance(vocabulary, GovernedLabelVocabulary):
        raise ValueError('typed candidate and governed vocabulary required')
    temporal = candidate.temporal
    if temporal.vocabulary_sha256 != vocabulary.lexicon.sha256:
        raise ValueError('candidate vocabulary mismatch')
    if temporal.status != 'terminated':
        return StructuralCandidateResult(None, ('label_decoding_not_terminated',))
    if type(num_loci) is not int or num_loci < 1:
        raise ValueError('explicit positive locus capacity required')
    if (not isinstance(source_extent, tuple) or len(source_extent) != 2
            or any(type(x) not in (int, float) for x in source_extent)):
        raise ValueError('finite increasing source extent required')
    try:
        finite = all(math.isfinite(x) for x in source_extent)
    except OverflowError:
        finite = False
    if not finite or not source_extent[0] < source_extent[1]:
        raise ValueError('finite increasing source extent required')
    if type(temporal.origin_seconds) not in (int, float) or temporal.origin_seconds != source_extent[0]:
        raise ValueError('candidate origin must match the declared source extent')
    count = len(temporal.events)
    if not count:
        return StructuralCandidateResult(None, ('empty_candidate',))
    for values in (referents, loci):
        if (not isinstance(values, tuple) or len(values) != count
                or any(x is not None and (type(x) is not int or x < 0) for x in values)):
            raise ValueError('one explicit nonnegative integer or unknown per event required')
    shape = (count, count, len(EDGE_TYPES))
    scores, valid = candidate.relation_logits, candidate.relation_valid
    if (candidate.relation_types != EDGE_TYPES or not isinstance(scores, torch.Tensor)
            or scores.dtype not in (torch.float32, torch.float64) or scores.shape != shape
            or not bool(torch.isfinite(scores).all())):
        raise ValueError('finite relation scores with exact codebook and shape required')
    domain = ~torch.eye(count, dtype=torch.bool, device=scores.device)
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.shape != (count, count) or not torch.equal(valid.to(scores.device), domain)):
        raise ValueError('candidate domain must exclude exactly self edges')
    if (not isinstance(selected_edges, torch.Tensor) or selected_edges.dtype != torch.bool
            or selected_edges.shape != shape):
        raise ValueError('explicit boolean edge decisions of exact shape required')
    selected = selected_edges.detach().to(device='cpu', copy=True)
    if bool(selected[torch.eye(count, dtype=torch.bool)].any()):
        return StructuralCandidateResult(None, ('self_edge',))
    events = []
    for i, event in enumerate(temporal.events):
        if event.label not in vocabulary.entries:
            raise ValueError('generated label outside bound vocabulary')
        events.append(SIREvent(i, event.label.kind, event.label.label_id,
                               event.start_seconds, event.end_seconds, referents[i], loci[i]))
    edges = [SIREdge(i, j, EDGE_TYPES[k]) for i, j, k in selected.nonzero().tolist()]
    graph = SIRGraph(events, edges)
    violations = validate_sir(graph, num_loci=num_loci)
    if not any(event.kind.is_manual for event in events):
        violations.append('manual_event_required')
    if 'invalid_interval' not in violations and any(
            e.t_start < source_extent[0] or e.t_end > source_extent[1] for e in events):
        violations.append('event_outside_source_extent')
    if violations:
        return StructuralCandidateResult(None, tuple(dict.fromkeys(violations)))
    payload = json.dumps(sir_to_dict(graph), sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return StructuralCandidateResult(payload, ())
