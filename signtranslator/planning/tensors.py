"""Lossless graph targets for governed SIR, not a gloss-token projection.

Label IDs remain in their governed lexicon namespace; they are not assumed to be
contiguous embedding indices. Padding is not an unknown linguistic label.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from ..grammar.sir import EdgeType, EventKind
from .supervision import ArtifactKind, GovernedArtifact, GovernedSIRAnnotation


SIR_TENSOR_SCHEMA_VERSION = 1
# Explicit stable codebooks. Reordering is a schema change, not an enum refactor.
EVENT_KINDS = (EventKind.MANUAL, EventKind.CLASSIFIER, EventKind.FINGERSPELL, EventKind.NONMANUAL)
EDGE_TYPES = (EdgeType.PRECEDENCE, EdgeType.OVERLAP, EdgeType.SCOPE, EdgeType.COREF, EdgeType.LOCUS)
_MAX_ID = torch.iinfo(torch.int64).max


@dataclass(frozen=True)
class SIRTargets:
    event_ids: torch.Tensor          # B,E, int64; -1 for padding
    event_kinds: torch.Tensor        # B,E, indices in EVENT_KINDS
    label_ids: torch.Tensor          # B,E, original governed IDs
    intervals: torch.Tensor          # B,E,2 float64 seconds
    event_valid: torch.Tensor        # B,E
    referent_ids: torch.Tensor       # B,E; -1 for absent/padded
    referent_present: torch.Tensor   # B,E; zero is a valid referent ID
    locus_ids: torch.Tensor          # B,E; -1 for absent/padded
    locus_present: torch.Tensor      # B,E
    edge_indices: torch.Tensor       # B,R,2; dense event row indices, not event IDs
    edge_types: torch.Tensor         # B,R, indices in EDGE_TYPES
    edge_valid: torch.Tensor         # B,R
    lexicon: GovernedArtifact
    convention: GovernedArtifact
    annotation_sha256: tuple[str, ...]
    schema_version: int = SIR_TENSOR_SCHEMA_VERSION


def tensorize_sir_annotations(
    annotations: Sequence[GovernedSIRAnnotation], *,
    expected_lexicon: GovernedArtifact, expected_convention: GovernedArtifact,
) -> SIRTargets:
    """Build CPU targets after verifying exact vocabulary and review bindings.

    No clipping, OOV substitution, interval quantization or edge dropping. This
    conversion does not grant corpus admission or substitute for evidence files.
    """
    for artifact, kind in ((expected_lexicon, ArtifactKind.SIR_LEXICON),
                           (expected_convention, ArtifactKind.ASL_CONVENTION)):
        if not isinstance(artifact, GovernedArtifact) or artifact.kind is not kind:
            raise ValueError("explicit typed lexicon and convention bindings are required")
    annotations = tuple(annotations)
    if not annotations or any(not isinstance(a, GovernedSIRAnnotation) for a in annotations):
        raise ValueError("a nonempty sequence of governed annotations is required")
    checked = tuple(GovernedSIRAnnotation.from_manifest(a.to_manifest()) for a in annotations)
    if any(a.lexicon != expected_lexicon or a.convention != expected_convention for a in checked):
        raise ValueError("annotation vocabulary/convention does not match target binding")
    graphs = [annotation.graph() for annotation in checked]
    for graph in graphs:
        for event in graph.events:
            for field in ("id", "label", "referent", "locus"):
                value = getattr(event, field)
                if value is not None and value > _MAX_ID:
                    raise ValueError(f"event {field} cannot be represented exactly as int64")
            if not math.isfinite(event.t_end - event.t_start):
                raise ValueError("event duration cannot be represented as finite float64")
    count = len(graphs)
    events = max(len(graph.events) for graph in graphs)
    edges = max(len(graph.edges) for graph in graphs)
    def ids(shape):
        return torch.full(shape, -1, dtype=torch.int64)

    event_ids, kinds, labels = (ids((count, events)) for _ in range(3))
    intervals = torch.zeros((count, events, 2), dtype=torch.float64)
    valid = torch.zeros((count, events), dtype=torch.bool)
    referents, loci = ids((count, events)), ids((count, events))
    referent_present = torch.zeros_like(valid)
    locus_present = torch.zeros_like(valid)
    edge_indices, edge_types = ids((count, edges, 2)), ids((count, edges))
    edge_valid = torch.zeros((count, edges), dtype=torch.bool)
    for row, graph in enumerate(graphs):
        positions = {event.id: index for index, event in enumerate(graph.events)}
        for index, event in enumerate(graph.events):
            event_ids[row, index] = event.id
            kinds[row, index] = EVENT_KINDS.index(event.kind)
            labels[row, index] = event.label
            intervals[row, index] = torch.tensor((event.t_start, event.t_end), dtype=torch.float64)
            valid[row, index] = True
            if event.referent is not None:
                referents[row, index] = event.referent
                referent_present[row, index] = True
            if event.locus is not None:
                loci[row, index] = event.locus
                locus_present[row, index] = True
        for index, edge in enumerate(graph.edges):
            edge_indices[row, index] = torch.tensor((positions[edge.source], positions[edge.target]))
            edge_types[row, index] = EDGE_TYPES.index(edge.type)
            edge_valid[row, index] = True
    return SIRTargets(event_ids, kinds, labels, intervals, valid, referents,
                      referent_present, loci, locus_present, edge_indices, edge_types,
                      edge_valid, expected_lexicon, expected_convention,
                      tuple(a.content_sha256() for a in checked))
