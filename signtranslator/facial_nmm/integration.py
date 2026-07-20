"""Facial/non-manual integration (docs/FACIAL_NMM.md §9).

Wires non-manual events into the Doc-03 SIR (NONMANUAL nodes + SCOPE edges to the
manual events they mark), and articulates active markers into a per-frame FLAME
expression sequence (Doc-04 face parts) that can drive Doc-08 blendshapes.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from ..grammar.sir import (
    EventKind, EdgeType, SIREvent, SIREdge, SIRGraph, validate_sir,
)
from .channels import NonmanualEvent
from .articulate import MarkerArticulator, marker_one_hot


def build_sir_with_nonmanual(manual_events: List[SIREvent],
                             nonmanual: List[NonmanualEvent],
                             scope_targets: Sequence[int]) -> SIRGraph:
    """Add NONMANUAL nodes for each non-manual event and SCOPE edges to the manual
    events they mark. ``scope_targets[i]`` is the id of the manual event that
    non-manual ``i`` scopes over (its scope must contain that event's interval).
    """
    events = list(manual_events)
    next_id = (max((e.id for e in manual_events), default=-1) + 1)
    edges: List[SIREdge] = []
    for nm, target_id in zip(nonmanual, scope_targets):
        node = SIREvent(id=next_id, kind=EventKind.NONMANUAL, label=int(nm.marker),
                        t_start=nm.t_s, t_end=nm.t_e)
        events.append(node)
        edges.append(SIREdge(next_id, target_id, EdgeType.SCOPE))
        next_id += 1
    return SIRGraph(events=events, edges=edges)


def articulate_frames(active_markers: torch.Tensor,
                      articulator: MarkerArticulator) -> torch.Tensor:
    """Per-frame marker intensities (T, num_markers) -> expression sequence (T, E)."""
    return articulator.expression(active_markers)


def events_to_frame_intensities(events: List[NonmanualEvent], num_markers: int,
                                num_frames: int, fps: float,
                                dtype=torch.get_default_dtype()) -> torch.Tensor:
    """Rasterise scoped events to a (T, num_markers) intensity grid (max over
    overlapping events of the same marker)."""
    grid = torch.zeros(num_frames, num_markers, dtype=dtype)
    for ev in events:
        f0 = max(0, int(ev.t_s * fps))
        f1 = min(num_frames, int(ev.t_e * fps) + 1)
        if f1 > f0:
            grid[f0:f1, int(ev.marker)] = torch.maximum(
                grid[f0:f1, int(ev.marker)],
                torch.full((f1 - f0,), ev.value, dtype=dtype))
    return grid
