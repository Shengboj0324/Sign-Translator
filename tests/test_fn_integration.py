"""Integration + cycle stress for the facial/non-manual layer.

Wires non-manual events into the Doc-03 SIR (validated), articulates them to a
FLAME expression sequence, distinguishes a synthetic marker-label fixture through
the whole chain, and runs a determinism/finiteness cycle-stress loop.
"""

import pytest
import torch

from signtranslator.grammar.sir import EventKind, EdgeType, SIREvent, validate_sir
from signtranslator.facial_nmm.channels import Channel, Marker, NonmanualEvent
from signtranslator.facial_nmm.articulate import MarkerArticulator
from signtranslator.facial_nmm.integration import (
    build_sir_with_nonmanual, articulate_frames, events_to_frame_intensities,
)

NUM_MARKERS = len(Marker)


def _manual():
    return [SIREvent(id=0, kind=EventKind.MANUAL, label=5, t_start=1.0, t_end=2.0)]


# ---------------------------------------------------------------------------
# SIR wiring
# ---------------------------------------------------------------------------
def test_nonmanual_events_land_in_valid_sir():
    manual = _manual()
    nm = [NonmanualEvent(Channel.BROW, Marker.YN_Q, 0.9, t_s=0.5, t_e=2.5)]  # contains [1,2]
    g = build_sir_with_nonmanual(manual, nm, scope_targets=[0])
    assert validate_sir(g) == []
    # exactly one NONMANUAL node and one SCOPE edge
    assert sum(1 for e in g.events if e.kind == EventKind.NONMANUAL) == 1
    assert len(g.edges_of(EdgeType.SCOPE)) == 1
    scope = g.edges_of(EdgeType.SCOPE)[0]
    assert scope.target == 0                                # scopes over the manual event


def test_scope_containment_is_both_a_validity_rule_and_a_loss():
    """A contradictory scope is rejected; the loss still supplies trainable signal."""
    import torch
    from signtranslator.facial_nmm.losses import scope_loss
    manual = _manual()                                       # manual [1, 2]
    # Marker lies inside the manual interval rather than containing it.
    nm = [NonmanualEvent(Channel.HEAD, Marker.NEG, 1.0, t_s=1.2, t_e=1.5)]
    g = build_sir_with_nonmanual(manual, nm, scope_targets=[0])
    assert "scope_time_contradiction" in validate_sir(g)
    # but the marker [1.2,1.5] does NOT contain the manual [1,2] -> containment loss > 0
    loss = scope_loss(torch.tensor([1.2]), torch.tensor([1.5]),
                      torch.tensor([1.0]), torch.tensor([2.0]))
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# articulation of a rasterised stream
# ---------------------------------------------------------------------------
def test_events_rasterise_and_articulate():
    ev = [NonmanualEvent(Channel.BROW, Marker.YN_Q, 0.8, t_s=0.5, t_e=2.5)]
    grid = events_to_frame_intensities(ev, NUM_MARKERS, num_frames=30, fps=10.0)
    # marker YN_Q active in frames [5, 25], zero elsewhere and in other markers
    assert torch.all(grid[5:26, int(Marker.YN_Q)] > 0)
    assert float(grid[:, int(Marker.WH_Q)].sum()) == 0.0
    art = MarkerArticulator(NUM_MARKERS, num_expr=10).double()
    expr = articulate_frames(grid.double(), art)
    assert expr.shape == (30, 10) and torch.isfinite(expr).all()


# ---------------------------------------------------------------------------
# synthetic marker-label comparison through the chain
# ---------------------------------------------------------------------------
def test_marker_fixture_changes_only_nonmanual_sir_labels():
    """Different fixture markers change labels while leaving manual events intact."""
    manual = _manual()
    yn = build_sir_with_nonmanual(
        manual, [NonmanualEvent(Channel.BROW, Marker.YN_Q, 1.0, 0.5, 2.5)], [0])
    wh = build_sir_with_nonmanual(
        manual, [NonmanualEvent(Channel.BROW, Marker.WH_Q, 1.0, 0.5, 2.5)], [0])
    nm_yn = [e.label for e in yn.events if e.kind == EventKind.NONMANUAL]
    nm_wh = [e.label for e in wh.events if e.kind == EventKind.NONMANUAL]
    assert nm_yn != nm_wh                                    # different marker labels
    # the manual channel is identical (same hands)
    man_yn = [e.label for e in yn.events if e.kind == EventKind.MANUAL]
    man_wh = [e.label for e in wh.events if e.kind == EventKind.MANUAL]
    assert man_yn == man_wh


# ---------------------------------------------------------------------------
# cycle stress
# ---------------------------------------------------------------------------
def test_cycle_stress_determinism_and_finiteness():
    art = MarkerArticulator(NUM_MARKERS, num_expr=8).double()
    art.eval()
    for s in range(60):
        g = torch.Generator().manual_seed(1000 + s)
        val = float(torch.rand(1, generator=g))
        ev = [NonmanualEvent(Channel.BROW, Marker.TOPIC, val, t_s=0.2, t_e=1.8)]
        grid = events_to_frame_intensities(ev, NUM_MARKERS, 20, 10.0).double()
        e1 = articulate_frames(grid, art)
        e2 = articulate_frames(grid, art)
        assert torch.equal(e1, e2) and torch.isfinite(e1).all()
