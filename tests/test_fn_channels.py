"""Verification of non-manual channels, scoped intervals, and the scope algebra.

Proves event validation, the grammatical marker->channel mapping, that concurrent
scopes are detected as properly nested vs partially crossing, and the nesting
forest.
"""

import pytest

from signtranslator.facial_nmm.channels import (
    Channel, Marker, MARKER_CHANNELS, GRAMMATICAL_MARKERS, NonmanualEvent,
    scope_relation, is_properly_nested, nesting_parents,
)
from signtranslator.grammar.temporal import AllenRelation


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def test_event_validation():
    e = NonmanualEvent(Channel.BROW, Marker.YN_Q, value=0.8, t_s=0.0, t_e=1.0)
    assert e.duration == 1.0 and e.is_grammatical
    with pytest.raises(ValueError):
        NonmanualEvent(Channel.BROW, Marker.YN_Q, 0.5, t_s=1.0, t_e=1.0)   # empty span
    with pytest.raises(ValueError):
        NonmanualEvent(Channel.BROW, Marker.YN_Q, 1.5, t_s=0.0, t_e=1.0)   # value > 1
    with pytest.raises(ValueError):
        NonmanualEvent(Channel.BROW, Marker.YN_Q, 0.5, 0.0, 1.0, confidence=2.0)


def test_grammatical_vs_affective_markers():
    yn = NonmanualEvent(Channel.BROW, Marker.YN_Q, 1.0, 0.0, 1.0)
    aff = NonmanualEvent(Channel.CHEEK, Marker.AFFECT, 1.0, 0.0, 1.0)
    assert yn.is_grammatical and not aff.is_grammatical
    assert Marker.AFFECT not in GRAMMATICAL_MARKERS


def test_marker_channels_mapping_is_concurrent():
    # a topic marker activates BOTH brow and head concurrently (not one channel)
    assert Channel.BROW in MARKER_CHANNELS[Marker.TOPIC]
    assert Channel.HEAD in MARKER_CHANNELS[Marker.TOPIC]
    # yn-question brow raise is positive, wh-question brow furrow is negative
    assert MARKER_CHANNELS[Marker.YN_Q][Channel.BROW] > 0
    assert MARKER_CHANNELS[Marker.WH_Q][Channel.BROW] < 0


# ---------------------------------------------------------------------------
# scope algebra
# ---------------------------------------------------------------------------
def test_nested_scopes_are_proper():
    # a WH-question scope [0,4] containing a topic scope [1,2]
    wh = NonmanualEvent(Channel.BROW, Marker.WH_Q, 1.0, 0.0, 4.0)
    topic = NonmanualEvent(Channel.HEAD, Marker.TOPIC, 1.0, 1.0, 2.0)
    assert scope_relation(topic, wh) == AllenRelation.DURING       # topic inside wh
    assert is_properly_nested([wh, topic])


def test_partially_crossing_scopes_are_rejected():
    a = NonmanualEvent(Channel.BROW, Marker.YN_Q, 1.0, 0.0, 2.0)
    b = NonmanualEvent(Channel.HEAD, Marker.NEG, 1.0, 1.0, 3.0)      # crosses a
    assert scope_relation(a, b) == AllenRelation.OVERLAPS
    assert not is_properly_nested([a, b])                           # ill-formed nesting


def test_disjoint_scopes_are_proper():
    a = NonmanualEvent(Channel.BROW, Marker.YN_Q, 1.0, 0.0, 1.0)
    b = NonmanualEvent(Channel.HEAD, Marker.NEG, 1.0, 2.0, 3.0)
    assert is_properly_nested([a, b])                               # disjoint is fine


def test_nesting_forest():
    # outer [0,10] contains mid [1,8] contains inner [2,3]; a separate root [12,14]
    outer = NonmanualEvent(Channel.BROW, Marker.WH_Q, 1.0, 0.0, 10.0)
    mid = NonmanualEvent(Channel.HEAD, Marker.TOPIC, 1.0, 1.0, 8.0)
    inner = NonmanualEvent(Channel.EYE_APERTURE, Marker.COND, 1.0, 2.0, 3.0)
    root2 = NonmanualEvent(Channel.HEAD, Marker.NEG, 1.0, 12.0, 14.0)
    events = [outer, mid, inner, root2]
    parents = nesting_parents(events)
    assert parents[0] is None                                       # outer is a root
    assert parents[1] == 0                                          # mid -> outer
    assert parents[2] == 1                                          # inner -> mid (smallest container)
    assert parents[3] is None                                       # root2 separate root
