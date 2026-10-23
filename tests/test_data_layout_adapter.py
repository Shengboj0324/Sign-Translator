"""Layout adapter: unavailable channels take the template layout; declared rotation equivalences relabel; the rest is
refused."""
from dataclasses import replace

import numpy as np
import pytest

from signtranslator.data.layout_adapter import conform
from signtranslator.data.multichannel import collate_multichannel
from tests.test_pose_resample import state


def channel_like(c, *, labels=None, frame=None, convention=None, valid=None):
    v = c.valid if valid is None else valid
    return replace(c, labels=labels or c.labels, coordinate_frame=frame or c.coordinate_frame,
                   convention=convention or c.convention, valid=v, observed=v & c.observed,
                   confidence=np.where(v, c.confidence, 0.), values=np.where(v[..., None], c.values, 0.))


def test_unavailable_channels_and_declared_rotation_frames_conform_and_collate():
    template = state(np.arange(6) / 24.)
    t = dict(template.channels)
    t["body"] = channel_like(t["body"], frame="two-view-triangulated-camera1-y-up", convention="sapiens-two-view-swing-ik-y-up-v1")
    template = replace(template, channels=t)
    mono = state(np.arange(4) / 30.)
    m = dict(mono.channels)
    m["body"] = channel_like(m["body"], frame="camera-aligned-y-up-metres", convention="mediapipe-tasks-swing-ik-y-up-v1")
    m["root_translation"] = channel_like(m["root_translation"], labels=("root_translation-unavailable",),
                                         frame="not-estimated", valid=np.zeros((4, 1), bool))
    mono = replace(mono, channels=m)
    with pytest.raises(ValueError):
        collate_multichannel([template, mono])
    fixed = conform(mono, template)
    batch = collate_multichannel([template, fixed])
    assert not batch.channels["root_translation"].valid[1].any() and batch.channels["body"].valid[1, :4].all()
    np.testing.assert_array_equal(fixed.channels["body"].values, mono.channels["body"].values)
    assert fixed.channels["body"].source_id == mono.channels["body"].source_id


def test_an_undeclared_layout_with_data_is_refused():
    template = state(np.arange(4) / 24.)
    other = dict(template.channels)
    other["body"] = channel_like(other["body"], frame="some-other-frame")
    with pytest.raises(ValueError, match="no declared equivalence"):
        conform(replace(template, channels=other), template)
