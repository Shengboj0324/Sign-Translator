"""Explicit layout adapter for training on mixed corpora (How2Sign two-view + ASLLRP / YouTube-ASL monocular).

``collate_multichannel`` refuses to batch states whose channel layouts (labels, coordinate frame, units, convention)
differ, and it should: a silent mix would train on incompatible numbers. Corpora estimated by different pipelines
differ in two ways that are safe to bridge, and only these are bridged here:

1. **An unavailable channel.** Monocular states carry ``root_translation`` / ``head_translation`` / gaze as explicit
   "unavailable" channels (no valid frame, their own placeholder labels). Such a channel is replaced by the template's
   layout with no valid frame, so the values are all zero and confidence zero, exactly the canonical absent payload.
2. **Declared-equivalent rotation layouts.** The body and hand rotations of both pipelines are the same joints, as
   swing-IK 6D rotations, y up, in camera 1's frame. Only the landmark source differs: Sapiens two-view triangulation
   or MediaPipe monocular world landmarks. ``EQUIVALENT`` lists exactly these (frame, convention) pairs; a valid
   channel whose labels and units match and whose (frame, convention) is declared equivalent takes the template's
   frame and convention names. Its values, validity, confidence and provenance (``source_id``, ``source_sha256``)
   are kept.

Anything else - different joint labels, units, or an undeclared frame or convention on a channel that has data - is
refused, so a new pipeline must be declared here before it can be mixed.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

# (template frame, template convention) <- (source frame, source convention), both swing-IK camera-1 y-up rotations
EQUIVALENT = {
    ("two-view-triangulated-camera1-y-up", "sapiens-two-view-swing-ik-y-up-v1"):
        {("camera-aligned-y-up-metres", "mediapipe-tasks-swing-ik-y-up-v1")},
}


def _absent_like(template, frames: int, source):
    shape = (frames,) + template.values.shape[1:]
    return replace(template, values=np.zeros(shape, template.values.dtype),
                   valid=np.zeros(shape[:2], bool), observed=np.zeros(shape[:2], bool),
                   confidence=np.zeros(shape[:2], template.confidence.dtype),
                   source_id=source.source_id, source_sha256=source.source_sha256, inference_method=None)


def conform(state, template):
    """``state`` in ``template``'s channel layouts, under the two rules above; raises on anything else."""
    frames = len(state.timestamps)
    channels = {}
    for name, ref in template.channels.items():
        c = state.channels[name]
        same = (c.labels, c.coordinate_frame, c.units, c.convention) == (ref.labels, ref.coordinate_frame, ref.units,
                                                                          ref.convention)
        if same:
            channels[name] = c
        elif not c.valid.any():
            channels[name] = _absent_like(ref, frames, c)
        elif (c.labels == ref.labels and c.units == ref.units and c.values.shape[1:] == ref.values.shape[1:]
              and (c.coordinate_frame, c.convention) in EQUIVALENT.get((ref.coordinate_frame, ref.convention), set())):
            channels[name] = replace(c, coordinate_frame=ref.coordinate_frame, convention=ref.convention)
        else:
            raise ValueError(f"{name}: layout ({c.coordinate_frame}, {c.convention}, {len(c.labels)} labels) has no "
                             f"declared equivalence to ({ref.coordinate_frame}, {ref.convention})")
    return replace(state, channels=channels)


__all__ = ["EQUIVALENT", "conform"]
