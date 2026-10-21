"""Referent loci as generator conditioning: from reference motion in training, from a planner at generation.

A clip's loci are its DP-vMF-means pointing clusters (``pointing``), ordered by the
first frame that points at each — the order in which referents are established. They
fill ``MAX_LOCI`` slots of (unit direction in the chest frame, present), flattened to
a ``SLOT_WIDTH`` vector that the generators project into the text encoder (zero-
initialised, so a model starts as its locus-free counterpart).

At generation there is no reference motion, so :func:`plan_loci` assigns slots from
the English with the contrastive policy of the research report (new third-person
referents alternate between the two sides; height only for iconic reasons) at the
azimuth and elevation of the loci measured on two-view How2Sign with outward pointing
(schema-2 grammar records, 2026-10-21: median 22.4° toward the signer's left, i.e.
contralateral for the 90% right-dominant signers, interquartile −6° to +40°; median
elevation +13.3°), mirrored for the second referent and further out for later ones.
These are the loci the locus-conditioned generator is trained on (``state_loci``), so
planned slots stay inside the training distribution. The earlier slots (25.5° right,
+22.6°) came from monocular states, where the dominant hand is misidentified in most
clips and index handshapes counted as points.
"""
from __future__ import annotations

import numpy as np

from .frames import body_frame_positions, dominant_side
from .pointing import loci_from_events, pointing_events
from .text import sentence_features

MAX_LOCI = 4
SLOT_WIDTH = 4 * MAX_LOCI
# (azimuth, elevation) degrees of planned slots; positive azimuth = signer's left (frames.py).
PLANNED_SLOTS = ((22.4, 13.3), (-22.4, 13.3), (50., 13.3), (-50., 13.3))


def direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])


def slots(directions) -> np.ndarray:
    """(SLOT_WIDTH,) flattened [x, y, z, present] × MAX_LOCI for up to MAX_LOCI unit directions."""
    out = np.zeros((MAX_LOCI, 4))
    for k, d in enumerate(list(directions)[:MAX_LOCI]):
        d = np.asarray(d, float)
        out[k, :3] = d / np.linalg.norm(d)
        out[k, 3] = 1.
    return out.ravel()


def state_loci(state) -> np.ndarray:
    """(SLOT_WIDTH,) slots of the loci the signer points at in ``state``, in order of establishment."""
    body = state.channels["body"]
    needed = {"Pelvis", "Chest", "LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm"}
    if not body.valid.any() or not needed <= set(body.labels):
        return slots([])
    joints, pos, ok = body_frame_positions(state)
    events = pointing_events(joints, pos, ok, state.timestamps, dominant_side(joints, pos, ok))
    loci = loci_from_events(events)
    loci.sort(key=lambda l: min(events[i].start for i in l.events))
    return slots([l.direction for l in loci])


def plan_loci(text: str) -> np.ndarray:
    """(SLOT_WIDTH,) planned slots: one per third-person referent (capped), or one deictic locus."""
    f = sentence_features(text)
    n = min(MAX_LOCI, int(f["third_person"]) or (1 if f["deixis"] else 0))
    return slots([direction(*PLANNED_SLOTS[k]) for k in range(n)])


__all__ = ["MAX_LOCI", "SLOT_WIDTH", "PLANNED_SLOTS", "direction", "slots", "state_loci", "plan_loci"]
