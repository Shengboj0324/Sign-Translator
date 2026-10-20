"""All grammar measurements of one canonical state, in one serialisable record."""
from __future__ import annotations

import numpy as np

from ..pose.multichannel import MultichannelMotion
from .frames import body_frame_positions, dominant_side, to_body_frame
from .nmm import head_gestures, nmm_tracks, span_fraction
from .pointing import frame_speed, loci_from_events, pointing_events
from .prosody import active_range, blink_onsets, boundary_evidence, final_lengthening, movement_units
from .role_shift import role_shifts

SCHEMA_VERSION = 1


def fps_of(state: MultichannelMotion) -> float:
    return float(1. / np.median(np.diff(state.timestamps))) if len(state.timestamps) > 1 else 24.


def gaze_of(state: MultichannelMotion):
    """(T, 3) cyclopean gaze in the camera-aligned frame, NaN where unavailable."""
    l, r = state.channels["left_gaze"], state.channels["right_gaze"]
    both = l.valid[:, 0] & r.valid[:, 0]
    if not both.any():
        return None
    g = l.values[:, 0] + r.values[:, 0]
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    return np.where(both[:, None], g, np.nan)


def grammar_record(state: MultichannelMotion, nmm_baselines: dict | None = None) -> dict:
    """JSON-ready grammar measurements: NMM spans, head gestures, pointing loci, role shifts, prosody.

    ``nmm_baselines`` (``nmm.pooled_baselines`` over the signer's or the reference set's
    clips) should be given whenever more than one clip is available; without it each
    clip is its own baseline.
    """
    fps, frames = fps_of(state), len(state.timestamps)
    record = dict(schema_version=SCHEMA_VERSION, sample_id=state.sample_id, frames=frames, fps=fps)
    face = state.channels["face"]
    nmm = None
    if face.valid.any() and "browInnerUp" in face.labels:
        nmm = nmm_tracks(face, fps, nmm_baselines)
        record["nmm"] = {k: dict(spans=v, fraction=span_fraction(v, frames), peak_z=float(np.max(nmm.z[k])),
                                 mean_z=float(np.mean(nmm.z[k][nmm.valid])) if nmm.valid.any() else 0.)
                         for k, v in nmm.spans.items()}
    head = state.channels["head_rotation"]
    gestures = None
    if head.valid.any() and head.values.shape[1] == 1:
        gestures = head_gestures(head, fps)
        record["head"] = dict(headshake=gestures["headshake"], nod=gestures["nod"])
    body = state.channels["body"]
    gaze_cam = gaze_of(state)
    canonical = {"Pelvis", "Chest", "LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm"} <= set(body.labels)
    if body.valid.any() and canonical:
        joints, pos, ok = body_frame_positions(state)
        side = dominant_side(joints, pos, ok)
        events = pointing_events(joints, pos, ok, state.timestamps, side)
        loci = loci_from_events(events)
        record["pointing"] = dict(dominant=side, events=[(e.start, e.stop) for e in events],
                                  loci=[dict(azimuth_deg=l.azimuth_deg, elevation_deg=l.elevation_deg,
                                             events=list(l.events), point=None if l.point is None else l.point.tolist())
                                        for l in loci])
        gaze_body = None
        if gaze_cam is not None:
            gaze_body = to_body_frame(state, np.nan_to_num(gaze_cam))
            gaze_body = np.where(np.isfinite(gaze_cam), gaze_body, np.nan)
        shifts, report = role_shifts(state, fps, gaze_body)
        record["role_shift"] = dict(spans=[(r.start, r.stop, r.direction) for r in shifts],
                                    fraction=span_fraction([(r.start, r.stop) for r in shifts], frames),
                                    change_points=report["change_points"], features=list(report["features"]))
        both = ok[:, joints.index("LeftHand")] & ok[:, joints.index("RightHand")]
        speed = (frame_speed(pos[:, joints.index("LeftHand")], state.timestamps)
                 + frame_speed(pos[:, joints.index("RightHand")], state.timestamps))
        heights = np.stack([pos[:, joints.index("LeftHand"), 1], pos[:, joints.index("RightHand"), 1]], 1)
        units = movement_units(speed, both, fps, span=active_range(heights, both))
        blink = state.channels["blink"]
        blinks = blink_onsets(blink.values[:, :, 0].mean(1), blink.valid.all(1), fps) if blink.valid.any() else []
        brow = nmm.z["brow_raise"] if nmm is not None else np.zeros(frames)
        yaw = gestures["yaw_deg"] if gestures is not None else np.zeros(frames)
        evidence = boundary_evidence(units, blinks, brow, yaw, final_window=int(round(.5 * fps)))
        record["prosody"] = dict(units=units, blinks=blinks, boundaries=[e["unit"] for e in evidence if e["boundary"]],
                                 final_lengthening=final_lengthening(units),
                                 final_blink=bool(evidence and evidence[-1]["blink"]))
    return record


__all__ = ["SCHEMA_VERSION", "grammar_record", "fps_of", "gaze_of"]
