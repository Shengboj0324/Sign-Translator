"""Two-view How2Sign states per recording session, admitted only through the gate.

Inputs (object store): Sapiens keypoints for frontal and lateral clips, and the
monocular MediaPipe states and landmarks for the frontal clips. For each session
(one source video recorded by both cameras): synchronise every clip pair, calibrate
the camera pair once with the bone-constrained bundle adjustment, triangulate, and
evaluate the declared gate (reprojection, synchronisation, triangulated vs monocular
bone-length constancy). Passing sessions get canonical states (triangulated body and
hands, MediaPipe face/head/blink on the same frontal frames); every session's gate
result is reported, pass or fail. Reports use the extraction schema, so
``build_how2sign_text_manifest.py`` reads them directly.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signtranslator.data.text_motion_dataset import R2Store  # noqa: E402
from signtranslator.data_engineering.two_view_states import (  # noqa: E402
    ClipKeypoints, aligned, calibrate_session, epipolar_synchronise, monocular_bone_cv, refined_points,
    rotation_inputs, session_gate,
    symmetry_error, synchronise, triangulated_bone_cv, triangulated_points, triangulated_state,
)
from signtranslator.data_engineering.video_landmarks import solve_rotations  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402

CLIP = re.compile(r"(?P<video>.+)_(?P<sentence>\d+)-(?P<camera>\d+)-rgb_(?P<view>front|side)")


def session_of(name: str) -> tuple[str, str, str] | None:
    match = CLIP.fullmatch(name)
    if not match:
        return None
    return f"{match['video']}-{match['camera']}", f"{match['video']}_{match['sentence']}-{match['camera']}", match["view"]


def load_keypoints(store, key, name) -> ClipKeypoints:
    with np.load(io.BytesIO(store.get(key)), allow_pickle=False) as data:
        return ClipKeypoints(name, data["times"], data["points"].astype(np.float64),
                             data["confidence"].astype(np.float64))


def load_state(store, key) -> tuple[MultichannelMotion, str]:
    payload = store.get(key)
    digest = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "s.npz"
        path.write_bytes(payload)
        return MultichannelMotion.load(path, expected_sha256=digest), digest


def discover_sessions(store, front_prefix: str, side_prefix: str) -> dict:
    """session -> sentence -> {"front": key, "side": key} from the keypoint prefixes."""
    sessions = collections.defaultdict(dict)
    for prefix, view in ((front_prefix, "front"), (side_prefix, "side")):
        for key in store.list(prefix.rstrip("/") + "/keypoints/"):
            parsed = session_of(Path(key).stem)
            if parsed and parsed[2] == view:
                sessions[parsed[0]].setdefault(parsed[1], {})[view] = key
    return sessions


def synchronised(store, sentences: dict) -> dict:
    """Load a session's clip pairs, find its view lag by epipolar geometry, align accepted clips."""
    clips = [(sentence, load_keypoints(store, keys["front"], sentence), load_keypoints(store, keys["side"], sentence))
             for sentence, keys in sorted(sentences.items()) if set(keys) == {"front", "side"}]
    if not clips:
        return dict(reason="no clip seen by both views", syncs=[], aligned=[], epipolar=None)
    epipolar = epipolar_synchronise(clips)
    lag = epipolar["lag_frames"]
    velocity = {s: synchronise(f, sd) for s, f, sd in clips}      # recorded for comparison only
    finite = lambda x: x if np.isfinite(x) else None                 # strict JSON: no infinities
    syncs = [dict(clip, sampson_px=finite(clip["sampson_px"]), lag_frames=lag,
                  velocity_lag_frames=velocity[clip["sentence"]]["lag_frames"],
                  velocity_correlation=finite(velocity[clip["sentence"]]["correlation"]))
             for clip in epipolar["clips"]]
    pairs = [(s, f, sd, lag, aligned(f, sd, lag)) for (s, f, sd), clip in zip(clips, epipolar["clips"]) if clip["passed"]]
    reason = None if pairs else (epipolar["reason"] or "no clip passed epipolar synchronisation")
    return dict(reason=reason, syncs=syncs, aligned=pairs, epipolar=epipolar, lag=lag)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-keypoints", required=True, help="prefix holding keypoints/<clip>.npz (frontal)")
    parser.add_argument("--side-keypoints", required=True, help="prefix holding keypoints/<clip>.npz (lateral)")
    parser.add_argument("--monocular", required=True, help="prefix holding states/ and landmarks/ (frontal)")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--intrinsics-key", help="shared camera intrinsics from calibrate_how2sign_cameras.py")
    parser.add_argument("--intrinsics-sha256", help="required with --intrinsics-key")
    args = parser.parse_args(argv)
    store = R2Store(os.environ)
    intrinsics = None
    if args.intrinsics_key:
        payload = store.get(args.intrinsics_key)
        if hashlib.sha256(payload).hexdigest() != args.intrinsics_sha256:
            raise SystemExit("the intrinsics file differs from --intrinsics-sha256")
        shared = json.loads(payload)["intrinsics"]
        intrinsics = (shared["focal1"], shared["focal2"], *shared["distortion"])
    sessions = discover_sessions(store, args.front_keypoints, args.side_keypoints)
    names = sorted(sessions)
    names = [n for i, n in enumerate(names) if i % args.shards == args.shard]
    results, gates = [], {}
    for session in names:
        session_sync = synchronised(store, sessions[session])
        syncs, aligned_pairs = session_sync["syncs"], session_sync["aligned"]
        if not aligned_pairs:
            gates[session] = dict(passed=False, reasons=[session_sync["reason"]], sync=syncs,
                                  epipolar_curve=(session_sync["epipolar"] or {}).get("curve"))
            continue
        epipolar, lag = session_sync["epipolar"], session_sync["lag"]
        calibration = calibrate_session([a[4][1:] for a in aligned_pairs], intrinsics=intrinsics)
        triangulated, mono_cvs = {}, []
        for sentence, front, side, lag, (times, fp, fc, sp, sc) in aligned_pairs:
            points, usable = triangulated_points(calibration, fp, fc, sp, sc)
            points = refined_points(calibration, points, usable, fp, fc, sp, sc)
            triangulated[sentence] = (points, usable, max(0, -lag), len(times))
            landmarks_key = f"{args.monocular.rstrip('/')}/landmarks/{sentence}-rgb_front.npz"
            if store.exists(landmarks_key):
                with np.load(io.BytesIO(store.get(landmarks_key)), allow_pickle=False) as mono:
                    mono_cvs.append(monocular_bone_cv(mono["pose_world"], mono["pose_visibility"]))
        all_points = np.concatenate([t[0] for t in triangulated.values()])
        tri_cv = triangulated_bone_cv(all_points)
        mono_cv = float(np.nanmedian(mono_cvs)) if mono_cvs else None
        gate = session_gate(calibration, syncs, tri_cv, mono_cv)
        gate.update(sync=syncs, lag_frames=lag, epipolar_sampson_px=epipolar["session_sampson_px"],
                    epipolar_curve=epipolar["curve"], shared_intrinsics=args.intrinsics_key, triangulated_bone_cv=tri_cv, monocular_bone_cv=mono_cv,
                    symmetry_error=symmetry_error(all_points), focal=[calibration.focal1, calibration.focal2],
                    reprojection_px=list(calibration.median_reprojection_px))
        gates[session] = gate
        if not gate["passed"]:
            continue
        for sentence, (points, usable, start, count) in triangulated.items():
            state_key = f"{args.monocular.rstrip('/')}/states/{sentence}-rgb_front.npz"
            if not store.exists(state_key):
                continue
            monocular, mono_sha = load_state(store, state_key)
            inputs = rotation_inputs(points, usable)
            rotations, hands = solve_rotations(inputs["world"], inputs["visible"], inputs["hands"])
            state = triangulated_state(monocular, slice(start, start + count), rotations, hands,
                                       source_id="how2sign_two_view", source_sha256=mono_sha,
                                       inference_method=("Sapiens-1B Goliath on frontal and lateral views, "
                                                         "bone-constrained two-view self-calibration, rigid smooth hand fit, "
                                                         "swing IK; "
                                                         "face/head/blink MediaPipe; values are estimates"))
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "s.npz"
                digest = state.save(path)
                key = f"{args.prefix.rstrip('/')}/states/{sentence}-rgb_front.npz"
                store.put(key, path.read_bytes())
            results.append(dict(sample_id=f"{sentence}-rgb_front", status="converted", state_key=key,
                                state_sha256=digest, seconds=float(state.timestamps[-1] - state.timestamps[0]),
                                available_channels=[n for n, c in state.channels.items() if c.valid.any()],
                                session=session))
    report = dict(schema_version=1, shard=args.shard, shards=args.shards, sessions=len(names),
                  sessions_passed=sum(g["passed"] for g in gates.values()), gates=gates, results=results)
    store.put(f"{args.prefix.rstrip('/')}/reports/shard-{args.shard:03d}-of-{args.shards:03d}.json",
              json.dumps(report, sort_keys=True, default=float).encode())
    print(json.dumps(dict(sessions=len(names), passed=report["sessions_passed"], states=len(results))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
