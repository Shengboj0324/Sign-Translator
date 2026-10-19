"""Shared intrinsics of How2Sign's two studio cameras, from many sessions at once.

One How2Sign session is often too short and its lateral-view keypoints too noisy for
the bone-constrained self-calibration to pin down both focal lengths (the cameras
fixate the signer, so focal lengths are not identifiable from epipolar geometry
alone). The green-screen studio's frontal and lateral cameras are the same for every
session, so their intrinsics are estimated jointly over a deterministic sample of
sessions (each with its own relative pose: the cameras may have been moved). The
result is written once with per-session diagnostics; ``triangulate_how2sign_sessions.py
--intrinsics-key`` then fits only each session's pose, and the declared gate still
refuses any session the shared intrinsics do not fit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402
from signtranslator.data_engineering.two_view_states import calibrate_cameras  # noqa: E402
from triangulate_how2sign_sessions import discover_sessions, synchronised  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path> for a local mirror")
    parser.add_argument("--front-keypoints", required=True)
    parser.add_argument("--side-keypoints", required=True)
    parser.add_argument("--sessions", type=int, default=40, help="sessions sampled evenly from the sorted list")
    parser.add_argument("--max-frames", type=int, default=600, help="frames per session in the joint fit")
    parser.add_argument("--output-key", required=True)
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    if store.exists(args.output_key):
        raise SystemExit("refusing to overwrite existing intrinsics")
    sessions = discover_sessions(store, args.front_keypoints, args.side_keypoints)
    names = sorted(sessions)
    step = max(1, len(names) // args.sessions)
    chosen, used = names[::step][:args.sessions], []
    for name in chosen:
        sync = synchronised(store, sessions[name])
        if sync["aligned"]:
            used.append((name, sync["lag"], [a[4][1:] for a in sync["aligned"]]))
        print(f"{name}: lag {sync.get('lag')} clips {len(sync['aligned'])} {sync['reason'] or ''}", flush=True)
    if len(used) < 2:
        raise SystemExit("fewer than two synchronised sessions; nothing to calibrate jointly")
    calibrations = calibrate_cameras([pairs for _, _, pairs in used], max_frames=args.max_frames)
    shared = calibrations[0]
    record = dict(
        schema_version=1, method="calibrate_jointly: shared focal lengths and division-model radial coefficients, "
                                  "per-session relative pose, bone-constrained bundle adjustment",
        intrinsics=dict(focal1=shared.focal1, focal2=shared.focal2, distortion=list(shared.distortion)),
        inputs=dict(front_keypoints=args.front_keypoints, side_keypoints=args.side_keypoints,
                    sessions_sampled=len(chosen), sessions_used=len(used), max_frames=args.max_frames),
        sessions=[dict(session=name, lag_frames=lag, reprojection_px=list(c.median_reprojection_px),
                       bone_length_cv=c.bone_length_cv, inlier_fraction=c.inlier_fraction, converged=c.converged)
                  for (name, lag, _), c in zip(used, calibrations)])
    payload = json.dumps(record, indent=2, sort_keys=True).encode()
    digest = store.put(args.output_key, payload)
    print(json.dumps(dict(output_key=args.output_key, sha256=digest, **record["intrinsics"],
                          sessions=len(used)), default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
