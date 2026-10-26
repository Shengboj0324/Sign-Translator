"""Per-signer dominant side from motion (round 5 amendment 2, for studio corpora such as RIT).

When both forearms are in view, tracking counts do not show which hand signs. This samples up to ``--per-signer``
clips of each signer (deterministic hash order). For each clip and side it takes the 90th percentile of the wrist's
height above its shoulder in the canonical skeleton (``canonical_skeleton.skeleton_positions``), over the frames
where that side's forearm is tracked: the signing hand is raised, the other rests. The signer's side is the one with
the higher median over clips. Mean wrist speed was tried first and rejected: on ASLLVD's right-dominant signers it
differed by only 5-20% between sides and called one signer left, while the height cue picked the right for all six,
by 0.05-0.19 in median. Writes ``{performer_sha256: {"side", "left_height", "right_height", "clips"}}``.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def wrist_heights(state) -> dict:
    """Per side: 90th percentile of wrist height above the shoulder over tracked-forearm frames (None if never)."""
    from signtranslator.generation.canonical_skeleton import skeleton_positions
    from canonicalize_asl_citizen import side_tracks
    joints, positions, _ = skeleton_positions(state)
    tracks = side_tracks(state)
    out = {}
    for side in ("Left", "Right"):
        ok = tracks[side][0]
        height = positions[:, joints.index(f"{side}Hand"), 1] - positions[:, joints.index(f"{side}Shoulder"), 1]
        out[side] = float(np.quantile(height[ok], .9)) if ok.any() else None
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest-key", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--per-signer", type=int, default=100)
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from evaluate_text_to_motion import load_state
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    by_signer = collections.defaultdict(list)
    for r in manifest["records"]:
        by_signer[r["performer_sha256"]].append(r)
    result = {}
    for signer, rows in sorted(by_signer.items()):
        rows = sorted(rows, key=lambda r: hashlib.sha256(r["sign"].encode()).hexdigest())[:args.per_signer]
        speeds = collections.defaultdict(list)
        for r in rows:
            sp = wrist_heights(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]))
            for side, v in sp.items():
                if v is not None:
                    speeds[side].append(v)
        left, right = (float(np.median(speeds[s])) if speeds[s] else -9. for s in ("Left", "Right"))
        result[signer] = dict(side="Left" if left > right else "Right", left_height=round(left, 4),
                              right_height=round(right, 4), clips=len(rows))
        print(json.dumps({signer[:8]: result[signer]}), flush=True)
    store.put(args.output_key, json.dumps(result, indent=2, sort_keys=True).encode())
    print(json.dumps(dict(output_key=args.output_key, signers=len(result),
                          sides=dict(collections.Counter(v["side"] for v in result.values())))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
