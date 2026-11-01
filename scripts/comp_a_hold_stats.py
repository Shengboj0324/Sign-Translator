"""Post-S2 A1's ``hold`` arm: the numbers it takes from real signing. Local, read-only on stored clips, nothing trains, no reader runs.
The output is committed and pinned by its hash (``reports/comp_a_hold_stats.json``).

A fixed draw of How2Sign train clips (never val, so not D20, C20 or the 1,000; c00 refused), and of ASLLRP train utterances for one height:
- ``left_speed_p99``: the left wrist's frame-to-frame speed in the chest frame (metres per second), p99 over pooled valid steps. It caps the
  hold's left arm.
- ``ready_pose``: the chordal mean of the left upper arm and forearm rotations (6D columns), for a sentence with no two-handed word.
- Per side, in the report's own keypoint view (shoulder units): speed, spread, height, and travel over ``WINDOW`` frames, the length of a
  bridge at A1's operating point. These are the reference the hold is read against. The right wrist is not one: real signing has the same
  left/right asymmetry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 24.
O = "acquisition/2026-10-17-v1/youtube-asl-v1"
EVAL_MIX = f"{O}/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
H2S = "how2sign-two-view-text-v1"
CLIPS, UTTERANCES, FRAMES_PER_CLIP = 400, 120, 24
WINDOW = 7
SEED = "comp-a-hold-stats"
WRISTS = dict(left=4, right=5)
LEFT_JOINTS = ("LeftUpperArm", "LeftForearm")


def train_clips(manifest, n: int = CLIPS, seed: str = SEED) -> list[dict]:
    """A fixed draw of How2Sign train records with a state. c00 is refused, never skipped."""
    pool = []
    for r in manifest["records"]:
        if r.get("corpus") != H2S or r.get("partition") != "train":
            continue
        if r.get("signer_component") == "c00":
            raise SystemExit(f"{r.get('sign')}: How2Sign signer 9 (c00) is never read")
        if (r.get("continuous") or {}).get("state_key"):
            pool.append(r)
    pool.sort(key=lambda r: r["sign"])
    return random.Random(seed).sample(pool, min(n, len(pool)))


def percentiles(values) -> dict:
    v = np.asarray(values, dtype=float)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), mean=round(float(v.mean()), 4), p50=round(float(np.percentile(v, 50)), 4),
                p90=round(float(np.percentile(v, 90)), 4), p99=round(float(np.percentile(v, 99)), 4))


def view_rows(state) -> dict:
    """Per side in the keypoint view: steps' speed, window travels, heights, and the clip's speed and spread."""
    from comp_a2_left_arm import keypoints
    points, valid = keypoints(state)
    out = {}
    for side, w in WRISTS.items():
        ok = valid[1:, w] & valid[:-1, w]
        steps = np.linalg.norm(points[1:, w] - points[:-1, w], axis=-1)[ok] * FPS
        both = valid[WINDOW:, w] & valid[:-WINDOW, w] if len(valid) > WINDOW else np.zeros(0, bool)
        travel = np.linalg.norm(points[WINDOW:, w] - points[:-WINDOW, w], axis=-1)[both] if len(both) else np.zeros(0)
        seen = valid[:, w]
        out[side] = dict(steps=steps, travel=travel, heights=points[seen, w][:, 1] if seen.any() else np.zeros(0),
                         speed=float(steps.mean()) if len(steps) else None,
                         spread=float(points[seen, w].std(0).mean()) if seen.sum() >= 2 else None)
    return out


def chest_steps(state, side: str) -> np.ndarray:
    from signtranslator.generation.composition_space import wrist_from_shoulder
    p, ok = wrist_from_shoulder(state, side)
    good = ok[1:] & ok[:-1]
    return np.linalg.norm(p[1:] - p[:-1], axis=-1)[good] * FPS


def left_rotations(state) -> dict:
    """{joint: valid (T, 3, 3) left-arm rotations}."""
    from signtranslator.generation.motion_metrics import _matrices
    body = state.channels["body"]
    labels = list(body.labels)
    return {j: _matrices(body.values[body.valid[:, labels.index(j)], labels.index(j)].astype(np.float64)) for j in LEFT_JOINTS}


def summarise(states) -> dict:
    """The file's numbers from an iterable of How2Sign train states."""
    from signtranslator.generation.composition_space import chordal_mean, to_6d
    pooled = {s: dict(steps=[], travel=[], heights=[], speed=[], spread=[]) for s in WRISTS}
    chest, rotations = {"Left": [], "Right": []}, {j: [] for j in LEFT_JOINTS}
    count = 0
    for state in states:
        count += 1
        for side, row in view_rows(state).items():
            for k in ("steps", "travel", "heights"):
                pooled[side][k].append(row[k])
            for k in ("speed", "spread"):
                if row[k] is not None:
                    pooled[side][k].append(row[k])
        for side in chest:
            chest[side].append(chest_steps(state, side))
        for j, R in left_rotations(state).items():
            if len(R):
                pick = np.linspace(0, len(R) - 1, min(FRAMES_PER_CLIP, len(R))).astype(int)
                rotations[j].append(R[pick])
    if not count or not all(rotations.values()):
        raise SystemExit("no How2Sign train clip with a tracked left arm")
    sides = {}
    for side, row in pooled.items():
        sides[side] = dict(step_speed=percentiles(np.concatenate(row["steps"])), travel_over_window=percentiles(np.concatenate(row["travel"])),
                           height=percentiles(np.concatenate(row["heights"])), clip_speed_mean=round(float(np.mean(row["speed"])), 4),
                           clip_spread_mean=round(float(np.mean(row["spread"])), 4))
    chest_left, chest_right = np.concatenate(chest["Left"]), np.concatenate(chest["Right"])
    return dict(clips=count, window_frames=WINDOW, sides=sides,
                right_over_left_clip_speed=round(sides["right"]["clip_speed_mean"] / sides["left"]["clip_speed_mean"], 4),
                chest_speed=dict(left=percentiles(chest_left), right=percentiles(chest_right)),
                left_speed_p99=round(float(np.percentile(chest_left, 99)), 4),
                ready_pose={j: [round(float(x), 6) for x in to_6d(chordal_mean(np.concatenate(v)))] for j, v in rotations.items()})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--output-file", required=True)
    args = p.parse_args(argv)
    if Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    from evaluate_text_to_motion import at_rate, load_state
    from round20_lib import GLOSS_MANIFEST
    from s2_lib import load_manifest_at
    from train_text_to_motion import make_store
    store = make_store(args.store)
    manifest = load_manifest_at(store, EVAL_MIX, EVAL_MIX_SHA)
    records = [dict(sign=r["sign"], key=r["continuous"]["state_key"], sha=r["continuous"]["state_sha256"]) for r in train_clips(manifest)]
    del manifest

    def states():
        for k, r in enumerate(records):
            yield at_rate(load_state(store, r["key"], r["sha"]), FPS)
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(records)} clips", flush=True)
    out = dict(kind="comp-a-hold-stats", seed=SEED, inputs=dict(key=EVAL_MIX, sha256=EVAL_MIX_SHA), how2sign_train=summarise(states()))
    gloss = load_manifest_at(store, *GLOSS_MANIFEST)
    train = sorted((r for r in gloss["records"] if r.get("partition") == "train" and (r.get("continuous") or {}).get("state_key")), key=lambda r: r["sign"])
    heights, speeds = [], []
    for r in random.Random(SEED).sample(train, min(UTTERANCES, len(train))):
        view = view_rows(at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), FPS))["left"]
        heights.append(view["heights"])
        if view["speed"] is not None:
            speeds.append(view["speed"])
    out["asllrp_train_left"] = dict(utterances=len(heights), height=percentiles(np.concatenate(heights)), clip_speed_mean=round(float(np.mean(speeds)), 4))
    data = json.dumps(out, indent=1, sort_keys=True).encode()
    with open(args.output_file, "xb") as stream:
        stream.write(data)
    print(args.output_file, hashlib.sha256(data).hexdigest(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
