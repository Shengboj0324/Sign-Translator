"""Post-S2 A6's rest pose for the held left arm: the chordal mean of the left upper arm and forearm over ASLLRP train frames inside annotated one-handed signs, every signer with equal weight.
Local, read-only on stored clips, nothing trains and no reader runs. The test partition (Ben) is never read: a record outside train stops the run. The output is committed and pinned
(``reports/comp_a6_rest_pose.json``)."""
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

SEED = "comp-a6-rest-pose"
UTTERANCES_PER_SIGNER = 60
FRAMES_PER_SIGNER = 2000
TOLERANCE = .02                      # as ``round20_lib``: a state more than this off its annotation's frames cannot place a span
JOINTS = ("LeftUpperArm", "LeftForearm")


def one_handed_spans(utterance: dict) -> list[tuple[int, int]]:
    """Inclusive frames, from the utterance's first, of its annotated one-handed signs. A sign with no annotation is no span."""
    from comp_a_lib import annotated_label
    u0, u1 = (utterance.get("frames") or [None, None])[:2]
    out = []
    if None in (u0, u1):
        return out
    for s in utterance.get("signs") or ():
        a, b = (s.get("frames") or [None, None])[:2]
        if None in (a, b) or not u0 <= a <= b <= u1:
            continue
        if annotated_label(dict(gloss=dict(handshapes=s.get("handshapes")))) is True:
            out.append((int(a - u0), int(b - u0)))
    return out


def train_utterances(utterances, records: dict) -> dict:
    """{signer: [(utterance, record)]} of train utterances with one-handed spans. Ben and every other partition stay out, by the gloss manifest's own split; a Ben record marked train stops the run."""
    from build_asllrp_gloss_manifest import PARTITION, signer
    out = {}
    for u in utterances:
        video = u.get("utterance_video")
        if not video:
            continue
        name = signer(u.get("source") or "")
        record = records.get(f"{u['collection']}:{Path(video).stem}")
        if name == "Ben" and (record or {}).get("partition") == "train":
            raise SystemExit(f"{record['sign']}: Ben is the test partition, never read")
        if PARTITION.get(name) != "train":
            continue
        if record is None or record.get("partition") != "train" or not (record.get("continuous") or {}).get("state_key"):
            continue
        if one_handed_spans(u):
            out.setdefault(name, []).append((u, record))
    return {k: sorted(v, key=lambda x: x[1]["sign"]) for k, v in sorted(out.items())}


def span_frames(state, utterance: dict) -> np.ndarray:
    """State frames inside the utterance's one-handed spans, empty when the state cannot place them."""
    u0, u1 = utterance["frames"][:2]
    scale = len(state.timestamps) / (u1 - u0 + 1)
    if abs(scale - 1.) > TOLERANCE:
        return np.zeros(0, dtype=int)
    keep = set()
    for a, b in one_handed_spans(utterance):
        keep.update(range(int(round(a * scale)), min(int(round(b * scale)) + 1, len(state.timestamps))))
    return np.array(sorted(keep), dtype=int)


def pose_rows(state, frames: np.ndarray) -> dict:
    """{joint: (n, 3, 3)} rotations on ``frames`` where both left joints are tracked."""
    from signtranslator.generation.motion_metrics import _matrices
    body = state.channels["body"]
    labels = list(body.labels)
    index = [labels.index(j) for j in JOINTS]
    ok = body.valid[frames][:, index].all(1)
    use = frames[ok]
    return {j: _matrices(body.values[use, k].astype(np.float64)) for j, k in zip(JOINTS, index)}


def thin(rows: list, limit: int) -> np.ndarray:
    """Up to ``limit`` of a signer's rotations, evenly over its frames."""
    v = np.concatenate(rows)
    return v[np.linspace(0, len(v) - 1, min(limit, len(v))).astype(int)]


def mean_pose(per_signer: dict) -> dict:
    """{joint: 6D}: the chordal mean over signers' equal shares of frames."""
    from signtranslator.generation.composition_space import chordal_mean, to_6d
    if not per_signer:
        raise SystemExit("no ASLLRP train frame on an annotated one-handed sign")
    size = min(min(len(np.concatenate(rows[j])) for j in JOINTS) for rows in per_signer.values())
    if size < 1:
        raise SystemExit("a signer has no tracked frame")
    return {j: [round(float(x), 6) for x in to_6d(chordal_mean(np.concatenate([thin(rows[j], min(size, FRAMES_PER_SIGNER)) for rows in per_signer.values()])))] for j in JOINTS}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--output-file", required=True)
    args = p.parse_args(argv)
    if Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    from comp_a2_left_arm import keypoints
    from evaluate_text_to_motion import load_state
    import round15_stage2_fit as fit
    from round20_lib import GLOSS_MANIFEST
    from s2_lib import load_manifest_at
    from train_text_to_motion import make_store
    store = make_store(args.store)
    gloss = load_manifest_at(store, *GLOSS_MANIFEST)
    records = {r["sign"]: r for r in gloss["records"]}
    items = json.loads(store.get(fit.KEYS["utterances"]))["items"]
    per_signer, counts, heights = {}, {}, {}
    for name, pairs in train_utterances(items, records).items():
        rows = {j: [] for j in JOINTS}
        heights[name] = []
        used = frames_read = 0
        for u, record in random.Random(f"{SEED}|{name}").sample(pairs, min(UTTERANCES_PER_SIGNER, len(pairs))):
            state = load_state(store, record["continuous"]["state_key"], record["continuous"]["state_sha256"])
            frames = span_frames(state, u)
            if not len(frames):
                continue
            for j, R in pose_rows(state, frames).items():
                if len(R):
                    rows[j].append(R)
            points, valid = keypoints(state)
            keep = frames[frames < len(points)]
            keep = keep[valid[keep, 4]]
            heights[name].append(points[keep, 4, 1])
            used += 1
            frames_read += len(frames)
        if all(rows[j] for j in JOINTS):
            per_signer[name] = rows
        counts[name] = dict(utterances_available=len(pairs), utterances_read=used, frames=frames_read,
                            left_wrist_height_median=round(float(np.median(np.concatenate(heights[name]))), 4) if heights[name] and len(np.concatenate(heights[name])) else None)
    out = dict(kind="comp-a6-rest-pose", seed=SEED, inputs=dict(gloss_manifest=GLOSS_MANIFEST[0], sha256=GLOSS_MANIFEST[1], utterances=fit.KEYS["utterances"]), signers=counts,
               partition="train only; the test partition (Ben) is never read", pose=mean_pose(per_signer))
    data = json.dumps(out, indent=1, sort_keys=True).encode()
    with open(args.output_file, "xb") as stream:
        stream.write(data)
    print(args.output_file, hashlib.sha256(data).hexdigest(), json.dumps(counts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
