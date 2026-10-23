"""Canonical-state manifest -> 2D keypoint shards through forward kinematics (``data.keypoints2d.from_state``).

The 2D back-translator is fine-tuned on these, and generated motion reaches it the same way, so fine-tuning and
judging share one domain. Each record's continuous state is read SHA-checked, resampled to ``--fps``, projected
and normalised (y up, hands local). Shards have the same format as ``build_uwb_keypoint_shards.py``, one series per
partition, written as ``<out-prefix>/<partition>/shard-NNNNN.npz``. The test partition is written too, so the final
check can read it, but trainers never read it.
"""
from __future__ import annotations

import argparse
import collections
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.keypoints2d import from_state, normalise  # noqa: E402

_WORKER = {}


def _init(store_spec: str, fps: float) -> None:
    from train_text_to_motion import make_store
    _WORKER.update(store=make_store(store_spec), fps=fps)


def convert(record: dict):
    from evaluate_text_to_motion import at_rate, load_state
    c = record["continuous"]
    try:
        state = at_rate(load_state(_WORKER["store"], c["state_key"], c["state_sha256"]), _WORKER["fps"])
        points, valid = from_state(state)
    except (KeyError, ValueError) as error:
        return record["sign"], None, str(error)[:200]
    return record["sign"], (normalise(points, valid, y_down=False), valid), None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--partition", action="append", default=[], help="default: train, val, test, heldout_official")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="records per partition (testing)")
    args = parser.parse_args(argv)
    from build_uwb_keypoint_shards import _write_shard
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from train_text_to_motion import make_store
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    parts, why = [], collections.Counter()
    with Pool(args.workers, initializer=_init, initargs=(args.store, args.fps)) as pool:
        for partition in args.partition or ["train", "val", "test", "heldout_official"]:
            records = [r for r in manifest["records"] if r.get("partition") == partition
                       and (r.get("continuous") or {}).get("status") == "converted" and (r.get("english") or r.get("text"))]
            if args.limit:
                records = records[:args.limit]
            by_sign = {r["sign"]: r for r in records}
            shards, pending = [], []
            for sign, result, error in pool.imap(convert, records, chunksize=16):
                if result is None:
                    why["unreadable"] += 1; print(f"{sign}: {error}", flush=True); continue
                points, valid = result
                if not (valid[:, 0] & valid[:, 1]).any():
                    why["no_shoulders"] += 1; continue
                r = by_sign[sign]
                pending.append(dict(clip_id=sign, video_id=sign.split("_")[0], text=r.get("english") or r["text"],
                                    partition=partition, frames=len(points), points=points, valid=valid))
                why[f"kept_{partition}"] += 1
                if len(pending) == args.shard_size:
                    shards.append(_write_shard(store, f"{args.out_prefix}/{partition}/shard-{len(shards):05d}.npz", pending))
                    pending = []
            if pending:
                shards.append(_write_shard(store, f"{args.out_prefix}/{partition}/shard-{len(shards):05d}.npz", pending))
            parts.append(dict(partition=partition, shards=shards))
    index = dict(corpus_id=args.corpus_id, source_fps=args.fps, manifest_key=args.manifest_key,
                 manifest_sha256=args.manifest_sha256, layout="signtranslator.data.keypoints2d (48 points, hands local)",
                 projection="canonical_skeleton FK, orthographic front (x, y of the camera-aligned y-up frame)",
                 counts=dict(why), parts=parts)
    store.put(f"{args.out_prefix}/index.json", json.dumps(index, indent=1, sort_keys=True).encode())
    print(json.dumps(dict(why)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
