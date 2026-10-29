"""ASL Citizen pixel-side check (recogniser v3; docs/NEXT_PHASE_PLAN.md, "Instrument finding and the recogniser v3
plan": "Image-right ASL Citizen signers are excluded after a pixel-side check"). Report-only, CPU.

ASL Citizen's stored landmarks hold pose points only (no image-space hands), so the hand landmarker is run again on the
original videos: ``--clips`` (about 150) **train and val** clips of the pinned manifest (the test partition is never
read: ``recognizer_corpora.check_loadable``), spread over signers (``tokenizer_domain_read.signer_spread_sample``:
signers in a seeded order, round-robin), each read from the verified ``ASL_Citizen.zip`` by ranged reads
(``extract_video_states.StoreRangeFile``) and passed through ``video_landmarks.extract_clip`` exactly as the corpus
was extracted (ROI crops, hand crops of 0.9 shoulder widths).

**Measure** (the Sem-Lex pixel-space measure, docs/NEXT_PHASE_PLAN.md "Sem-Lex orientation test" and "Sem-Lex
per-signer sides"): per clip, the image half (normalised image x < 0.5: ``image_left``) of the more active hand
detection (``active_half``): every detected hand is put in the half of its wrist's image x on that frame; a half's
motion is the summed frame-to-frame displacement, averaged over the 21 hand points, over consecutive frames with a
detection in that half; the clip's side is the half with more motion, undecided when the halves together have fewer
than ``MIN_PAIRS`` consecutive detected pairs or the motions tie. A signer's side is the majority of their decided
clips (undecided on a tie or none). Natural right-handed video puts the active hand on the image **left** (YouTube-ASL
baseline: 72 of 100 clips); a signer whose side is ``image_right`` is listed for exclusion, with the
``--exclude-signers`` spec the recogniser takes (``performer_sha256`` prefixes of 16 hex digits).
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

MANIFEST = "acquisition/2026-10-17-v1/derived/asl-citizen-v1/manifest-v1/manifest.json"
MANIFEST_SHA256 = "708af3a6748742c0ce8d147a984ef1de57abcec94da47ff0ce82391b067661a6"
ZIP_KEY = "acquisition/2026-10-17-v1/asl-citizen-v1/ASL_Citizen.zip"
ZIP_SHA256 = "349c0e33f13b876f19ad4dee72205f4d2eae02937613e275200aa92283268eab"
CORPUS = "asl-citizen-video-estimate-v1"
MIN_PAIRS = 5
HALVES = ("image_left", "image_right")
PREFIX_DIGITS = 16


# --------------------------------------------------------------------------------------------- pixel-space side

def half_motion(hands: list[list[np.ndarray]]) -> dict:
    """``hands[t]``: the (21, 2) normalised image points of every hand detected on frame t. Per half: consecutive
    detected pairs and the summed mean point displacement over them (the first detection of a half per frame)."""
    tracks = {h: [None] * len(hands) for h in HALVES}
    for t, detections in enumerate(hands):
        for points in detections:
            half = "image_left" if points[0, 0] < .5 else "image_right"
            if tracks[half][t] is None:
                tracks[half][t] = np.asarray(points, dtype=np.float64)
    out = {}
    for half, track in tracks.items():
        pairs = [(a, b) for a, b in zip(track[:-1], track[1:]) if a is not None and b is not None]
        out[half] = dict(pairs=len(pairs), motion=float(sum(np.linalg.norm(b - a, axis=1).mean() for a, b in pairs)),
                         frames=sum(p is not None for p in track))
    return out


def active_half(motion: dict) -> str | None:
    a, b = motion["image_left"], motion["image_right"]
    if a["pairs"] + b["pairs"] < MIN_PAIRS or a["motion"] == b["motion"]:
        return None
    return "image_left" if a["motion"] > b["motion"] else "image_right"


def clip_hands(clip) -> list[list[np.ndarray]]:
    """Per frame, the detected hands' (21, 2) image points of a ``ClipLandmarks``."""
    found = np.asarray(clip.hands_found, dtype=bool)
    return [[np.asarray(clip.hands_image[t, k], dtype=np.float64) for k in np.flatnonzero(found[t])]
            for t in range(len(found))]


def group_sides(rows, group: str = "signer") -> dict:
    """{signer: dict(side, image_left, image_right, undecided)}: the majority of the decided clips."""
    by: dict = collections.defaultdict(list)
    for r in rows:
        by[r[group]].append(r["side"])
    out = {}
    for g, sides in sorted(by.items()):
        left, right = sides.count("image_left"), sides.count("image_right")
        out[g] = dict(side="image_left" if left > right else "image_right" if right > left else None,
                      clips=len(sides), image_left=left, image_right=right, undecided=sides.count(None))
    return out


def quantiles(values) -> dict | None:
    """min / median / max of per-signer clip counts (a side decided on few clips is weak evidence)."""
    if not values:
        return None
    return dict(min=int(min(values)), median=float(np.median(values)), max=int(max(values)))


def exclusion_spec(signers: dict, corpus: str = CORPUS) -> str | None:
    """``--exclude-signers`` spec of the image-right signers (performer SHA-256 prefixes), or None when there is none."""
    right = sorted(s[:PREFIX_DIGITS] for s, v in signers.items() if v["side"] == "image_right")
    return f"{corpus}:{','.join(right)}" if right else None


def sample_records(manifest: dict, clips: int, seed: int) -> list[dict]:
    """``clips`` train/val records spread over signers (round-robin over a seeded signer order)."""
    from signtranslator.generation.recognizer_corpora import check_loadable
    from tokenizer_domain_read import signer_spread_sample
    by_sign = {}
    for r in manifest["records"]:
        if r.get("partition") not in ("train", "val") or not r.get("performer_sha256"):
            continue
        check_loadable(r)
        by_sign[r["sign"]] = r
    picked = signer_spread_sample([(s, r["performer_sha256"]) for s, r in by_sign.items()], clips, seed)
    return [by_sign[s] for s, _ in picked]


# ------------------------------------------------------------------------------------------------------ workers

def _init(zip_key, models_dir):
    import extract_video_states
    extract_video_states._init(zip_key, models_dir)


def measure(job) -> dict:
    """One clip's active half; refusals and failures are recorded, never fatal."""
    sign, member, signer, partition = job
    import extract_video_states
    from signtranslator.data_engineering.video_landmarks import extract_clip
    row = dict(sign=sign, signer=signer, partition=partition)
    try:
        payload = extract_video_states._WORKER["archive"].read(member)
        with tempfile.TemporaryDirectory() as t:
            video = Path(t) / ("clip" + (Path(member).suffix.lower() or ".mp4"))
            video.write_bytes(payload)
            clip = extract_clip(video, extract_video_states._WORKER["models"], roi=True, hand_scale=.9)
    except (ValueError, OSError) as error:
        return dict(row, side=None, status="refused", reason=str(error)[:200])
    motion = half_motion(clip_hands(clip))
    return dict(row, side=active_half(motion), status="measured", clip_sha256=hashlib.sha256(payload).hexdigest(),
                frames=len(clip.times), **{f"{h}_{k}": v for h, m in motion.items() for k, v in m.items()})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--clips", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--manifest-key", default=MANIFEST)
    p.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    p.add_argument("--zip-key", default=ZIP_KEY)
    p.add_argument("--zip-sha256", default=ZIP_SHA256)
    p.add_argument("--models-dir", required=True, help="the MediaPipe Tasks models (verify_models checks their hashes)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", required=True, help="local JSON report")
    p.add_argument("--output-key", help="also write the report once to the store")
    args = p.parse_args(argv)
    import io
    import zipfile
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from extract_video_states import StoreRangeFile
    store = R2Store(os.environ)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: the report is written once")
    provenance = json.loads(store.get(args.zip_key + ".provenance.json"))
    if provenance.get("sha256") != args.zip_sha256 or not provenance.get("verified_by_r2_reread"):
        raise SystemExit("the zip is not the verified acquisition named by --zip-sha256")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    records = sample_records(manifest, args.clips, args.seed)
    archive = zipfile.ZipFile(io.BufferedReader(StoreRangeFile(store.s3, store.bucket, args.zip_key), 1 << 20))
    members = {Path(n).stem: n for n in archive.namelist() if n.lower().endswith((".mp4", ".mov"))}
    jobs = [(r["sign"], members[r["sign"]], r["performer_sha256"], r["partition"]) for r in records if r["sign"] in members]
    print(f"{len(records)} sampled over {len({r['performer_sha256'] for r in records})} signers; "
          f"{len(jobs)} found in the zip", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.zip_key, args.models_dir)) as pool:
        rows = list(pool.map(measure, jobs))
    signers = group_sides([r for r in rows if r["status"] == "measured"])
    decided = {s: v for s, v in signers.items() if v["side"] is not None}
    report = dict(
        declared="docs/NEXT_PHASE_PLAN.md, Instrument finding and the recogniser v3 plan (2026-10-29 06:37 UTC)",
        measure="active-hand image half per clip, signer majority (the Sem-Lex pixel-space measure)",
        manifest=dict(key=args.manifest_key, sha256=args.manifest_sha256), zip=dict(key=args.zip_key,
                                                                                   sha256=args.zip_sha256),
        sampling=dict(clips=args.clips, seed=args.seed, sampled=len(records), measured=len(jobs),
                      rule="tokenizer_domain_read.signer_spread_sample over train and val records"),
        clips=dict(measured=sum(r["status"] == "measured" for r in rows),
                   refused=sum(r["status"] == "refused" for r in rows),
                   **{h: sum(r["side"] == h for r in rows) for h in HALVES},
                   undecided=sum(r["status"] == "measured" and r["side"] is None for r in rows)),
        signers=dict(total=len(signers), decided=len(decided),
                     sampled_clips_per_signer=dict(sorted(collections.Counter(r["performer_sha256"][:PREFIX_DIGITS]
                                                                              for r in records).items())),
                     measured_clips_per_signer=quantiles([v["clips"] for v in signers.values()]),
                     image_left=sum(v["side"] == "image_left" for v in decided.values()),
                     image_right=sum(v["side"] == "image_right" for v in decided.values())),
        per_signer=signers, image_right_signers=sorted(s for s, v in signers.items() if v["side"] == "image_right"),
        exclude_signers_spec=exclusion_spec(signers), per_clip=rows)
    payload = json.dumps(report, indent=1, sort_keys=True).encode()
    Path(args.output).write_bytes(payload)
    if args.output_key:
        store.put(args.output_key, payload)
    print(json.dumps({k: report[k] for k in ("clips", "signers", "exclude_signers_spec")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
