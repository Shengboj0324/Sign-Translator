"""Per-signer hand dominance from wrist motion, across corpora: the declared Sem-Lex handedness test (round 14 data,
docs/NEXT_PHASE_PLAN.md, 2026-10-28 18:01 UTC).

**Measure (per clip).** Forward kinematics of the stored state (``canonical_skeleton.skeleton_positions``): each
wrist's position relative to its own shoulder. The dominant hand is the one whose wrist moves more: the larger **path
length**, summed over consecutive frame pairs on which **both** arms are solved (forearm and upper arm valid,
``compositional.side_tracks``). Using the same frames for both sides keeps tracking out of the comparison: a side
that is solved on more frames does not get a longer path for that reason (forearm tracking is confounded by the
right elbow's visibility in webcam framing). A clip with fewer than ``--min-pairs`` such pairs, or equal paths, is
undecided. For contrast only, the summary also gives the confounded variants: each side's path over its own solved
frames, and round 5's height cue (90th percentile of wrist height above the shoulder) on the shared frames.

**Per signer:** the majority side over their decided clips (a tie is undecided). **Per corpus:** the right-dominant
fraction of decided signers, and of decided clips. Signer units: Sem-Lex's ``signer_id`` (metadata); the manifest's
``performer_sha256`` (ASLLVD, RIT, ASL Citizen); How2Sign and YouTube-ASL have no signer ids in our manifests, so their
units are the source video (How2Sign: the first 11 characters of the sign id; one signer per video) and the YouTube
channel (``source.channel_id``).

**Declared rule:** Sem-Lex is judged **mirrored** iff its right-dominant signer fraction is below 0.25 while How2Sign's
and ASLLVD's are above 0.6.

Sources (each optional): ``--semlex`` (the video states of ``--semlex-splits`` through their extraction reports, with
the manifest builder's latest-final-result rule; shards still extracting are read as far as reported), ``--manifest
KEY@SHA`` (round10-mixed-v1: How2Sign, ASLLVD, RIT, YouTube-ASL), ``--citizen KEY`` (ASL Citizen's source manifest,
the stored unmirrored states; read-only, recognition-only data, nothing derived is kept). At most ``--per-signer``
clips per signer, in SHA-256 order of the clip id, and at most ``--max-units`` signer units per corpus. Every state
is hashed against its record. Writes a local JSON (``--output``) and optionally ``--output-key`` under the smoke prefix.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

SMOKE_BASE = "acquisition/2026-10-17-v1/sem-lex-mirror-v1/adapter-dev/smoke"
CITIZEN_KEY = "acquisition/2026-10-17-v1/derived/asl-citizen-v1/manifest-v1/manifest.json"
RULE = dict(semlex_below=.25, how2sign_above=.6, asllvd_above=.6)
_WORKER: dict = {}


def wrist_motion(state, min_pairs: int = 8) -> dict:
    """Per side: wrist path length (relative to its shoulder) over the pairs of consecutive frames on which both arms
    are solved, the path over the side's own solved frames, and the wrist-height cue; and the clip's side."""
    from signtranslator.generation.canonical_skeleton import skeleton_positions
    from signtranslator.generation.compositional import side_tracks
    joints, pos, _ = skeleton_positions(state)
    tracks = side_tracks(state)
    solved = {s: np.asarray(tracks[s][0], dtype=bool) for s in ("Left", "Right")}
    both = solved["Left"] & solved["Right"]
    shared = both[1:] & both[:-1]
    out = dict(frames=int(len(both)), shared_pairs=int(shared.sum()))
    for side in ("Left", "Right"):
        rel = pos[:, joints.index(f"{side}Hand")] - pos[:, joints.index(f"{side}Shoulder")]
        step = np.linalg.norm(np.diff(rel, axis=0), axis=1)
        own = solved[side][1:] & solved[side][:-1]
        out[f"{side.lower()}_path"] = round(float(step[shared].sum()), 5)
        out[f"{side.lower()}_path_own_frames"] = round(float(step[own].sum()), 5)
        out[f"{side.lower()}_height"] = round(float(np.quantile(rel[both, 1], .9)), 5) if both.any() else None
    left, right = out["left_path"], out["right_path"]
    out["side"] = None if out["shared_pairs"] < min_pairs or left == right else ("Right" if right > left else "Left")
    lo, ro = out["left_path_own_frames"], out["right_path_own_frames"]
    out["side_own_frames"] = None if lo == ro else ("Right" if ro > lo else "Left")
    lh, rh = out["left_height"], out["right_height"]
    out["side_height"] = None if lh is None or out["shared_pairs"] < min_pairs or lh == rh else \
        ("Right" if rh > lh else "Left")
    return out


def _init(store_spec: str) -> None:
    import warnings
    warnings.simplefilter("error")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    from convert_semlex_poses import make_store
    _WORKER["store"] = make_store(store_spec)


def measure(task) -> dict:
    from signtranslator.pose.multichannel import MultichannelMotion
    corpus, unit, clip, key, sha, min_pairs = task
    payload = _WORKER["store"].get(key)
    if hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError(f"{key}: state bytes differ from the record")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "s.npz"
        path.write_bytes(payload)
        state = MultichannelMotion.load(path, expected_sha256=sha)
    return dict(wrist_motion(state, min_pairs), corpus=corpus, unit=unit, clip=clip)


def sample(rows: list[tuple], per_unit: int, max_units: int | None) -> list[tuple]:
    """(corpus, unit, clip, key, sha) rows: at most ``per_unit`` per unit, in SHA-256 order of the clip id; at most
    ``max_units`` units (in SHA-256 order of the unit)."""
    by = collections.defaultdict(list)
    for row in rows:
        by[row[1]].append(row)
    units = sorted(by, key=lambda u: hashlib.sha256(str(u).encode()).hexdigest())[:max_units]
    out = []
    for u in sorted(units, key=str):
        out += sorted(by[u], key=lambda r: hashlib.sha256(r[2].encode()).hexdigest())[:per_unit]
    return out


def manifest_rows(manifest: dict) -> list[tuple]:
    rows = []
    for r in manifest["records"]:
        c = r.get("continuous") or {}
        if not c.get("state_sha256"):
            continue
        corpus = r.get("corpus")
        if corpus == "how2sign-two-view-text-v1":
            unit = r["sign"][:11]
        elif corpus == "youtube-asl-v1":
            unit = (r.get("source") or {}).get("channel_id")
        elif corpus in ("asllvd-text-v1", "rit-text-r6"):
            unit = r.get("performer_sha256")
        else:
            continue
        if unit:
            rows.append((corpus, unit, r["sign"], c["state_key"], c["state_sha256"]))
    return rows


def citizen_rows(manifest: dict) -> list[tuple]:
    return [("asl-citizen", r["performer_sha256"], r["sign"], r["continuous"]["state_key"],
             r["continuous"]["state_sha256"]) for r in manifest["records"]
            if (r.get("continuous") or {}).get("state_sha256") and r.get("performer_sha256")]


def semlex_rows(store, args) -> list[tuple]:
    from build_semlex_manifest import VIDEO_PREFIX, video_results
    from convert_semlex_poses import METADATA_KEY, METADATA_SHA256, load_metadata
    splits = tuple(args.semlex_splits.split(","))
    results, _ = video_results(store, VIDEO_PREFIX, splits, require_all_shards=False)
    unmirrored = None
    if args.semlex_states_prefix:                  # verification: the re-solved states of the same clips
        from build_semlex_manifest import unmirror_results
        unmirrored, _ = unmirror_results(store, args.semlex_states_prefix.rstrip("/"), splits, require_all_shards=False)
    metadata = load_metadata(store, METADATA_KEY, METADATA_SHA256)
    rows = []
    for (video_id, split), r in sorted(results.items()):
        if not r.get("state_sha256") or r.get("status") not in ("converted", "resumed", "already_present"):
            continue
        if unmirrored is not None:
            u = unmirrored.get((video_id, split))
            if u is None or u.get("status") not in ("converted", "already_present") or \
                    u.get("source_state_sha256") != r["state_sha256"]:
                continue
            r = dict(r, state_key=u["state_key"], state_sha256=u["state_sha256"])
        signers = {m["signer_id"] for m in metadata.get(video_id, [])}
        if len(signers) != 1:
            continue
        rows.append((f"semlex-{split}", signers.pop(), f"{split}/{video_id}", r["state_key"], r["state_sha256"]))
    return rows


def summarise(results: list[dict]) -> dict:
    out = {}
    for corpus in sorted({r["corpus"] for r in results}):
        rs = [r for r in results if r["corpus"] == corpus]
        corpus_out = dict(clips=len(rs))
        for name, field in (("path_shared_frames", "side"), ("path_own_frames", "side_own_frames"),
                            ("height_shared_frames", "side_height")):
            units = collections.defaultdict(collections.Counter)
            for r in rs:
                units[r["unit"]][r[field] or "undecided"] += 1
            sides = {}
            for u, c in units.items():
                sides[u] = "undecided" if c["Right"] == c["Left"] else ("Right" if c["Right"] > c["Left"] else "Left")
            decided = [s for s in sides.values() if s != "undecided"]
            clips = [r[field] for r in rs if r[field]]
            corpus_out[name] = dict(
                units=len(units), units_decided=len(decided),
                right_fraction_units=round(sum(s == "Right" for s in decided) / len(decided), 4) if decided else None,
                clips_decided=len(clips),
                right_fraction_clips=round(sum(s == "Right" for s in clips) / len(clips), 4) if clips else None)
            if name == "path_shared_frames":
                corpus_out["per_unit"] = {str(u)[:16]: dict(side=sides[u], right=units[u]["Right"], left=units[u]["Left"],
                                                            undecided=units[u]["undecided"]) for u in sorted(units, key=str)}
        corpus_out["shared_pairs_median"] = float(np.median([r["shared_pairs"] for r in rs]))
        out[corpus] = corpus_out
    return out


def verdict(summary: dict) -> dict:
    """The declared rule on the per-signer right-dominant fractions of the primary measure."""
    def frac(*corpora):
        units = [summary[c]["path_shared_frames"] for c in corpora if c in summary]
        right = sum(u["right_fraction_units"] * u["units_decided"] for u in units if u["units_decided"])
        n = sum(u["units_decided"] for u in units)
        return round(right / n, 4) if n else None
    semlex, how2sign, asllvd = frac("semlex-all"), frac("how2sign-two-view-text-v1"), frac("asllvd-text-v1")
    ready = None not in (semlex, how2sign, asllvd)
    mirrored = ready and semlex < RULE["semlex_below"] and how2sign > RULE["how2sign_above"] and \
        asllvd > RULE["asllvd_above"]
    return dict(rule=f"mirrored iff Sem-Lex right-dominant signer fraction < {RULE['semlex_below']} and How2Sign > "
                     f"{RULE['how2sign_above']} and ASLLVD > {RULE['asllvd_above']}",
                semlex_right_fraction=semlex, how2sign_right_fraction=how2sign, asllvd_right_fraction=asllvd,
                mirrored=bool(mirrored) if ready else None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--semlex", action="store_true")
    p.add_argument("--semlex-splits", default="val,train")
    p.add_argument("--semlex-states-prefix", help="verification: read Sem-Lex from unmirror_semlex_states.py's states "
                   "under this prefix (the same clips, re-solved)")
    p.add_argument("--manifest", help="KEY@SHA of a mix (round10-mixed-v1)")
    p.add_argument("--citizen", nargs="?", const=CITIZEN_KEY, help="ASL Citizen's source manifest key (reference)")
    p.add_argument("--per-signer", type=int, default=30)
    p.add_argument("--max-units", type=int, default=60)
    p.add_argument("--min-pairs", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--output-key")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    from convert_semlex_poses import make_store
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    if args.output_key and not args.output_key.startswith(SMOKE_BASE + "/"):
        raise SystemExit(f"--output-key must be under {SMOKE_BASE}/")
    store = make_store(args.store)
    rows, inputs = [], {}
    if args.semlex:
        rows += semlex_rows(store, args)
        inputs["semlex_states_prefix"] = args.semlex_states_prefix or "states-video-v1 (as extracted)"
    if args.manifest:
        key, _, sha = args.manifest.rpartition("@")
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "m.json"
            path.write_bytes(store.get(key) + b"\n")
            rows += manifest_rows(load_manifest(path, expected_sha256=sha))
        inputs["manifest"] = args.manifest
    if args.citizen:
        payload = store.get(args.citizen)
        inputs["citizen"] = f"{args.citizen}@{hashlib.sha256(payload).hexdigest()}"
        rows += citizen_rows(json.loads(payload))
    tasks = []
    for corpus in sorted({r[0] for r in rows}):
        tasks += [r + (args.min_pairs,) for r in sample([r for r in rows if r[0] == corpus], args.per_signer,
                                                        args.max_units)]
    print(f"{len(tasks)} clips from {len(rows)} candidates", flush=True)
    if args.workers <= 0:
        _init(args.store)
        results = [measure(t) for t in tasks]
    else:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init,
                                 initargs=(args.store,)) as pool:
            results = list(pool.map(measure, tasks, chunksize=8))
    # Sem-Lex's signers appear in both splits: the verdict reads one signer set over both ("semlex-all")
    summary = summarise(results + [dict(r, corpus="semlex-all") for r in results if r["corpus"].startswith("semlex-")])
    report = dict(inputs=inputs, per_signer=args.per_signer, max_units=args.max_units, min_pairs=args.min_pairs,
                  verdict=verdict(summary), corpora=summary, clips=results)
    payload = json.dumps(report, indent=1, sort_keys=True).encode()
    with open(args.output, "xb") as stream:
        stream.write(payload)
    if args.output_key:
        if store.exists(args.output_key):
            raise SystemExit(f"refusing to overwrite {args.output_key}")
        store.put(args.output_key, payload)
    print(json.dumps(dict(verdict=report["verdict"], corpora={c: {k: v for k, v in s.items() if k != "per_unit"}
                                                              for c, s in summary.items()}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
