"""Round 18's training mix (docs/NEXT_PHASE_PLAN.md "Round 18: the UWB 4x-data test", declared 2026-10-30 13:03 UTC):
``round18-mixed-v1`` = ``round14b-mixed-v1`` (every record byte-identical) + the lifted UWB clips as corpus
``youtube-asl-uwb-lifted-v2`` (training weight 1, the trainer's default, as ``youtube-asl-v1``).

**Input.** The converted UWB prefix of ``convert_uwb_clips.py`` (lifter-v2 bundle, visibility ``head``): its
``index.json`` must be complete (``--allow-incomplete`` only for smoke runs) and carry exactly the declared settings
(``--expect-*``: bundle SHA, visibility, fps table SHA, stream check SHA, overlap table SHA, ``fps_sources
estimated``); every unit report the index lists is read, and every converted clip must be ``estimated`` fps with the
same bundle and rule (a mismatch refuses the build).

**Selection** (counts printed per reason):
1. ``uwb_manifest.select_training_records`` against the base's ``youtube-asl-v1`` records (train and monitor): leak
   drops (the builder's own ``EvalIndex`` over ``--eval-manifest``, whose SHA-256 must equal the converter's, and
   every record's flag must agree with it), cues we already hold in a selected record (ours kept), and our partitions.
2. Only ``train``: UWB's dev clips and our monitor videos/channels (``monitor``) stay out, and so does any clip whose
   channel is a monitor channel by ``build_youtube_asl_manifest.monitor_channel`` (share ``--monitor-share``, 0.01 as
   ``youtube-asl-v1``).
3. ``youtube-asl-v1``'s segment rule where the lifted record has the field (``build_youtube_asl_manifest.keep_segment``
   with its defaults: notes, at least 2 words, 1-20 s, 0.5-4 words/s, at most 300 characters, hand coverage >= 0.5
   read as the larger per-side hand fraction), the caption cleaned by ``clean_text``, and a text repeated within one
   video kept once. Face coverage, yaw and second-person fields do not exist for lifted clips and are not applied.

**Refusals.** Any kept text that the evaluation index (``--eval-manifest``) or the base's How2Sign / ASLLRP
val, test and heldout sentences would flag (exact or >= 0.8 5-gram Jaccard); a mix whose first 1,536 / 2,048
validation records (the trainer's fixed batches) or whose pilot sample (``pilot_content_check.validation_sentences``,
1,000 clips) differ from the base's; an existing output key (write-once).

Writes the lifted corpus manifest (``--lifted-key``) and the mix (``--manifest-key``), both canonical, and prints the
counts per corpus and partition, the lifted hours and both SHA-256s (``MANIFEST_SHA=...`` for the training job).
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json, load_manifest  # noqa: E402
from signtranslator.data_engineering.uwb_manifest import select_training_records  # noqa: E402

ACQ = "acquisition/2026-10-17-v1"
BASE = f"{ACQ}/youtube-asl-v1/round14b-mixed-v1/manifest.json@710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d"
CORPUS = "youtube-asl-uwb-lifted-v2"
MIX_ID = "round18-mixed-v1"
OUR_CORPUS = "youtube-asl-v1"
EVAL_CORPORA = ("how2sign-two-view-text-v1", "asllrp-dai2-plan-v2")
EVAL_PARTITIONS = ("val", "test", "heldout_official")
LICENCE = (f"{CORPUS}: UWB YouTube-ASL Clip Keypoint Dataset (CC BY 4.0) lifted to motion states by our z-free "
           "regressors (lifter v2, trained on How2Sign CC BY-NC 4.0 and our research-only YouTube-ASL states); "
           "captions YouTube-ASL (CC BY 4.0, Google Research); research use, private store, no redistribution")
SEGMENT_RULE = SimpleNamespace(min_words=2, min_seconds=1., max_seconds=20., max_words_per_second=4.,
                               min_words_per_second=.5, max_chars=300, min_hand_coverage=.5, min_face_coverage=.9,
                               max_abs_yaw=30., max_other_person=.1)


def fetch_manifest(store, spec: str) -> tuple[str, dict]:
    key, _, sha = spec.rpartition("@")
    if not key or len(sha) != 64:
        raise SystemExit(f"{spec!r}: expected KEY@<64-hex SHA-256>")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(key) + b"\n")
        return sha, load_manifest(path, expected_sha256=sha)


def converted_records(store, prefix: str, *, allow_incomplete: bool = False) -> tuple[dict, list[dict]]:
    """(index, every clip record of every unit the index lists)."""
    index = json.loads(store.get(f"{prefix.rstrip('/')}/index.json"))
    if not index.get("complete") and not allow_incomplete:
        raise SystemExit(f"{prefix}/index.json is not complete ({index.get('units_finished')} of "
                         f"{index.get('units_total')} units): finish every shard and --merge-index first")
    records = []
    for unit in index["units"]:
        report = json.loads(store.get(unit["key"]))
        if report.get("settings") != index["settings"]:
            raise SystemExit(f"{unit['key']}: its settings differ from the index's")
        records += report["clips"]
    return index, records


def check_settings(settings: dict, expect: dict) -> None:
    differ = {k: (settings.get(k), v) for k, v in expect.items() if v is not None and settings.get(k) != v}
    if differ:
        raise SystemExit("the conversion's settings are not the declared ones: "
                         + json.dumps({k: dict(converted=a, declared=b) for k, (a, b) in differ.items()}))


def check_converted(records: list[dict], settings: dict) -> collections.Counter:
    """Every converted clip is estimated fps, with the index's bundle and visibility rule; counts by status."""
    statuses = collections.Counter()
    for r in records:
        statuses[r.get("status")] += 1
        if r.get("status") != "converted":
            continue
        if r.get("fps_source") != "estimated" and settings.get("fps_sources") == "estimated":
            raise SystemExit(f"{r.get('clip_id')}: converted at an assumed rate in an estimated-only conversion")
        if r.get("regressors_sha256") != settings.get("regressors_sha256") or r.get("visibility") != settings.get("visibility"):
            raise SystemExit(f"{r.get('clip_id')}: bundle or visibility rule differs from the index's settings")
    return statuses


def segment_filter(records: list[dict], *, monitor_share: float) -> tuple[list[dict], collections.Counter]:
    """Steps 2 and 3 of the docstring: train only, no monitor channel, ``youtube-asl-v1``'s segment rule, one text per
    video. ``records`` come from ``select_training_records``."""
    from build_youtube_asl_manifest import clean_text, keep_segment, monitor_channel
    kept, why, seen = [], collections.Counter(), collections.defaultdict(set)
    for r in sorted(records, key=lambda r: (r["video_id"], r.get("first_frame", 0), r["clip_id"])):
        if r["partition"] != "train":
            why[f"partition_{r['partition']}"] += 1
            continue
        if r.get("our_channel_id") and monitor_channel(r["our_channel_id"], monitor_share):
            why["monitor_channel"] += 1
            continue
        hands = r.get("hand_fraction") or {}
        segment = dict(text=r["text"], start=0., end=float(r["seconds"]),
                       hand_coverage=max(hands.values()) if hands else 0.)
        reason = keep_segment(segment, args=SEGMENT_RULE)
        text = clean_text(r["text"])
        if reason is None and text.lower() in seen[r["video_id"]]:
            reason = "repeated_text"
        if reason:
            why[reason] += 1
            continue
        seen[r["video_id"]].add(text.lower())
        kept.append(dict(r, text=text))
        why["kept"] += 1
    return kept, why


def lifted_record(r: dict) -> dict:
    """A manifest record in ``youtube-asl-v1``'s layout, with the lifted provenance."""
    return dict(sign=f"{CORPUS}:{r['clip_id']}", text=r["text"], partition="train", performer_sha256=None,
                source=dict(video_id=r["video_id"], clip_id=r["clip_id"], first_frame=r["first_frame"],
                            last_frame=r["last_frame"], fps=r["fps"], fps_source=r["fps_source"],
                            channel_id=r.get("our_channel_id"), zip=r["zip"], member=r["member"],
                            member_sha256=r["member_sha256"], uwb_partition=r.get("uwb_partition"),
                            partition_source=r.get("partition_source"), licence=r.get("licence")),
                quality=dict(hand_coverage=max((r.get("hand_fraction") or {"none": 0.}).values()),
                             hand_fraction=r.get("hand_fraction"), forearm_fraction=r.get("forearm_fraction"),
                             face_fraction=r.get("face_fraction")),
                lifted=dict(adapter=r.get("adapter"), regressors_sha256=r.get("regressors_sha256"),
                            visibility=r.get("visibility"), eval_index_sha256=r.get("eval_index_sha256")),
                continuous=dict(status="converted", state_key=r["state_key"], state_sha256=r["state_sha256"],
                                seconds=r["seconds"], available_channels=r["available_channels"]))


def refusal_index(base: dict, eval_sentences: list[str]):
    from build_youtube_asl_manifest import EvalIndex
    sentences = list(eval_sentences) + [r.get("english") or r.get("text") for r in base["records"]
                                        if r.get("corpus") in EVAL_CORPORA and r.get("partition") in EVAL_PARTITIONS
                                        and (r.get("english") or r.get("text"))]
    return EvalIndex(sentences)


def eval_sentences_of(store, specs: list[str]) -> list[str]:
    out = []
    for spec in specs:
        _, m = fetch_manifest(store, spec)
        out += [r.get("english") or r.get("text") for r in m["records"]
                if r.get("partition") in EVAL_PARTITIONS and (r.get("english") or r.get("text"))]
    return out


def pilot_sample(manifest: dict, clips: int = 1000) -> list[str]:
    from pilot_content_check import validation_sentences
    return [r["sign"] for r in validation_sentences(manifest, clips)]


def build(store, args) -> dict:
    from build_mixed_text_manifest import extend, val_prefix_unchanged
    import convert_uwb_clips as cu
    for key in (args.lifted_key, args.manifest_key):
        if key and store.exists(key) and not args.dry_run:
            raise SystemExit(f"refusing to overwrite {key}")
    base_sha, base = fetch_manifest(store, args.base)
    index, records = converted_records(store, args.converted_prefix, allow_incomplete=args.allow_incomplete)
    settings = index["settings"] or {}
    check_settings(settings, dict(regressors_sha256=args.expect_bundle_sha, visibility=args.expect_visibility,
                                  fps_table_sha256=args.expect_fps_table_sha, overlap_table_sha256=args.expect_overlap_sha,
                                  stream_check_sha256=args.expect_stream_check_sha, fps_sources=args.expect_fps_sources,
                                  eval_manifests=sorted(args.eval_manifest)))
    statuses = check_converted(records, settings)
    eval_index, eval_sha, _ = cu.eval_index(args.eval_manifest, store)
    if eval_sha != settings.get("eval_index_sha256"):
        raise SystemExit("the builder's evaluation index differs from the one the conversion checked against")
    ours = [r for r in base["records"] if r.get("corpus") == OUR_CORPUS]
    selected, why_select = select_training_records(records, eval_index_sha256=eval_sha, our_records=ours,
                                                   eval_index=eval_index)
    kept, why_segment = segment_filter(selected, monitor_share=args.monitor_share)
    guard = refusal_index(base, eval_sentences_of(store, args.eval_manifest))
    leaks = [r["clip_id"] for r in kept if guard.leaks(r["text"])]
    if leaks:
        raise SystemExit(f"{len(leaks)} lifted records carry an evaluation text (e.g. {leaks[:3]}): refusing")
    if not kept:
        raise SystemExit(f"no lifted record was kept: {dict(why_select)} {dict(why_segment)}")
    inputs = {"uwb-converted-index": hashlib.sha256(store.get(f"{args.converted_prefix.rstrip('/')}/index.json")).hexdigest(),
              "dedup-base": base_sha, "eval-index": eval_sha}
    lifted = build_corpus_manifest(corpus_id=CORPUS, records=[lifted_record(r) for r in kept], inputs=inputs,
                                   split=dict(kind="continuous", signer_disjoint=False, counts={"train": len(kept)},
                                              source="train only: UWB dev clips, our monitor videos and monitor "
                                                     "channels stay out (round 18)"))
    lifted["license"] = LICENCE
    lifted["filters"] = dict(conversion_statuses=dict(statuses), select_training_records=why_select,
                             segment_rule=dict(vars(SEGMENT_RULE), applied="text, duration, rate, length, hand "
                                               "coverage (larger per-side hand fraction); face, yaw and second-person "
                                               "fields absent for lifted clips"),
                             segment=dict(why_segment), monitor_share=args.monitor_share)
    lifted["provenance"] = dict(converted_prefix=args.converted_prefix, conversion_settings=settings,
                                conversion_counts=index.get("counts"), eval_manifests=sorted(args.eval_manifest),
                                rule="docs/NEXT_PHASE_PLAN.md, Round 18 (declared 2026-10-30 13:03 UTC)")
    lifted_bytes = canonical_json(lifted)
    lifted_sha = hashlib.sha256(lifted_bytes).hexdigest()
    mix = extend(base_sha, base, [(lifted_sha, lifted)], corpus_id=MIX_ID)
    prefix = {str(n): val_prefix_unchanged(base, mix, n) for n in (1536, 2048)}
    if not all(prefix.values()):
        raise SystemExit(f"the trainer's fixed validation batches moved: {prefix}")
    if pilot_sample(base) != pilot_sample(mix):
        raise SystemExit("the pilot's 1,000 How2Sign validation clips differ from the base's")
    mix_bytes = canonical_json(mix)
    mix_sha = hashlib.sha256(mix_bytes).hexdigest()
    by_corpus = collections.Counter((r.get("corpus"), r.get("partition")) for r in mix["records"])
    summary = dict(base=args.base, lifted_key=args.lifted_key, lifted_sha256=lifted_sha, lifted_records=len(kept),
                   lifted_hours=round(sum(r["seconds"] for r in kept) / 3600, 2),
                   lifted_channels=len({r.get("our_channel_id") for r in kept}),
                   largest_channel_share=round(max(collections.Counter(r.get("our_channel_id") for r in kept).values())
                                               / len(kept), 4),
                   youtube_asl_v1_hours=round(sum(r["continuous"]["seconds"] for r in ours) / 3600, 2),
                   manifest_key=args.manifest_key, manifest_sha256=mix_sha, records=len(mix["records"]),
                   by_corpus={f"{c}/{p}": n for (c, p), n in sorted(by_corpus.items(), key=str)},
                   conversion_statuses=dict(statuses), select_training_records=why_select, segment=dict(why_segment),
                   val_prefix_unchanged=prefix, pilot_sample_unchanged=True, dry_run=bool(args.dry_run))
    if not args.dry_run:
        for key in (args.lifted_key, args.manifest_key):
            if store.exists(key):
                raise SystemExit(f"refusing to overwrite {key}")
        if store.put(args.lifted_key, lifted_bytes) != lifted_sha or store.put(args.manifest_key, mix_bytes) != mix_sha:
            raise SystemExit("the store returned another SHA-256 than the bytes written")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="r2", help="r2, or dir:<path>")
    p.add_argument("--base", default=BASE, help="KEY@SHA of the mix to extend (round14b-mixed-v1)")
    p.add_argument("--converted-prefix", required=True, help="convert_uwb_clips.py --out-prefix (with index.json)")
    p.add_argument("--eval-manifest", action="append", default=[], required=True,
                   help="KEY@SHA, repeatable: exactly the conversion's --eval-manifest list")
    p.add_argument("--lifted-key", required=True)
    p.add_argument("--manifest-key", required=True)
    p.add_argument("--expect-bundle-sha", required=True)
    p.add_argument("--expect-visibility", default="head")
    p.add_argument("--expect-fps-table-sha", required=True)
    p.add_argument("--expect-stream-check-sha", required=True)
    p.add_argument("--expect-overlap-sha", required=True)
    p.add_argument("--expect-fps-sources", default="estimated")
    p.add_argument("--monitor-share", type=float, default=.01)
    p.add_argument("--allow-incomplete", action="store_true", help="smoke runs only: an index without every unit")
    p.add_argument("--dry-run", action="store_true", help="build and check, print the counts, write nothing")
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    summary = build(store, args)
    print(json.dumps(summary, indent=1))
    print(f"LIFTED_SHA={summary['lifted_sha256']}")
    print(f"MANIFEST_SHA={summary['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
