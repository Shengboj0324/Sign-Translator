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
   For a clip of one of our videos the partition is our video's (or channel's), so a UWB dev clip of one of our
   training videos trains; UWB's dev clips of other videos stay ``monitor``. No evaluation data is exposed either
   way: evaluation is How2Sign / ASLLRP, and leaks are filtered.
2. **Rejected cues do not re-enter** (amendment 2 item 1). Every lifted clip is joined to the cue of ours it pairs
   with (``duplicate_of_our_cue``; otherwise its largest overlap with cover >= 0.5): the cue's fate in
   ``youtube-asl-v1`` (manifest-r3b2) is recomputed from our extraction reports with that manifest's own filters
   (:func:`our_cue_reasons`, a mirror of ``build_youtube_asl_manifest.build``, checked against the snapshot). Kept:
   clips with no cue of ours (``never_extracted``) and clips whose cue was rejected for a reason the lifter
   addresses (``LIFTER_ADDRESSABLE``: hands not tracked by our extraction). Dropped, by reason: every other rejection
   (second person, not frontal, face not visible, the video gate, the channel cap, and the caption and timing
   rejections), a cue our filters would keep but the snapshot lacks, and a cue missing from our reports.
3. Only ``train``, and no clip of a monitor channel by ``build_youtube_asl_manifest.monitor_channel`` (share
   ``--monitor-share``, 0.01 as ``youtube-asl-v1``).
4. ``youtube-asl-v1``'s segment rule where the lifted record has the field (``build_youtube_asl_manifest.keep_segment``
   with its defaults: notes, at least 2 words, 1-20 s, 0.5-4 words/s, at most 300 characters, hand coverage >= 0.5
   read as the larger per-side hand fraction), the caption cleaned by ``clean_text``; a cleaned text that leaks
   (``evaluation_leak_cleaned``, the normalisation ``youtube-asl-v1`` checked); a text repeated within one video,
   counting our own selected texts of that video (``repeated_text``; compared as cleaned lower-case words). Face coverage, yaw and second-person fields do
   not exist for lifted clips and are not applied.

**Refusals.** Fewer than ``MIN_LIFTED`` (20,000) kept records: ``ROUND18-NO-MEANINGFUL-GAIN``, exit 5, nothing
written (amendment 2: round 18 is then not trained). Any kept text that the evaluation index or the base's How2Sign /
ASLLRP val, test and heldout sentences would flag (an assertion after step 4); a mix whose first 1,536 / 2,048
validation records (the trainer's fixed batches) or whose pilot sample (``pilot_content_check.validation_sentences``,
1,000 clips) differ from the base's; an existing output key (write-once); any conversion setting other than the
declared ones (``--expect-*``).

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
OUR_MANIFEST = (f"{ACQ}/youtube-asl-v1/manifest-r3b2/manifest.json"
                "@f69ad828f1e428b82ce62e414e1b0a694a22902fe8ca2fd8384f5918a761e070")   # round14b's youtube-asl-v1
OUR_REPORTS = f"{ACQ}/youtube-asl-v1/videos"
LIFTER_ADDRESSABLE = ("hands_not_visible",)          # our extraction did not track the hands; the lifter may
MIN_LIFTED = 20_000                                   # amendment 2: fewer -> no meaningful data gain, not trained
NO_GAIN_EXIT = 5
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


def our_cue_reasons(reports, *, args, eval_index=None) -> dict[str, str]:
    """{our sample id: its fate in a ``build_youtube_asl_manifest.build`` run with ``args``}: ``kept`` or the reason
    (a video-level reason is given to every segment of the video). The same code path as ``build``, step by step,
    including the channel cap; tested against ``build`` itself."""
    from build_youtube_asl_manifest import clean_text, keep_segment, monitor_channel, video_gate  # noqa: F401
    reasons, kept = {}, []
    for report in reports:
        segments = report.get("segments", [])
        video = None
        if report.get("status") != "converted":
            video = f"video_{report.get('status')}"
        elif report.get("caption_kind") != "manual" and not args.allow_auto:
            video = "video_auto_captions"
        else:
            video = video_gate(report, args=args)
        if video:
            reasons.update({seg["sample_id"]: video for seg in segments if seg.get("sample_id")})
            continue
        seen = set()
        for seg in segments:
            reason = keep_segment(seg, args=args)
            text = clean_text(seg["text"])
            if reason is None and text.lower() in seen:
                reason = "repeated_text"
            if reason is None and eval_index is not None and eval_index.leaks(text):
                reason = "evaluation_leak"
            if reason:
                reasons[seg["sample_id"]] = reason
                continue
            seen.add(text.lower())
            reasons[seg["sample_id"]] = "kept"
            kept.append((f"{args.corpus_id}:{seg['sample_id']}", (report.get("meta") or {}).get("channel_id"),
                         seg["sample_id"]))
    if args.channel_cap < 1 and kept:
        cap = max(1, int(args.channel_cap * len(kept)))
        per = collections.Counter()
        for sign, channel, sample in sorted(kept, key=lambda k: hashlib.sha256(k[0].encode()).hexdigest()):
            if per[channel] < cap:
                per[channel] += 1
            else:
                reasons[sample] = "channel_cap"
    return reasons


def our_cue_of(r: dict) -> str | None:
    """The cue of ours a lifted clip is: its pair, else its largest overlap covering >= half of the clip."""
    if r.get("duplicate_of_our_cue"):
        return r["duplicate_of_our_cue"]
    best = max(r.get("our_cue_overlaps") or [], key=lambda o: o.get("cover", 0.), default=None)
    return best["sample_id"] if best and best.get("cover", 0.) >= .5 else None


def rejected_cue_filter(records: list[dict], reasons: dict, snapshot: set) -> tuple[list[dict], collections.Counter,
                                                                                    collections.Counter]:
    """Step 2 of the docstring: (kept records with ``our_cue_category``, drops by reason, kept by category)."""
    kept, dropped, categories = [], collections.Counter(), collections.Counter()
    for r in records:
        cue = our_cue_of(r)
        if cue is None:
            category = "never_extracted"
        elif cue in snapshot:                   # select_training_records drops these; never kept here either
            dropped["our_cue_selected"] += 1
            continue
        elif cue not in reasons:
            dropped["our_cue_not_in_reports"] += 1
            continue
        elif reasons[cue] == "kept":
            dropped["our_cue_not_in_snapshot"] += 1
            continue
        elif reasons[cue] in LIFTER_ADDRESSABLE:
            category = f"our_rejected:{reasons[cue]}"
        else:
            dropped[f"our_rejected:{reasons[cue]}"] += 1
            continue
        categories[category] += 1
        kept.append(dict(r, our_cue_category=category))
    return kept, dropped, categories


def segment_filter(records: list[dict], *, monitor_share: float, leak_index=None,
                   our_texts: dict | None = None) -> tuple[list[dict], collections.Counter]:
    """Steps 3 and 4 of the docstring: train only, no monitor channel, ``youtube-asl-v1``'s segment rule, no cleaned-
    text leak (``leak_index``: one or more ``EvalIndex``), one text per video counting ``our_texts`` (video id -> our
    selected texts). ``records`` come from ``select_training_records``."""
    from build_youtube_asl_manifest import _WORD, clean_text, keep_segment, monitor_channel

    def key(text):                              # words only: "We went home late." and "we went home late" are one
        return " ".join(_WORD.findall(clean_text(text).lower()))
    indexes = [] if leak_index is None else leak_index if isinstance(leak_index, (list, tuple)) else [leak_index]
    kept, why = [], collections.Counter()
    seen = collections.defaultdict(set)
    for video, texts in (our_texts or {}).items():
        seen[video] |= {key(t) for t in texts}
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
        if reason is None and any(index.leaks(text) for index in indexes):
            reason = "evaluation_leak_cleaned"
        if reason is None and key(text) in seen[r["video_id"]]:
            reason = "repeated_text"
        if reason:
            why[reason] += 1
            continue
        seen[r["video_id"]].add(key(text))
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
                            partition_source=r.get("partition_source"), our_cue_category=r.get("our_cue_category"),
                            licence=r.get("licence")),
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


def load_our_cue_reasons(store, our_manifest: str, reports_prefix: str, base: dict, workers: int = 16):
    """(reasons, snapshot sample ids, check): ``our_cue_reasons`` over every extraction report under
    ``reports_prefix`` with the filters and evaluation manifests ``our_manifest`` (manifest-r3b2) was built with.
    Refused unless ``our_manifest`` holds exactly the base's ``youtube-asl-v1`` records; ``check`` says whether the
    report list is the one it was built from and how much of its selection the recomputation reproduces."""
    from concurrent.futures import ThreadPoolExecutor
    from build_youtube_asl_manifest import EvalIndex
    sha, ours = fetch_manifest(store, our_manifest)
    snapshot = {r["sign"].split(":", 1)[-1] for r in ours["records"]}
    in_base = {r["sign"].split(":", 1)[-1] for r in base["records"] if r.get("corpus") == OUR_CORPUS}
    if snapshot != in_base:
        raise SystemExit(f"{our_manifest} does not hold exactly the base's {OUR_CORPUS} records")
    filters = dict(ours["filters"])
    filters.pop("reasons", None)
    sentences = eval_sentences_of(store, filters.get("eval_manifest") or [])
    keys = sorted(k for k in store.list(reports_prefix.rstrip("/") + "/") if k.endswith(".json"))
    with ThreadPoolExecutor(workers) as pool:
        reports = list(pool.map(lambda k: json.loads(store.get(k)), keys))
    reasons = our_cue_reasons(reports, args=SimpleNamespace(**filters),
                              eval_index=EvalIndex(sentences) if sentences else None)
    kept = {k for k, v in reasons.items() if v == "kept"}
    check = dict(our_manifest_sha256=sha, reports=len(keys),
                 report_list_is_the_snapshots=hashlib.sha256("".join(keys).encode()).hexdigest()
                 == (ours.get("inputs") or {}).get("reports"),
                 snapshot=len(snapshot), snapshot_reproduced=len(snapshot & kept), snapshot_not_reproduced=len(
                     snapshot - kept), kept_not_in_snapshot=len(kept - snapshot),
                 reasons=dict(collections.Counter(reasons.values())))
    return reasons, snapshot, check


def select_lifted(records: list[dict], *, base: dict, eval_index, eval_sha: str, reasons: dict, snapshot: set,
                  leak_indexes: list, monitor_share: float) -> tuple[list[dict], dict]:
    """Steps 1-4 of the docstring over converted clip records: (kept records, counts per step)."""
    ours = [r for r in base["records"] if r.get("corpus") == OUR_CORPUS]
    selected, why_select = select_training_records(records, eval_index_sha256=eval_sha, our_records=ours,
                                                   eval_index=eval_index)
    joined, why_cue, categories = rejected_cue_filter(selected, reasons, snapshot)
    our_texts = collections.defaultdict(list)
    for r in ours:
        our_texts[(r.get("source") or {}).get("video_id")].append(r["text"])
    kept, why_segment = segment_filter(joined, monitor_share=monitor_share, leak_index=leak_indexes,
                                       our_texts=our_texts)
    for index in leak_indexes:                  # the guard: an assertion after the cleaned-text filter
        leaks = [r["clip_id"] for r in kept if index.leaks(r["text"])]
        if leaks:
            raise SystemExit(f"{len(leaks)} lifted records carry an evaluation text (e.g. {leaks[:3]}): refusing")
    return kept, dict(select_training_records=dict(why_select), rejected_cues=dict(why_cue),
                      kept_by_category_before_segment_rule=dict(categories), segment=dict(why_segment),
                      kept_by_category=dict(collections.Counter(r["our_cue_category"] for r in kept)))


def pilot_sample(manifest: dict, clips: int = 1000) -> list[str]:
    from pilot_content_check import validation_sentences
    return [r["sign"] for r in validation_sentences(manifest, clips)]


class NoMeaningfulGain(Exception):
    pass


def build(store, args) -> dict:
    from build_mixed_text_manifest import extend, val_prefix_unchanged
    import convert_uwb_clips as cu
    for key in (args.lifted_key, args.manifest_key):
        if key and store.exists(key) and not args.dry_run:
            raise SystemExit(f"refusing to overwrite {key}")
    base_sha, base = fetch_manifest(store, args.base)
    index, records = converted_records(store, args.converted_prefix, allow_incomplete=args.allow_incomplete)
    settings = index["settings"] or {}
    from signtranslator.data_engineering.uwb_keypoints import ADAPTER
    check_settings(settings, dict(regressors_sha256=args.expect_bundle_sha, visibility=args.expect_visibility,
                                  fps_table_sha256=args.expect_fps_table_sha, overlap_table_sha256=args.expect_overlap_sha,
                                  stream_check_sha256=args.expect_stream_check_sha, fps_sources=args.expect_fps_sources,
                                  eval_manifests=sorted(args.eval_manifest), world=args.expect_world,
                                  face=args.expect_face, head=args.expect_head, flags=args.expect_flags,
                                  min_frames=args.expect_min_frames, adapter=ADAPTER))
    statuses = check_converted(records, settings)
    eval_index, eval_sha, _ = cu.eval_index(args.eval_manifest, store)
    if eval_sha != settings.get("eval_index_sha256"):
        raise SystemExit("the builder's evaluation index differs from the one the conversion checked against")
    ours = [r for r in base["records"] if r.get("corpus") == OUR_CORPUS]
    reasons, snapshot, our_check = load_our_cue_reasons(store, args.our_manifest, args.our_reports, base)
    guard = refusal_index(base, eval_sentences_of(store, args.eval_manifest))
    kept, counts = select_lifted(records, base=base, eval_index=eval_index, eval_sha=eval_sha, reasons=reasons,
                                 snapshot=snapshot, leak_indexes=[eval_index, guard], monitor_share=args.monitor_share)
    print(json.dumps(dict(our_cues=our_check, **counts), indent=1), flush=True)
    if len(kept) < args.min_lifted:
        print(f"ROUND18-NO-MEANINGFUL-GAIN {len(kept)} lifted records < {args.min_lifted}: no manifest written")
        raise NoMeaningfulGain(len(kept))
    inputs = {"uwb-converted-index": hashlib.sha256(store.get(f"{args.converted_prefix.rstrip('/')}/index.json")).hexdigest(),
              "dedup-base": base_sha, "eval-index": eval_sha}
    lifted = build_corpus_manifest(corpus_id=CORPUS, records=[lifted_record(r) for r in kept], inputs=inputs,
                                   split=dict(kind="continuous", signer_disjoint=False, counts={"train": len(kept)},
                                              source="train only: monitor partitions and monitor channels stay "
                                                     "out; UWB dev clips of our training videos train (round 18)"))
    lifted["license"] = LICENCE
    lifted["filters"] = dict(conversion_statuses=dict(statuses), our_cues=our_check, **counts,
                             segment_rule=dict(vars(SEGMENT_RULE), applied="text, duration, rate, length, hand "
                                               "coverage (larger per-side hand fraction); face, yaw and second-person "
                                               "fields absent for lifted clips"),
                             monitor_share=args.monitor_share)
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
                   conversion_statuses=dict(statuses), our_cues=our_check, **counts,
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
    p.add_argument("--our-manifest", default=OUR_MANIFEST, help="KEY@SHA of youtube-asl-v1 (manifest-r3b2)")
    p.add_argument("--our-reports", default=OUR_REPORTS, help="prefix of our extraction reports (<video>.json)")
    p.add_argument("--min-lifted", type=int, default=MIN_LIFTED)
    p.add_argument("--expect-world", default="learned")
    p.add_argument("--expect-face", default="blendshapes")
    p.add_argument("--expect-head", default="none")
    p.add_argument("--expect-flags", default="generator")
    p.add_argument("--expect-min-frames", type=int, default=8)
    p.add_argument("--allow-incomplete", action="store_true", help="smoke runs only: an index without every unit")
    p.add_argument("--dry-run", action="store_true", help="build and check, print the counts, write nothing")
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    try:
        summary = build(store, args)
    except NoMeaningfulGain:
        return NO_GAIN_EXIT
    print(json.dumps(summary, indent=1))
    print(f"LIFTED_SHA={summary['lifted_sha256']}")
    print(f"MANIFEST_SHA={summary['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
