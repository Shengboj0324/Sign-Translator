"""Round 19's training mix (docs/NEXT_PHASE_PLAN.md "Round 19: refresh our own YouTube data", declared 2026-10-30
14:16 UTC): ``round19-mixed-v1`` = ``round14b-mixed-v1`` with its YouTube component rebuilt from the current extraction
reports, everything else byte-identical.

**How round14b's YouTube component was built** (verified, not assumed): its 100,454 ``youtube-asl-v1`` records are
byte-identical (but for the ``corpus`` tag) to ``youtube-asl-v1/manifest-r3b2`` (sha ``f69ad828...``), which
``build_youtube_asl_manifest.py`` built on 2026-10-24 from the extraction reports of that time. They reached round14b
unchanged through round3b-mixed-v2 -> round3c -> round6 -> round10 (whose trims touch only the one-word corpora) ->
round14b (round10 + ``semlex-text-v2``, the Sem-Lex corpus without image-right signers 11, 23, 46, 76, from
``states-video-v1``). Corpus weights are training flags, unchanged.

**The rebuild.**
1. ``build_youtube_asl_manifest.build`` over every report under ``--reports-prefix`` (sorted keys, as its ``main``),
   with manifest-r3b2's own recorded filters (minimum words, duration, rates, hand / face coverage, yaw, second person,
   caption coverage, manual captions only, channel cap 0.05, monitor share 0.01). The cap is recomputed over the new
   kept set, as the builder does.
2. Its evaluation-text filter: manifest-r3b2's two evaluation manifests (How2Sign full-v1, ASLLRP plan v2) **plus**
   round14b's own val, test and heldout texts (every corpus). ``build`` compares the cleaned caption (``clean_text``),
   which is the text that trains, so cleaned-text leaks are covered; the result is asserted leak-free afterwards.
3. The mix: round14b's records with ``youtube-asl-v1`` replaced by the rebuilt component (``usable`` recomputed by
   ``build_corpus_manifest``, as ``build_mixed_text_manifest.extend``). Every other top-level field is kept; ``replaces``
   records what changed.

**Refusals:** round14b's YouTube records are not manifest-r3b2's; any non-YouTube record not byte-identical (canonical
JSON) to round14b's; a YouTube record outside ``train``/``monitor``; a moved validation prefix (1,536 / 2,048 records,
the trainer's fixed batches); a changed 1,000-clip pilot sample (``pilot_content_check.validation_sentences``); a kept
YouTube text that leaks; an existing output key (write-once). ``--dry-run`` writes nothing.

Prints per-corpus records and hours for round14b and round19, the YouTube delta (kept, dropped by reason, new), and
``MANIFEST_SHA=...``.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json  # noqa: E402

ACQ = "acquisition/2026-10-17-v1"
BASE = f"{ACQ}/youtube-asl-v1/round14b-mixed-v1/manifest.json@710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d"
OUR_MANIFEST = (f"{ACQ}/youtube-asl-v1/manifest-r3b2/manifest.json"
                "@f69ad828f1e428b82ce62e414e1b0a694a22902fe8ca2fd8384f5918a761e070")
REPORTS = f"{ACQ}/youtube-asl-v1/videos"
YOUTUBE_KEY = f"{ACQ}/youtube-asl-v1/manifest-r19/manifest.json"
MIX_KEY = f"{ACQ}/youtube-asl-v1/round19-mixed-v1/manifest.json"
YT = "youtube-asl-v1"
MIX_ID = "round19-mixed-v1"
EVAL_PARTITIONS = ("val", "test", "heldout_official")
BUILD_ONLY = ("reasons", "manifest_key", "store", "prefix", "licence", "eval_manifest")   # not filter settings


def strip(record: dict, *keys) -> dict:
    return {k: v for k, v in record.items() if k not in keys}


def check_base(base: dict, ours: dict) -> None:
    """round14b's YouTube records are manifest-r3b2's, byte for byte (but the corpus tag)."""
    yt = {r["sign"]: strip(r, "corpus") for r in base["records"] if r.get("corpus") == YT}
    theirs = {r["sign"]: r for r in ours["records"]}
    if set(yt) != set(theirs) or any(canonical_json(yt[k]) != canonical_json(theirs[k]) for k in yt):
        raise SystemExit("round14b's youtube-asl-v1 records are not manifest-r3b2's: refusing to rebuild from it")


def eval_sentences(store, specs, base: dict) -> list[str]:
    from build_round18_manifest import eval_sentences_of
    return eval_sentences_of(store, specs) + [r.get("english") or r.get("text") for r in base["records"]
                                              if r.get("partition") in EVAL_PARTITIONS
                                              and (r.get("english") or r.get("text"))]


def rebuild_youtube(reports: list[dict], keys: list[str], ours: dict, sentences: list[str]) -> tuple[dict, dict]:
    """(the rebuilt youtube-asl-v1 manifest, the build's reason counts), with manifest-r3b2's own filters."""
    from build_youtube_asl_manifest import EvalIndex, build
    filters = dict(ours["filters"])
    settings = {k: v for k, v in filters.items() if k not in BUILD_ONLY}
    records, why = build(reports, args=SimpleNamespace(**settings), corpus_id=YT, eval_index=EvalIndex(sentences))
    leaks = EvalIndex(sentences)
    leaked = [r["sign"] for r in records if leaks.leaks(r["text"])]
    if leaked:
        raise SystemExit(f"{len(leaked)} rebuilt YouTube records carry an evaluation text (e.g. {leaked[:3]})")
    counts = collections.Counter(r["partition"] for r in records)
    manifest = build_corpus_manifest(corpus_id=YT, records=records,
                                     inputs={"reports": hashlib.sha256("".join(keys).encode()).hexdigest()},
                                     split=dict(kind="continuous", source="all train; a hashed share of channels as monitor",
                                                counts=dict(counts)))
    manifest["license"] = ours["license"]
    manifest["filters"] = dict(settings, eval_manifest=filters.get("eval_manifest"), reasons=why,
                               eval_texts_also="round14b-mixed-v1's own val, test and heldout texts (round 19)",
                               rebuilt_from=dict(manifest_r3b2=OUR_MANIFEST, reports=len(keys)))
    return manifest, why


def replace_youtube(base_sha: str, base: dict, yt_sha: str, yt: dict, our_sha: str) -> dict:
    """round14b with its youtube-asl-v1 records replaced by ``yt``'s; every other record unchanged."""
    records = [strip(r, "usable") for r in base["records"] if r.get("corpus") != YT]
    records += [dict(strip(r, "usable"), corpus=YT) for r in yt["records"]]
    split = dict(base.get("split") or {})
    split["counts"] = {p: sum(r.get("partition") == p for r in records) for p in (split.get("counts") or {})}
    split["source"] = (split.get("source") or "") + "; youtube-asl-v1 rebuilt from the current reports (round 19)"
    manifest = build_corpus_manifest(corpus_id=MIX_ID, records=records,
                                     inputs={"round14b-mixed-v1": base_sha, f"{YT}-r19": yt_sha}, split=split)
    for key, value in base.items():
        if key not in manifest and key not in ("extends",):
            manifest[key] = value
    manifest["license"] = base.get("license")
    manifest["extends"] = base.get("extends")
    manifest["replaces"] = dict(base="round14b-mixed-v1", base_sha256=base_sha, corpus=YT, old=OUR_MANIFEST.split("@")[0],
                                old_sha256=our_sha, new_sha256=yt_sha,
                                rule="every non-YouTube record of the base unchanged (byte-identical canonical JSON); "
                                     "youtube-asl-v1 rebuilt with manifest-r3b2's filters from the current reports")
    by = {r["sign"]: r for r in manifest["records"]}
    for r in base["records"]:
        if r.get("corpus") != YT and canonical_json(by.get(r["sign"])) != canonical_json(r):
            raise SystemExit(f"{r['sign']}: a non-YouTube record changed")
    bad = [r["sign"] for r in manifest["records"] if r.get("corpus") == YT and r.get("partition") not in ("train", "monitor")]
    if bad:
        raise SystemExit(f"YouTube records outside train/monitor: {bad[:3]}")
    return manifest


def describe(manifest: dict) -> dict:
    out = collections.defaultdict(lambda: dict(records=0, hours=0.))
    for r in manifest["records"]:
        d = out[f"{r.get('corpus')}/{r.get('partition')}"]
        d["records"] += 1
        d["hours"] += ((r.get("continuous") or {}).get("seconds") or 0.) / 3600
    return {k: dict(records=v["records"], hours=round(v["hours"], 2)) for k, v in sorted(out.items())}


def youtube_delta(old: dict, new: dict, reports: list[dict], ours: dict, sentences: list[str]) -> dict:
    """Old (round14b) vs rebuilt YouTube records: kept, dropped (by the rebuilt build's reason), new."""
    from build_round18_manifest import our_cue_reasons
    from build_youtube_asl_manifest import EvalIndex
    settings = {k: v for k, v in ours["filters"].items() if k not in BUILD_ONLY}
    reasons = our_cue_reasons(reports, args=SimpleNamespace(**settings), eval_index=EvalIndex(sentences))
    o = {r["sign"].split(":", 1)[1]: r for r in old["records"] if r.get("corpus", YT) == YT}
    n = {r["sign"].split(":", 1)[1]: r for r in new["records"]}
    changed = sum(canonical_json(strip(o[k], "corpus", "usable")) != canonical_json(strip(n[k], "corpus", "usable"))
                  for k in set(o) & set(n))
    hours = lambda ids, src: round(sum(src[k]["continuous"]["seconds"] for k in ids) / 3600, 2)  # noqa: E731
    gone, added = set(o) - set(n), set(n) - set(o)
    return dict(kept=len(set(o) & set(n)), kept_but_changed=changed, dropped=len(gone), dropped_hours=hours(gone, o),
                dropped_by_reason=dict(collections.Counter(reasons.get(k, "not_in_reports") for k in gone)),
                new=len(added), new_hours=hours(added, n),
                new_partitions=dict(collections.Counter(n[k]["partition"] for k in added)))


def pilot_sample(manifest: dict) -> list[str]:
    from pilot_content_check import validation_sentences
    return [r["sign"] for r in validation_sentences(manifest, 1000)]


def build(store, args) -> dict:
    from build_mixed_text_manifest import val_prefix_unchanged
    from build_round18_manifest import fetch_manifest
    for key in (args.youtube_key, args.manifest_key):
        if store.exists(key) and not args.dry_run:
            raise SystemExit(f"refusing to overwrite {key}")
    base_sha, base = fetch_manifest(store, args.base)
    our_sha, ours = fetch_manifest(store, args.our_manifest)
    check_base(base, ours)
    sentences = eval_sentences(store, ours["filters"].get("eval_manifest") or [], base)
    keys = sorted(k for k in store.list(args.reports_prefix.rstrip("/") + "/") if k.endswith(".json"))
    with ThreadPoolExecutor(args.workers) as pool:
        reports = list(pool.map(lambda k: json.loads(store.get(k)), keys))
    yt, why = rebuild_youtube(reports, keys, ours, sentences)
    yt_bytes = canonical_json(yt)
    yt_sha = hashlib.sha256(yt_bytes).hexdigest()
    mix = replace_youtube(base_sha, base, yt_sha, yt, our_sha)
    prefix = {str(n): val_prefix_unchanged(base, mix, n) for n in (1536, 2048)}
    if not all(prefix.values()):
        raise SystemExit(f"the trainer's fixed validation batches moved: {prefix}")
    if pilot_sample(base) != pilot_sample(mix):
        raise SystemExit("the pilot's 1,000 How2Sign validation clips differ from round14b's")
    mix_bytes = canonical_json(mix)
    mix_sha = hashlib.sha256(mix_bytes).hexdigest()
    before, after = describe(base), describe(mix)
    summary = dict(base=args.base, youtube_key=args.youtube_key, youtube_sha256=yt_sha, youtube_reasons=why,
                   reports=len(keys), manifest_key=args.manifest_key, manifest_sha256=mix_sha,
                   records=dict(round14b=len(base["records"]), round19=len(mix["records"])),
                   youtube_delta=youtube_delta(ours, yt, reports, ours, sentences),
                   by_corpus=dict(round14b=before, round19=after),
                   delta={k: dict(records=after.get(k, {}).get("records", 0) - before.get(k, {}).get("records", 0),
                                  hours=round(after.get(k, {}).get("hours", 0.) - before.get(k, {}).get("hours", 0.), 2))
                          for k in sorted(set(before) | set(after))
                          if after.get(k) != before.get(k)},
                   non_youtube_identical=True, val_prefix_unchanged=prefix, pilot_sample_unchanged=True,
                   dry_run=bool(args.dry_run))
    if not args.dry_run:
        for key in (args.youtube_key, args.manifest_key):
            if store.exists(key):
                raise SystemExit(f"refusing to overwrite {key}")
        if store.put(args.youtube_key, yt_bytes) != yt_sha or store.put(args.manifest_key, mix_bytes) != mix_sha:
            raise SystemExit("the store returned another SHA-256 than the bytes written")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="r2", help="r2, or dir:<path>")
    p.add_argument("--base", default=BASE, help="KEY@SHA of round14b-mixed-v1")
    p.add_argument("--our-manifest", default=OUR_MANIFEST, help="KEY@SHA of round14b's YouTube component (manifest-r3b2)")
    p.add_argument("--reports-prefix", default=REPORTS)
    p.add_argument("--youtube-key", default=YOUTUBE_KEY)
    p.add_argument("--manifest-key", default=MIX_KEY)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    summary = build(store, args)
    print(json.dumps(summary, indent=1))
    print(f"YOUTUBE_SHA={summary['youtube_sha256']}")
    print(f"MANIFEST_SHA={summary['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
