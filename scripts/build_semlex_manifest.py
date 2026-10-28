"""Corpus manifest of Sem-Lex one-word records from the converter's learned-path states (local; writes a local file).

Reads ``convert_semlex_poses.py``'s per-shard reports under ``--prefix`` (default the pinned learned path,
``states-semlex-holistic-learned-world-v2-<sha12>/``), the pinned Sem-Lex metadata CSV and the ASL-LEX 2.0
``signdata.csv`` (pinned by its provenance record), and writes one manifest in the schema of our other lexical
corpora (``corpus_manifest.build_corpus_manifest``; records like ``rit-text-r6`` / ``wlasl-text-r6``).

**Text.**
- ``asllex`` rows: the label is an ASL-LEX EntryID (``cat_3``); its row gives the Code and the LemmaID; the text is the
  lemma with a trailing variant suffix removed and underscores read as spaces (``cat_3`` -> "cat",
  ``hard_of_hearing`` -> "hard of hearing"). The variant stays in ``gloss.variant`` and the code in
  ``gloss.asl_lex_code``.
- ``freetext`` rows: kept only if, after trimming spaces and reading underscores as spaces (a trailing ``_<digits>``
  variant is dropped), the label is 1-4 lower-case words of letters (internal ``'`` or ``-`` allowed) and every word is
  a lower-case entry of the English word list given by ``--wordlist`` (its SHA-256 is recorded). Proper nouns,
  misspellings, digits and symbols are refused.
- ``signbank`` rows: the metadata gives a Signbank id but no English, so they are skipped (counted, with how many
  ASL-LEX would map through ``SignBankReferenceID``, for information).

**Partitions.** Sem-Lex's own splits, unchanged: train, val, test (held out). They share signers, so the split is
recorded as not signer-disjoint, with the overlap. ``performer_sha256`` is the SHA-256 of ``semlex-signer:<id>``.

**Repeated ids.** The metadata repeats 2,854 video ids (2,974 extra rows). Exact repeats collapse to one row. An id
whose rows disagree on label type or label is skipped (``skip:ambiguous_label``); an id listed in train and val (995)
is kept in train only (``info:train_val_listing_kept_in_train``); an id listed in test and in another split, or with
rows that disagree on duration within its split, is skipped. Conversion results are matched by (id, split).

**Filters** (each counted): no converted state (refused, failed, error or missing in the reports); below
``--min-fps``; a hand (either side) tracked on fewer than ``--min-hand`` of the frames; then the trim of
``build_trimmed_word_manifest.py`` (``sign_window``, the dominant wrist's active span on the state at 24 fps, the same
dominant-side and tracked-arm rules, recorded as ``continuous.window_seconds``).

**Coverage** of the R5 words: how many of the uncovered words now have Sem-Lex *train* records whose text equals the
word (the rule of the R5 derivation), and the projected ceiling.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

BASE = "acquisition/2026-10-17-v1/sem-lex-mirror-v1"
ASLLEX_KEY = "acquisition/2026-10-17-v1/asl-lex-2.0/signdata.csv"
CORPUS_ID = "semlex-text-v1"
LICENCE = ("semlex-text-v1: Sem-Lex Benchmark (Kezar et al. 2023, arXiv 2310.00196), CC BY-NC-SA 4.0 (the "
           "publisher's terms, accepted by the owner): research, non-commercial, attribution, share-alike; obtained "
           "from the Hugging Face mirror tylerxia8/sem-lex-pilot-mirror (labelled Apache-2.0, which is not the licence); "
           "text from ASL-LEX 2.0 lemmas (CC BY 4.0); private store, no redistribution")
CITATION = ("Kezar, L., Thomason, J., Caselli, N., Sehyr, Z., Pontecorvi, E. (2023). The Sem-Lex Benchmark: Modeling "
            "ASL Signs and Their Phonemes. arXiv:2310.00196")
WORD = re.compile(r"[a-z]+(?:['-][a-z]+)*")


def performer(signer_id: str) -> str:
    return hashlib.sha256(f"semlex-signer:{signer_id}".encode()).hexdigest()


def asllex_text(lemma: str) -> str:
    """ASL-LEX LemmaID -> English text: a trailing ``_<digits>`` variant removed, underscores read as spaces."""
    return re.sub(r"_\d+$", "", lemma.strip()).replace("_", " ").strip()


def freetext_text(label: str, words: set[str]) -> str | None:
    """A free-text label as English text, or None when it fails the strict check."""
    text = re.sub(r"_\d+$", "", label.strip()).replace("_", " ").strip()
    tokens = text.split(" ")
    if not text or len(tokens) > 4 or any(not WORD.fullmatch(t) for t in tokens):
        return None
    if any(t not in words and t.replace("'", "") not in words for t in tokens):
        return None
    return " ".join(tokens)


def load_wordlist(path: Path) -> tuple[set[str], str]:
    payload = Path(path).read_bytes()
    words = {w.strip() for w in payload.decode("utf-8", "replace").splitlines()}
    return {w for w in words if w and w == w.lower()}, hashlib.sha256(payload).hexdigest()


def latest_results(store, prefix: str) -> tuple[dict, list[str], set[str]]:
    """Per video id, the last result with a stored state across all shard reports (later reports win), the report
    keys read, and the bundle SHA-256s the reports name."""
    keys = sorted(k for k in store.list(f"{prefix}/reports/") if k.endswith(".json"))
    results, bundles = {}, set()
    for key in keys:
        report = json.loads(store.get(key))
        bundles.add(report.get("regressors_sha256"))
        for r in report["results"]:
            key_ = (r["video_id"], r.get("split"))
            previous = results.get(key_)
            if r.get("state_sha256") or previous is None or not previous.get("state_sha256"):
                results[key_] = r
    return results, keys, bundles


def resolve_rows(rows: list[dict]) -> tuple[dict | None, str | None]:
    """One metadata row for a video id with (possibly) repeated rows, or (None, reason)."""
    if len({(r["label_type"], r["label"]) for r in rows}) > 1:
        return None, "skip:ambiguous_label"
    splits = {r["split"] for r in rows}
    if "test" in splits and len(splits) > 1:
        return None, "skip:test_and_other_split"
    note = None
    if splits == {"train", "val"}:
        rows, note = [r for r in rows if r["split"] == "train"], "info:train_val_listing_kept_in_train"
    elif len(splits) > 1:
        return None, "skip:conflicting_splits"
    if len({r["duration"] for r in rows}) > 1:
        return None, "skip:conflicting_durations"
    return rows[0], note


def trim_window(state, threshold: float, margin: float, min_tracked: float, min_seconds: float):
    """``build_trimmed_word_manifest``'s rule on one state, at 24 fps: (window, reason) (the rule lives in
    ``signtranslator.generation.compositional`` since round 15, moved unchanged)."""
    from signtranslator.generation.compositional import trim_window as rule
    from signtranslator.pose.resample import resample_motion
    return rule(resample_motion(state, 24.), threshold, margin, min_tracked, min_seconds)


def build(store, args) -> tuple[dict, dict]:
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest
    from signtranslator.data_engineering.semlex_pose import LEARNED_ADAPTER
    from signtranslator.pose.multichannel import MultichannelMotion
    from convert_semlex_poses import METADATA_KEY, METADATA_SHA256, load_metadata
    metadata = load_metadata(store, args.metadata_key or METADATA_KEY, args.metadata_sha256 or METADATA_SHA256)
    provenance = json.loads(store.get(args.asllex_key + ".provenance.json"))
    asllex_bytes = store.get(args.asllex_key)
    asllex_sha = hashlib.sha256(asllex_bytes).hexdigest()
    if provenance.get("sha256") != asllex_sha or not provenance.get("verified_by_r2_reread"):
        raise SystemExit("ASL-LEX signdata.csv is not the verified acquisition")
    asllex = {r["EntryID"]: r for r in csv.DictReader(io.StringIO(asllex_bytes.decode("utf-8", "replace")))}
    signbank_ids = {r.get("SignBankReferenceID", "").split(".")[0] for r in asllex.values()}
    words, wordlist_sha = load_wordlist(Path(args.wordlist))
    results, report_keys, bundles = latest_results(store, args.prefix)
    bundles.discard(None)
    if len(bundles) > 1:
        raise SystemExit(f"the reports name more than one regressor bundle: {sorted(bundles)}")
    counts = collections.Counter()
    records, signers_by_split = [], collections.defaultdict(set)
    for video_id, rows in sorted(metadata.items()):
        row, why = resolve_rows(rows)
        if row is None:
            counts[why] += 1
            continue
        if why:
            counts[why] += 1
        split = row["split"]
        counts[f"metadata:{row['label_type']}"] += 1
        if row["label_type"] == "asllex":
            entry = asllex.get(row["label"])
            if entry is None:
                counts["skip:asllex_label_not_in_asllex"] += 1
                continue
            text = asllex_text(entry["LemmaID"] or entry["EntryID"])
            gloss = dict(label=row["label"], variant=row["label"], asl_lex_code=entry["Code"], status="asllex_label",
                         lemma=entry["LemmaID"])
        elif row["label_type"] == "freetext":
            text = freetext_text(row["label"], words)
            if text is None:
                counts["skip:freetext_not_plain_english"] += 1
                continue
            gloss = dict(label=row["label"], status="semlex_freetext")
        else:
            counts["skip:signbank_no_english_in_metadata"] += 1
            if row["label"].split(".")[0] in signbank_ids:
                counts["info:signbank_mappable_through_asllex"] += 1
            continue
        result = results.get((video_id, split))
        if result is None:
            counts["drop:no_conversion_result"] += 1
            continue
        if not result.get("state_sha256"):
            counts[f"drop:conversion_{result['status']}"] += 1
            continue
        payload = store.get(result["state_key"])
        if hashlib.sha256(payload).hexdigest() != result["state_sha256"]:
            raise SystemExit(f"{result['state_key']}: state bytes differ from the report")
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "s.npz"
            path.write_bytes(payload)
            state = MultichannelMotion.load(path, expected_sha256=result["state_sha256"])
        frames = len(state.timestamps)
        fps = frames / (float(row["duration"]) / 1000.)
        if fps < args.min_fps:
            counts["drop:fps_below_min"] += 1
            continue
        hand = (state.channels["left_hand"].valid.all(1) | state.channels["right_hand"].valid.all(1)).mean()
        if hand < args.min_hand:
            counts["drop:hand_tracked_below_min"] += 1
            continue
        window, reason = trim_window(state, args.threshold, args.margin, args.min_tracked, args.min_seconds)
        counts[f"trim:{reason}"] += 1
        if window is None:
            continue
        available = sorted(n for n, c in state.channels.items() if c.valid.any())
        quality = dict(frames=frames, fps=round(fps, 3), hand_fraction=round(float(hand), 4))
        records.append(dict(
            sign=f"semlex-{video_id}", corpus=CORPUS_ID, partition=split, text=text, gloss=gloss,
            performer_sha256=performer(row["signer_id"]), quality=quality,
            source=dict(video_id=video_id, signer_id=row["signer_id"], label_type=row["label_type"],
                        semlex_split=split, adapter=result.get("adapter", LEARNED_ADAPTER),
                        regressors_sha256=result.get("regressors_sha256")),
            continuous=dict(status="converted", state_key=result["state_key"], state_sha256=result["state_sha256"],
                            available_channels=available, window_seconds=list(window),
                            seconds=round(window[1] - window[0] + 1 / 24., 4))))
        signers_by_split[split].add(row["signer_id"])
        counts[f"kept:{split}"] += 1
    overlap = {f"{a}&{b}": len(signers_by_split[a] & signers_by_split[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    split = dict(kind="continuous", source="Sem-Lex's own splits, unchanged (test held out)", signer_disjoint=False,
                 signer_overlap=overlap,
                 counts={p: sum(r["partition"] == p for r in records) for p in ("train", "val", "test")})
    inputs = {"semlex_metadata": args.metadata_sha256 or METADATA_SHA256, "asl_lex_signdata": asllex_sha,
              "english_wordlist": wordlist_sha}
    for key in report_keys:
        inputs[f"report:{key.rsplit('/', 1)[-1]}"] = hashlib.sha256(store.get(key)).hexdigest()
    manifest = build_corpus_manifest(corpus_id=CORPUS_ID, inputs=inputs, records=records, split=split)
    bundle = next(iter(bundles), None)
    manifest["license"] = LICENCE
    manifest["provenance"] = dict(citation=CITATION, mirror="huggingface.co/datasets/tylerxia8/sem-lex-pilot-mirror@"
                                  "bf340cbfc4522ecc4dcd0fc247e33c4cb3498275", adapter=LEARNED_ADAPTER,
                                  regressors_sha256=bundle, states_prefix=args.prefix, convention="generator")
    manifest["trim"] = dict(rule="one-word clips cut to the dominant wrist's active span (height above rest + threshold x "
                                 "(peak - rest)) +- margin (build_trimmed_word_manifest.sign_window)",
                            threshold=args.threshold, margin=args.margin, min_tracked=args.min_tracked,
                            min_seconds=args.min_seconds)
    manifest["filters"] = dict(min_fps=args.min_fps, min_hand=args.min_hand, counts=dict(sorted(counts.items())))
    coverage = None
    if args.r5:
        derivation = json.loads(Path(args.r5).read_text())
        r5 = [w.upper() for w in derivation["r5"]]
        uncovered = sorted(Path(args.uncovered).read_text().split()) if args.uncovered else \
            sorted(derivation["round13"]["uncovered"])
        train_texts = collections.Counter(r["text"].lower() for r in records if r["partition"] == "train")
        now = {w: train_texts.get(w.lower(), 0) for w in uncovered}
        newly = sorted(w for w, n in now.items() if n)
        covered_before = len(r5) - len(uncovered)
        coverage = dict(r5_words=len(r5), covered_before=covered_before, uncovered=len(uncovered),
                        now_with_semlex_train_clips=len(newly), words=newly, clips_per_word=now,
                        still_uncovered=sorted(set(uncovered) - set(newly)),
                        ceiling_before=round(covered_before / len(r5), 4),
                        projected_ceiling=round((covered_before + len(newly)) / len(r5), 4))
    return manifest, dict(counts=dict(sorted(counts.items())), coverage=coverage, records=len(records),
                          split=split, regressors_sha256=bundle)


def main(argv=None) -> int:
    from signtranslator.data_engineering.world_regressor import WORLD_REGRESSORS
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2", help="'r2' or 'dir:/path' (tests)")
    p.add_argument("--prefix", default=f"{BASE}/states-semlex-holistic-learned-world-v2-{WORLD_REGRESSORS['sha256'][:12]}")
    p.add_argument("--metadata-key")
    p.add_argument("--metadata-sha256")
    p.add_argument("--asllex-key", default=ASLLEX_KEY)
    p.add_argument("--wordlist", default="/usr/share/dict/words", help="English word list for free-text labels")
    p.add_argument("--min-fps", type=float, default=20.)
    p.add_argument("--min-hand", type=float, default=.3)
    p.add_argument("--threshold", type=float, default=.1)
    p.add_argument("--margin", type=float, default=.25)
    p.add_argument("--min-tracked", type=float, default=.3)
    p.add_argument("--min-seconds", type=float, default=.5)
    p.add_argument("--r5", help="R5 derivation JSON (the 150 words, round13 covered/uncovered)")
    p.add_argument("--uncovered", help="the uncovered R5 words (whitespace-separated); default: from --r5")
    p.add_argument("--output", required=True, help="local manifest path (written once)")
    p.add_argument("--summary", help="local summary JSON path")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    from signtranslator.data_engineering.corpus_manifest import canonical_json
    from convert_semlex_poses import make_store
    args.prefix = args.prefix.rstrip("/")
    manifest, summary = build(make_store(args.store), args)
    payload = canonical_json(manifest)
    with open(args.output, "xb") as stream:
        stream.write(payload + b"\n")
    summary["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "coverage"} | {
        "coverage": {k: v for k, v in (summary["coverage"] or {}).items() if k != "clips_per_word"}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
