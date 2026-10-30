"""Text -> motion manifest of a sentence-video corpus extracted by ``extract_video_corpus.py`` (instrument v2;
docs/NEXT_PHASE_PLAN.md, "Instrument v2: design and declared gates").

Reads the per-video reports ``<prefix>/videos/<group>/<stem>.json`` (the extractor's layout: ``<group>`` is the
split directory, or ``--group-column``'s value in zip mode, e.g. the ASL STEM Wiki participant) and keeps a video only
if it converted, has text (``--text-field``, e.g. ``"sentence text"`` for ASL STEM Wiki, ``sentence`` for
2M-Flores) of at least ``--min-words`` words, lasts ``--min-seconds``..``--max-seconds`` and has the better-tracked
hand on at least ``--min-hand-coverage`` of its frames. Every drop is counted by reason.

Partitions:
- ``--crossdomain`` (2M-Flores): every record is ``crossdomain``; ``bt_corpora`` gives it the cross-domain role and the
  back-translator's trainer refuses it (the corpus is held out entirely, gate G4);
- otherwise ``train``, except the held-out groups, ``holdout`` (reported only): ``--holdout-groups G,G,...`` names
  them, or ``--holdout-count N`` takes the first N groups in the order of SHA-256("<corpus id>\\n<group>") (a fixed,
  content-free choice), recorded in the manifest (``holdout_groups``).

Each record keeps the report's text fields verbatim under ``source``, its group as ``source.signer_id`` (the
participant: signer-disjoint splits), the extraction report's key and SHA-256, and the corpus licence is written into
the manifest (``--licence``). Written once (an existing manifest key is refused).

    python scripts/build_video_corpus_manifest.py --prefix acquisition/2026-10-30-v1/asl-stem-wiki/extract-v1 \\
        --corpus-id asl-stem-wiki-v1 --text-field "sentence text" --holdout-count 4 \\
        --licence "Microsoft Research License (instruments only)" --manifest-key .../asl-stem-wiki-v1/manifest.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STATE_FIELDS = ("base", "status", "state_key", "state_sha256", "landmarks_key", "landmarks_sha256", "frames", "seconds",
                "hand_coverage", "face_coverage", "median_abs_yaw_deg", "available_channels", "reason")


def group_order(corpus_id: str, groups) -> list[str]:
    return sorted(groups, key=lambda g: hashlib.sha256(f"{corpus_id}\n{g}".encode()).hexdigest())


def holdout_of(corpus_id: str, groups, *, names=None, count: int = 0) -> list[str]:
    groups = sorted(set(groups))
    if names:
        unknown = sorted(set(names) - set(groups))
        if unknown:
            raise SystemExit(f"--holdout-groups: no report of group(s) {unknown}")
        chosen = sorted(set(names))
    else:
        chosen = sorted(group_order(corpus_id, groups)[:count])
    if chosen and len(chosen) >= len(groups):
        raise SystemExit("every group would be held out")
    return chosen


def build(reports, *, corpus_id: str, text_field: str, crossdomain: bool, holdout, min_words: int,
          min_seconds: float, max_seconds: float, min_hand_coverage: float) -> tuple[list[dict], dict]:
    """(records, drop counts) from ``(report key, report bytes)`` pairs."""
    records, why = [], collections.Counter()
    for key, payload in reports:
        report = json.loads(payload)
        group = key.rsplit("/", 2)[-2]
        if report.get("status") != "converted":
            why[f"video_{report.get('status')}"] += 1
            continue
        text = re.sub(r"\s+", " ", str(report.get(text_field) or "")).strip()
        if len(_WORD.findall(text)) < min_words:
            why["few_words"] += 1
            continue
        if not min_seconds <= float(report["seconds"]) <= max_seconds:
            why["duration"] += 1
            continue
        if (report.get("hand_coverage") or 0.) < min_hand_coverage:
            why["hands_not_visible"] += 1
            continue
        partition = "crossdomain" if crossdomain else ("holdout" if group in holdout else "train")
        fields = {k: v for k, v in report.items() if k not in STATE_FIELDS}
        records.append(dict(
            sign=f"{corpus_id}:{report['base']}", text=text, partition=partition, performer_sha256=None,
            source=dict(fields, group=group, signer_id=group, report_key=key,
                        report_sha256=hashlib.sha256(payload).hexdigest()),
            quality=dict(hand_coverage=report.get("hand_coverage"), face_coverage=report.get("face_coverage")),
            continuous=dict(status="converted", state_key=report["state_key"], state_sha256=report["state_sha256"],
                            seconds=float(report["seconds"]), available_channels=report["available_channels"])))
        why[f"kept:{partition}"] += 1
    return records, dict(why)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="r2", help="r2, or dir:<path>")
    p.add_argument("--prefix", required=True, help="extract_video_corpus.py --out")
    p.add_argument("--corpus-id", required=True)
    p.add_argument("--manifest-key", required=True)
    p.add_argument("--text-field", required=True)
    p.add_argument("--licence", required=True)
    p.add_argument("--crossdomain", action="store_true", help="every record is crossdomain (never trained on)")
    p.add_argument("--holdout-groups", help="comma-separated groups held out (reported only)")
    p.add_argument("--holdout-count", type=int, default=0, help="or: hold out the first N groups by SHA-256 order")
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--min-seconds", type=float, default=.5)
    p.add_argument("--max-seconds", type=float, default=20.)
    p.add_argument("--min-hand-coverage", type=float, default=.5)
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json
    if args.crossdomain and (args.holdout_groups or args.holdout_count):
        raise SystemExit("--crossdomain holds the whole corpus out: no holdout groups")
    if args.holdout_groups and args.holdout_count:
        raise SystemExit("--holdout-groups or --holdout-count, not both")
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    prefix = args.prefix.rstrip("/")
    keys = sorted(k for k in store.list(f"{prefix}/videos/") if k.endswith(".json"))
    if not keys:
        raise SystemExit(f"no reports under {prefix}/videos/")
    groups = {k.rsplit("/", 2)[-2] for k in keys}
    holdout = holdout_of(args.corpus_id, groups, names=args.holdout_groups.split(",") if args.holdout_groups else None,
                         count=args.holdout_count)
    reports = [(k, store.get(k)) for k in keys]
    records, why = build(reports, corpus_id=args.corpus_id, text_field=args.text_field, crossdomain=args.crossdomain,
                         holdout=set(holdout), min_words=args.min_words, min_seconds=args.min_seconds,
                         max_seconds=args.max_seconds, min_hand_coverage=args.min_hand_coverage)
    if not records:
        raise SystemExit(f"no video passed the filters: {why}")
    reports_digest = hashlib.sha256(b"".join(hashlib.sha256(k.encode() + b"\0" + v).digest() for k, v in reports))
    counts = collections.Counter(r["partition"] for r in records)
    manifest = build_corpus_manifest(corpus_id=args.corpus_id, records=records,
                                     inputs={"reports": reports_digest.hexdigest()},
                                     split=dict(kind="continuous", counts=dict(counts),
                                                source="crossdomain: held out entirely" if args.crossdomain else
                                                "train; held-out groups (signer-disjoint) as holdout"))
    manifest.update(license=args.licence, holdout_groups=holdout, groups=sorted(groups), crossdomain=args.crossdomain,
                    filters=dict({k: v for k, v in vars(args).items() if k != "store"}, reasons=why))
    digest = store.put(args.manifest_key, canonical_json(manifest))
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, records=len(records),
                          partitions=dict(counts), holdout_groups=holdout, groups=len(groups), reasons=why)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
