"""Text -> motion manifest of an acquired YouTube ASL corpus (plan item 7, stage Y4).

Reads the per-video reports ``acquire_youtube_asl.py`` wrote under ``--prefix/videos/`` and keeps a caption segment
only if every check passes:

- the video's captions are the uploader's (``manual``); auto-generated captions only with ``--allow-auto``;
- the text carries words: not a bracketed or music-only note (``[demonstrates name sign]``, ``(music)``, ``♪``), not
  a URL-only line, at least ``--min-words`` words;
- duration ``--min-seconds``..``--max-seconds`` and words per second at most ``--max-words-per-second`` (a caption
  much faster than signing is not that segment's translation);
- the better-tracked hand is valid on at least ``--min-hand-coverage`` of the frames (the signer is in view);
- the same text is not repeated within a video (intro/outro boilerplate);
- the video passes the published gates (10 s-5 h, >= 360 px high, 15-60 fps, captions covering >= 40% of it), the
  caption is <= 300 characters at 0.5-4 words per second, the face is tracked on >= 90% of frames at a median head yaw
  <= 30 degrees, and it is not an evaluation sentence (exact or >= 0.8 word-5-gram Jaccard with any val/test sentence of
  ``--eval-manifest``); no channel exceeds ``--channel-cap`` of the records
  (docs/research/YOUTUBE_ASL_RESEARCH.md section 1.3).

Partitions: every kept segment is ``train``, except the segments of a small set of channels (``--monitor-share`` of
channels, chosen by hashing the channel id) labelled ``monitor``. It is a channel-disjoint split for watching
pretraining; it is never a selection or evaluation set, which stay How2Sign and ASLLRP. The licence of the result
carries the corpus's terms, and every record keeps its video id, channel, upload date and YouTube licence field.
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
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json  # noqa: E402

_NOTE = re.compile(r"^\s*[\[(][^\])]*[\])]\s*$")
_MUSIC = re.compile(r"[♪♫]")
_URL = re.compile(r"^\s*(https?://|www\.)\S+\s*$")
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def clean_text(text: str) -> str:
    """Caption text without speaker labels, bracketed notes inside the line, and extra spaces."""
    t = re.sub(r"[\[(][^\])]*[\])]", " ", text)
    t = re.sub(r"^\s*[A-Z][A-Z .'-]{1,30}:\s+", "", t)          # "NARRATOR: ..."
    return re.sub(r"\s+", " ", t).strip()


def keep_segment(segment: dict, *, args) -> str | None:
    """None if kept, else the reason."""
    text = segment["text"]
    if _NOTE.match(text) or _MUSIC.search(text) or _URL.match(text):
        return "note"
    words = _WORD.findall(clean_text(text))
    if len(words) < args.min_words:
        return "few_words"
    seconds = segment["end"] - segment["start"]
    if not args.min_seconds <= seconds <= args.max_seconds:
        return "duration"
    if len(words) / max(seconds, 1e-6) > args.max_words_per_second:
        return "too_fast"
    if len(words) / max(seconds, 1e-6) < args.min_words_per_second:
        return "too_slow"
    if len(text) > args.max_chars:
        return "too_long"
    if segment.get("hand_coverage", 0.) < args.min_hand_coverage:
        return "hands_not_visible"
    if segment.get("face_coverage") is not None and segment["face_coverage"] < args.min_face_coverage:
        return "face_not_visible"
    if segment.get("median_abs_yaw_deg") is not None and segment["median_abs_yaw_deg"] > args.max_abs_yaw:
        return "not_frontal"
    return None


def video_gate(report: dict, *, args) -> str | None:
    """Published YouTube-ASL / YouTube-SL-25 video heuristics: duration, resolution, frame rate, caption coverage."""
    meta = report.get("meta") or {}
    duration = meta.get("duration") or 0
    if not 10 <= duration <= 5 * 3600:
        return "video_duration"
    if (meta.get("height") or 0) < 360:
        return "video_resolution"
    if not 15 <= (meta.get("fps") or 0) <= 60:
        return "video_fps"
    if report.get("caption_seconds") is not None and report["caption_seconds"] < .4 * duration:
        return "video_caption_coverage"
    return None


def five_grams(text: str) -> set:
    w = _WORD.findall(text.lower())
    return {tuple(w[i:i + 5]) for i in range(max(len(w) - 4, 1))} if w else set()


class EvalIndex:
    """Normalised texts and word 5-grams of every evaluation sentence (How2Sign, ASLLRP, FLEURS-ASL val/test):
    a training caption that equals one, or shares >= ``jaccard`` of its 5-grams with one, is leakage."""

    def __init__(self, sentences, *, jaccard: float = .8):
        self.exact = {" ".join(_WORD.findall(s.lower())) for s in sentences}
        self.grams = [five_grams(s) for s in sentences]
        self.by_gram = collections.defaultdict(set)
        for i, g in enumerate(self.grams):
            for x in g:
                self.by_gram[x].add(i)
        self.jaccard = jaccard

    def leaks(self, text: str) -> bool:
        if " ".join(_WORD.findall(text.lower())) in self.exact:
            return True
        g = five_grams(text)
        candidates = set().union(*(self.by_gram.get(x, set()) for x in g)) if g else set()
        return any(len(g & self.grams[i]) / max(len(g | self.grams[i]), 1) >= self.jaccard for i in candidates)


def monitor_channel(channel_id: str, share: float) -> bool:
    return int(hashlib.sha256((channel_id or "").encode()).hexdigest()[:8], 16) / 16 ** 8 < share


def build(reports, *, args, corpus_id: str, eval_index: EvalIndex | None = None) -> tuple[list[dict], dict]:
    records, why = [], collections.Counter()
    for report in reports:
        if report.get("status") != "converted":
            why[f"video_{report.get('status')}"] += 1
            continue
        if report.get("caption_kind") != "manual" and not args.allow_auto:
            why["video_auto_captions"] += len(report.get("segments", [])); continue
        gate = video_gate(report, args=args)
        if gate:
            why[gate] += 1; continue
        meta = report.get("meta") or {}
        partition = "monitor" if monitor_channel(meta.get("channel_id"), args.monitor_share) else "train"
        seen = set()
        for seg in report.get("segments", []):
            reason = keep_segment(seg, args=args)
            text = clean_text(seg["text"])
            if reason is None and text.lower() in seen:
                reason = "repeated_text"
            if reason is None and eval_index is not None and eval_index.leaks(text):
                reason = "evaluation_leak"
            if reason:
                why[reason] += 1; continue
            seen.add(text.lower())
            records.append(dict(sign=f"{corpus_id}:{seg['sample_id']}", text=text, partition=partition,
                                performer_sha256=None,
                                source=dict(video_id=report["video_id"], cue=seg["cue"], start=seg["start"],
                                            end=seg["end"], channel_id=meta.get("channel_id"),
                                            upload_date=meta.get("upload_date"), youtube_license=meta.get("license"),
                                            caption_kind=report.get("caption_kind")),
                                quality=dict(hand_coverage=seg.get("hand_coverage")),
                                continuous=dict(status="converted", state_key=seg["state_key"],
                                                state_sha256=seg["state_sha256"], seconds=seg["seconds"],
                                                available_channels=seg["available_channels"])))
            why["kept"] += 1
    if args.channel_cap < 1 and records:                     # no channel above its share of the kept records
        cap = max(1, int(args.channel_cap * len(records)))
        per, kept = collections.Counter(), []
        for r in sorted(records, key=lambda r: hashlib.sha256(r["sign"].encode()).hexdigest()):
            c = r["source"]["channel_id"]
            if per[c] < cap:
                per[c] += 1; kept.append(r)
            else:
                why["channel_cap"] += 1
        why["kept"] = len(kept)
        records = sorted(kept, key=lambda r: r["sign"])
    return records, dict(why)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path>")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--licence", required=True, help="the corpus's terms, recorded in the manifest")
    parser.add_argument("--allow-auto", action="store_true")
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--min-seconds", type=float, default=1.)
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--max-words-per-second", type=float, default=4.)
    parser.add_argument("--min-hand-coverage", type=float, default=.5)
    parser.add_argument("--monitor-share", type=float, default=.01)
    parser.add_argument("--min-words-per-second", type=float, default=.5)
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--min-face-coverage", type=float, default=.9)
    parser.add_argument("--max-abs-yaw", type=float, default=30.)
    parser.add_argument("--channel-cap", type=float, default=.05, help="largest share of records from one channel")
    parser.add_argument("--eval-manifest", action="append", default=[],
                        help="KEY@SHA of a manifest whose val/test sentences must not appear in training (repeatable)")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    keys = sorted(k for k in store.list(f"{args.prefix.rstrip('/')}/videos/") if k.endswith(".json"))
    reports = [json.loads(store.get(k)) for k in keys]
    sentences = []
    for spec in args.eval_manifest:
        import tempfile
        from signtranslator.data_engineering.corpus_manifest import load_manifest
        key, _, sha = spec.rpartition("@")
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.json"; p.write_bytes(store.get(key) + b"\n")
            m = load_manifest(p, expected_sha256=sha)
        sentences += [r.get("english") or r.get("text") for r in m["records"]
                      if r.get("partition") in ("val", "test", "heldout_official") and (r.get("english") or r.get("text"))]
    records, why = build(reports, args=args, corpus_id=args.corpus_id,
                         eval_index=EvalIndex(sentences) if sentences else None)
    if not records:
        raise SystemExit(f"no segment passed the filters: {why}")
    digest_inputs = {"reports": hashlib.sha256("".join(keys).encode()).hexdigest()}
    counts = collections.Counter(r["partition"] for r in records)
    manifest = build_corpus_manifest(corpus_id=args.corpus_id, records=records, inputs=digest_inputs,
                                     split=dict(kind="continuous", source="all train; a hashed share of channels as monitor",
                                                counts=dict(counts)))
    manifest["license"] = args.licence
    manifest["filters"] = dict(vars(args) | {"reasons": why})
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.manifest_key, canonical_json(manifest))
    hours = sum(r["continuous"]["seconds"] for r in records) / 3600
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, records=len(records),
                          hours=round(hours, 2), partitions=dict(counts), reasons=why)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
