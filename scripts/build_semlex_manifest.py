"""Corpus manifest of Sem-Lex one-word records, from the converter's pose-path states or from our own video extraction.

Two sources (``--source``), one text mapping, one set of filters and one trim:

- ``pose`` (default): ``convert_semlex_poses.py``'s per-shard reports under ``--prefix`` (default the pinned learned
  path, ``states-semlex-holistic-learned-world-v2-<sha12>/``); one regressor bundle across the reports.
- ``video``: ``extract_semlex_videos.py``'s per-shard reports under ``--prefix`` (default ``states-video-v1/``): Sem-Lex's
  original val and train videos through our own MediaPipe Tasks extraction and solver under the generator convention.
  Only ``--splits`` (default val, train) are built; the Sem-Lex **test** split is not in these states (pose files only)
  and is left out of this manifest: it is a separate held-out set. Reports (several per shard after reruns; a crashed
  pass may have written none, or one that is not a full pass):
  - a report counts only if it is a full production pass: not a smoke run, and the whole archive was streamed to its
    pinned SHA-256 (``archive_stream.matches_pinned``); others are listed in the summary and not used;
  - every shard of every built split needs at least one such report (a smoke run with ``--limit`` may lack some);
  - all used reports must name the same MediaPipe Tasks models, convention, metadata and (per split) archive;
  - per clip, the **latest** used report (by its start time) whose result is final wins: a state (``converted``,
    ``resumed``, ``already_present``) or ``refused``. ``error``, ``failed`` and ``exists_different`` are not final, so an
    earlier final result stands; two final results naming different states are refused (the stores are write-once); a
    later refusal never replaces a stored state (counted);
  - a clip whose best result is still ``error``, ``failed`` or ``exists_different`` stops the build (rerun the
    extraction shard, or pass ``--allow-unfinished`` to drop them, counted), so the write-once manifest never loses
    clips a rerun would recover;
  - a metadata id of a built split with no result is a pinned archive gap (``semlex_known_missing.json``) or an error.
  - ``--states-prefix states-video-v1-unmirrored``: the states come from ``unmirror_semlex_states.py`` (the
    recordings judged mirrored, round 14's handedness test): per clip its latest final un-mirror result, which must
    name exactly the extracted state the extraction reports name (``source_state_sha256``) and sit at its key; a
    clip without one stops the build (``--allow-unfinished`` drops it). The manifest records ``unmirrored: true``
    (top level and provenance) and each record's ``source.unmirrored``, extracted state key and SHA-256.

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

**Partitions.** Sem-Lex's own splits, unchanged. They share signers, so the split is recorded as not signer-disjoint
(``signer_disjoint: false``), with the overlap. ``performer_sha256`` is the SHA-256 of ``semlex-signer:<id>``.

**Repeated ids.** The metadata repeats 2,854 video ids (2,974 extra rows). Exact repeats collapse to one row. An id
whose rows disagree on label type or label is skipped (``skip:ambiguous_label``); an id listed in train and val (995)
is kept in train only (``info:train_val_listing_kept_in_train``); an id listed in test and in another split, or with
rows that disagree on duration within its split, is skipped. Results are matched by (id, split).

**Filters** (each counted): no state (refused, failed, error or missing in the reports); below ``--min-fps`` (frames
over the metadata duration); a hand (either side) tracked on fewer than ``--min-hand`` of the frames; then the trim of
``build_trimmed_word_manifest.py`` (``sign_window``, the dominant wrist's active span on the state at 24 fps, the same
dominant-side and tracked-arm rules, recorded as ``continuous.window_seconds``). Every state's bytes are hashed against
its report before use.

**Coverage** of the R5 words, by ``exposure_diagnostic.py``'s matching rule (a record's text, upper-cased with spaces
removed, equals the word), on *train* records. ``--coverage-mix KEY@SHA`` derives the 150 words from that mix
(``pilot_content_check.py``'s rule: the recogniser's reliable alphabetic glosses that occur in the mix's How2Sign
training sentences, sorted, the first 150; recogniser validation pinned by ``--recognizer-validation``) and counts the
words' existing one-word training clips (``--one-word-corpus``); ``--r5`` reads a derivation JSON instead. Reported:
how many uncovered words now have at least 1 and at least 5 Sem-Lex training clips, and the projected ceiling.

**Outputs.** ``--output`` (local, written once, canonical bytes plus a newline: ``load_manifest``'s file form);
``--output-key`` / ``--summary-key`` upload the manifest (canonical bytes, no newline, as every manifest in the store:
its SHA-256 is the manifest SHA) and the summary with the store's ``put`` after checking neither key exists. A
``--limit`` (smoke) build may only upload under the smoke prefix. ``--workers N`` reads the states in N spawned
processes (the result is the same as in one).
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

BASE = "acquisition/2026-10-17-v1/sem-lex-mirror-v1"
SMOKE_BASE = f"{BASE}/adapter-dev/smoke"
ASLLEX_KEY = "acquisition/2026-10-17-v1/asl-lex-2.0/signdata.csv"
VIDEO_PREFIX = f"{BASE}/states-video-v1"
VIDEO_EXTRACTOR = "semlex-video-v1"
VIDEO_SPLITS = ("val", "train")
FINAL_STATES = ("converted", "resumed", "already_present")
RECOGNIZER_VALIDATION = ("training-runs/sign-recognizer-v2/validation.json@"
                         "a23117eca843c520b84ed6d66fe885f9952a8c3cdd0abe87cc5ac5981d98ee05")
ONE_WORD_CORPORA = ("asllvd-text-v1", "rit-text-r6", "wlasl-text-r6")
CORPUS_ID = "semlex-text-v1"
LICENCE = ("semlex-text-v1: Sem-Lex Benchmark (Kezar et al. 2023, arXiv 2310.00196), CC BY-NC-SA 4.0 (the "
           "publisher's terms, accepted by the owner): research, non-commercial, attribution, share-alike; obtained "
           "from the Hugging Face mirror tylerxia8/sem-lex-pilot-mirror (labelled Apache-2.0, which is not the licence); "
           "text from ASL-LEX 2.0 lemmas (CC BY 4.0); private store, no redistribution")
CITATION = ("Kezar, L., Thomason, J., Caselli, N., Sehyr, Z., Pontecorvi, E. (2023). The Sem-Lex Benchmark: Modeling "
            "ASL Signs and Their Phonemes. arXiv:2310.00196")
MIRROR = "huggingface.co/datasets/tylerxia8/sem-lex-pilot-mirror@bf340cbfc4522ecc4dcd0fc247e33c4cb3498275"
WORD = re.compile(r"[a-z]+(?:['-][a-z]+)*")
_WORKER: dict = {}


class StateMismatch(Exception):
    """A stored state whose bytes differ from its report."""


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


def parse_pinned(spec: str) -> tuple[str, str]:
    key, _, sha = spec.rpartition("@")
    if not key or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise SystemExit(f"{spec!r}: expected KEY@<64-hex SHA-256>")
    return key, sha


def get_pinned(store, spec: str) -> tuple[bytes, str]:
    key, sha = parse_pinned(spec)
    payload = store.get(key)
    if hashlib.sha256(payload).hexdigest() != sha:
        raise SystemExit(f"{key}: bytes differ from the pinned SHA-256")
    return payload, key


# ------------------------------------------------------------------------------------------------------------ reports

def latest_results(store, prefix: str) -> tuple[dict, list[str], set[str]]:
    """Pose path: per (video id, split), the last result with a stored state across all shard reports (later reports
    win), the report keys read, and the bundle SHA-256s the reports name."""
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


def _final(result: dict | None) -> bool:
    if result is None:
        return False
    return (result.get("status") in FINAL_STATES and bool(result.get("state_sha256"))) or result.get("status") == "refused"


def video_results(store, prefix: str, splits, *, require_all_shards: bool = True) -> tuple[dict, dict]:
    """Video path: per (video id, split), the result of the latest used report whose result for that clip is final
    (see the module docstring), and what was read: used and ignored reports with their SHA-256, the shards, the
    models, convention, inference methods, metadata and archive hashes."""
    import convert_semlex_poses
    from convert_semlex_poses import shard_of
    KNOWN_MISSING = convert_semlex_poses.KNOWN_MISSING
    keys = sorted(k for k in store.list(f"{prefix}/reports/") if k.endswith(".json"))
    used, ignored = [], []
    for key in keys:
        payload = store.get(key)
        report = json.loads(payload)
        if report.get("extractor") != VIDEO_EXTRACTOR:
            raise SystemExit(f"{key}: not a {VIDEO_EXTRACTOR} report ({report.get('extractor')!r})")
        if report.get("states_prefix") != prefix:
            raise SystemExit(f"{key}: names states prefix {report.get('states_prefix')!r}, not {prefix!r}")
        why = None
        if report.get("split") not in splits:
            why = f"split {report.get('split')!r} not built"
        elif report.get("smoke") or report.get("limit") or report.get("match"):
            why = "smoke run"
        elif (report.get("archive_stream") or {}).get("matches_pinned") is not True:
            why = "not a full pass (the archive was not streamed to its pinned SHA-256)"
        if why:
            ignored.append(dict(report=key, reason=why, sha256=hashlib.sha256(payload).hexdigest()))
            continue
        used.append((report["started"], key, hashlib.sha256(payload).hexdigest(), report))
    used.sort(key=lambda t: (t[0], t[1]))
    if not used:
        raise SystemExit(f"no usable {VIDEO_EXTRACTOR} report under {prefix}/reports/")

    def one(field, name):
        values = {json.dumps(r.get(field), sort_keys=True) for *_, r in used}
        if len(values) != 1:
            raise SystemExit(f"the reports name more than one {name}: {sorted(values)}")
        return json.loads(values.pop())
    models, convention = one("models", "MediaPipe Tasks model set"), one("convention", "convention")
    metadata_sha256 = one("metadata_sha256", "metadata SHA-256")
    pinned = json.loads(KNOWN_MISSING.read_text())["splits"]
    archives, shards, known = {}, {}, {}
    for split in splits:
        here = [r for *_, r in used if r["split"] == split]
        if not here:
            raise SystemExit(f"split {split}: no usable report")
        archive = {r["archive_sha256"] for r in here}
        count = {r["shards"] for r in here}
        if len(archive) != 1 or len(count) != 1:
            raise SystemExit(f"split {split}: the reports name {len(archive)} archives and {len(count)} shard counts")
        archives[split], n = archive.pop(), count.pop()
        entry = pinned.get(split) or {}
        if entry.get("archive_sha256") != archives[split] or entry.get("format") != "webm":
            raise SystemExit(f"split {split}: archive {archives[split]} is not pinned as .webm videos")
        known[split] = set(entry.get("missing") or ())
        covered = {r["shard"] for r in here}
        missing = sorted(set(range(n)) - covered)
        if missing and require_all_shards:
            raise SystemExit(f"split {split}: shard(s) {missing} of {n} have no full-pass report")
        shards[split] = dict(shards=n, covered=sorted(covered), missing=missing,
                             reports_per_shard={str(s): sum(r["shard"] == s for r in here) for s in sorted(covered)})
    results, info = {}, collections.Counter()
    for _, key, _, report in used:
        for r in report["results"]:
            if r.get("split") != report["split"]:
                raise SystemExit(f"{key}: result {r.get('video_id')} of split {r.get('split')} in a "
                                 f"{report['split']} report")
            if shard_of(r["video_id"], report["shards"]) != report["shard"]:
                raise SystemExit(f"{key}: {r['video_id']} belongs to shard {shard_of(r['video_id'], report['shards'])}, "
                                 f"not {report['shard']}")
            k = (r["video_id"], r["split"])
            previous = results.get(k)
            if _final(r):
                if previous is not None and _final(previous) and previous.get("state_sha256") and \
                        r.get("state_sha256") and previous["state_sha256"] != r["state_sha256"]:
                    raise SystemExit(f"{r['video_id']} ({r['split']}): two reports name different states "
                                     f"({previous['state_sha256']}, {r['state_sha256']})")
                if previous is not None and previous.get("state_sha256") and r["status"] == "refused":
                    info["info:refused_after_a_stored_state_kept_the_state"] += 1   # the hashed state stands
                    continue
                if previous is not None and not _final(previous):
                    info["info:final_after_a_non_final_result"] += 1
                results[k] = dict(r, report=key)
            elif previous is None or not _final(previous):
                results[k] = dict(r, report=key)
            else:
                info[f"info:later_{r.get('status')}_kept_earlier_final"] += 1
    read = dict(used=[dict(report=k, sha256=s, split=r["split"], shard=r["shard"], started=r["started"])
                      for _, k, s, r in used], ignored=ignored, shards=shards, models=models, convention=convention,
                inference_methods=sorted({r["inference_method"] for *_, r in used}), metadata_sha256=metadata_sha256,
                archives=archives, landmarks_prefixes=sorted({r.get("landmarks_prefix") for *_, r in used}),
                known_missing=known, info=dict(sorted(info.items())))
    return results, read


UNMIRROR_JOB = "semlex-unmirror-v1"


def unmirror_results(store, prefix: str, splits, *, require_all_shards: bool = True) -> tuple[dict, dict]:
    """``unmirror_semlex_states.py``'s results under ``prefix``: per (video id, split), the latest full (non-smoke)
    report's final result (a state: ``converted``, ``already_present``), else its latest result; every shard of each
    built split needs a report (except smoke builds). Two final results naming different states are refused."""
    from convert_semlex_poses import shard_of
    keys = sorted(k for k in store.list(f"{prefix}/reports/") if k.endswith(".json"))
    used = []
    for key in keys:
        payload = store.get(key)
        report = json.loads(payload)
        if report.get("job") != UNMIRROR_JOB or report.get("states_prefix") != prefix:
            raise SystemExit(f"{key}: not a {UNMIRROR_JOB} report for {prefix}")
        if report.get("split") in splits and not report.get("smoke"):
            used.append((report["started"], key, hashlib.sha256(payload).hexdigest(), report))
    used.sort(key=lambda t: (t[0], t[1]))
    shards = {}
    for split in splits:
        here = [r for *_, r in used if r["split"] == split]
        counts = {r["shards"] for r in here}
        if len(counts) > 1:
            raise SystemExit(f"un-mirror reports of {split} name {len(counts)} shard counts")
        n = counts.pop() if counts else None
        missing = sorted(set(range(n)) - {r["shard"] for r in here}) if n else ["all"]
        if missing and require_all_shards:
            raise SystemExit(f"un-mirror split {split}: shard(s) {missing} have no report")
        shards[split] = dict(shards=n, missing=missing)
    results = {}
    for _, key, _, report in used:
        for r in report["results"]:
            if shard_of(r["video_id"], report["shards"]) != report["shard"] or r.get("split") != report["split"]:
                raise SystemExit(f"{key}: {r['video_id']} is not of this report's split and shard")
            k = (r["video_id"], r["split"])
            final = r.get("status") in ("converted", "already_present") and bool(r.get("state_sha256"))
            previous = results.get(k)
            prev_final = previous is not None and previous.get("status") in ("converted", "already_present") \
                and bool(previous.get("state_sha256"))
            if final and prev_final and previous["state_sha256"] != r["state_sha256"]:
                raise SystemExit(f"{r['video_id']} ({r['split']}): two un-mirror reports name different states")
            if final or not prev_final:
                results[k] = dict(r, report=key)
    return results, dict(used=[dict(report=k, sha256=s, split=r["split"], shard=r["shard"]) for _, k, s, r in used],
                         shards=shards, method_suffixes=sorted({r.get("method_suffix") for *_, r in used}))


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


# --------------------------------------------------------------------------------------------------------- per clip

def trim_window(state, threshold: float, margin: float, min_tracked: float, min_seconds: float):
    """``build_trimmed_word_manifest``'s rule on one state, at 24 fps: (window, reason) (the rule lives in
    ``signtranslator.generation.compositional`` since round 15, moved unchanged)."""
    from signtranslator.generation.compositional import trim_window as rule
    from signtranslator.pose.resample import resample_motion
    return rule(resample_motion(state, 24.), threshold, margin, min_tracked, min_seconds)


def dominant_side(state) -> str:
    """The side the trim rule takes as dominant (the tracked wrist height above its shoulder that varies most), on
    the state at 24 fps; for reporting only. Not a handedness classifier: on right-dominant RIT and ASLLVD clips it
    picks the left about half the time (16 and 23 of 40 sampled, 2026-10-28)."""
    from signtranslator.generation.canonical_skeleton import skeleton_positions
    from signtranslator.generation.compositional import side_tracks
    from signtranslator.pose.resample import resample_motion
    state = resample_motion(state, 24.)
    joints, pos, _ = skeleton_positions(state)
    tracks = side_tracks(state)
    best = None
    for side in ("Right", "Left"):
        ok = np.asarray(tracks[side][0], dtype=bool)
        h = pos[:, joints.index(f"{side}Hand"), 1] - pos[:, joints.index(f"{side}Shoulder"), 1]
        spread = float(np.ptp(h[ok])) if ok.any() else -1.
        if best is None or spread > best[0]:
            best = (spread, side)
    return best[1]


def _init_worker(store_spec: str) -> None:
    import warnings
    warnings.simplefilter("error")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    from convert_semlex_poses import make_store
    _WORKER["store"] = make_store(store_spec)


def examine(task: tuple) -> dict:
    """One clip: the state fetched and hashed against its report, decoded; frames, frame rate over the metadata
    duration, hand fraction, then (if both filters pass) the trim window, the untrimmed length and the dominant side."""
    from signtranslator.pose.multichannel import MultichannelMotion
    state_key, state_sha256, duration_ms, p = task
    payload = _WORKER["store"].get(state_key)
    if hashlib.sha256(payload).hexdigest() != state_sha256:
        raise StateMismatch(f"{state_key}: state bytes differ from the report")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "s.npz"
        path.write_bytes(payload)
        state = MultichannelMotion.load(path, expected_sha256=state_sha256)
    frames = len(state.timestamps)
    fps = frames / (float(duration_ms) / 1000.)
    out = dict(frames=frames, fps=round(fps, 3))
    if fps < p["min_fps"]:
        return dict(out, outcome="drop:fps_below_min")
    hand = (state.channels["left_hand"].valid.all(1) | state.channels["right_hand"].valid.all(1)).mean()
    out["hand_fraction"] = round(float(hand), 4)
    if hand < p["min_hand"]:
        return dict(out, outcome="drop:hand_tracked_below_min")
    window, reason = trim_window(state, p["threshold"], p["margin"], p["min_tracked"], p["min_seconds"])
    out.update(outcome=f"trim:{reason}", window=None if window is None else list(window))
    if window is not None:
        out.update(available=sorted(n for n, c in state.channels.items() if c.valid.any()),
                   clip_seconds=round(float(state.timestamps[-1] - state.timestamps[0]), 4),
                   side=dominant_side(state))
    return out


def examine_all(tasks: list, store_spec: str, workers: int) -> list[dict]:
    """``examine`` over the tasks, in order: in this process (``workers`` 0) or a spawned pool."""
    try:
        if workers <= 0:
            _init_worker(store_spec)
            return [examine(t) for t in tasks]
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(store_spec,)) as pool:
            return list(pool.map(examine, tasks, chunksize=max(1, min(64, len(tasks) // (4 * workers) or 1))))
    except StateMismatch as error:
        raise SystemExit(str(error)) from error


# --------------------------------------------------------------------------------------------------------- coverage

def r5_from_mix(validation: dict, mix: dict, words: int = 150) -> list[str]:
    """``pilot_content_check.py``'s R5 rule: the recogniser's reliable alphabetic glosses whose lower case occurs in the
    mix's How2Sign training sentences, sorted, the first ``words``."""
    train = [r for r in mix["records"] if r.get("corpus") == "how2sign-two-view-text-v1" and r.get("partition") == "train"]
    vocab = {w for r in train for w in re.findall(r"[a-z]+", r["text"].lower())}
    return sorted(g for g in validation["per_gloss"]["reliable_glosses"] if g.isalpha() and g.lower() in vocab)[:words]


def word_label(text: str) -> str:
    """``exposure_diagnostic.py``'s label of a record's text: upper case, spaces removed."""
    return text.upper().replace(" ", "")


def coverage(r5: list[str], uncovered: list[str], records: list[dict], before: dict | None = None) -> dict:
    """How many of the uncovered R5 words now have Sem-Lex training clips (at least 1, at least 5), and the ceilings.
    ``before``: per word, its existing one-word training clips (enables the at-least-5 ceilings)."""
    train = collections.Counter(word_label(r["text"]) for r in records if r["partition"] == "train")
    now = {w: train.get(w, 0) for w in uncovered}
    newly = sorted(w for w, n in now.items() if n)
    newly5 = sorted(w for w, n in now.items() if n >= 5)
    covered_before = len(r5) - len(uncovered)
    out = dict(r5_words=len(r5), covered_before=covered_before, uncovered=len(uncovered),
               now_with_semlex_train_clips=len(newly), now_with_5_semlex_train_clips=len(newly5), words=newly,
               words_with_5=newly5, clips_per_word=now, still_uncovered=sorted(set(uncovered) - set(newly)),
               ceiling_before=round(covered_before / len(r5), 4),
               projected_ceiling=round((covered_before + len(newly)) / len(r5), 4),
               rule="train records whose text, upper-cased with spaces removed, equals the word (exposure_diagnostic)")
    if before is not None:
        after = {w: before.get(w, 0) + train.get(w, 0) for w in r5}
        out.update(covered_before_5=sum(before.get(w, 0) >= 5 for w in r5),
                   covered_after=sum(n >= 1 for n in after.values()), covered_after_5=sum(n >= 5 for n in after.values()),
                   ceiling_before_5=round(sum(before.get(w, 0) >= 5 for w in r5) / len(r5), 4),
                   projected_ceiling_5=round(sum(n >= 5 for n in after.values()) / len(r5), 4),
                   semlex_train_clips_on_covered_words=sum(train.get(w, 0) for w in r5 if before.get(w, 0)))
    return out


def prepare_coverage(store, args):
    """(R5 words, uncovered words, existing clips per word or None, provenance) from ``--coverage-mix`` or ``--r5``,
    or None when neither is given."""
    if args.coverage_mix:
        return coverage_inputs_from_mix(store, args)
    if args.r5:
        derivation = json.loads(Path(args.r5).read_text())
        r5 = [w.upper() for w in derivation["r5"]]
        uncovered = sorted(Path(args.uncovered).read_text().split()) if args.uncovered else \
            sorted(derivation["round13"]["uncovered"])
        return r5, uncovered, None, {}
    return None


def coverage_inputs_from_mix(store, args):
    """The R5 words of ``--coverage-mix`` and their existing one-word training clips. ``pilot_content_check.py`` also
    keeps only glosses in the recogniser's labels; the reliable glosses are computed per recogniser class, so they
    are labels by construction (the derived list equals the pilots' pinned 150, checked 2026-10-28)."""
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    key, sha = parse_pinned(args.coverage_mix)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(key) + b"\n")
        mix = load_manifest(path, expected_sha256=sha)
    validation, vkey = get_pinned(store, args.recognizer_validation)
    r5 = r5_from_mix(json.loads(validation), mix)
    one_word = set(args.one_word_corpus or ONE_WORD_CORPORA)
    if not one_word <= {r.get("corpus") for r in mix["records"]}:
        raise SystemExit(f"--one-word-corpus names corpora not in {key}: {sorted(one_word - {r.get('corpus') for r in mix['records']})}")
    labels = collections.Counter(word_label(r["text"]) for r in mix["records"]
                                 if r.get("corpus") in one_word and r.get("partition") == "train" and r.get("text"))
    before = {w: labels.get(w, 0) for w in r5}
    meta = dict(mix=key, mix_sha256=sha, recognizer_validation=vkey, one_word_corpora=sorted(one_word),
                r5_list_sha256=hashlib.sha256(",".join(r5).encode()).hexdigest())
    return r5, sorted(w for w in r5 if not before[w]), before, meta


# ------------------------------------------------------------------------------------------------------------ build

def quantiles(values) -> dict | None:
    values = [float(v) for v in values]
    if not values:
        return None
    return dict(n=len(values), min=round(min(values), 3), p10=round(float(np.percentile(values, 10)), 3),
                median=round(float(np.median(values)), 3), p90=round(float(np.percentile(values, 90)), 3),
                max=round(max(values), 3), mean=round(float(np.mean(values)), 3))


def describe(records: list[dict], examined: dict, splits) -> dict:
    """Trimmed and untrimmed durations, signers, glosses and sides, per split and overall. Sides: the trim rule's
    chosen side per clip, and (video path) each signer's side by round 5's per-signer rule: the forearm tracked on
    more frames, summed over the signer's clips (ASL Citizen: 50 of 52 signers left, mirrored webcams)."""
    out = {}
    for part in list(splits) + ["all"]:
        rs = [r for r in records if part == "all" or r["partition"] == part]
        signers = collections.Counter(r["source"]["signer_id"] for r in rs)
        sides = collections.Counter(examined[r["sign"]]["side"] for r in rs)
        per_signer = collections.defaultdict(collections.Counter)
        forearm = collections.defaultdict(collections.Counter)
        for r in rs:
            per_signer[r["source"]["signer_id"]][examined[r["sign"]]["side"]] += 1
            for side, fraction in ((r["quality"].get("forearm_tracked") or {}).items()):
                forearm[r["source"]["signer_id"]][side] += fraction * r["quality"]["frames"]
        left_signers = sorted(s for s, c in forearm.items() if c["Left"] > c["Right"])
        out[part] = dict(
            records=len(rs), trimmed_seconds=quantiles(r["continuous"]["seconds"] for r in rs),
            untrimmed_seconds=quantiles(examined[r["sign"]]["clip_seconds"] for r in rs),
            kept_fraction_of_clip=quantiles(r["continuous"]["seconds"] / max(examined[r["sign"]]["clip_seconds"], 1e-6)
                                            for r in rs),
            signers=len(signers), clips_per_signer=quantiles(signers.values()),
            texts=len({r["text"] for r in rs}), gloss_labels=len({r["gloss"]["label"] for r in rs}),
            asl_lex_codes=len({r["gloss"].get("asl_lex_code") for r in rs if r["gloss"].get("asl_lex_code")}),
            label_types=dict(sorted(collections.Counter(r["source"]["label_type"] for r in rs).items())),
            trim_side=dict(sorted(sides.items())),
            signers_mostly_left_trim_side=sum(c["Left"] > c["Right"] for c in per_signer.values()))
        if forearm:
            out[part].update(signers_left_by_forearm_tracking=len(left_signers),
                             clips_of_left_signers_by_forearm_tracking=sum(signers[s] for s in left_signers),
                             clips_tracking_left_forearm_more=sum(
                                 (r["quality"].get("forearm_tracked") or {}).get("Left", 0) >
                                 (r["quality"].get("forearm_tracked") or {}).get("Right", 0) for r in rs))
    return out


def build(store, args) -> tuple[dict, dict]:
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest
    from convert_semlex_poses import METADATA_KEY, METADATA_SHA256, load_metadata
    video = args.source == "video"
    splits = tuple(args.splits.split(",")) if video else ("train", "val", "test")
    if video and (not splits or set(splits) - set(VIDEO_SPLITS)):
        raise SystemExit(f"--splits must name some of {VIDEO_SPLITS} (the test split is not in the video states)")
    metadata_sha256 = args.metadata_sha256 or METADATA_SHA256
    metadata = load_metadata(store, args.metadata_key or METADATA_KEY, metadata_sha256)
    provenance = json.loads(store.get(args.asllex_key + ".provenance.json"))
    asllex_bytes = store.get(args.asllex_key)
    asllex_sha = hashlib.sha256(asllex_bytes).hexdigest()
    if provenance.get("sha256") != asllex_sha or not provenance.get("verified_by_r2_reread"):
        raise SystemExit("ASL-LEX signdata.csv is not the verified acquisition")
    asllex = {r["EntryID"]: r for r in csv.DictReader(io.StringIO(asllex_bytes.decode("utf-8", "replace")))}
    signbank_ids = {r.get("SignBankReferenceID", "").split(".")[0] for r in asllex.values()}
    words, wordlist_sha = load_wordlist(Path(args.wordlist))
    read, bundle = None, None
    if video:
        results, read = video_results(store, args.prefix, splits, require_all_shards=not args.limit)
        if read["metadata_sha256"] != metadata_sha256:
            raise SystemExit(f"the reports were run on metadata {read['metadata_sha256']}, not {metadata_sha256}")
        report_hashes = {f"report:{u['report'].rsplit('/', 1)[-1]}": u["sha256"] for u in read["used"]}
        unmirrored = bool(args.states_prefix) and args.states_prefix.rstrip("/") != args.prefix
        if unmirrored:
            args.states_prefix = args.states_prefix.rstrip("/")
            um_results, um_read = unmirror_results(store, args.states_prefix, splits, require_all_shards=not args.limit)
            report_hashes.update({f"report:unmirror:{u['report'].rsplit('/', 1)[-1]}": u["sha256"]
                                  for u in um_read["used"]})
    else:
        unmirrored = False
        from signtranslator.data_engineering.semlex_pose import LEARNED_ADAPTER
        results, report_keys, bundles = latest_results(store, args.prefix)
        bundles.discard(None)
        if len(bundles) > 1:
            raise SystemExit(f"the reports name more than one regressor bundle: {sorted(bundles)}")
        bundle = next(iter(bundles), None)
        report_hashes = {f"report:{k.rsplit('/', 1)[-1]}": hashlib.sha256(store.get(k)).hexdigest()
                         for k in report_keys}
    counts = collections.Counter(read["info"] if read else {})
    pending = []                                       # (row, text, gloss, result) for clips with a stored state
    for video_id, rows in sorted(metadata.items()):
        row, why = resolve_rows(rows)
        if row is None:
            counts[why] += 1
            continue
        if why:
            counts[why] += 1
        split = row["split"]
        if split not in splits:
            counts[f"skip:{split}_split_not_built" + ("_held_out" if split == "test" else "")] += 1
            continue
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
            if video and video_id in read["known_missing"].get(split, ()):
                counts["drop:not_in_archive_pinned_gap"] += 1
            elif video:
                counts["drop:no_extraction_result"] += 1
            else:
                counts["drop:no_conversion_result"] += 1
            continue
        if not result.get("state_sha256") or (video and result["status"] not in FINAL_STATES):
            counts[f"drop:{'extraction' if video else 'conversion'}_{result['status']}"] += 1
            continue
        if video and result["state_key"] != f"{args.prefix}/{split}/{video_id}.npz":
            raise SystemExit(f"{video_id}: the report names state {result['state_key']}")
        if unmirrored:                                 # the re-solved state of exactly this extracted state
            u = um_results.get((video_id, split))
            if u is None or u.get("status") not in ("converted", "already_present") or not u.get("state_sha256"):
                counts[f"drop:unmirror_{'missing' if u is None else u.get('status')}"] += 1
                continue
            if u.get("source_state_sha256") != result["state_sha256"] or \
                    u["state_key"] != f"{args.states_prefix}/{split}/{video_id}.npz":
                raise SystemExit(f"{video_id}: the un-mirrored state is not of the extracted state "
                                 f"({u.get('source_state_sha256')} != {result['state_sha256']}) or not at its key")
            result = dict(result, state_key=u["state_key"], state_sha256=u["state_sha256"],
                          extracted_state_key=result["state_key"], extracted_state_sha256=result["state_sha256"],
                          unmirror_report=u["report"], tracked=u.get("tracked") or result.get("tracked"))
        if args.limit and len(pending) >= args.limit:
            counts["smoke:beyond_limit"] += 1
            continue
        pending.append((row, text, gloss, result))
    if video and counts["drop:no_extraction_result"] and not args.limit:
        raise SystemExit(f"{counts['drop:no_extraction_result']} metadata clips of the built splits have no result "
                         "in any report and are not pinned archive gaps")
    unfinished = {k: n for k, n in counts.items() if k in (
        "drop:extraction_error", "drop:extraction_failed", "drop:extraction_exists_different") or
        k.startswith("drop:unmirror_")}
    if video and unfinished and not args.limit and not args.allow_unfinished:
        raise SystemExit(f"clips without a final extraction (or un-mirror) result: {unfinished}; rerun those shards "
                         "(they resume), or pass --allow-unfinished to drop them")
    cov_inputs = prepare_coverage(store, args)         # before the long pass: a bad pin fails in seconds
    params = dict(min_fps=args.min_fps, min_hand=args.min_hand, threshold=args.threshold, margin=args.margin,
                  min_tracked=args.min_tracked, min_seconds=args.min_seconds)
    tasks = [(res["state_key"], res["state_sha256"], float(row["duration"]), params) for row, _, _, res in pending]
    outcomes = examine_all(tasks, args.store, args.workers)
    records, examined, signers_by_split = [], {}, collections.defaultdict(set)
    for (row, text, gloss, result), out in zip(pending, outcomes):
        counts[out["outcome"]] += 1
        if out.get("window") is None:
            continue
        video_id, split = row["video_id"], row["split"]
        window = out["window"]
        quality = dict(frames=out["frames"], fps=out["fps"], hand_fraction=out["hand_fraction"])
        if video:
            quality.update(measured_fps=result.get("fps"), duration_ratio=result.get("duration_ratio"),
                           image_size=result.get("image_size"), upright_root_applied=result.get("upright_root_applied"),
                           forearm_tracked=(result.get("tracked") or {}).get("forearm"))
            source = dict(video_id=video_id, signer_id=row["signer_id"], label_type=row["label_type"],
                          semlex_split=split, extraction=VIDEO_EXTRACTOR, video_sha256=result.get("source_sha256"),
                          landmarks_key=result.get("landmarks_key"), landmarks_sha256=result.get("landmarks_sha256"),
                          report=result["report"].rsplit("/", 1)[-1])
            if unmirrored:
                source.update(unmirrored=True, extracted_state_key=result["extracted_state_key"],
                              extracted_state_sha256=result["extracted_state_sha256"],
                              unmirror_report=result["unmirror_report"].rsplit("/", 1)[-1])
        else:
            from signtranslator.data_engineering.semlex_pose import LEARNED_ADAPTER
            source = dict(video_id=video_id, signer_id=row["signer_id"], label_type=row["label_type"],
                          semlex_split=split, adapter=result.get("adapter", LEARNED_ADAPTER),
                          regressors_sha256=result.get("regressors_sha256"))
        sign = f"semlex-{video_id}"
        records.append(dict(
            sign=sign, corpus=CORPUS_ID, partition=split, text=text, gloss=gloss,
            performer_sha256=performer(row["signer_id"]), quality=quality, source=source,
            continuous=dict(status="converted", state_key=result["state_key"], state_sha256=result["state_sha256"],
                            available_channels=out["available"], window_seconds=window,
                            seconds=round(window[1] - window[0] + 1 / 24., 4))))
        examined[sign] = out
        signers_by_split[split].add(row["signer_id"])
        counts[f"kept:{split}"] += 1
    pairs = [(a, b) for k, a in enumerate(splits) for b in splits[k + 1:]] if video else \
        [("train", "val"), ("train", "test"), ("val", "test")]
    overlap = {f"{a}&{b}": len(signers_by_split[a] & signers_by_split[b]) for a, b in pairs}
    split_source = ("Sem-Lex's own train and val splits, unchanged; the test split is held out (a separate set, not in "
                    "this manifest)" if video else "Sem-Lex's own splits, unchanged (test held out)")
    split = dict(kind="continuous", source=split_source, signer_disjoint=False, signer_overlap=overlap,
                 counts={p: sum(r["partition"] == p for r in records) for p in ("train", "val", "test")})
    inputs = {"semlex_metadata": metadata_sha256, "asl_lex_signdata": asllex_sha, "english_wordlist": wordlist_sha}
    inputs.update(report_hashes)
    manifest = build_corpus_manifest(corpus_id=CORPUS_ID, inputs=inputs, records=records, split=split)
    manifest["license"] = LICENCE
    if video:
        manifest["provenance"] = dict(
            citation=CITATION, mirror=MIRROR, source="video_extraction", extractor=VIDEO_EXTRACTOR,
            script="scripts/extract_semlex_videos.py", convention="generator", convention_settings=read["convention"],
            mediapipe_tasks_models=read["models"], inference_methods=read["inference_methods"],
            states_prefix=args.prefix, landmarks_prefixes=read["landmarks_prefixes"],
            raw_landmarks_key="<landmarks prefix>/<split>/<video_id>.npz (each record's source.landmarks_key, "
                              "with its SHA-256)", archives=read["archives"], splits=list(splits),
            test_split="held out: not in these video states and not in this manifest", unmirrored=unmirrored)
        if unmirrored:
            manifest["provenance"].update(
                states_prefix=args.states_prefix, extraction_states_prefix=args.prefix, unmirror_job=UNMIRROR_JOB,
                unmirror_script="scripts/unmirror_semlex_states.py", unmirror_method_suffixes=um_read["method_suffixes"],
                unmirror_reason="the recordings were judged mirrored (round 14 handedness test): each state is "
                                "re-solved from its stored raw landmarks flipped horizontally")
        manifest["unmirrored"] = unmirrored
    else:
        from signtranslator.data_engineering.semlex_pose import LEARNED_ADAPTER
        manifest["provenance"] = dict(citation=CITATION, mirror=MIRROR, adapter=LEARNED_ADAPTER,
                                      regressors_sha256=bundle, states_prefix=args.prefix, convention="generator")
    manifest["trim"] = dict(rule="one-word clips cut to the dominant wrist's active span (height above rest + threshold x "
                                 "(peak - rest)) +- margin (build_trimmed_word_manifest.sign_window)",
                            threshold=args.threshold, margin=args.margin, min_tracked=args.min_tracked,
                            min_seconds=args.min_seconds)
    manifest["filters"] = dict(min_fps=args.min_fps, min_hand=args.min_hand, counts=dict(sorted(counts.items())))
    if args.limit:
        manifest["smoke"] = dict(limit=args.limit, note="a smoke build: the first clips with a state only")
    cov = None
    if cov_inputs is not None:
        r5, uncovered, before, meta = cov_inputs
        cov = dict(coverage(r5, uncovered, records, before), **meta)
    summary = dict(source=args.source, counts=dict(sorted(counts.items())), coverage=cov, records=len(records),
                   split=split, describe=describe(records, examined, splits if video else ("train", "val", "test")))
    if video:
        summary["reports"] = {k: read[k] for k in ("used", "ignored", "shards", "inference_methods", "archives")}
    else:
        summary["regressors_sha256"] = bundle
    return manifest, summary


def main(argv=None) -> int:
    from signtranslator.data_engineering.world_regressor import WORLD_REGRESSORS
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2", help="'r2' or 'dir:/path' (tests)")
    p.add_argument("--source", choices=("pose", "video"), default="pose")
    p.add_argument("--prefix", help="states prefix; default the pinned learned path (pose) or states-video-v1 (video)")
    p.add_argument("--states-prefix", help="video: the un-mirrored states (unmirror_semlex_states.py's prefix) that "
                   "replace the extracted ones; each must re-solve exactly the extracted state the reports name")
    p.add_argument("--splits", default=",".join(VIDEO_SPLITS), help="video: the splits to build (val, train)")
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
    p.add_argument("--coverage-mix", help="KEY@SHA of the mix the R5 words and their existing clips come from")
    p.add_argument("--recognizer-validation", default=RECOGNIZER_VALIDATION, help="KEY@SHA of the recogniser's "
                   "validation report (its reliable glosses)")
    p.add_argument("--one-word-corpus", action="append", help=f"existing one-word corpora (default {ONE_WORD_CORPORA})")
    p.add_argument("--allow-unfinished", action="store_true", help="video: drop (and count) clips whose best result is "
                   "error, failed or exists_different instead of stopping")
    p.add_argument("--limit", type=int, help="smoke: only the first N clips that have a state (in id order)")
    p.add_argument("--workers", type=int, default=0, help="processes reading the states (0: this process)")
    p.add_argument("--output", required=True, help="local manifest path (written once)")
    p.add_argument("--summary", help="local summary JSON path")
    p.add_argument("--output-key", help="also upload the manifest here (refused if the key exists)")
    p.add_argument("--summary-key", help="also upload the summary here (refused if the key exists)")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    from signtranslator.data_engineering.corpus_manifest import canonical_json
    from convert_semlex_poses import make_store
    if args.prefix is None:
        args.prefix = VIDEO_PREFIX if args.source == "video" else \
            f"{BASE}/states-semlex-holistic-learned-world-v2-{WORLD_REGRESSORS['sha256'][:12]}"
    args.prefix = args.prefix.rstrip("/")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    store = make_store(args.store)
    for key in (args.output_key, args.summary_key):
        if key and args.limit and not key.startswith(SMOKE_BASE + "/"):
            raise SystemExit(f"a --limit build uploads only under {SMOKE_BASE}/")
        if key and store.exists(key):
            raise SystemExit(f"refusing to overwrite {key}")
    if Path(args.output).exists():                  # fail before the build; the write below is exclusive as well
        raise FileExistsError(f"{args.output} exists (written once)")
    manifest, summary = build(store, args)
    payload = canonical_json(manifest)
    with open(args.output, "xb") as stream:
        stream.write(payload + b"\n")
    summary["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    if args.output_key:
        if store.exists(args.output_key):
            raise SystemExit(f"refusing to overwrite {args.output_key}")
        # the store holds the canonical bytes (no newline), as every manifest in R2: readers append the newline
        if store.put(args.output_key, payload) != summary["manifest_sha256"]:
            raise SystemExit("upload digest mismatch")
        summary["manifest_key"] = args.output_key
    if args.summary_key:
        if store.exists(args.summary_key):
            raise SystemExit(f"refusing to overwrite {args.summary_key}")
        store.put(args.summary_key, json.dumps(summary, indent=1, sort_keys=True).encode())
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("coverage", "reports")} | {
        "coverage": {k: v for k, v in (summary["coverage"] or {}).items() if k != "clips_per_word"}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
