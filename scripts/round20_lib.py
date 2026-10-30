"""Round 20 (docs/NEXT_PHASE_PLAN.md, "Round 20: composition v2", declared 2026-10-30 19:09 UTC): the shared pieces of
the round's evaluation tooling. Nothing here trains; nothing here changes a round-15 default.

- **Sentence sets** (``validation_pool``, ``draw_sets``, ``set_payload``, ``load_sentence_set``): D20 and C20 drawn from
  the How2Sign val clips of the pilot's signer groups (c01, c03, c05: ``pilot_content_check.validation_sentences``'s
  own filter, never c00) outside the 1,000 and the 300, with texts shared with the 1,000 excluded (normalised as the
  selection normalises, ``compositional.normalise``).
- **Vocabulary** (``semlex_only_entries``, ``coverage``): entries whose every training record is Sem-Lex (they have no
  ASLLVD annotation, so the sign-fitted timing would place their core at ASLLVD's pooled median fraction; round 20 uses
  the activity core instead), and the selection's coverage statistics.
- **Continuous forms** (``asllrp_segments``, ``segment_clip``, ``continuous_material``): real word clips cut from
  ASLLRP continuous utterances at their annotated sign spans (the Sign Bank utterances' frames), mapped to vocabulary
  entries by the stage-2 fit's own per-sign mapping (``round15_stage2_fit.gloss_table``: ``map_sign`` tiers with
  index, single entry). Utterances of the gloss manifest's train and val partitions only (test, Ben, never read).
- **Word cache** (``entry_seed``, ``entry_label``, ``cached_material``): one stored generated sign per entry.
- **Paired statistics** (``paired_mean``, ``likelihood_summary``).
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 24.
H2S = "how2sign-two-view-text-v1"
COMPONENTS = ("c01", "c03", "c05")
SET_SEED = 20
SEMLEX_CORPORA = ("semlex-text-v1", "semlex-text-v2")
GLOSS_MANIFEST = ("acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/gloss-manifest-v1/manifest.json",
                  "f8c40aead29ad1cf97c4a3986d326d475f07bf539deec87219ba69cd47e72944")
SEGMENT_PARTITIONS = ("train", "val")
SEGMENT_CORPUS = "asllrp-continuous-segment"
SEGMENT_CONTEXT = .5          # seconds of the utterance kept either side of a segment's sign (approach / return)
CACHE_SEED_BASE = 40_000_000  # an entry's sample 0: CACHE_SEED_BASE + (first 4 bytes of sha256(entry)) mod 10^7


# ---- sentence sets -------------------------------------------------------------------------------------------------

def text_key(text) -> str:
    from signtranslator.generation.compositional import normalise
    return " ".join(normalise(text))


def validation_pool(manifest) -> list[dict]:
    """``validation_sentences``'s candidates, in manifest order (no sampling): How2Sign val, c01/c03/c05, a text."""
    return [r for r in manifest["records"] if r.get("corpus") == H2S and r.get("partition") == "val"
            and r.get("signer_component") in set(COMPONENTS) and r.get("text")]


def draw_sets(pool, excluded_signs, excluded_texts, *, dev: int = 500, confirm: int = 1000,
              seed: int = SET_SEED) -> tuple[list, list, dict]:
    """(D, C, counts): ``dev`` records drawn with ``random.Random(seed).sample`` from the pool outside
    ``excluded_signs`` and without a normalised text in ``excluded_texts``; then, with the same generator, ``confirm``
    from what remains, also without a text of D (C is disjoint from D in clips and in texts)."""
    signs = [r["sign"] for r in pool]
    if len(set(signs)) != len(signs):
        raise ValueError("the pool's sign ids are not unique")
    excluded_signs, excluded_texts = set(excluded_signs), set(excluded_texts)
    remainder = [r for r in pool if r["sign"] not in excluded_signs and text_key(r["text"]) not in excluded_texts]
    rng = random.Random(seed)
    if len(remainder) < dev:
        raise ValueError(f"only {len(remainder)} candidates for {dev} development sentences")
    d = rng.sample(remainder, dev)
    d_signs, d_texts = {r["sign"] for r in d}, {text_key(r["text"]) for r in d}
    rest = [r for r in remainder if r["sign"] not in d_signs and text_key(r["text"]) not in d_texts]
    if len(rest) < confirm:
        raise ValueError(f"only {len(rest)} candidates for {confirm} confirmation sentences")
    c = rng.sample(rest, confirm)
    counts = dict(pool=len(pool), excluded_signs=int(sum(r["sign"] in excluded_signs for r in pool)),
                  excluded_texts_only=int(sum(r["sign"] not in excluded_signs and text_key(r["text"]) in excluded_texts
                                              for r in pool)),
                  remainder=len(remainder), dev=len(d), confirm_candidates=len(rest), confirm=len(c))
    return d, c, counts


def set_payload(name: str, records, *, manifest_key: str, manifest_sha256: str, rule: str, counts: dict) -> dict:
    return dict(kind="round20-sentence-set", set=name, rule=rule, seed=SET_SEED, manifest_key=manifest_key,
                manifest_sha256=manifest_sha256, counts=counts, sentences=len(records),
                records=[dict(sign=r["sign"], text=r["text"], signer_component=r.get("signer_component"),
                              state_key=r["continuous"]["state_key"], state_sha256=r["continuous"]["state_sha256"])
                         for r in records])


def load_sentence_set(payload: dict, manifest, manifest_key: str, manifest_sha256: str) -> list[dict]:
    """The set's manifest records in the set's order, after checking it was drawn from this manifest and that every
    record is a validation-pool record with the stored text and state."""
    if payload.get("kind") != "round20-sentence-set":
        raise SystemExit("not a round-20 sentence set")
    if payload.get("manifest_key") != manifest_key or payload.get("manifest_sha256") != manifest_sha256:
        raise SystemExit("the sentence set was drawn from another manifest")
    by_sign = {r["sign"]: r for r in validation_pool(manifest)}
    out = []
    for row in payload["records"]:
        r = by_sign.get(row["sign"])
        if r is None or r["text"] != row["text"] or r["continuous"]["state_sha256"] != row["state_sha256"]:
            raise SystemExit(f"{row['sign']}: not this manifest's validation record")
        out.append(r)
    if len({r["sign"] for r in out}) != len(out):
        raise SystemExit("the sentence set repeats a clip")
    return out


# ---- vocabulary ----------------------------------------------------------------------------------------------------

def semlex_only_entries(pools) -> list[str]:
    """Entries whose every training record is a Sem-Lex record (``pools``: ``round15_oracle.word_pools``)."""
    return sorted(e for e, rs in pools.items() if rs and all(r.get("corpus") in SEMLEX_CORPORA for r in rs))


def coverage(details) -> dict:
    """``compositional.selection_statistics`` plus the distribution of kept entries per sentence and the most common
    unmatched and stop tokens."""
    from signtranslator.generation.compositional import KEPT, selection_statistics
    out = selection_statistics(details)
    words = [sum(r["kind"] in KEPT for r in rows) for rows in details]
    if words:
        out["entries_per_sentence"] = dict(mean=round(float(np.mean(words)), 3), median=float(np.median(words)),
                                           p10=float(np.percentile(words, 10)), p90=float(np.percentile(words, 90)))
    for kind in ("unmatched", "stop"):
        out[f"top_{kind}"] = collections.Counter(t for rows in details for r in rows if r["kind"] == kind
                                                 for t in r["tokens"]).most_common(30)
    return out


# ---- continuous forms (ASLLRP segments) ---------------------------------------------------------------------------

def asllrp_segments(utterances, gloss_manifest, vocab, signbank, codes, semlex) -> tuple[dict, dict]:
    """({entry: [segment]}, counts): every timed sign of the gloss manifest's train/val utterances that maps to a
    single vocabulary entry (``map_sign`` tiers with index); segments sorted by id. A segment holds its utterance's
    manifest record and its sign's frames in the utterance (inclusive, from the utterance's first frame)."""
    import asllrp_gloss_corpus as agc
    from build_asllrp_gloss_manifest import PARTITION, signer
    from round15_stage2_fit import sign_key
    records = {r["sign"]: r for r in gloss_manifest["records"] if r.get("partition") in SEGMENT_PARTITIONS}
    counts = collections.Counter()
    table, pools = {}, collections.defaultdict(list)
    for u in utterances:
        video = u.get("utterance_video")
        if not video:
            continue
        sign_id = f"{u['collection']}:{Path(video).stem}"
        record = records.get(sign_id)
        if record is None or PARTITION.get(signer(u.get("source") or "")) not in SEGMENT_PARTITIONS:
            continue
        u0, u1 = (u.get("frames") or [None, None])[:2]
        if None in (u0, u1):
            counts["utterance_untimed"] += 1
            continue
        for k, s in enumerate(u.get("signs") or ()):
            if not s.get("gloss"):
                continue
            sign = dict(gloss=s["gloss"], variant=s.get("variant", ""), occurrence=s.get("occurrence", ""),
                        sign_type=s.get("sign_type", ""), category=agc.category(s["gloss"], s.get("sign_type", "")))
            key = sign_key(sign)
            if key not in table:
                table[key] = agc.map_sign(sign, vocab, signbank, codes, semlex)
            m = table[key]
            if m["tier"] not in agc.WITH_INDEX or len(m["entries"]) != 1:
                counts["unmapped"] += 1
                continue
            a, b = (s.get("frames") or [None, None])[:2]
            if None in (a, b) or not u0 <= a <= b <= u1:
                counts["sign_untimed"] += 1
                continue
            entry = m["entries"][0]
            pools[entry].append(dict(sign=f"{sign_id}#{k}", corpus=SEGMENT_CORPUS, entry=entry, gloss=s["gloss"],
                                     tier=m["tier"], record=record, start=int(a - u0), end=int(b - u0),
                                     utterance_frames=int(u1 - u0 + 1), partition=record["partition"]))
            counts["segments"] += 1
    pools = {e: sorted(v, key=lambda x: x["sign"]) for e, v in sorted(pools.items())}
    counts.update(entries=len(pools), utterances=len({x["record"]["sign"] for v in pools.values() for x in v}))
    return pools, dict(sorted(counts.items()))


def utterance_clip(record, *, cache, fetch, template):
    """(the utterance conformed to the layout and at 24 fps, its native frame times from its first frame)."""
    from signtranslator.data.layout_adapter import conform
    from signtranslator.data_engineering.corpus_manifest import load_sample
    from evaluate_text_to_motion import at_rate
    state = load_sample(record, "continuous", cache=cache, fetch=fetch)
    native = np.asarray(state.timestamps, dtype=float) - float(state.timestamps[0])
    return at_rate(conform(state, template), FPS), native


SPAN_TOLERANCE = .03          # a state up to 3% longer or shorter than its annotation is mapped proportionally


def segment_clip(segment, utterance, native, *, margin: float = .1, context: float = SEGMENT_CONTEXT,
                 tolerance: float = SPAN_TOLERANCE) -> dict | None:
    """A piece (``{"state", "lo", "hi", "reason", "span_scale"}``) for one segment of a loaded utterance
    (``utterance_clip``): the 24 fps frames from ``context`` seconds before the sign's first frame to ``context`` after
    its last, with the core the sign's span widened by ``margin`` seconds (at least one frame). The annotation's frame
    indices are used as they are when the state has the annotation's frame count; a count off by at most
    ``tolerance`` (the RIT sentence videos' states run about 1% long) maps them proportionally (``span_scale``); a
    larger difference returns None (the span cannot be placed)."""
    from signtranslator.generation.compositional import take_frames
    scale = len(native) / segment["utterance_frames"]
    if abs(scale - 1.) > tolerance:
        return None
    at = lambda k: min(len(native) - 1, int(round(k * scale)))         # noqa: E731
    ta, tb = float(native[at(segment["start"])]), float(native[at(segment["end"])])
    t = np.asarray(utterance.timestamps, dtype=float) - float(utterance.timestamps[0])
    keep = np.flatnonzero((t >= ta - context - 1e-9) & (t <= tb + context + 1e-9))
    if not len(keep):
        keep = np.array([int(np.argmin(np.abs(t - (ta + tb) / 2)))])
    state = take_frames(utterance, keep)
    tt = t[keep]
    core = np.flatnonzero((tt >= ta - margin - 1e-9) & (tt <= tb + margin + 1e-9))
    if not len(core):
        core = np.array([int(np.argmin(np.abs(tt - (ta + tb) / 2)))])
    lo, hi = int(core[0]), int(core[-1])
    return dict(state=state, lo=lo, hi=hi, side=None, whole=lo == 0 and hi == len(keep) - 1, reason="continuous",
                span_scale=round(scale, 5))


def continuous_material(job, clip: int, segment_pools, pools, *, loaded: dict, load_utterance, load_word, rule,
                        timing: str, fractions, margin: float, tighten: bool = False) -> dict:
    """``round15_stage2.oracle_material`` with each word's clip taken from an ASLLRP continuous segment where the entry
    has one (drawn by ``round15_oracle.choose_record`` in the job's block, from the entry's segments), else exactly
    the isolated oracle's piece (the same draw and core). Each segment row says its ``source``."""
    import round15_stage2 as r2
    from round15_oracle import choose_record
    from signtranslator.generation.compositional import parts_from_pieces
    pieces, rows = [], []
    for w in job["words"]:
        piece, row = None, None
        if w in segment_pools:
            seg = choose_record(segment_pools, clip, w, job["block"])
            usign = seg["record"]["sign"]
            if usign not in loaded:
                loaded[usign] = load_utterance(seg["record"])
            piece = segment_clip(seg, *loaded[usign], margin=margin)
            if piece is not None:
                row = dict(word=w, frames=len(piece["state"].timestamps), sign=seg["sign"], corpus=SEGMENT_CORPUS,
                           gloss=seg["gloss"], lo=piece["lo"], hi=piece["hi"], reason="continuous", source="continuous",
                           span_scale=piece["span_scale"])
        if piece is None:
            one = dict(job, words=[w])
            iso = r2.oracle_material(one, clip, pools, loaded=loaded, load=load_word, rule=rule, timing=timing,
                                     fractions=fractions, margin=margin, tighten=tighten,
                                     records=[choose_record(pools, clip, w, job["block"])])
            piece = iso["pieces"][0]
            row = dict(iso["segments"][0], source="isolated" if w not in segment_pools else "isolated_span_mismatch")
        pieces.append(piece)
        rows.append(row)
    return dict(mode="oracle_stage2", parts=parts_from_pieces(pieces), segments=rows, pieces=pieces)


# ---- word cache ------------------------------------------------------------------------------------------------------

def entry_seed(entry: str) -> int:
    """An entry's fixed sample-0 seed (``rerank.rerank_seeds`` gives the others), independent of any sentence."""
    return CACHE_SEED_BASE + int.from_bytes(hashlib.sha256(f"round20-cache|{entry}".encode()).digest()[:4], "big") % 10 ** 7


def entry_label(entry: str, labels) -> str | None:
    """The recogniser label of an entry: the entry upper-cased (``sentence_word_spotting.sentence_labels``'s rule),
    for a multi-word entry its words joined without spaces or with hyphens; None when no form is a label."""
    labels = set(labels)
    words = entry.upper().split()
    for form in (" ".join(words), "".join(words), "-".join(words)):
        if form in labels:
            return form
    return None


def cached_material(job, states: dict, *, cores=None, rule=None) -> dict:
    """``compositional.compose_cores``'s core_crossfade material from cached word clips (``states``: entry -> state):
    each word's stored clip, cut to its core by ``cores(word, clip)`` (the sign-fitted rule) or ``core_window``."""
    from signtranslator.generation.compositional import core_window, parts_from_pieces
    pieces, rows = [], []
    for w in job["words"]:
        clip = states[w]
        window = cores(w, clip) if cores else core_window(clip, rule)
        pieces.append(dict(state=clip, **window))
        rows.append(dict(word=w, frames=len(clip.timestamps), seed=None, lo=window["lo"], hi=window["hi"],
                         reason=window["reason"], cached=True,
                         **({"tightened": window["tightened"]} if "tightened" in window else {})))
    return dict(mode=job["mode"], parts=parts_from_pieces(pieces), segments=rows, pieces=pieces)


# ---- paired statistics -----------------------------------------------------------------------------------------------

def paired_mean(a, b=None, *, boot: int = 1000, seed: int = 0) -> dict | None:
    """Mean of a - b (b omitted: of a) over the items where both are numbers, with a 95% bootstrap interval over items
    (resampled jointly); None without items."""
    a = [None if x is None else float(x) for x in a]
    b = [0.] * len(a) if b is None else [None if x is None else float(x) for x in b]
    if len(a) != len(b):
        raise ValueError("paired values differ in length")
    d = np.array([x - y for x, y in zip(a, b) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)])
    if not len(d):
        return None
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (boot, len(d)))].mean(1) if boot else np.array([d.mean()])
    return dict(difference=round(float(d.mean()), 4),
                ci95=[round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
                n=int(len(d)))


def likelihood_summary(rows, forms, pairs: dict, *, boot: int, other: dict | None = None) -> dict:
    """The reference log-likelihood per token under the back-translator (``rows``: the merged clips, each with
    ``likelihood[form] = [own, mismatched, tokens]``): per form its mean, the paired per-sentence difference to blank
    (the selection metric), to its own deranged references, to the forms of ``pairs`` and, with ``other`` (another
    merged report's ``likelihood``), to the same form there (``paired_<form>``)."""
    own = {f: [r["likelihood"][f][0] if f in r["likelihood"] else None for r in rows] for f in forms}
    out = {}
    for f in forms:
        mism = [r["likelihood"][f][1] if f in r["likelihood"] else None for r in rows]
        row = dict(mean=paired_mean(own[f], boot=boot), minus_mismatched=paired_mean(own[f], mism, boot=boot))
        if f != "blank" and "blank" in own:
            row["minus_blank"] = paired_mean(own[f], own["blank"], boot=boot)
        for label, g in (pairs.get(f) or {}).items():
            if g in own:
                row[f"minus_{label}"] = paired_mean(own[f], own[g], boot=boot)
        if other and f in (other.get("per_form") or {}):
            row[f"minus_paired_{f}"] = paired_mean(own[f], other["per_form"][f], boot=boot)
        out[f] = row
    return dict(metric="mean log-probability per reference token, teacher-forced (back-translator)",
                forms=out, per_form=own)


__all__ = ["FPS", "H2S", "COMPONENTS", "SET_SEED", "SEMLEX_CORPORA", "GLOSS_MANIFEST", "text_key", "validation_pool",
           "draw_sets", "set_payload", "load_sentence_set", "semlex_only_entries", "coverage", "asllrp_segments",
           "utterance_clip", "segment_clip", "continuous_material", "entry_seed", "entry_label", "cached_material",
           "paired_mean", "likelihood_summary"]
