"""Post-S2 A3 (reports/post_s2_plan.md section 3): the pre-registered list of unsigned tokens and the source clips each would use.
Local, read-only on the store, nothing trains. The output is committed and pinned by its hash before any read.

**Ranks.** How2Sign train texts of the round-10 mix (never val, so not D20, C20 or the 1,000; c00 refused), counted by the
composition's own selection on r10's vocabulary: a token ranks when ``selection_details`` calls it ``unmatched``. Ties go
alphabetically. D20's texts are used only for the coverage line, which no rank depends on.

**Contractions** (``contraction_entry``), adding no clip. ``n't`` always means ``not`` (``asllrp_gloss_corpus._CONTRACTIONS``), including
``can't`` and ``won't``: the first piece would reverse the meaning, so it is never used. Any other contraction takes its first piece's entry
(``it's`` -> ``it``): ``alias_r10`` when that is an r10 entry, ``alias_list`` when it is a token of this list. Every ``n't`` token of the train
texts is listed in ``negation_tokens`` and maps in every arm, whatever its rank.

**Sources: two kinds, and the one with more well-tracked clips serves the token** (a tie goes to isolated). The count of each is stated per
token in ``source_rule``, before any read.
- Isolated clips: train records of the round-14b mix whose normalised text is the token, in the allowed corpora. A single-letter entry takes
  no isolated clip: the selection's own rule treats those as letters (ASLLVD's ``i`` is annotated handshape I).
- ASLLRP continuous segments, train utterances only, by the repository's existing tables: ``PRONOUNS`` for index signs, ``spelled_word`` for
  lexicalised and fingerspelled glosses (``#SO``, ``fs-OR``; never a name sign), ``name_candidates`` for lexical glosses. No new mapping, no
  ASL Citizen map and no gloss in the selection. A fingerspelled source is flagged: it is a spelling, not the lexical sign.
Partial-meaning substitutes (``RECENT-PAST`` for ``just``, ``REALLY`` for ``actually``) are not used; ``options_not_used`` lists them for a person.

**Short spans.** ASLLRP annotates most index signs on one to three frames (the hold of the point), so at margin 0 such a piece would be
one still pose. A segment's piece is therefore widened evenly to at least ``MIN_SEGMENT_SECONDS`` (``min_seconds`` in its row), about
ASLLRP's median sign (0.234 s) and short of the 0.1 s margins the wrist diagnosis blames. No other word's core changes.

**Well tracked.** On the piece the composition would cut from (the trimmed clip, or the annotated span): the dominant forearm and
upper arm tracked on at least ``MIN_ARM`` of the frames and the dominant hand on at least ``MIN_HAND``. At most ``MAX_CHECKED``
candidates per token are checked (a fixed draw) and at most ``MAX_KEPT`` kept. A segment whose span cannot be placed is dropped.

**Flags for a person.** ``pointing``: an index or possessive sign, or a pronoun by the repository's word classes. Its direction
is not controlled, so a Deaf reviewer should say whether a citation point reads. ``shared_gloss``: other tokens the same
segments serve. ``prior_train_records``: how many isolated train records the prior saw under this exact word (0: generating the
word has no word-level training behind it).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from comp_a_lib import ALLOWED_CORPORA, D20_KEY, D20_SHA, SEGMENT_CORPUS, TOPS, check_source_record, median_seconds, require_d20  # noqa: E402

O = "acquisition/2026-10-17-v1/youtube-asl-v1"
EVAL_MIX = f"{O}/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
SOURCE_MIX = f"{O}/round14b-mixed-v1/manifest.json"
SOURCE_MIX_SHA = "710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d"
H2S = "how2sign-two-view-text-v1"
MIN_ARM, MIN_HAND = .8, .5
MAX_CHECKED, MAX_KEPT = 40, 16
MIN_SEGMENT_SECONDS = .2
RANKS = max(TOPS)
PARTIAL_OPTIONS = dict(just=["RECENT-PAST"], actually=["REALLY"])     # listed for a person, never used
RULE = (f"ranks: How2Sign train texts, unmatched under r10 by the composition's selection, ties alphabetical. Well tracked: dominant "
        f"arm >= {MIN_ARM} and dominant hand >= {MIN_HAND} of the piece's frames; a segment's piece widened to at least "
        f"{MIN_SEGMENT_SECONDS} s; the kind with more well-tracked clips serves a token. A rank without a source stays unsigned; nothing "
        "slides up. Vocabulary is train only; ASL Citizen, STEM Wiki, c00, test and held-out are refused.")


def ranked_tokens(manifest, vocab) -> list[tuple[str, int]]:
    """[(token, count)]: every token the selection leaves unmatched in How2Sign train texts, most frequent first."""
    from signtranslator.generation.compositional import selection_details
    counts = collections.Counter()
    for r in manifest["records"]:
        if r.get("corpus") != H2S or r.get("partition") != "train" or not r.get("text"):
            continue
        if r.get("signer_component") == "c00":
            raise SystemExit(f"{r.get('sign')}: How2Sign signer 9 (c00) is never read")
        for row in selection_details(r["text"], vocab):
            if row["kind"] == "unmatched":
                counts.update(row["tokens"])
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def contraction_entry(token: str, vocab, listed) -> tuple[str | None, str | None]:
    """(entry, status) of a contraction (module docstring); (None, "no_source") when its entry is neither r10's nor this list's."""
    from signtranslator.generation.compositional import _matchable
    if token.endswith("n't"):
        return ("not", "alias_r10") if _matchable("not", vocab) else (None, "no_source")
    piece = token.split("'")[0]
    if _matchable(piece, vocab):
        return piece, "alias_r10"
    return (piece, "alias_list") if piece in listed else (None, "no_source")


def negation_tokens(manifest, vocab) -> list[str]:
    """Every token the selection leaves unmatched in How2Sign train texts that ends in n't (rank does not matter)."""
    return sorted(t for t, _ in ranked_tokens(manifest, vocab) if t.endswith("n't"))


def isolated_candidates(manifest, tokens) -> dict:
    """{token: records}: the allowed corpora's train records with that exact normalised text and usable channels."""
    from signtranslator.data_engineering.corpus_manifest import usable_channels
    from signtranslator.generation.compositional import normalise
    wanted, out = set(tokens), collections.defaultdict(list)
    for r in manifest["records"]:
        if r.get("corpus") not in ALLOWED_CORPORA or r.get("partition") != "train" or not r.get("text"):
            continue
        entry = " ".join(normalise(r["text"]))
        if entry in wanted and usable_channels(r, "continuous") and (r.get("continuous") or {}).get("seconds"):
            check_source_record(r)
            out[entry].append(r)
    return {t: sorted(v, key=lambda r: r["sign"]) for t, v in out.items()}


def token_segments(utterances, gloss_manifest, tokens) -> tuple[dict, dict]:
    """({token: segments}, {token: other tokens served by the same glosses}): train utterances only; ``PRONOUNS`` for index
    signs, ``name_candidates`` (single entries) for the rest. The same segment shape as ``round20_lib.asllrp_segments``."""
    import asllrp_gloss_corpus as agc
    from build_asllrp_gloss_manifest import PARTITION, signer
    records = {r["sign"]: r for r in gloss_manifest["records"] if r.get("partition") == "train"}
    wanted, pools, shared = set(tokens), collections.defaultdict(list), collections.defaultdict(set)
    for u in utterances:
        video = u.get("utterance_video")
        if not video or PARTITION.get(signer(u.get("source") or "")) != "train":
            continue
        sign_id = f"{u['collection']}:{Path(video).stem}"
        record = records.get(sign_id)
        u0, u1 = (u.get("frames") or [None, None])[:2]
        if record is None or None in (u0, u1):
            continue
        for k, s in enumerate(u.get("signs") or ()):
            gloss = s.get("gloss")
            a, b = (s.get("frames") or [None, None])[:2]
            if not gloss or None in (a, b) or not u0 <= a <= b <= u1:
                continue
            kind = agc.category(gloss, s.get("sign_type", ""))
            if kind == "index":
                served = agc.PRONOUNS.get(agc.index_key(gloss, s.get("occurrence", "")), ())
            elif kind in ("lexicalised_fs", "fingerspelled"):
                served = (agc.spelled_word(gloss),)
            else:
                served = tuple(alt[0] for alt in agc.name_candidates(gloss) if len(alt) == 1)
            hits = [w for w in served if w in wanted]
            for w in hits:
                shared[w].update(x for x in hits if x != w)
                pools[w].append(dict(sign=f"{sign_id}#{k}", corpus=SEGMENT_CORPUS, gloss=gloss, gloss_kind=kind, occurrence=s.get("occurrence", ""),
                                     partition="train", record_sign=sign_id, start=int(a - u0), end=int(b - u0),
                                     utterance_frames=int(u1 - u0 + 1), min_seconds=MIN_SEGMENT_SECONDS))
    return {t: sorted(v, key=lambda x: x["sign"]) for t, v in pools.items()}, {t: sorted(v) for t, v in shared.items()}


def tracked_shares(state) -> dict:
    from signtranslator.generation.compositional import side_tracks
    arm, hand = side_tracks(state)["Right"]
    return dict(arm=round(float(arm.mean()), 4) if len(arm) else 0., hand=round(float(hand.mean()), 4) if len(hand) else 0.)


def well_tracked(shares: dict) -> bool:
    return shares["arm"] >= MIN_ARM and shares["hand"] >= MIN_HAND


def draw(candidates: list, token: str, k: int = MAX_CHECKED) -> list:
    """A fixed draw of at most ``k`` candidates, by the token, in sign order."""
    if len(candidates) <= k:
        return list(candidates)
    return sorted(random.Random(f"comp-a3|{token}").sample(candidates, k), key=lambda c: c["sign"])


def pick_source(isolated: int, segments: int) -> str | None:
    """The kind with more well-tracked clips serves a token; a tie goes to isolated, the form every r10 word uses."""
    if not isolated and not segments:
        return None
    return "isolated" if isolated >= segments else "segment"


def is_pointing(token: str, sources) -> bool:
    import asllrp_gloss_corpus as agc
    return token in agc.WORD_CLASSES["pronoun"] or any(s.get("corpus") == SEGMENT_CORPUS and agc.category(s["gloss"]) == "index"
                                                       for s in sources)


def build(manifest10, manifest14, utterances, gloss_manifest, *, load_clip, load_utterance, trimmed_over_annotated: float,
          ranks: int = RANKS, d20_texts=None) -> dict:
    """The list. ``load_clip(record)``: a word clip as the composition serves it. ``load_utterance(record)``: as
    ``round20_lib.utterance_clip``."""
    from signtranslator.generation.compositional import isolated_corpora, training_vocabulary
    from round20_lib import segment_clip
    vocab = training_vocabulary(manifest10, isolated_corpora(manifest10))
    every = ranked_tokens(manifest10, vocab)
    ranked = every[:ranks]
    listed = {t for t, _ in ranked}
    negations = sorted(t for t, _ in every if t.endswith("n't"))
    plain = [t for t in listed if "'" not in t]
    iso = isolated_candidates(manifest14, plain)
    segs, shared = token_segments(utterances, gloss_manifest, plain)
    prior_records = {t: len(iso.get(t, ())) for t in plain}
    rows, utterance_cache = [], {}
    by_sign = {r["sign"]: r for r in gloss_manifest["records"]}
    for rank, (token, count) in enumerate(ranked, 1):
        row = dict(rank=rank, token=token, train_count=count, status="no_source", entry=None, source_kind=None, sources=[],
                   typical_seconds=None, candidates=dict(isolated=len(iso.get(token, ())), segments=len(segs.get(token, ()))),
                   checked=0, flags={})
        if "'" in token:
            entry, status = contraction_entry(token, vocab, listed)
            row.update(status=status, entry=entry)
            row["flags"].update(contraction=True, negation=token.endswith("n't"),
                                pointing=is_pointing(token.split("'")[0], ()) if not token.endswith("n't") else False)
            row["source_rule"] = f"contraction: {status}, entry {entry}"
            rows.append(row)
            continue
        single = len(token) == 1
        row["flags"]["single_letter"] = single
        kept_iso, kept_seg, checked_iso, checked_seg = [], [], 0, 0
        if not single:
            for r in draw(iso.get(token, []), token):
                checked_iso += 1
                shares = tracked_shares(load_clip(r))
                if well_tracked(shares):
                    kept_iso.append(dict(sign=r["sign"], corpus=r["corpus"], partition=r["partition"],
                                         signer_component=r.get("signer_component"), tracked=shares))
        for s in draw(segs.get(token, []), token):
            sign = s["record_sign"]
            if sign not in utterance_cache:
                utterance_cache[sign] = load_utterance(by_sign[sign])
            piece = segment_clip(s, *utterance_cache[sign], margin=0.)
            checked_seg += 1
            if piece is None:
                continue
            shares = tracked_shares(piece["state"])
            if well_tracked(shares):
                kept_seg.append(dict(s, tracked=shares, seconds=round(len(piece["state"].timestamps) / 24., 4)))
        kind = pick_source(len(kept_iso), len(kept_seg))
        kept = (kept_iso if kind == "isolated" else kept_seg)[:MAX_KEPT]
        row["checked"] = checked_iso + checked_seg
        row["source_rule"] = (f"most well-tracked clips: isolated {len(kept_iso)} of {checked_iso} checked, segment {len(kept_seg)} of "
                              f"{checked_seg} checked; {kind or 'none'} serves it")
        for source in kept:
            check_source_record(source)
        if kept:
            typical = (median_seconds([r["continuous"]["seconds"] for r in iso[token]]) if kind == "isolated"
                       else median_seconds([k["seconds"] for k in kept]) * trimmed_over_annotated)
            row.update(status="clips", entry=token, source_kind=kind, sources=kept, typical_seconds=round(typical, 4))
        gloss_kinds = sorted({k.get("gloss_kind") for k in kept if k.get("gloss_kind")})
        row["flags"].update(pointing=is_pointing(token, kept or segs.get(token, [])), shared_gloss=shared.get(token, []),
                            prior_train_records=prior_records[token], no_well_tracked_source=not kept, single_clip=len(kept) == 1,
                            fingerspelled_source=any(k in ("lexicalised_fs", "fingerspelled") for k in gloss_kinds))
        if token in PARTIAL_OPTIONS:
            row["options_not_used"] = PARTIAL_OPTIONS[token]
        rows.append(row)
    entries = {r["token"] for r in rows if r["status"] == "clips"}
    for r in rows:                    # a contraction whose entry found no clip has no source after all
        if r["status"] == "alias_list" and r["entry"] not in entries:
            r["flags"]["no_well_tracked_source"] = True
    out = dict(kind="comp-a3-tokens", rule=RULE, min_arm=MIN_ARM, min_hand=MIN_HAND, trimmed_over_annotated=trimmed_over_annotated,
               min_segment_seconds=MIN_SEGMENT_SECONDS, negation_tokens=negations,
               inputs=dict(rank_mix=dict(key=EVAL_MIX, sha256=EVAL_MIX_SHA), source_mix=dict(key=SOURCE_MIX, sha256=SOURCE_MIX_SHA)),
               tokens=rows)
    if d20_texts is not None:
        out["d20_coverage_text_only"] = coverage_lines(d20_texts, vocab, rows, negations)
    return out


def unsigned_negation(details) -> int:
    """Sentences holding an n't token that stays unmatched."""
    return int(sum(any(r["kind"] == "unmatched" and r["tokens"][0].endswith("n't") for r in rows) for rows in details))


def coverage_lines(texts, vocab, rows, negations=()) -> dict:
    """Words per sentence and token shares on D20's text under r10 and each arm. No rank depends on this; no motion is read."""
    from signtranslator.generation.compositional import selection_details, selection_statistics
    from comp_a_lib import apply_token_map, arm_of
    base = [selection_details(t, vocab) for t in texts]
    out = dict(r10=dict(_coverage(base, selection_statistics), sentences_with_unsigned_negation=unsigned_negation(base)))
    for top in TOPS:
        arm = arm_of(dict(tokens=rows, negation_tokens=list(negations)), top)
        details, _ = apply_token_map(base, arm["token_map"])
        out[f"top{top}"] = dict(_coverage(details, selection_statistics), new_entries=sorted(arm["entries"]),
                                unsigned_ranks=sorted(arm["dropped"]), sentences_with_unsigned_negation=unsigned_negation(details))
    return out


def _coverage(details, stats) -> dict:
    s = stats(details)
    return dict(words_per_sentence=s["words_per_sentence_mean"], token_share=s["token_share"], words=s["words_total"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--output-file", required=True)
    p.add_argument("--cache-dir", default=str(Path(tempfile.gettempdir()) / "sign-translator-cache"))
    args = p.parse_args(argv)
    if Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from train_motion_tokens import layout_template
    from train_text_to_motion import make_store
    from round15_oracle import TEMPLATE_CORPUS, word_clip
    from round15_stage2_fit import KEYS, fit_model, agc
    from round20_lib import GLOSS_MANIFEST, utterance_clip
    from s2_lib import load_manifest_at
    store = make_store(args.store)
    manifest10, manifest14 = load_manifest_at(store, EVAL_MIX, EVAL_MIX_SHA), load_manifest_at(store, SOURCE_MIX, SOURCE_MIX_SHA)
    gloss = load_manifest_at(store, *GLOSS_MANIFEST)
    utterances = json.loads(store.get(KEYS["utterances"]))["items"]
    from signtranslator.generation.compositional import isolated_corpora
    model, _ = fit_model(agc._store(args.store), EVAL_MIX, EVAL_MIX_SHA, isolated_corpora(manifest10))
    cache = ContentCache(Path(args.cache_dir), 3 * 2 ** 30)
    template = layout_template(manifest10, TEMPLATE_CORPUS, cache, store.get)
    require_d20(D20_KEY, D20_SHA)
    raw = store.get(D20_KEY)
    if hashlib.sha256(raw).hexdigest() != D20_SHA:
        raise SystemExit(f"{D20_KEY} differs from its pin")
    out = build(manifest10, manifest14, utterances, gloss,
                load_clip=lambda r: word_clip(r, cache=cache, fetch=store.get, template=template),
                load_utterance=lambda r: utterance_clip(r, cache=cache, fetch=store.get, template=template),
                trimmed_over_annotated=float(model["timing"]["trimmed_over_annotated"]),
                d20_texts=[r["text"] for r in json.loads(raw)["records"]])
    data = json.dumps(out, indent=1, sort_keys=True).encode()
    with open(args.output_file, "xb") as stream:
        stream.write(data)
    print(args.output_file, hashlib.sha256(data).hexdigest(), flush=True)
    for r in out["tokens"]:
        print(r["rank"], r["token"], r["train_count"], r["status"], r["entry"], r["source_kind"], len(r["sources"]), r["candidates"],
              r["checked"], {k: v for k, v in r["flags"].items() if v}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
