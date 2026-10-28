"""Round 15, stage 1: a compositional sentence generator at inference time, against end-to-end generation
(docs/NEXT_PHASE_PLAN.md, "Round 15 declared", 2026-10-28 14:59 UTC; the declaration is fixed).

**Clips:** the pilot's (``pilot_content_check.validation_sentences``: How2Sign val, signer groups c01/c03/c05, seed 0,
``--clips``), real states at 24 fps; the reference duration is ``frames / 24`` as in the pilot.

**Word selection** (``signtranslator.generation.compositional.select_words``, no evaluator): the training vocabulary is
the English texts of the prior manifest's one-word train records (``--corpora``, default the isolated-sign corpora
present in the manifest: asllvd-text-v1, rit-text-r6, wlasl-text-r6 and, when present, semlex-text-v1), matched
greedily longest first, never a single-letter entry, with the declared inflection fallback (-s/-es/-ies, -ing, -ed) for
tokens that start no entry, English order, the declared stop words dropped (amendment 1, 15:29 UTC). The report's
``selection`` (also per variant) gives words per sentence and the fallback, stop-word and unmatched shares. Durations: ``fitted_durations``, each word's
typical duration the median trimmed ``continuous.seconds`` of its train records, floor 0.3 s.

**Forms** (each back-translated by the pilot's instrument, ``back-translator-kp2d-uwb-ft-v1``, beam 4):
- ``real``, ``blank`` (``evaluate_text_to_motion.blanked``) and ``end_to_end``: the pilot's generated form by the pilot's
  own sampler (``compositional_baseline.generate_end_to_end``: batches of 32, seeds 1000 + i, reference duration);
- per variant V of ``--variants`` (``crossfade`` = V1, ``chained`` = V2, ``chained_trim`` = V3; ``compose``):
  - ``V``: the sentence's selected words, durations fitted to the reference duration (crossfade: plus the frames its
    joins overlap, as the baseline), word k's seed ``100000 + 1000 i + k``;
  - ``V/shuffled``: the same words, durations and seeds in a seeded non-identity order (``compositional_baseline.
    shuffle_order(n, clip, 0)``, the same order for every variant): content without order. Crossfade reorders its
    word clips, which is exactly the composition of the reordered words (each clip has its own seed: tested); chained
    modes generate the reordered chain;
  - ``V/mismatched_words``: another sentence's selected words (a seeded derangement, ``--seed`` + 1), durations fitted
    to *this* clip's reference duration, seeds ``+ 10^7``: the same mode and duration, other words.
  A sentence with no selected word reads as its blank clip in that form (counted).
``--word-workers`` spreads the compositions over processes (each loads the prior on ``--device``); every composition
draws from its own seeds, so the states do not depend on it.

**Scores:**
- corpus BLEU with the pilot's bootstrap interval; the deranged-reference control (``--seed``); paired differences
  against end_to_end, blank, the variant's shuffled and mismatched-words controls and the other variants
  (``sentence_exposure_diagnostic.score_forms``);
- ``sentence_word_spotting`` (its functions, unchanged: ``--recognizer-run``, every label, windows 0.6/0.9/1.2 s, hop
  0.1 s, top-5) on real, blank, end_to_end and each variant: recall against the sentence's words and against the next
  sentence's words (its mismatched control), the instrument's ``separated`` reading, and the paired per-sentence
  difference (reported, not part of the rule);
- motion statistics (``compositional_baseline.motion_rows``): duration and its ratio to the reference, wrist speed,
  hand tracking, wrist spread, the fraction over the translator's 20 s window; and their ratios to real.

**Verdict (declared):** a variant is PROMISING only if (1) its paired BLEU against end_to_end has a 95% lower bound
above 0, (2) its paired BLEU against its mismatched-words control has a lower bound above 0, and (3) its word-spotting
recall is above its mismatched control (the instrument's reading: the recall's lower bound above the control's upper
bound). Otherwise NOT PROMISING.

Writes one JSON report to ``--output-key`` (or ``--output-file`` for local checks; both write-once) and prints a summary
with the verdicts.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 24.
TRANSLATOR_RUN = "back-translator-kp2d-uwb-ft-v1"
SEED_BASE = 100_000                 # word k of clip i: SEED_BASE + MAX_WORDS i + k (+ MISMATCHED_BLOCK)
MAX_WORDS = 1000
MISMATCHED_BLOCK = 10_000_000
CONTROLS = ("", "/shuffled", "/mismatched_words")
SPOT_WINDOWS = (.6, .9, 1.2)

CHAIN_DESIGN = (
    "chained: segment k is the prior's continuation (continue_from, the prefix forms' sampler) of every earlier word's "
    "generated codes, with word k's text alone in the cross-attention (its own planned loci; the empty text for "
    "guidance). The clock runs uniformly from 0 over history and segment, so segment k starts where segment k-1 "
    "ended; the duration input is the cumulative end of word k (continue_from's count / 24), so the phase "
    "seconds/duration reaches 1 exactly at word k's end, as an isolated training clip's does at its end. Word k's own "
    "duration with that clock would give phases far above 1, a clock restarting per word would put the history at "
    "negative or repeated times, and the sentence's total duration would leave every word but the last without an end. "
    "Cost: word k starts at phase start_k/end_k, near 1 for late words. chained_trim: after each segment, its frames "
    "outside build_trimmed_word_manifest's dominant-wrist window (the manifest's trim parameters) are cut from the "
    "codes before the next word continues; a segment the rule cannot trim is kept whole.")

CAVEATS = [
    "Word order: composed forms sign English words in English order; How2Sign is interpreted from English and the "
    "translator was fine-tuned on it, so the English order may favour composed forms. V/shuffled measures how much of "
    "any gain is order.",
    "Vocabulary (amendment 1): single-letter entries (ASLLVD's fingerspelled letters) are never matched, so the "
    "pronoun 'I' is dropped. The inflection fallback maps by spelling only and can mislead (toes -> to, a stop word; a "
    "noun plural onto a homograph verb); it is reported (fallback_share, per_clip selection). Contractions, "
    "possessives (it's, don't, mother's) and digits stay out of vocabulary, as the corpora write none; irregular forms "
    "(went, made) and synonyms are not mapped. Function words that have a sign (and, with, for, but) are kept.",
    "Durations: fitted words are short (the baseline's median was about 0.5 s for the content words) while the prior's "
    "isolated training clips are trimmed to about 1.7-2.0 s; words at the 0.3 s floor make a composition longer than "
    "its reference (reported).",
    "Chained clock: word k starts at phase start_k/end_k (near 1 for late words), a phase isolated clips only see near "
    "their end; the prior may then begin the word as an ending. The duration input is the cumulative end, not the "
    "word's own duration (see chain_design).",
    "Rest trim (V3): the rule's 0.25 s margins and 0.5 s minimum window are the training manifest's, fitted to 1-2 s "
    "isolated clips; on 0.5-1 s generated segments it cuts little (3-clip F12 smoke: every segment 'trimmed', 98.4% "
    "of frames kept), so V3 may differ little from V2 (reported: trim_reasons, frames_kept_fraction). Trimmed "
    "sentences are shorter than their reference (reported: seconds_over_reference).",
    "Joins: V1's 4-frame crossfade joins unrelated end and start poses; V2/V3 rely on the prior's own continuation, "
    "which in training only followed the same text.",
    "Controls: mismatched_words keeps the clip's duration and mode but changes the words (and their number); shuffled "
    "keeps words and durations. A long word list fitted into a short reference hits the 0.3 s floor, so a "
    "mismatched-words clip can be longer than its reference (3-clip F12 smoke: seconds_over_reference 1.9-2.7 on "
    "average, one short clip). The deranged-reference control reads each form against other sentences' English.",
    "Instrument: back-translator-kp2d-uwb-ft-v1 reads arms and hands only (48 2D points, no face). Blank reads about 2 "
    "BLEU (the translator's text prior); read paired differences and controls, not point gaps. Word spotting uses the "
    "isolated-sign recogniser (v2) on windows; its labels never select words.",
    "Decoder gains are applied per clip about the clip mean: per word for V1 (as the pilot lexicon), once over the "
    "whole chain for V2/V3 (as end-to-end).",
    "Sampling noise: one sample per word occurrence; V1, V2 and V3 share seeds, so the variants' differences are "
    "seed-paired, not sample-identical: V1 conditions each word on its fitted seconds, V2/V3 on whole frames (and the "
    "decoder gains are applied per word for V1, over the chain for V2/V3). V2 and V3 share their first segment's codes.",
]


# ---- pure logic --------------------------------------------------------------------------------------------------

def word_seeds(clip: int, count: int, *, mismatched: bool = False) -> list[int]:
    """The seeds of a composition's words: one per word, unique over clips, words and the two blocks."""
    if count > MAX_WORDS:
        raise ValueError(f"more than {MAX_WORDS} words in a sentence")
    return [SEED_BASE + MAX_WORDS * clip + k + (MISMATCHED_BLOCK if mismatched else 0) for k in range(count)]


def plan_durations(mode: str, words, typical, reference_seconds: float, overlap: int) -> list[float]:
    """Fitted durations of a composition: the reference duration, plus the frames crossfade joins overlap."""
    from signtranslator.generation.compositional import fitted_durations
    total = reference_seconds + ((len(words) - 1) * overlap / FPS if mode == "crossfade" and words else 0.)
    return fitted_durations(words, typical, total)


def plan_jobs(variants, selected, typical, seconds, content_perm, orders, overlap) -> dict:
    """{(variant, control, clip): job dict (mode, words, durations, seeds) or None (no word: the blank clip)}.
    Crossfade's shuffled form is not a job (its matched clips are reordered)."""
    jobs = {}
    for v in variants:
        for i, words in enumerate(selected):
            if not words:
                jobs[(v, "", i)] = jobs[(v, "/shuffled", i)] = None
            else:
                d = plan_durations(v, words, typical, seconds[i], overlap)
                s = word_seeds(i, len(words))
                jobs[(v, "", i)] = dict(mode=v, words=list(words), durations=d, seeds=s)
                o = orders[i]
                jobs[(v, "/shuffled", i)] = None if v == "crossfade" else dict(
                    mode=v, words=[words[k] for k in o], durations=[d[k] for k in o], seeds=[s[k] for k in o])
            other = selected[content_perm[i]]
            jobs[(v, "/mismatched_words", i)] = dict(
                mode=v, words=list(other), durations=plan_durations(v, other, typical, seconds[i], overlap),
                seeds=word_seeds(i, len(other), mismatched=True)) if other else None
    return jobs


def form_pairs(variants) -> dict:
    """``score_forms``'s paired comparisons: each variant against end_to_end, its two controls and the other variants;
    each control against end_to_end."""
    pairs = {}
    for v in variants:
        pairs[v] = {"end_to_end": "end_to_end", "shuffled": f"{v}/shuffled",
                    "mismatched_words": f"{v}/mismatched_words",
                    **{f"variant_{w}": w for w in variants if w != v}}
        for c in CONTROLS[1:]:
            pairs[f"{v}{c}"] = {"end_to_end": "end_to_end"}
    return pairs


def verdict(bleu_row: dict, spotting: dict, spotting_mismatched: dict) -> dict:
    """The declared rule on one variant: its BLEU row (``score_forms``) and its word-spotting recall rows."""
    from strict_prior_review import separated
    lower_e2e = bleu_row["minus_end_to_end"]["ci95"][0]
    lower_mism = bleu_row["minus_mismatched_words"]["ci95"][0]
    a, b = (spotting or {}).get("ci95"), (spotting_mismatched or {}).get("ci95")
    spot = bool(a and b) and separated(a, b)
    checks = dict(bleu_vs_end_to_end_lower_above_0=bool(lower_e2e > 0),
                  bleu_vs_mismatched_words_lower_above_0=bool(lower_mism > 0),
                  spotting_recall_above_mismatched=bool(spot))
    return dict(checks, verdict="PROMISING" if all(checks.values()) else "NOT PROMISING",
                minus_end_to_end=bleu_row["minus_end_to_end"], minus_mismatched_words=bleu_row["minus_mismatched_words"],
                spotting_recall=spotting, spotting_recall_mismatched=spotting_mismatched)


def paired_recall(rows, rows_mismatched, seed: int = 0) -> dict:
    """Mean per-sentence recall minus the mismatched recall (sentences with both), with a bootstrap 95% interval."""
    from strict_prior_review import bootstrap_mean
    d = np.array([a["recall"] - b["recall"] for a, b in zip(rows, rows_mismatched)
                  if a["recall"] is not None and b["recall"] is not None], dtype=float)
    if not len(d):
        return dict(difference=None, ci95=None, n=0)
    return dict(difference=round(float(d.mean()), 4), ci95=bootstrap_mean(d, seed=seed), n=int(len(d)))


def trim_parameters(manifest) -> dict:
    """The rest-trim parameters the prior's manifest was cut with (its ``trim`` block), else the rule's defaults."""
    from signtranslator.generation.compositional import TRIM
    block = manifest.get("trim") or {}
    return {k: float(block.get(k, v)) for k, v in TRIM.items()}


# ---- generation --------------------------------------------------------------------------------------------------

def compose_job(sample, job: dict, *, guidance: float, temperature, crossfade: int, trim: dict):
    """``compose`` of one job with the loaded generator (``load_generator``'s sample: its prior, tokenizer and accepted
    decoder gains), post-processed as the pilot (``exposure_diagnostic.make_post``)."""
    from signtranslator.generation.compositional import compose
    from exposure_diagnostic import make_post
    state, info = compose(sample.prior, sample.tokenizer, job["words"], job["durations"], job["mode"], job["seeds"],
                          guidance=guidance, temperature=temperature, post=make_post(sample.decoder_gains),
                          crossfade=crossfade, trim=trim)
    info.pop("codes", None)
    return state, info


_WORKER = {}


def _init_worker(store_spec, checkpoint_key, mix_sha256, device, temperature, threads, settings):
    """A composing process: its own store and generator, loaded as the main process loads them."""
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from evaluate_text_to_motion import load_generator
    from train_text_to_motion import make_store
    torch.set_num_threads(threads)
    _WORKER["sample"] = load_generator(make_store(store_spec), checkpoint_key, mix_sha256, device,
                                       temperature=temperature)[2]
    _WORKER["settings"] = settings


def _run(job):
    return compose_job(_WORKER["sample"], job, **_WORKER["settings"])


def run_jobs(sample, jobs: list, settings: dict, *, workers: int = 1, worker_args=None, progress=None,
             every: int = 50) -> list:
    """``compose_job`` of each job, in order: in this process, or in ``workers`` spawned processes (longest jobs
    first). Every job has its own seeds, so the result does not depend on the workers."""
    if workers <= 1:
        out = []
        for k, job in enumerate(jobs):
            out.append(compose_job(sample, job, **settings))
            if progress and ((k + 1) % every == 0 or k + 1 == len(jobs)):
                progress(k + 1)
        return out
    import multiprocessing
    order = sorted(range(len(jobs)), key=lambda k: -sum(jobs[k]["durations"]))
    out = [None] * len(jobs)
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker,
                                                   initargs=(*worker_args, settings)) as pool:
        for done, (k, result) in enumerate(zip(order, pool.imap(_run, [jobs[k] for k in order], chunksize=1))):
            out[k] = result
            if progress and ((done + 1) % every == 0 or done + 1 == len(jobs)):
                progress(done + 1)
    return out


# ---- the run -----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    from signtranslator.generation.compositional import ISOLATED_CORPORA, MODES
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", required=True)
    p.add_argument("--mix-key", required=True, help="the manifest the prior was trained on (its vocabulary)")
    p.add_argument("--mix-sha256", required=True)
    p.add_argument("--clips", type=int, default=300)
    p.add_argument("--variants", default=",".join(MODES))
    p.add_argument("--corpora", default=",".join(ISOLATED_CORPORA), help="the isolated-sign corpora of the vocabulary")
    p.add_argument("--part-temperature", default="body=1.1,left_hand=1.1,right_hand=1.1,face=1.0")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--crossfade", type=float, default=.15, help="seconds of overlap at each crossfade join (V1)")
    p.add_argument("--translator-run", default=TRANSLATOR_RUN)
    p.add_argument("--recognizer-run", default="sign-recognizer-v2", help="the word-spotting instrument")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0, help="the reference derangement (the words derangement: seed + 1)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--word-workers", type=int, default=1, help="processes composing sentences (each loads the prior "
                   "on --device; every composition has its own seeds, so the states do not depend on this)")
    p.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    p.add_argument("--output-key", help="the store key of the JSON report (write-once)")
    p.add_argument("--output-file", help="a local path for the JSON report instead (write-once; local checks)")
    args = p.parse_args(argv)
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    variants = [v for v in args.variants.split(",") if v]
    if not variants or any(v not in MODES for v in variants) or len(set(variants)) != len(variants):
        p.error(f"--variants: distinct names from {','.join(MODES)}")
    corpora = tuple(c for c in args.corpora.split(",") if c)
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.generation.compositional import (KEPT, STOP_WORDS, concatenate, crossfade_frames,
                                                         selection_details, selection_statistics, training_vocabulary)
    from compositional_baseline import (generate_end_to_end, hypothesis_length_ratio, motion_rows, shuffle_order,
                                        MOTION_KEYS)
    from evaluate_text_to_motion import at_rate, blanked, load_generator, load_state
    from pilot_content_check import validation_sentences
    from score_back_translation import load_translator, translate
    from sentence_exposure_diagnostic import derangement, mean_statistics, ratio_to, score_forms
    import sentence_word_spotting as sws
    from strict_prior_review import separated
    from train_text_to_motion import make_store
    import validate_sign_recognizer as vsr
    clock = {"start": time.time()}
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    vocab = training_vocabulary(manifest, corpora)
    if not vocab:
        raise SystemExit(f"the manifest has no train records of {corpora}: no training vocabulary")
    present = sorted({r.get("corpus") for r in manifest["records"]} & set(corpora))
    trim = trim_parameters(manifest)
    val = validation_sentences(manifest, args.clips)
    texts = [r["text"] for r in val]
    n = len(texts)
    if n < 2:
        raise SystemExit("need at least two clips (the controls are derangements)")
    refs = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), FPS) for r in val]
    seconds = [float(s.timestamps[-1] - s.timestamps[0]) + 1 / FPS for s in refs]
    blanks = blanked(refs)
    details = [selection_details(t, vocab) for t in texts]
    selected = [[r["entry"] for r in rows if r["kind"] in KEPT] for rows in details]
    if max(len(w) for w in selected) > MAX_WORDS:
        raise SystemExit(f"a sentence selects more than {MAX_WORDS} words")
    if SEED_BASE + MAX_WORDS * n >= MISMATCHED_BLOCK:
        raise SystemExit("too many clips: word seeds would reach the mismatched block")
    reference_perm = derangement(n, args.seed)
    content_perm = derangement(n, args.seed + 1)
    orders = [shuffle_order(len(w), i, 0) for i, w in enumerate(selected)]
    overlap = crossfade_frames(args.crossfade)
    jobs = plan_jobs(variants, selected, vocab, seconds, content_perm, orders, overlap)
    work = [k for k, j in jobs.items() if j is not None]
    print(f"{n} clips, {sum(seconds):.0f} s of reference; vocabulary {len(vocab)} entries ({','.join(present)}); "
          f"{sum(len(w) for w in selected)} words selected; {len(work)} compositions, "
          f"{sum(sum(jobs[k]['durations']) for k in work):.0f} s to generate", flush=True)

    temperature = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature.split(","))}
    _, _, sample = load_generator(store, args.checkpoint_key, args.mix_sha256, args.device, temperature=temperature)
    end_to_end = generate_end_to_end(sample, texts, seconds, args.guidance)
    clock["end_to_end"] = time.time()
    print(f"end to end done ({clock['end_to_end'] - clock['start']:.0f} s)", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()           # the batched sampler's cached blocks, before the workers take their share
    settings =dict(guidance=args.guidance, temperature=temperature, crossfade=overlap, trim=trim)
    worker = (args.store, args.checkpoint_key, args.mix_sha256, args.device, temperature,
              max(1, torch.get_num_threads() // max(args.word_workers, 1)))
    results = run_jobs(sample, [jobs[k] for k in work], settings, workers=args.word_workers, worker_args=worker,
                       progress=lambda k: print(f"composition {k}/{len(work)} ({time.time() - clock['start']:.0f} s)",
                                                flush=True))
    done = dict(zip(work, results))
    clock["compose"] = time.time()

    forms = {"real": refs, "blank": blanks, "end_to_end": end_to_end}
    per_clip = [dict(sign=r["sign"], text=t, words=w, shuffled_order=o, mismatched_from=c,
                     fallback=[[" ".join(x["tokens"]), x["entry"]] for x in rows_ if x["kind"] == "fallback"],
                     unmatched=[x["tokens"][0] for x in rows_ if x["kind"] == "unmatched"],
                     frames_reference=len(ref.timestamps), frames_end_to_end=len(e.timestamps))
                for r, t, w, o, c, ref, e, rows_ in zip(val, texts, selected, orders, content_perm, refs, end_to_end,
                                                        details)]
    for v in variants:
        for c in CONTROLS:
            states = []
            for i in range(n):
                key = (v, c, i)
                if key in done:
                    states.append(done[key][0])
                elif c == "/shuffled" and jobs[(v, "", i)] is not None:      # crossfade: the matched clips reordered
                    clips = done[(v, "", i)][1]["clips"]
                    states.append(concatenate([clips[k] for k in orders[i]], overlap,
                                              sample_id=f"compositional-crossfade-shuffled:{i}"))
                else:
                    states.append(blanks[i])
                if c == "":
                    job = jobs[key]
                    row = dict(frames=len(states[-1].timestamps), empty=job is None)
                    if job is not None:
                        row.update(durations=[round(d, 4) for d in job["durations"]],
                                   segments=[{k2: v2 for k2, v2 in s.items() if k2 != "seed"}
                                             for s in done[key][1]["segments"]])
                    per_clip[i][v] = row
            forms[f"{v}{c}"] = states
    clock["assemble"] = time.time()

    # selection and timing statistics
    counts = [len(w) for w in selected]
    tokens = [len(t.split()) for t in texts]
    stats = selection_statistics(details)
    fallback_pairs = collections.Counter((" ".join(x["tokens"]), x["entry"]) for rows in details for x in rows
                                         if x["kind"] == "fallback")
    plans = {v: [d for i in range(n) if jobs[(v, "", i)] for d in jobs[(v, "", i)]["durations"]] for v in variants}
    words_flat = [w for ws in selected for w in ws]
    selection = dict(
        vocabulary_entries=len(vocab), vocabulary_multiword=sum(" " in w for w in vocab), corpora=present,
        words_per_whitespace_token=round(sum(counts) / max(sum(tokens), 1), 4),
        multiword_words=int(sum(" " in w for w in words_flat)),
        shuffle_unchanged=int(sum([w[k] for k in o] == w for w, o in zip(selected, orders) if w)),
        single_letter_words=int(sum(len(w) == 1 for w in words_flat)), **stats,
        most_common=collections.Counter(words_flat).most_common(30),
        fallback_most_common=[[t, e, c] for (t, e), c in fallback_pairs.most_common(30)],
        unmatched_most_common=collections.Counter(x["tokens"][0] for rows in details for x in rows
                                                  if x["kind"] == "unmatched").most_common(30))
    # per variant (amendment 1): every variant signs the same selection; its mismatched-words control signs the
    # deranged sentences' selections, which have the same statistics
    selection["per_variant"] = {v: dict(stats) for v in variants}
    timing = {}
    for v in variants:
        main_states = forms[v]
        ratio = [len(s.timestamps) / len(r.timestamps) for s, r, row in zip(main_states, refs, per_clip)
                 if not row[v]["empty"]]
        timing[v] = dict(fitted_word_seconds_median=round(float(np.median(plans[v])), 4) if plans[v] else None,
                         words_at_floor=round(float(np.mean([d <= .3 + 1e-9 for d in plans[v]])), 4) if plans[v]
                         else None,
                         frames_over_reference_median=round(float(np.median(ratio)), 4) if ratio else None,
                         frames_over_reference_p10_p90=[round(float(np.percentile(ratio, q)), 4) for q in (10, 90)]
                         if ratio else None)
        if v == "chained_trim":
            segs = [s for i in range(n) if jobs[(v, "", i)] for s in done[(v, "", i)][1]["segments"]]
            timing[v]["trim_reasons"] = dict(collections.Counter(s["trim"] for s in segs))
            timing[v]["frames_kept_fraction"] = round(sum(s["kept"] for s in segs) / max(sum(s["frames"] for s in segs),
                                                                                          1), 4)

    # back-translation
    kp = json.loads(store.get(f"training-runs/{args.translator_run}/best.json"))["checkpoint_key"]
    from pilot_content_check import MIX_SHA
    translator, kind = load_translator(store, kp, MIX_SHA)                  # the translator's own manifest (round 3c)
    translator = translator.to(torch.device(args.device))
    translator_fps = float(getattr(getattr(translator, "config", None), "fps", FPS))
    hyps = {}
    for name, states in forms.items():
        hyps[name] = list(translate(translator, kind, states))
        print(f"translated {name} ({time.time() - clock['start']:.0f} s)", flush=True)
    clock["translation"] = time.time()
    scores = score_forms(hyps, texts, reference_perm, boot=args.bootstrap, pairs=form_pairs(variants))
    for name, row in scores.items():
        row["hypothesis_over_reference_length"] = hypothesis_length_ratio(hyps[name], texts)
    clock["scoring"] = time.time()

    # word spotting (sentence_word_spotting's functions, unchanged)
    recogniser, labels, best = vsr.load_model(store, args.recognizer_run, args.device)
    index = {g: i for i, g in enumerate(labels)}
    spot_words = [sws.sentence_labels(t, set(labels)) for t in texts]
    spot_mismatched = spot_words[1:] + spot_words[:1]                     # the next sentence's words
    spotting, spot_rows = {}, {}
    for name in ["real", "blank", "end_to_end"] + variants:
        preds = [sws.window_predictions(recogniser, at_rate(s, FPS), args.device, windows=SPOT_WINDOWS, hop=.1,
                                        k=args.top_k) for s in forms[name]]
        own, other = sws.score_form(preds, spot_words, index), sws.score_form(preds, spot_mismatched, index)
        spot_rows[name] = own
        spotting[name] = dict(recall=sws.summarise([x["recall"] for x in own]),
                              order=sws.summarise([x["order"] for x in own]))
        spotting[f"{name}_mismatched"] = dict(recall=sws.summarise([x["recall"] for x in other]),
                                              order=sws.summarise([x["order"] for x in other]))
        a, b = spotting[name]["recall"]["ci95"], spotting[f"{name}_mismatched"]["recall"]["ci95"]
        spotting[name]["above_mismatched"] = bool(a and b and separated(a, b))
        spotting[name]["minus_mismatched_paired"] = paired_recall(own, other)
        print(f"spotted {name} ({time.time() - clock['start']:.0f} s)", flush=True)
    clock["spotting"] = time.time()

    motion = {name: mean_statistics(motion_rows(states, refs, translator_fps)) for name, states in forms.items()}
    motion_ratio = {name: ratio_to(m, motion["real"]) for name, m in motion.items()}
    verdicts = {v: verdict(scores[v], spotting[v]["recall"], spotting[f"{v}_mismatched"]["recall"]) for v in variants}
    clock["statistics"] = time.time()

    report = dict(
        round=15, stage=1, checkpoint_key=args.checkpoint_key, mix_key=args.mix_key, mix_sha256=args.mix_sha256,
        translator_run=args.translator_run, translator_key=kp, recognizer_run=args.recognizer_run,
        recognizer_checkpoint=best.get("checkpoint_key") if isinstance(best, dict) else None, clips=n,
        variants=variants, guidance=args.guidance, part_temperature=temperature,
        decoder_gains=getattr(sample, "decoder_gains", None), crossfade=dict(seconds=args.crossfade, frames=overlap),
        trim=trim, chain_design=CHAIN_DESIGN, corpora=list(corpora), floor_seconds=.3,
        stop_words=sorted(STOP_WORDS),
        typical_seconds={w: round(vocab[w], 4) for w in sorted(set(words_flat))},
        seeds=dict(end_to_end="1000 + batch start (batches of 32)", word=f"{SEED_BASE} + {MAX_WORDS} clip + word index",
                   mismatched_words=f"word seed + {MISMATCHED_BLOCK}", shuffle="compositional_baseline.shuffle_order"
                   "(n, clip, 0)", reference_derangement=args.seed, words_derangement=args.seed + 1),
        bootstrap=args.bootstrap, beam=4, selection=selection, timing=timing, references=texts,
        reference_derangement=reference_perm, words_derangement=content_perm, forms=scores,
        spotting=dict(windows=list(SPOT_WINDOWS), hop=.1, top_k=args.top_k, vocab="all", labels=len(labels),
                      scores=spotting), motion=motion, motion_ratio_to_real=motion_ratio, verdicts=verdicts,
        seconds={k: round(v - clock["start"], 1) for k, v in clock.items() if k != "start"},
        caveats=CAVEATS, per_clip=per_clip)
    for i, row in enumerate(per_clip):
        row["spotting"] = {name: spot_rows[name][i]["found"] for name in spot_rows}
    summary(report, MOTION_KEYS)
    payload = json.dumps(report, indent=1, default=float).encode()
    if args.output_file:
        with open(args.output_file, "xb") as stream:                     # write-once
            stream.write(payload)
    if args.output_key:
        if store.exists(args.output_key):
            raise SystemExit(f"{args.output_key} appeared during the run: refusing to overwrite")
        store.put(args.output_key, payload)
    return 0


def summary(report: dict, motion_keys) -> None:
    """BLEU and controls per form, word spotting, motion, and the declared verdict per variant."""
    def ci(d):
        return f"{d['difference']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"

    def rc(d):
        return f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]" if d.get("ci95") else "n/a"
    forms = report["forms"]
    print(f"BLEU on {report['clips']} clips (form, BLEU [95% CI], minus deranged refs, minus blank, minus end_to_end, "
          f"controls, hyp/ref length):", flush=True)
    for name, row in forms.items():
        extra = "".join(f"  -{k[6:]} {ci(v)}" for k, v in row.items()
                        if k.startswith("minus_") and k not in ("minus_mismatched", "minus_blank", "minus_end_to_end"))
        print(f"  {name:28s} {row['bleu']:6.2f} [{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]"
              f"  -mism {ci(row['minus_mismatched'])}"
              + (f"  -blank {ci(row['minus_blank'])}" if "minus_blank" in row else "")
              + (f"  -e2e {ci(row['minus_end_to_end'])}" if "minus_end_to_end" in row else "")
              + extra + f"  len {row['hypothesis_over_reference_length']}", flush=True)
    print("word spotting recall (form, recall, against the next sentence's words, above):", flush=True)
    s = report["spotting"]["scores"]
    for name in [k for k in s if not k.endswith("_mismatched")]:
        print(f"  {name:28s} {rc(s[name]['recall'])}  mism {rc(s[name + '_mismatched']['recall'])}  "
              f"above {s[name]['above_mismatched']}", flush=True)
    print("motion (" + ", ".join(motion_keys) + "):", flush=True)
    for name, m in report["motion"].items():
        print(f"  {name:28s} " + "  ".join(f"{m.get(k):7.3f}" if m.get(k) is not None else "    nan"
                                           for k in motion_keys), flush=True)
    print(json.dumps(dict(selection={k: v for k, v in report["selection"].items()
                                     if not k.endswith("most_common") and k != "per_variant"},
                          timing=report["timing"], seconds=report["seconds"])), flush=True)
    for v, d in report["verdicts"].items():
        print(f"VERDICT {v}: {d['verdict']}  (1) BLEU - end_to_end {ci(d['minus_end_to_end'])} "
              f"{'ok' if d['bleu_vs_end_to_end_lower_above_0'] else 'FAIL'}; (2) BLEU - mismatched words "
              f"{ci(d['minus_mismatched_words'])} {'ok' if d['bleu_vs_mismatched_words_lower_above_0'] else 'FAIL'}; "
              f"(3) spotting {rc(d['spotting_recall'] or {})} vs {rc(d['spotting_recall_mismatched'] or {})} "
              f"{'ok' if d['spotting_recall_above_mismatched'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
