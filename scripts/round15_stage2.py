"""Round 15, stage 2: the compositional generator with ASLLRP-fitted grammar and sign-core timing, against end-to-end
generation, stage 1 and the real-word oracle (docs/NEXT_PHASE_PLAN.md, "Round 15 stage 2 groundwork", 2026-10-28
21:16 UTC). Built and tested only; it runs on the 300 How2Sign val sentences only after the owner's declaration,
which fixes the verdict rule (``--rule``) and may pin the fitted model (``--expect-model-sha256``).

**Clips and pairing:** the stored stage-1 report's (``--stage1-report-key``): its manifest, its 300 How2Sign val clips
(``pilot_content_check.validation_sentences``), references, derangements and stage-1 word selection, all recomputed
and checked clip by clip (``round15_oracle.check_pairing``); the stored oracle report's references are checked too.

**The stage-2 model** (``round15_stage2_fit.fit_model``): fitted at run time on ASLLRP *train* only (the How2Sign clips
are read after it, and the fit refuses any record that is not an ASLLRP train utterance), or read from
``--model-file`` and checked against its hash. ``--expect-model-sha256`` stops a run whose model is not the declared one.

**Plan of a sentence** (``plan_clip``): the stage-2 selection and structure (``compositional_grammar.selection_rows``
then ``apply_structure``: lexicon, keep / map / insert IX / drop, reorder moves), each entry's sentence-sign target
(``target_seconds``) and its generation length (its typical isolated duration, the vocabulary's median trimmed
seconds); frames by ``plan_frames`` with the model's gap and leading share and ``--total-fit``.

**Variants** (``--variants``):
- ``core_crossfade``: each word generated alone at its typical isolated duration, cut to its sign core
  (``compositional.core_window``: the manifests' dominant-wrist activity rule, margin ``--core-margin``), resampled to
  its target, joined by interpolated gaps (``compose_cores``, ``render``);
- ``core_chained``: the prior continues the kept codes of the earlier words (stage 1's chained sampler and clock);
  the codes after each core are dropped before the next word, and the prior's own frames between two cores are the
  transition, resampled to the gap;
- ``oracle_stage2``: real isolated clips of the same words (``round15_oracle.choose_record``, the stage-1 oracle's
  draws for the same (clip, word)), cut and timed exactly like ``core_crossfade``: the timing and structure gain on
  real words, apart from word quality.
Controls of each: ``/shuffled`` (the same words, targets and seeds in ``shuffle_order(n, clip, 0)``; crossfade and
oracle reorder their pieces, chained regenerates) and ``/mismatched_words`` (another sentence's stage-2 words, the
stage-1 words derangement, fitted to this clip's reference; seeds and draws in their own block). A sentence with no
entry reads as its blank clip. Further ``--total-fit`` values render the same generated material again (main form
only, ``V@fit``).

**Comparators** (stored hypotheses, scored here with the same references and bootstrap): ``end_to_end``,
``stage1_chained``, ``stage1_crossfade`` (the stage-1 report) and ``oracle_stage1`` (the oracle report's fitted
oracle, 2.58); ``real`` and ``blank`` are translated afresh (reproduction reported).

**Scores:** corpus BLEU with the pilot's bootstrap; paired differences of every variant against end_to_end,
stage1_chained, oracle_stage1, its controls and the other variants; ``sentence_word_spotting`` (stage 1's settings)
against the mismatched control; motion statistics with wrist speed, as ratios to real.

**Verdict:** ``--rule`` (declared by the owner): comma-separated checks per variant, each ``bleu:<comparison>[>x]``
(the paired difference's 95% lower bound above x, default 0; comparison one of the form's ``minus_*`` keys) or
``spotting`` (recall's lower bound above the mismatched control's upper bound). Without a rule the verdict is
``UNDECLARED``.

Writes one JSON report (``--output-key`` or ``--output-file``, write-once).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 24.
STAGE1_REPORT = "acquisition/2026-10-17-v1/youtube-asl-v1/round15/compositional-f12-step60000.json"
ORACLE_REPORT = "acquisition/2026-10-17-v1/youtube-asl-v1/round15/oracle-composition-f12-vocab.json"
VARIANTS = ("core_crossfade", "core_chained", "oracle_stage2")
GENERATED = ("core_crossfade", "core_chained")
CONTROLS = ("", "/shuffled", "/mismatched_words")
STORED = {"end_to_end": ("stage1", "end_to_end"), "stage1_chained": ("stage1", "chained"),
          "stage1_crossfade": ("stage1", "crossfade"), "oracle_stage1": ("oracle", "oracle")}
SEED_BASE = 20_000_000              # word k of clip i: SEED_BASE + MAX_WORDS i + k (+ MISMATCHED_BLOCK)
MAX_WORDS = 1000
MISMATCHED_BLOCK = 10_000_000

CAVEATS = [
    "Sign cores: the activity rule's window (margin 0) is about 1.8x the annotated ASLLVD sign span (median 1.13 s "
    "against 0.62 s on ASLLVD train: it starts about 0.21 s before the sign and ends 0.31 s after), so a core carries "
    "part of the raise and the return; resampled to the ASLLRP sign duration it is compressed more than the sign "
    "itself (reported: timing.core_compression; motion wrist_speed against real).",
    "Targets: the sentence-sign duration model is fitted on ASLLRP's signers and domain (lab-elicited sentences); "
    "How2Sign's signers and pace may differ. --total-fit pad keeps ASLLRP's pace and pads with the first word's "
    "approach and the last word's return; scale stretches cores and gaps to the reference.",
    "Structure: fitted on ASLLRP train (1,652 utterances); its learned maps (competitive linking) and drops are "
    "ASLLRP-specific, listed in the model. Pointing signs (IX, POSS) are the vocabulary's citation forms: their "
    "direction is not controlled. Topic-first reordering is not attempted (not detectable from English).",
    "The gold the structure was fitted against is the ASLLRP gloss sequence mapped to the vocabulary; signs without a "
    "vocabulary entry (classifiers, gestures, name signs, most fingerspelling) are absent from it.",
    "Chained: as stage 1, word k starts at phase start_k / end_k of its clock; the core is cut on the segment decoded "
    "while it was generated, the sentence decoded once from the kept codes (the decoder's context differs slightly).",
    "Comparators are the stored stage-1 and oracle hypotheses (their GPU run); real and blank are retranslated here.",
]


# ---- pure logic --------------------------------------------------------------------------------------------------

def word_seeds(clip: int, count: int, *, mismatched: bool = False) -> list[int]:
    """One seed per word, unique over clips, words and the two blocks, and apart from stage 1's blocks."""
    if count > MAX_WORDS:
        raise ValueError(f"more than {MAX_WORDS} words in a sentence")
    return [SEED_BASE + MAX_WORDS * clip + k + (MISMATCHED_BLOCK if mismatched else 0) for k in range(count)]


def plan_clip(text, model, vocab) -> dict:
    """The stage-2 plan of one sentence: entries, the selection rows and structure trace, each entry's target
    (seconds, source) and generation seconds."""
    from signtranslator.generation import compositional_grammar as cg
    rows = cg.selection_rows(text, vocab, model["lexicon"])
    entries, trace = cg.apply_structure(rows, model["structure"], vocab)
    targets = [cg.target_seconds(e, model["timing"], vocab) for e in entries]
    return dict(entries=entries, rows=rows, trace=trace, targets=[t for t, _ in targets],
                target_sources=[s for _, s in targets], generate=[float(vocab[e]) for e in entries])


def frame_plan(targets, reference_seconds, model, total_fit, gap_seconds=None) -> dict:
    from signtranslator.generation.compositional_grammar import plan_frames
    t = model["timing"]
    return plan_frames(targets, t["gap"] if gap_seconds is None else gap_seconds, reference_seconds, total_fit,
                       t["lead_share"])


def plan_jobs(variants, plans, content_perm, orders) -> dict:
    """{(variant, control, clip): job (mode, words, generate seconds, targets, seeds, order) or None}. Pieces-based
    variants (core_crossfade, oracle_stage2) reorder their matched pieces for /shuffled: no job of their own."""
    jobs = {}
    for v in variants:
        for i, plan in enumerate(plans):
            words = plan["entries"]
            if not words:
                jobs[(v, "", i)] = jobs[(v, "/shuffled", i)] = None
            else:
                s = word_seeds(i, len(words))
                jobs[(v, "", i)] = dict(mode=v, words=list(words), generate=list(plan["generate"]),
                                        targets=list(plan["targets"]), seeds=s, block="matched")
                o = orders[i]
                jobs[(v, "/shuffled", i)] = None if v != "core_chained" else dict(
                    mode=v, words=[words[k] for k in o], generate=[plan["generate"][k] for k in o],
                    targets=[plan["targets"][k] for k in o], seeds=[s[k] for k in o], block="matched")
            other = plans[content_perm[i]]
            jobs[(v, "/mismatched_words", i)] = dict(
                mode=v, words=list(other["entries"]), generate=list(other["generate"]),
                targets=list(other["targets"]), seeds=word_seeds(i, len(other["entries"]), mismatched=True),
                block="mismatched_words") if other["entries"] else None
    return jobs


def form_pairs(variants, extra_fits=()) -> dict:
    pairs = {}
    for v in variants:
        pairs[v] = {"end_to_end": "end_to_end", "stage1_chained": "stage1_chained", "oracle_stage1": "oracle_stage1",
                    "stage1_crossfade": "stage1_crossfade", "shuffled": f"{v}/shuffled",
                    "mismatched_words": f"{v}/mismatched_words", **{f"variant_{w}": w for w in variants if w != v}}
        for c in CONTROLS[1:]:
            pairs[f"{v}{c}"] = {"end_to_end": "end_to_end"}
        for f in extra_fits:
            pairs[f"{v}@{f}"] = {"end_to_end": "end_to_end", "stage1_chained": "stage1_chained",
                                 "oracle_stage1": "oracle_stage1", "primary_fit": v}
    for name in ("stage1_chained", "stage1_crossfade", "oracle_stage1"):
        pairs[name] = {"end_to_end": "end_to_end"}
    return pairs


_CHECK = re.compile(r"^(?:bleu:(?P<pair>[a-z0-9_]+)(?:>(?P<x>-?[0-9.]+))?|(?P<spot>spotting))$")


def parse_rule(rule: str | None) -> list[dict]:
    """``--rule``: comma-separated ``bleu:<comparison>[>x]`` or ``spotting`` checks (module docstring)."""
    if not rule:
        return []
    checks = []
    for part in (p.strip() for p in rule.split(",") if p.strip()):
        m = _CHECK.match(part)
        if not m:
            raise ValueError(f"unknown rule check {part!r}")
        checks.append(dict(kind="spotting") if m["spot"] else
                      dict(kind="bleu", pair=m["pair"], above=float(m["x"]) if m["x"] else 0.))
    return checks


def verdict(checks, bleu_row: dict, spotting: dict, spotting_mismatched: dict) -> dict:
    """The declared rule on one variant; UNDECLARED without checks."""
    from strict_prior_review import separated
    if not checks:
        return dict(verdict="UNDECLARED", checks={})
    out = {}
    for c in checks:
        if c["kind"] == "spotting":
            a, b = (spotting or {}).get("ci95"), (spotting_mismatched or {}).get("ci95")
            out["spotting_recall_above_mismatched"] = bool(a and b and separated(a, b))
        else:
            row = bleu_row.get(f"minus_{c['pair']}")
            if row is None:
                raise ValueError(f"the rule names {c['pair']!r}, which this form is not compared with")
            out[f"bleu_vs_{c['pair']}_lower_above_{c['above']:g}"] = bool(row["ci95"][0] > c["above"])
    return dict(verdict="PROMISING" if all(out.values()) else "NOT PROMISING", checks=out)


# ---- generation --------------------------------------------------------------------------------------------------

def generate_job(sample, job: dict, *, guidance: float, temperature, rule: dict):
    """The generated material of one job (``compose_cores``) with the loaded generator, post-processed as the pilot
    (``exposure_diagnostic.make_post``)."""
    from signtranslator.generation.compositional import compose_cores
    from exposure_diagnostic import make_post
    out = compose_cores(sample.prior, sample.tokenizer, job["words"], job["generate"], job["mode"], job["seeds"],
                        guidance=guidance, temperature=temperature, post=make_post(sample.decoder_gains), rule=rule)
    out.pop("codes", None)
    return out


_WORKER = {}


def _init_worker(store_spec, checkpoint_key, mix_sha256, device, temperature, threads, settings):
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
    return generate_job(_WORKER["sample"], job, **_WORKER["settings"])


def run_jobs(sample, jobs: list, settings: dict, *, workers: int = 1, worker_args=None, progress=None,
             every: int = 50) -> list:
    """``generate_job`` of each job in order, here or in ``workers`` spawned processes (longest first); every job
    has its own seeds, so the result does not depend on the workers."""
    if workers <= 1:
        out = []
        for k, job in enumerate(jobs):
            out.append(generate_job(sample, job, **settings))
            if progress and ((k + 1) % every == 0 or k + 1 == len(jobs)):
                progress(k + 1)
        return out
    import multiprocessing
    order = sorted(range(len(jobs)), key=lambda k: -sum(jobs[k]["generate"]))
    out = [None] * len(jobs)
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker,
                                                   initargs=(*worker_args, settings)) as pool:
        for done, (k, result) in enumerate(zip(order, pool.imap(_run, [jobs[k] for k in order], chunksize=1))):
            out[k] = result
            if progress and ((done + 1) % every == 0 or done + 1 == len(jobs)):
                progress(done + 1)
    return out


def oracle_material(job, clip: int, pools, *, loaded: dict, load, rule) -> dict:
    """Real clips of a job's words (``choose_record`` in the job's block: the stage-1 oracle's draw for the same
    clip and word), cut to their cores like ``core_crossfade``."""
    from signtranslator.generation.compositional import core_window, parts_from_pieces
    from round15_oracle import choose_record
    records = [choose_record(pools, clip, w, job["block"]) for w in job["words"]]
    pieces = []
    for r in records:
        if r["sign"] not in loaded:
            loaded[r["sign"]] = load(r)
        state = loaded[r["sign"]]
        pieces.append(dict(state=state, **core_window(state, rule)))
    rows = [dict(word=w, frames=len(p["state"].timestamps), sign=r["sign"], corpus=r["corpus"], lo=p["lo"],
                 hi=p["hi"], reason=p["reason"]) for w, r, p in zip(job["words"], records, pieces)]
    return dict(mode="oracle_stage2", parts=parts_from_pieces(pieces), segments=rows, pieces=pieces)


# ---- the run -----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    from signtranslator.generation.compositional import ISOLATED_CORPORA
    from round15_stage2_fit import DEFAULTS
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", help="the prior (required unless only oracle_stage2 runs)")
    p.add_argument("--stage1-report-key", default=STAGE1_REPORT)
    p.add_argument("--oracle-report-key", default=ORACLE_REPORT)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--total-fit", default="pad", help="pad, scale or none; more than one (comma-separated) renders "
                   "the same material again for the main forms (the first is the primary)")
    p.add_argument("--core-margin", type=float, default=0., help="seconds added either side of the active window")
    p.add_argument("--core-threshold", type=float, default=.1, help="the activity threshold (the manifests' 0.1)")
    p.add_argument("--gap-seconds", type=float, help="the gap between signs (default: the model's ASLLRP median)")
    p.add_argument("--model-file", help="a fitted stage-2 model (round15_stage2_fit.py); default: fit it now")
    p.add_argument("--expect-model-sha256", help="stop unless the stage-2 model has this hash (the declared model)")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--fit-{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--rule", help="the declared verdict rule (module docstring); default: UNDECLARED")
    p.add_argument("--part-temperature", default="body=1.1,left_hand=1.1,right_hand=1.1,face=1.0")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--recognizer-run", help="the word-spotting instrument (default: the stage-1 report's)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--word-workers", type=int, default=1)
    p.add_argument("--fetch-threads", type=int, default=8)
    p.add_argument("--cache-dir", default=str(Path(tempfile.gettempdir()) / "sign-translator-cache"))
    p.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    p.add_argument("--output-key")
    p.add_argument("--output-file")
    args = p.parse_args(argv)
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    variants = [v for v in args.variants.split(",") if v]
    if not variants or any(v not in VARIANTS for v in variants) or len(set(variants)) != len(variants):
        p.error(f"--variants: distinct names from {','.join(VARIANTS)}")
    if any(v in GENERATED for v in variants) and not args.checkpoint_key:
        p.error("--checkpoint-key is required for generated variants")
    from signtranslator.generation.compositional_grammar import TOTAL_FITS
    fits = [f for f in args.total_fit.split(",") if f]
    if not fits or any(f not in TOTAL_FITS for f in fits) or len(set(fits)) != len(fits):
        p.error(f"--total-fit: distinct values from {','.join(TOTAL_FITS)}")
    checks = parse_rule(args.rule)
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest
    from signtranslator.generation.compositional import (CORE_RULE, KEPT, crossfade_frames, parts_from_pieces, render,
                                                         selection_details, training_vocabulary)
    from compositional_baseline import MOTION_KEYS, hypothesis_length_ratio, motion_rows, shuffle_order
    from evaluate_text_to_motion import at_rate, blanked, load_generator, load_state
    from pilot_content_check import MIX_SHA, validation_sentences
    from round15_compositional import paired_recall
    from round15_oracle import TEMPLATE_CORPUS, check_pairing, word_clip, word_pools
    import round15_stage2_fit as fit
    from score_back_translation import load_translator, translate
    from sentence_exposure_diagnostic import derangement, mean_statistics, ratio_to, score_forms
    import sentence_word_spotting as sws
    from strict_prior_review import separated
    from train_motion_tokens import layout_template
    from train_text_to_motion import make_store
    import validate_sign_recognizer as vsr
    clock = {"start": time.time()}
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    stored = {"stage1": json.loads(store.get(args.stage1_report_key)),
              "oracle": json.loads(store.get(args.oracle_report_key))}
    s1 = stored["stage1"]
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(s1["mix_key"]) + b"\n")
        manifest = load_manifest(path, expected_sha256=s1["mix_sha256"])
    corpora = tuple(s1["corpora"])
    vocab = training_vocabulary(manifest, corpora)

    # the stage-2 model: ASLLRP train only, before any How2Sign clip is read
    if args.model_file:
        model = json.loads(Path(args.model_file).read_text())
        fit.check_model(model)
        if model.get("mix_key") != s1["mix_key"] or model["inputs"].get(s1["mix_key"]) != s1["mix_sha256"]:
            raise SystemExit("the stage-2 model was fitted on another vocabulary")
    else:
        params = {k: getattr(args, f"fit_{k}") for k in DEFAULTS}
        model, _ = fit.fit_model(fit.agc._store(args.store), s1["mix_key"], s1["mix_sha256"], corpora, params)
    if args.expect_model_sha256 and model["sha256"] != args.expect_model_sha256:
        raise SystemExit(f"the stage-2 model is {model['sha256']}, not the declared {args.expect_model_sha256}")
    clock["model"] = time.time()

    # the stage-1 clips, checked against the stored reports
    val = validation_sentences(manifest, s1["clips"])
    texts = [r["text"] for r in val]
    n = len(texts)
    details = [selection_details(t, vocab) for t in texts]
    selected1 = [[r["entry"] for r in rows if r["kind"] in KEPT] for rows in details]
    reference_perm = derangement(n, s1["seeds"]["reference_derangement"])
    content_perm = derangement(n, s1["seeds"]["words_derangement"])
    check_pairing(s1, val, texts, selected1, vocab, reference_perm, content_perm,
                  [shuffle_order(len(w), i, 0) for i, w in enumerate(selected1)],
                  crossfade_frames(float(s1["crossfade"]["seconds"])))
    if stored["oracle"].get("references") != texts:
        raise SystemExit("the oracle report's references are not the stage-1 clips")
    refs = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), FPS) for r in val]
    seconds = [float(s.timestamps[-1] - s.timestamps[0]) + 1 / FPS for s in refs]
    blanks = blanked(refs)
    plans = [plan_clip(t, model, vocab) for t in texts]
    orders = [shuffle_order(len(pl["entries"]), i, 0) for i, pl in enumerate(plans)]
    if max(len(pl["entries"]) for pl in plans) > MAX_WORDS:
        raise SystemExit(f"a sentence plans more than {MAX_WORDS} signs")
    jobs = plan_jobs(variants, plans, content_perm, orders)
    rule = dict(CORE_RULE, margin=args.core_margin, threshold=args.core_threshold)
    gen_keys = [k for k, j in jobs.items() if j is not None and k[0] in GENERATED]
    print(f"{n} clips; stage-2 model {model['sha256'][:12]}; {sum(len(pl['entries']) for pl in plans)} signs planned "
          f"(stage 1: {sum(len(w) for w in selected1)}); {len(gen_keys)} generated compositions, "
          f"{sum(sum(jobs[k]['generate']) for k in gen_keys):.0f} s to generate", flush=True)

    temperature = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature.split(","))}
    material = {}
    if gen_keys:
        _, _, sample = load_generator(store, args.checkpoint_key, s1["mix_sha256"], args.device,
                                      temperature=temperature)
        settings = dict(guidance=args.guidance, temperature=temperature, rule=rule)
        worker = (args.store, args.checkpoint_key, s1["mix_sha256"], args.device, temperature,
                  max(1, torch.get_num_threads() // max(args.word_workers, 1)))
        results = run_jobs(sample, [jobs[k] for k in gen_keys], settings, workers=args.word_workers,
                           worker_args=worker, progress=lambda k: print(
                               f"composition {k}/{len(gen_keys)} ({time.time() - clock['start']:.0f} s)", flush=True))
        material.update(zip(gen_keys, results))
        del sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    clock["generate"] = time.time()
    if "oracle_stage2" in variants:
        pools = word_pools(manifest, corpora)
        cache = ContentCache(Path(args.cache_dir), 8 * 2 ** 30)
        template = layout_template(manifest, TEMPLATE_CORPUS, cache, store.get)
        loaded = {}
        load = lambda r: word_clip(r, cache=cache, fetch=store.get, template=template)    # noqa: E731
        for k, j in jobs.items():
            if j is not None and k[0] == "oracle_stage2" and k[1] != "/shuffled":
                material[k] = oracle_material(j, k[2], pools, loaded=loaded, load=load, rule=rule)
    clock["oracle"] = time.time()

    # render every form
    forms = {"real": refs, "blank": blanks}
    per_clip = [dict(sign=r["sign"], text=t, stage1_words=w1, entries=pl["entries"], shuffled_order=o,
                     mismatched_from=c, targets=[round(x, 4) for x in pl["targets"]],
                     target_sources=pl["target_sources"], trace=pl["trace"],
                     lexicon=[[x["text"], x["entry"], x["tier"]] for x in pl["rows"] if x["kind"] == "lexicon"],
                     frames_reference=len(ref.timestamps))
                for r, t, w1, pl, o, c, ref in zip(val, texts, selected1, plans, orders, content_perm, refs)]
    timing_rows = collections.defaultdict(list)
    for v in variants:
        for c in CONTROLS:
            states = []
            for i in range(n):
                key, job = (v, c, i), jobs[(v, c, i)]
                if c == "/shuffled" and v != "core_chained":
                    job = jobs[(v, "", i)]
                if job is None:
                    states.append(blanks[i])
                    continue
                if c == "/shuffled" and v != "core_chained":
                    o = orders[i]
                    parts = parts_from_pieces(material[(v, "", i)]["pieces"], o)
                    targets = [job["targets"][k] for k in o]
                else:
                    parts, targets = material[key]["parts"], job["targets"]
                fp = frame_plan(targets, seconds[i], model, fits[0], args.gap_seconds)
                states.append(render(parts, fp, sample_id=f"round15-stage2-{v}{c}:{i}"))
                if c == "":
                    segs = material[key]["segments"]
                    per_clip[i][v] = dict(frames=fp["frames"], plan=fp, segments=[
                        {k2: v2 for k2, v2 in s.items() if k2 != "seed"} for s in segs])
                    for s, f in zip(segs, fp["cores"]):
                        timing_rows[v].append(dict(core=s["hi"] - s["lo"] + 1, planned=f, reason=s["reason"],
                                                   fit=fp["fit"]))
            forms[f"{v}{c}"] = states
        for f in fits[1:]:
            forms[f"{v}@{f}"] = [blanks[i] if jobs[(v, "", i)] is None else
                                 render(material[(v, "", i)]["parts"],
                                        frame_plan(jobs[(v, "", i)]["targets"], seconds[i], model, f, args.gap_seconds),
                                        sample_id=f"round15-stage2-{v}@{f}:{i}") for i in range(n)]
    clock["render"] = time.time()

    # back-translation
    kp = json.loads(store.get(f"training-runs/{s1['translator_run']}/best.json"))["checkpoint_key"]
    if kp != s1["translator_key"]:
        raise SystemExit(f"the translator run now points at {kp}, not the stored report's {s1['translator_key']}")
    translator, kind = load_translator(store, kp, MIX_SHA)
    translator = translator.to(torch.device(args.device))
    translator_fps = float(getattr(getattr(translator, "config", None), "fps", FPS))
    hyps = {}
    for name, states in forms.items():
        hyps[name] = list(translate(translator, kind, states))
        print(f"translated {name} ({time.time() - clock['start']:.0f} s)", flush=True)
    for name, (report, source) in STORED.items():
        hyps[name] = list(stored[report]["forms"][source]["hypotheses"])
        if len(hyps[name]) != n:
            raise SystemExit(f"the stored {source} hypotheses are not {n} clips")
    reproduction = {name: dict(identical=int(sum(a == b for a, b in zip(hyps[name], s1["forms"][name]["hypotheses"]))),
                               clips=n) for name in ("real", "blank")}
    clock["translation"] = time.time()
    pairs = form_pairs(variants, fits[1:])
    scores = score_forms(hyps, texts, reference_perm, boot=args.bootstrap, pairs=pairs)
    for name, row in scores.items():
        row["hypothesis_over_reference_length"] = hypothesis_length_ratio(hyps[name], texts)
    for name, (report, source) in STORED.items():
        reproduction[name] = dict(bleu=scores[name]["bleu"], stored_bleu=stored[report]["forms"][source]["bleu"])
    clock["scoring"] = time.time()

    # word spotting (sentence_word_spotting's functions, unchanged; stage 1's settings)
    spot = s1["spotting"]
    recognizer_run = args.recognizer_run or s1["recognizer_run"]
    recogniser, labels, best = vsr.load_model(store, recognizer_run, args.device)
    index = {g: i for i, g in enumerate(labels)}
    spot_words = [sws.sentence_labels(t, set(labels)) for t in texts]
    spot_mismatched = spot_words[1:] + spot_words[:1]
    spotting, spot_rows = {}, {}
    for name in variants:
        preds = [sws.window_predictions(recogniser, at_rate(s, FPS), args.device, windows=tuple(spot["windows"]),
                                        hop=spot["hop"], k=spot["top_k"]) for s in forms[name]]
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
    for name, source in (("end_to_end", "end_to_end"), ("stage1_chained", "chained"),
                         ("stage1_crossfade", "crossfade"), ("real", "real")):
        for suffix in ("", "_mismatched"):
            spotting[f"stored_{name}{suffix}"] = spot["scores"].get(f"{source}{suffix}")
    for suffix in ("", "_mismatched"):
        spotting[f"stored_oracle_stage1{suffix}"] = stored["oracle"]["spotting"]["scores"].get(f"oracle{suffix}")
    clock["spotting"] = time.time()

    motion = {name: mean_statistics(motion_rows(states, refs, translator_fps)) for name, states in forms.items()
              if name != "blank"}
    motion_ratio = {name: ratio_to(m, motion["real"]) for name, m in motion.items()}
    motion_ratio["stored_end_to_end"] = s1["motion_ratio_to_real"]["end_to_end"]
    motion_ratio["stored_stage1_chained"] = s1["motion_ratio_to_real"]["chained"]
    motion_ratio["stored_oracle_stage1"] = stored["oracle"]["motion_ratio_to_real"].get("oracle")
    verdicts = {v: verdict(checks, scores[v], spotting[v]["recall"], spotting[f"{v}_mismatched"]["recall"])
                for v in variants}

    timing = {}
    for v, rows in timing_rows.items():
        comp = [r["core"] / r["planned"] for r in rows]
        timing[v] = dict(core_frames_median=float(np.median([r["core"] for r in rows])),
                         planned_core_frames_median=float(np.median([r["planned"] for r in rows])),
                         core_compression_median=round(float(np.median(comp)), 4),
                         core_compression_p10_p90=[round(float(np.percentile(comp, q)), 4) for q in (10, 90)],
                         core_reasons=dict(collections.Counter(r["reason"] for r in rows)),
                         sentence_fits=dict(collections.Counter(per_clip[i][v]["plan"]["fit"] for i in range(n)
                                                                if v in per_clip[i])),
                         frames_over_reference_median=round(float(np.median(
                             [per_clip[i][v]["frames"] / per_clip[i]["frames_reference"] for i in range(n)
                              if v in per_clip[i]])), 4))
    ops = collections.Counter(t["op"] for pl in plans for t in pl["trace"])
    sources = collections.Counter(s for pl in plans for s in pl["target_sources"])
    selection = dict(signs_per_sentence_mean=round(float(np.mean([len(pl["entries"]) for pl in plans])), 3),
                     stage1_words_per_sentence_mean=round(float(np.mean([len(w) for w in selected1])), 3),
                     empty_sentences=int(sum(not pl["entries"] for pl in plans)), operations=dict(ops),
                     moved=int(sum(bool(t.get("moved")) for pl in plans for t in pl["trace"])),
                     lexicon_spans=int(sum(len(r["lexicon"]) for r in per_clip)), target_sources=dict(sources),
                     target_seconds_median=round(float(np.median([x for pl in plans for x in pl["targets"]])), 4)
                     if any(pl["targets"] for pl in plans) else None,
                     most_common=collections.Counter(e for pl in plans for e in pl["entries"]).most_common(30))
    report = dict(
        round=15, stage=2, checkpoint_key=args.checkpoint_key, stage1_report_key=args.stage1_report_key,
        oracle_report_key=args.oracle_report_key, mix_key=s1["mix_key"], mix_sha256=s1["mix_sha256"],
        translator_run=s1["translator_run"], translator_key=kp, recognizer_run=recognizer_run,
        recognizer_checkpoint=best.get("checkpoint_key") if isinstance(best, dict) else None, clips=n,
        variants=variants, total_fit=fits, core_rule=rule, gap_seconds=args.gap_seconds or model["timing"]["gap"],
        guidance=args.guidance, part_temperature=temperature, rule=args.rule, rule_checks=checks,
        model_sha256=model["sha256"], model_parameters=model["parameters"],
        model_timing={k: v for k, v in model["timing"].items() if k not in ("sentence_seconds", "sentence_signs",
                                                                              "annotated")},
        model_structure=dict(moves=model["structure"]["moves"], pointing=model["structure"]["pointing"]),
        seeds=dict(word=f"{SEED_BASE} + {MAX_WORDS} clip + word index", mismatched_words=f"+ {MISMATCHED_BLOCK}",
                   shuffle="compositional_baseline.shuffle_order(n, clip, 0)",
                   oracle="round15_oracle.choose_record (blocks matched / mismatched_words)"),
        bootstrap=args.bootstrap, beam=4, references=texts, reference_derangement=reference_perm,
        words_derangement=content_perm, reproduction=reproduction, selection=selection, timing=timing, forms=scores,
        spotting=dict(windows=spot["windows"], hop=spot["hop"], top_k=spot["top_k"], labels=len(labels),
                      scores=spotting),
        motion=motion, motion_ratio_to_real=motion_ratio, verdicts=verdicts,
        seconds={k: round(v - clock["start"], 1) for k, v in clock.items() if k != "start"}, caveats=CAVEATS,
        per_clip=per_clip)
    for i, row in enumerate(per_clip):
        row["spotting"] = {name: spot_rows[name][i]["found"] for name in spot_rows}
    summary(report, MOTION_KEYS)
    payload = json.dumps(report, indent=1, default=float).encode()
    if args.output_file:
        with open(args.output_file, "xb") as stream:
            stream.write(payload)
    if args.output_key:
        if store.exists(args.output_key):
            raise SystemExit(f"{args.output_key} appeared during the run: refusing to overwrite")
        store.put(args.output_key, payload)
    return 0


def summary(report: dict, motion_keys) -> None:
    def ci(d):
        return f"{d['difference']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"

    def rc(d):
        return f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]" if d and d.get("ci95") else "n/a"
    print(f"BLEU on {report['clips']} clips (form, BLEU [95% CI], paired comparisons, hyp/ref length):", flush=True)
    for name, row in report["forms"].items():
        extra = "".join(f"  -{k[6:]} {ci(v)}" for k, v in row.items() if k.startswith("minus_") and k != "minus_blank")
        print(f"  {name:34s} {row['bleu']:6.2f} [{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]{extra}"
              f"  len {row['hypothesis_over_reference_length']}", flush=True)
    print("reproduction: " + json.dumps(report["reproduction"]), flush=True)
    s = report["spotting"]["scores"]
    print("word spotting recall (form, recall, mismatched, above):", flush=True)
    for name in [k for k in s if not k.endswith("_mismatched") and s[k] and not k.startswith("stored_")]:
        print(f"  {name:34s} {rc(s[name]['recall'])}  mism {rc(s[name + '_mismatched']['recall'])}  "
              f"above {s[name].get('above_mismatched')}", flush=True)
    print("motion (" + ", ".join(motion_keys) + "):", flush=True)
    for name, m in report["motion"].items():
        print(f"  {name:34s} " + "  ".join(f"{m.get(k):7.3f}" if m.get(k) is not None else "    nan"
                                           for k in motion_keys), flush=True)
    print("wrist speed / real: " + json.dumps({k: (v or {}).get("wrist_speed")
                                              for k, v in report["motion_ratio_to_real"].items()}), flush=True)
    print(json.dumps(dict(selection={k: v for k, v in report["selection"].items() if k != "most_common"},
                          timing=report["timing"], seconds=report["seconds"])), flush=True)
    for v, d in report["verdicts"].items():
        print(f"VERDICT {v}: {d['verdict']} " + json.dumps(d["checks"]), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
