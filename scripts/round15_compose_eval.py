"""Round 15: a larger paired evaluation of a composition on the pilot's How2Sign val clips, sharded (docs/NEXT_PHASE_PLAN.md,
amendment 4 result, 2026-10-29 11:52 UTC). Built and tested only; it runs on How2Sign once declared.

Two subcommands. Both are write-once (``--output-key`` or ``--output-file``).

**run** (one shard, GPU for generated variants): the clips are ``pilot_content_check.validation_sentences(eval mix,
--clips)`` (the pilot's own sample; ``--clips 1000`` is the pilot BLEU's), with no stored stage-1 report:
- the stage-1 selection is computed with stage 1's own code path (``compositional.selection_details`` on the training
  vocabulary of ``--corpora`` of the evaluation mix, the entries of kind exact or fallback). With
  ``--stage1-report-key`` every clip also in that report (by sign id) must have the report's words; the counts are
  reported. The first 300 of the 1000 are not the stage-1 report's 300 (``random.sample`` draws k = 300 and k = 1000
  by different methods from the 3,096 candidates); 289 of the 300 are among the 1000;
- the stage-2 model (``round15_stage2_fit.fit_model`` on ASLLRP train, or ``--model-file``), its hash pinned by
  ``--expect-model-sha256``; plans, jobs, generation, rendering, real-clip material and timing modes are
  ``round15_stage2``'s own (``plan_words`` / ``plan_clip``, ``plan_jobs``, ``run_jobs``, ``oracle_material``,
  ``render_form``), with ``--selection``, ``--timing``, ``--core-margin`` and ``--generator-mix-key`` as there;
- derangements are over all N clips (``--seed``: references; ``--seed`` + 1: the mismatched words), shuffle orders
  ``shuffle_order(n, clip, 0)``, word seeds ``round15_stage2.word_seeds`` by the global clip index: a clip's forms do not
  depend on the sharding;
- **position, not identity:** word seeds, the oracle's real-clip draws (``choose_record`` by clip index), shuffle
  orders and mismatched-words partners all follow a clip's *position* in the N-sample. A clip shared with the
  300-clip runs sits at another position among the 1,000, so a ``--clips 1000`` run does not reproduce amendment 4 (or
  any 300-clip report) clip for clip; a ``--clips 300`` run does (``validation_sentences(mix, 300)`` is the same
  sample). The report states this (``indexing``);
- the vocabulary's corpora default to ``compositional.isolated_corpora(eval mix)``: round 15's four for
  round10-mixed-v1, so the default reproduces the declared stage-2 model 42389ca4;
- **shards**: clip i belongs to shard ``(i // --block) % --shards``; translation runs per block of ``--block`` (32,
  the translator's batch) clips, so a sharded and an unsharded run translate identical batches. Each shard stores, per
  clip, every form's hypothesis, the spotting rows (own words and the next sentence's words) and the motion rows.

**--rerank N** (default 1: unchanged): each core_crossfade word is the best of N samples by the prior's own
teacher-forced PMI, log p(codes | word) - log p(codes | empty text) per code (``signtranslator.generation.rerank``;
sample 0 is the word's existing seed, sample k seed + k x 10^8, so N = 1 reproduces the plain run exactly). Only the
prior scores; no recogniser, back-translator or other evaluator enters the choice. Each shard row records, per word,
the N seeds, scores and the chosen index (``rerank``, for the matched and mismatched-words compositions; /shuffled
reorders the matched words). Not for core_chained or the stage1 timing.

**oracle_stage2** (``--variants oracle_stage2``, no checkpoint, CPU): the real-word ceiling on the same clips.

**Composition knobs** (``signtranslator.generation.composition_space``; each independently switchable, all off by
default, and then nothing in the run or its report changes):
- ``--space-normalise sentence|how2sign``: one signing space per sentence (torso and dominant-hand centroid moved to
  the sentence's mean or How2Sign train's, hand excursion scaled to How2Sign train's), with the statistics of
  ``scripts/round15_space_stats.py`` pinned by ``--space-stats-key`` / ``--space-stats-sha256``;
- ``--bridge minimum-jerk``: quintic bridges matching the cores' boundary values and velocities;
- ``--tighten-cores`` (sign-fitted): median-position cores (``entry`` / ``all``) cut again by the activity rule inside
  their window, never extended; the shard and merged reports count them (``core_counts``).
Space normalisation needs word pieces (core_crossfade, oracle_stage2); none of the knobs applies to the stage1 timing.

**--refiner-checkpoint-key** (the sentence refiner, ``signtranslator.generation.refiner``; ``--refiner-sha256`` pins
it): after composition, every rendered form of every variant (the main form, /shuffled, /mismatched_words and the
new control /rest_only) is tokenized with the refiner's own tokenizer, refined (``refine_codes``: the input's length,
``--refiner-guidance`` against the empty source, ``--refiner-temperature`` (default: ``--part-temperature``; one
number for every part, or the same part list), one batch per global block with seed
``REFINE_SEED`` + block, so every form of a clip shares its random numbers and shards equal the unsharded run) and
decoded as a generation (the tokenizer's accepted gains, the generator's post-processing): ``<form>@refined``.
``<variant>@recon`` (the main form tokenized and decoded, not refined) separates the tokenizer's round trip from the
refinement. **/rest_only** (only with a refiner): the same plan and frames with every word clip replaced by its own
first frame (its rest before the sign) held for the clip's length (``rest_material``; core_chained: the chain's first
frame). With ``--space-normalise`` the held clips are normalised like the composition's (a static pose has no
excursion, so its scale stays 1); with ``--tighten-cores`` the cores are cut from the original clips, then held.

**What each refined control tests.** ``/mismatched_words@refined`` (another sentence's words, refined) is the
*content* control: a refinement that reads as well from the wrong words did not carry this sentence's content.
``/rest_only@refined`` only detects *input-ignoring* (collapse into the prior: a refiner that signs from its own prior
reads as well from rest poses); passing it says nothing about content. ``@recon`` (the input tokenized and decoded) is
the comparator of the refinement itself: refined minus recon is the refiner's effect, refined minus unrefined adds the
tokenizer's round trip (reported, not the rule). Spotting is also scored on ``<variant>@refined``. The merge's
``--refined-rule`` is the declared rule of the refined forms; the design's rule is ``REFINED_RULE``
(``bleu:recon,bleu:mismatched_words,bleu:rest_only,spotting``); comparisons available: ``recon``, ``unrefined``,
``mismatched_words``, ``rest_only``, ``shuffled`` (all refined), ``end_to_end``. It refuses spotting read with any
recogniser but ``INSTRUMENT`` (sign-recognizer-v3d).

**Memorisation (report-only, with a refiner):** each shard reads the refined codes, the refiner's input codes and the
real clips' codes against the refiner's training targets (``memorisation.CodeIndex``, the index its trainer wrote);
the merge reads the hypotheses of real, unrefined, @recon, @refined, /rest_only@refined and /mismatched_words@refined
against How2Sign train texts (best sentence BLEU, exact matches; ``memorisation.TextIndex``). ``real`` is the chance
level of both.

**merge** (CPU): the shards' partial reports (all of 0..K-1, the same settings and clips, disjoint and complete) are
combined in clip order and scored as ``round15_stage2`` scores: corpus BLEU with the pilot's bootstrap, paired
differences (``score_forms``) of each variant against its controls and the other variants, against ``end_to_end``
(``--end-to-end-report``: a pilot report's ``meaning.generated`` hypotheses; its references must be these clips' texts in
order) and against ``--paired-with`` (another merged report on the same clips: each variant against the same variant
there, ``paired_<variant>``); the declared rule (``--rule``); spotting recall against its mismatched control (and the
paired difference); motion statistics with wrist speed as ratios to real.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 24.
EVAL_MIX = "acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
TRANSLATOR_RUN = "back-translator-kp2d-uwb-ft-v1"
SPOT = dict(windows=(.6, .9, 1.2), hop=.1, top_k=5)
CONTROLS = ("", "/shuffled", "/mismatched_words")
INDEXING = ("word seeds, oracle real-clip draws, shuffle orders and mismatched-words partners follow each clip's position "
            "in validation_sentences(mix, clips); a clip shared with a run of another --clips sits at another position, "
            "so runs of different N are not paired clip for clip (a --clips 300 run reproduces the 300-clip reports)")
SETTINGS = ("clips", "shards", "block", "seed", "mix_key", "mix_sha256", "corpora", "variants", "selection", "timing",
            "rerank", "core_margin", "total_fit", "guidance", "part_temperature", "checkpoint_key", "generator_mix",
            "model_sha256", "translator_key", "recognizer_run", "recognizer_checkpoint", "space_normalise", "bridge",
            "tighten_cores", "space_stats", "refiner")
REST_ONLY = "/rest_only"
REFINED, RECON = "@refined", "@recon"
INSTRUMENT = "sign-recognizer-v3d"   # the spotting instrument of the refined rule (recogniser v3 amendment 2)
REFINED_RULE = "bleu:recon,bleu:mismatched_words,bleu:rest_only,spotting"     # the design's rule (review of a1bdda3)
REFINE_SEED = 30_000_000             # refinement of global block b: REFINE_SEED + b, the same for every form


# ---- pure logic --------------------------------------------------------------------------------------------------

def shard_of(clip: int, shards: int, block: int = 32) -> int:
    """The shard a clip belongs to: blocks of ``block`` clips dealt round-robin."""
    return (clip // block) % shards


def shard_clips(n: int, shard: int, shards: int, block: int = 32) -> list[int]:
    if not 0 <= shard < shards:
        raise ValueError(f"shard {shard} is not in 0..{shards - 1}")
    return [i for i in range(n) if shard_of(i, shards, block) == shard]


def blocks(clips, block: int = 32) -> list[list[int]]:
    """The shard's clips grouped by their global block (so every translation batch is the unsharded run's)."""
    out: dict[int, list[int]] = {}
    for i in clips:
        out.setdefault(i // block, []).append(i)
    return [out[k] for k in sorted(out)]


def stage1_selection(texts, vocab) -> tuple[list, list]:
    """(details, words) of round 15 stage 1's selection, its own code path (``round15_compositional.main``)."""
    from signtranslator.generation.compositional import KEPT, selection_details
    details = [selection_details(t, vocab) for t in texts]
    return details, [[r["entry"] for r in rows if r["kind"] in KEPT] for rows in details]


def check_stage1_report(val, words, stored) -> dict:
    """Every clip also in a stored stage-1 report (by sign id) must have its words; counts."""
    by_sign = {row["sign"]: row["words"] for row in stored.get("per_clip", [])}
    shared = [(r["sign"], w) for r, w in zip(val, words) if r["sign"] in by_sign]
    differ = [s for s, w in shared if by_sign[s] != w]
    if differ:
        raise SystemExit(f"the stage-1 selection differs from the stored report on {len(differ)} clips: {differ[:5]}")
    return dict(stored_clips=len(by_sign), shared_clips=len(shared), identical=len(shared))


def merge_partials(partials: list[dict]) -> dict:
    """The shards' per-clip rows in clip order, after checking they are one run: the same settings, every shard
    0..K-1 once, clips disjoint and complete."""
    if not partials:
        raise SystemExit("no shards to merge")
    first = partials[0]
    for p in partials[1:]:
        diff = [k for k in SETTINGS if p["settings"].get(k) != first["settings"].get(k)]
        diff += [k for k in ("texts", "signs") if p.get(k) != first.get(k)]
        if diff:
            raise SystemExit(f"shard {p['shard']} differs from shard {first['shard']} in {diff}")
    K, n = first["settings"]["shards"], first["settings"]["clips"]
    if sorted(p["shard"] for p in partials) != list(range(K)):
        raise SystemExit(f"shards {sorted(p['shard'] for p in partials)} are not 0..{K - 1}")
    rows = {}
    for p in partials:
        for row in p["clips"]:
            if row["clip"] in rows:
                raise SystemExit(f"clip {row['clip']} is in two shards")
            rows[row["clip"]] = row
    if sorted(rows) != list(range(n)):
        raise SystemExit(f"the shards cover {len(rows)} of {n} clips")
    return dict(settings=first["settings"], rows=[rows[i] for i in range(n)], texts=first["texts"],
                signs=first["signs"])


def merge_core_counts(partials) -> dict:
    """{"core_counts": summed over the shards} when the shards recorded them (a knob was on), else {}."""
    import collections
    parts = [p["core_counts"] for p in partials if p.get("core_counts")]
    if not parts:
        return {}
    out = {}
    for v in parts[0]:
        reasons = collections.Counter()
        for p in parts:
            reasons.update(p[v]["reasons"])
        out[v] = dict(reasons=dict(reasons), **{k: int(sum(p[v][k] for p in parts))
                                                for k in ("tightened_cores", "tightened_frames", "cores")})
    return {"core_counts": out}


def form_pairs(variants, paired=(), refined: bool = False) -> dict:
    pairs = {}
    for v in variants:
        pairs[v] = {"end_to_end": "end_to_end", "shuffled": f"{v}/shuffled", "mismatched_words": f"{v}/mismatched_words",
                    **{f"variant_{w}": w for w in variants if w != v}}
        if v in paired:
            pairs[v][f"paired_{v}"] = f"paired:{v}"
        for c in CONTROLS[1:] + ((REST_ONLY,) if refined else ()):
            pairs[f"{v}{c}"] = {"end_to_end": "end_to_end"}
        if refined:
            pairs[f"{v}{REFINED}"] = {"end_to_end": "end_to_end", "unrefined": v, "recon": f"{v}{RECON}",
                                      "shuffled": f"{v}/shuffled{REFINED}",
                                      "mismatched_words": f"{v}/mismatched_words{REFINED}",
                                      "rest_only": f"{v}{REST_ONLY}{REFINED}",
                                      **{f"variant_{w}": f"{w}{REFINED}" for w in variants if w != v}}
            pairs[f"{v}{RECON}"] = {"end_to_end": "end_to_end", "unrefined": v}
            for c in CONTROLS[1:] + (REST_ONLY,):
                pairs[f"{v}{c}{REFINED}"] = {"end_to_end": "end_to_end", "unrefined": f"{v}{c}"}
    return pairs


def held(state, frames: int | None = None):
    """``state``'s first frame held for ``frames`` frames (default: its own length) on a uniform clock."""
    from signtranslator.generation.compositional import take_frames
    return take_frames(state, np.zeros(len(state.timestamps) if frames is None else frames, dtype=int))


def rest_material(material: dict) -> dict:
    """The rest-only counterpart of a composition's material: every word clip replaced by its own first frame (the
    rest before the sign) held for the clip's length, so ``round15_stage2.render_form`` lays the same plan, frames and
    joins over rest poses. Pieces (core_crossfade, oracle_stage2: cores re-cut from the held clips), clips and the
    chained state (the stage1 timing), and core_chained's parts (every core and the lead and trail: the chain's first
    frame; transitions bridged)."""
    from signtranslator.generation.compositional import parts_from_pieces
    out = dict(material)
    if material.get("pieces") is not None:
        out["pieces"] = [dict(p, state=held(p["state"])) for p in material["pieces"]]
        if material.get("parts") is not None:
            out["parts"] = parts_from_pieces(out["pieces"])
    elif material.get("parts") is not None:
        parts = material["parts"]
        rest = held(parts["lead"][0], 1)
        out["parts"] = dict(cores=[held(rest, len(c.timestamps)) for c in parts["cores"]],
                            transitions=[None] * (len(parts["cores"]) - 1), lead=(rest, 0), trail=(rest, 0))
    if material.get("clips") is not None:
        out["clips"] = [held(c) for c in material["clips"]]
    if material.get("state") is not None:
        out["state"] = held(material["state"])
    return out


# ---- run -----------------------------------------------------------------------------------------------------------

def run(args) -> dict:
    import collections
    import torch
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest
    from signtranslator.generation.compositional import (CORE_RULE, crossfade_frames, isolated_corpora,
                                                         training_vocabulary)
    from compositional_baseline import motion_rows, shuffle_order
    from evaluate_text_to_motion import at_rate, blanked, load_generator, load_state
    from pilot_content_check import MIX_SHA, validation_sentences
    from round15_oracle import TEMPLATE_CORPUS, word_clip, word_pools
    import round15_stage2 as r2
    import round15_stage2_fit as fit
    from score_back_translation import load_translator, translate
    from sentence_exposure_diagnostic import derangement
    import sentence_word_spotting as sws
    from train_motion_tokens import layout_template
    from train_text_to_motion import make_store
    import validate_sign_recognizer as vsr
    clock = {"start": time.time()}
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    corpora = (tuple(c for c in args.corpora.split(",") if c) if args.corpora
               else isolated_corpora(manifest))               # round 15's four for the round10 mix
    vocab = training_vocabulary(manifest, corpora)
    generator_mix = dict(key=args.mix_key, sha256=args.mix_sha256, check=None)
    if args.generator_mix_key:
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "g.json"; path.write_bytes(store.get(args.generator_mix_key) + b"\n")
            gm = load_manifest(path, expected_sha256=args.generator_mix_sha256)
        generator_mix = dict(key=args.generator_mix_key, sha256=args.generator_mix_sha256,
                             check=r2.check_generator_mix(manifest, gm))
        del gm
    # the stage-2 model: ASLLRP train only, before any How2Sign clip is read
    if args.model_file:
        model = json.loads(Path(args.model_file).read_text())
        fit.check_model(model)
        if model.get("mix_key") != args.mix_key or model["inputs"].get(args.mix_key) != args.mix_sha256 \
                or list(model.get("corpora") or ()) != list(corpora):
            raise SystemExit("the stage-2 model was fitted on another vocabulary")
    else:
        model, _ = fit.fit_model(fit.agc._store(args.store), args.mix_key, args.mix_sha256, corpora)
    if args.expect_model_sha256 and model["sha256"] != args.expect_model_sha256:
        raise SystemExit(f"the stage-2 model is {model['sha256']}, not the declared {args.expect_model_sha256}")
    if args.timing == "sign-fitted" and not model["timing"].get("default_seconds"):
        raise SystemExit("the stage-2 model has no default_seconds: the sign-fitted gap ratio is undefined")
    space = None
    if args.space_normalise != "off":           # How2Sign train statistics, pinned by their hash
        import hashlib
        payload = store.get(args.space_stats_key)
        if hashlib.sha256(payload).hexdigest() != args.space_stats_sha256:
            raise SystemExit(f"{args.space_stats_key} does not match --space-stats-sha256")
        space = dict(stats=json.loads(payload)["statistics"], target=args.space_normalise)

    val = validation_sentences(manifest, args.clips)
    n = len(val)
    if n < 2:
        raise SystemExit("need at least two clips (the controls are derangements)")
    texts = [r["text"] for r in val]
    details, words1 = stage1_selection(texts, vocab)
    stage1_check = (check_stage1_report(val, words1, json.loads(store.get(args.stage1_report_key)))
                    if args.stage1_report_key else None)
    plans = ([r2.plan_words(w, model, vocab, rows=d) for w, d in zip(words1, details)] if args.selection == "stage1"
             else [r2.plan_clip(t, model, vocab) for t in texts])
    reference_perm, content_perm = derangement(n, args.seed), derangement(n, args.seed + 1)
    orders = [shuffle_order(len(pl["entries"]), i, 0) for i, pl in enumerate(plans)]
    mine = shard_clips(n, args.shard, args.shards, args.block)
    refs = {i: at_rate(load_state(store, val[i]["continuous"]["state_key"], val[i]["continuous"]["state_sha256"]), FPS)
            for i in mine}
    seconds = [float(refs[i].timestamps[-1] - refs[i].timestamps[0]) + 1 / FPS if i in refs else 1. for i in range(n)]
    variants = [v for v in args.variants.split(",") if v]
    overlap = crossfade_frames(.15)                                  # stage 1's 4 frames
    jobs = r2.plan_jobs(variants, plans, content_perm, orders, seconds=seconds, vocab=vocab, timing=args.timing,
                        overlap=overlap)
    jobs = {k: j for k, j in jobs.items() if k[2] in refs}
    margin = args.core_margin if args.core_margin is not None else (.1 if args.timing == "sign-fitted" else 0.)
    rule = dict(CORE_RULE, margin=margin)
    fractions = fit.sign_fractions(manifest, vocab) if args.timing == "sign-fitted" else None
    mode = dict(timing=args.timing, fractions=fractions, margin=margin, tighten=args.tighten_cores)
    knobs_on = args.space_normalise != "off" or args.bridge != "linear" or args.tighten_cores
    temperature = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature.split(","))}
    gen_keys = [k for k, j in jobs.items() if j is not None and k[0] in r2.GENERATED]
    print(f"shard {args.shard}/{args.shards}: {len(mine)} of {n} clips; model {model['sha256'][:12]}; "
          f"{len(gen_keys)} generated compositions", flush=True)
    material = {}
    if gen_keys:
        _, _, sample = load_generator(store, args.checkpoint_key, generator_mix["sha256"], args.device,
                                      temperature=temperature)
        settings = dict(guidance=args.guidance, temperature=temperature, rule=rule, rerank=args.rerank, **mode)
        worker = (args.store, args.checkpoint_key, generator_mix["sha256"], args.device, temperature,
                  max(1, torch.get_num_threads() // max(args.word_workers, 1)))
        results = r2.run_jobs(sample, [jobs[k] for k in gen_keys], settings, workers=args.word_workers,
                              worker_args=worker, progress=lambda k: print(
                                  f"composition {k}/{len(gen_keys)} ({time.time() - clock['start']:.0f} s)", flush=True))
        material.update(zip(gen_keys, results))
        del sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if "oracle_stage2" in variants:
        pools = word_pools(manifest, corpora)
        cache = ContentCache(Path(args.cache_dir), 8 * 2 ** 30)
        template = layout_template(manifest, TEMPLATE_CORPUS, cache, store.get)
        loaded = {}
        load = lambda r: word_clip(r, cache=cache, fetch=store.get, template=template)    # noqa: E731
        for k, j in jobs.items():
            if j is not None and k[0] == "oracle_stage2" and k[1] != "/shuffled":
                material[k] = r2.oracle_material(j, k[2], pools, loaded=loaded, load=load, rule=rule, **mode)
    clock["compose"] = time.time()

    # render: {form: {clip: state}}
    blanks = dict(zip(mine, blanked([refs[i] for i in mine])))
    forms = {"real": dict(refs), "blank": blanks}
    plans_out = {i: {} for i in mine}
    refining = bool(args.refiner_checkpoint_key)
    controls = CONTROLS + ((REST_ONLY,) if refining else ())
    for v in variants:
        for c in controls:
            states = {}
            for i in mine:
                job = jobs[(v, c, i)] if c in CONTROLS and not (c == "/shuffled" and v != "core_chained") \
                    else jobs[(v, "", i)]
                if job is None:
                    states[i] = blanks[i]
                    continue
                reordered = c == "/shuffled" and v != "core_chained"
                source = (material[(v, "", i)] if reordered else rest_material(material[(v, "", i)])
                          if c == REST_ONLY else material[(v, c, i)])
                states[i], fp = r2.render_form(source, job, seconds[i], model, timing=args.timing,
                                               total_fit=args.total_fit, order=orders[i] if reordered else None,
                                               sample_id=f"round15-eval-{v}{c}:{i}", space=space, bridge=args.bridge)
                if c == "":
                    plans_out[i][v] = dict(frames=fp["frames"], fit=fp["fit"], overlong=fp["overlong"])
                    if knobs_on:
                        plans_out[i][v].update(space=fp.get("space"), bridge=args.bridge, cores=[
                            {k2: s.get(k2) for k2 in ("reason", "lo", "hi", "tightened")}
                            for s in material[(v, c, i)]["segments"]])
            forms[f"{v}{c}"] = states
    refiner_info = None
    if refining:                     # every composed form refined, per global block (module docstring)
        from signtranslator.generation.refiner import load_refiner, load_target_index, refine_states, tokenize_state
        from evaluate_text_to_motion import accepted_decoder_gains
        from exposure_diagnostic import make_post
        refiner, rtok, rmeta = load_refiner(store, args.refiner_checkpoint_key, args.refiner_sha256, args.device)
        target_index, index_meta = load_target_index(store, rmeta["target_index_key"], rmeta["target_index_sha256"])
        rpost = make_post(accepted_decoder_gains(store, rmeta["tokenizer_key"]))
        r_temperature = temperature if args.refiner_temperature is None else (
            {k: float(v) for k, v in (x.split("=") for x in args.refiner_temperature.split(","))}
            if "=" in args.refiner_temperature else {k: float(args.refiner_temperature) for k in temperature})
        refiner_info = dict(checkpoint_key=args.refiner_checkpoint_key, sha256=rmeta["sha256"], step=rmeta["step"],
                            tokenizer_key=rmeta["tokenizer_key"], guidance=args.refiner_guidance,
                            temperature=r_temperature,
                            seed_base=REFINE_SEED, target_index_key=rmeta["target_index_key"],
                            target_index_sha256=rmeta["target_index_sha256"])

        def nearest(codes):
            q = target_index.query(codes)
            return dict(q, nearest=None if q["nearest"] is None else index_meta["signs"][q["nearest"]])
        # report-only memorisation read (module docstring); the real clips' codes are the chance level, read from
        # the references served as the training targets are (build_recon_states.serve: usable channels, window,
        # conformed to the template, 24 fps)
        from build_recon_states import serve
        r_cache = ContentCache(Path(args.cache_dir), 8 * 2 ** 30)
        r_template = layout_template(manifest, TEMPLATE_CORPUS, r_cache, store.get)
        memorised = {i: {"real": nearest(tokenize_state(rtok, serve(val[i], r_template, cache=r_cache,
                                                                    fetch=store.get), args.device)["codes"])}
                     for i in mine}
        for v in variants:
            for c, recon in [(c, False) for c in controls] + [("", True)]:
                name = f"{v}{c}"
                out = {}
                for group in blocks(mine, args.block):
                    states, codes, inputs = refine_states(
                        refiner, rtok, [forms[name][i] for i in group], seed=REFINE_SEED + group[0] // args.block,
                        guidance=args.refiner_guidance, temperature=r_temperature, post=rpost, device=args.device,
                        recon=recon, return_codes=True)
                    out.update(zip(group, states))
                    if c == "" and not recon:            # the refined codes and the refiner's input codes
                        for i, out_codes, in_codes in zip(group, codes, inputs):
                            memorised[i][f"{v}{REFINED}"] = nearest(out_codes)
                            memorised[i][v] = nearest(in_codes)
                forms[f"{name}{RECON if recon else REFINED}"] = out
                print(f"{'reconstructed' if recon else 'refined'} {name} ({time.time() - clock['start']:.0f} s)",
                      flush=True)
        del refiner, rtok
        clock["refine"] = time.time()
    # translate per global block (the unsharded run's batches)
    kp = json.loads(store.get(f"training-runs/{TRANSLATOR_RUN}/best.json"))["checkpoint_key"]
    translator, kind = load_translator(store, kp, MIX_SHA)
    translator = translator.to(torch.device(args.device))
    translator_fps = float(getattr(getattr(translator, "config", None), "fps", FPS))
    hyps = {name: {} for name in forms}
    for name, states in forms.items():
        for group in blocks(mine, args.block):
            for i, h in zip(group, translate(translator, kind, [states[i] for i in group])):
                hyps[name][i] = h
        print(f"translated {name} ({time.time() - clock['start']:.0f} s)", flush=True)
    clock["translation"] = time.time()
    # spotting (own words, the next sentence's words) and motion, per clip
    recogniser, labels, best = vsr.load_model(store, args.recognizer_run, args.device)
    index = {g: k for k, g in enumerate(labels)}
    spot_words = [sws.sentence_labels(t, set(labels)) for t in texts]
    spot = {}
    for v in variants + ([f"{w}{REFINED}" for w in variants] if refining else []):
        spot[v] = {}
        for i in mine:
            pred = sws.window_predictions(recogniser, at_rate(forms[v][i], FPS), args.device, windows=SPOT["windows"],
                                          hop=SPOT["hop"], k=SPOT["top_k"])
            own = sws.score_form([pred], [spot_words[i]], index)[0]
            other = sws.score_form([pred], [spot_words[(i + 1) % n]], index)[0]
            spot[v][i] = dict(own=own, mismatched=other)
    motion = {name: dict(zip(mine, motion_rows([states[i] for i in mine], [refs[i] for i in mine], translator_fps)))
              for name, states in forms.items() if name != "blank"}
    clock["scoring"] = time.time()
    rows = []
    for i in mine:
        rows.append(dict(clip=i, sign=val[i]["sign"], entries=plans[i]["entries"], stage1_words=words1[i],
                         shuffled_order=orders[i], mismatched_from=content_perm[i], frames_reference=len(refs[i].timestamps),
                         plans=plans_out[i], hypotheses={name: hyps[name][i] for name in forms},
                         spotting={v: spot[v][i] for v in spot},
                         **({"memorisation_codes": memorised[i]} if refining else {}),
                         motion={name: motion[name][i] for name in motion},
                         rerank={v: {block: [s.get("rerank") for s in material[(v, c, i)]["segments"]]
                                     for block, c in (("matched", ""), ("mismatched_words", "/mismatched_words"))
                                     if (v, c, i) in material}
                                 for v in variants if v in r2.GENERATED} if args.rerank > 1 else None))
    settings = dict(clips=n, shards=args.shards, block=args.block, seed=args.seed, mix_key=args.mix_key,
                    mix_sha256=args.mix_sha256, corpora=list(corpora), variants=variants, selection=args.selection,
                    timing=args.timing, core_margin=margin, rerank=args.rerank, total_fit=args.total_fit,
                    guidance=args.guidance,
                    part_temperature=temperature, checkpoint_key=args.checkpoint_key, generator_mix=generator_mix,
                    model_sha256=model["sha256"], translator_key=kp, recognizer_run=args.recognizer_run,
                    recognizer_checkpoint=best.get("checkpoint_key") if isinstance(best, dict) else None)
    if refiner_info:
        settings["refiner"] = refiner_info
    core_counts = None
    if knobs_on:                                 # recorded only when a knob is on (default reports are unchanged)
        settings.update(space_normalise=args.space_normalise, bridge=args.bridge, tighten_cores=args.tighten_cores,
                        space_stats=dict(key=args.space_stats_key, sha256=args.space_stats_sha256) if space else None)
        core_counts = {}
        for v in variants:
            segs = [s for i in mine for c in ("", "/mismatched_words") if (v, c, i) in material
                    for s in material[(v, c, i)]["segments"]]
            core_counts[v] = dict(reasons=dict(collections.Counter(s.get("reason") for s in segs)),
                                  tightened_cores=int(sum(bool(s.get("tightened")) for s in segs)),
                                  tightened_frames=int(sum(s.get("tightened") or 0 for s in segs)),
                                  cores=len(segs))
    return dict(kind="round15-compose-eval-shard", **({"core_counts": core_counts} if core_counts else {}), shard=args.shard, settings=settings, texts=texts, indexing=INDEXING,
                signs=[r["sign"] for r in val], reference_derangement=reference_perm, words_derangement=content_perm,
                stage1_check=stage1_check, spot_labels=len(labels),
                timing_counts=dict(collections.Counter(p[v]["fit"] for p in plans_out.values() for v in p)),
                seconds={k: round(v - clock["start"], 1) for k, v in clock.items() if k != "start"}, clips=rows)


# ---- merge ---------------------------------------------------------------------------------------------------------

def train_texts(store, settings) -> list[str]:
    """The How2Sign train texts of the evaluation mix (the refiner's training sentences are among them)."""
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from round15_oracle import TEMPLATE_CORPUS
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(settings["mix_key"]) + b"\n")
        manifest = load_manifest(path, expected_sha256=settings["mix_sha256"])
    return [r["text"] for r in manifest["records"] if r.get("corpus") == TEMPLATE_CORPUS
            and r.get("partition") == "train" and r.get("text")]


def memorisation_report(store, settings, rows, hyps, variants) -> dict:
    """Report-only memorisation reads (``signtranslator.generation.memorisation``): each listed form's hypotheses
    against How2Sign train texts (best sentence BLEU, exact matches) and the shards' code n-gram reads against the
    refiner's training targets; ``real`` (val clips the refiner never saw) is the chance level of both."""
    from signtranslator.generation.memorisation import TextIndex, code_summary
    index = TextIndex(train_texts(store, settings))
    names = ["real"] + [f for v in variants for f in (v, f"{v}{RECON}", f"{v}{REFINED}", f"{v}{REST_ONLY}{REFINED}",
                                                       f"{v}/mismatched_words{REFINED}")]
    codes = list(rows[0].get("memorisation_codes") or {})
    return dict(train_texts=len(index.tokens), text={n: index.summary(hyps[n]) for n in names if n in hyps},
                codes={n: code_summary([r["memorisation_codes"][n] for r in rows]) for n in codes},
                note="report-only; real is the chance level (val clips, never in the refiner's training)")


def merge(args, store) -> dict:
    from round15_compositional import paired_recall
    import round15_stage2 as r2
    from sentence_exposure_diagnostic import derangement, mean_statistics, ratio_to, score_forms
    import sentence_word_spotting as sws
    from strict_prior_review import separated
    partials = [json.loads(store.get(k)) for k in args.shard_keys.split(",") if k] + \
               [json.loads(Path(f).read_text()) for f in (args.shard_files or "").split(",") if f]
    merged = merge_partials(partials)
    s, rows, texts = merged["settings"], merged["rows"], merged["texts"]
    n, variants = s["clips"], s["variants"]
    reference_perm = derangement(n, s["seed"])
    if partials[0]["reference_derangement"] != reference_perm:
        raise SystemExit("the shards' reference derangement is not this seed's")
    forms = list(rows[0]["hypotheses"])
    hyps = {name: [r["hypotheses"][name] for r in rows] for name in forms}
    comparators = {}
    if args.end_to_end_report:
        pilot = json.loads(store.get(args.end_to_end_report))
        meaning = pilot["meaning"]
        if meaning.get("references") != texts:
            raise SystemExit("the end-to-end report's references are not these clips' texts in order")
        hyps["end_to_end"] = list(meaning["generated"]["hypotheses"])
        comparators["end_to_end"] = dict(key=args.end_to_end_report, checkpoint_key=pilot.get("checkpoint_key"),
                                         stored_bleu=meaning["generated"]["bleu"],
                                         real_identical=int(sum(a == b for a, b in zip(meaning["real"]["hypotheses"],
                                                                                       hyps["real"]))),
                                         blank_identical=int(sum(a == b for a, b in zip(meaning["blank"]["hypotheses"],
                                                                                        hyps["blank"]))))
    paired = ()
    if args.paired_with:
        other = json.loads(store.get(args.paired_with))
        if other.get("references") != texts or other.get("signs") != merged["signs"]:
            raise SystemExit("--paired-with: another report's clips")
        paired = tuple(v for v in variants if v in other["forms"])
        for v in paired:
            hyps[f"paired:{v}"] = list(other["forms"][v]["hypotheses"])
        comparators["paired_with"] = dict(key=args.paired_with, settings=other.get("settings"),
                                          variants=list(paired))
    refined = bool(s.get("refiner"))
    if args.refined_rule and not refined:
        raise SystemExit("--refined-rule: these shards were not refined (no --refiner-checkpoint-key)")
    if args.refined_rule and s.get("recognizer_run") != INSTRUMENT:
        raise SystemExit(f"--refined-rule: spotting must be read with {INSTRUMENT} (the instrument since recogniser v3 "
                         f"amendment 2), not {s.get('recognizer_run')}")
    checks, refined_checks = r2.parse_rule(args.rule), r2.parse_rule(args.refined_rule)
    pairs = form_pairs(variants, paired, refined=refined)
    for flag, cs, names in (("--rule", checks, variants),
                            ("--refined-rule", refined_checks, [f"{v}{REFINED}" for v in variants])):
        for c in cs:
            for v in names:
                if c["kind"] == "bleu" and c["pair"] not in set(pairs[v]) | {"mismatched", "blank"}:
                    raise SystemExit(f"{flag}: {v} is not compared with {c['pair']!r}")
                if c["kind"] == "bleu" and c["pair"] == "end_to_end" and "end_to_end" not in hyps:
                    raise SystemExit(f"{flag} names end_to_end: give --end-to-end-report")
    scores = score_forms(hyps, texts, reference_perm, boot=args.bootstrap, pairs=pairs)
    spotting = {}
    for v in rows[0]["spotting"]:
        own = [r["spotting"][v]["own"] for r in rows]
        other = [r["spotting"][v]["mismatched"] for r in rows]
        spotting[v] = dict(recall=sws.summarise([x["recall"] for x in own]),
                           order=sws.summarise([x["order"] for x in own]))
        spotting[f"{v}_mismatched"] = dict(recall=sws.summarise([x["recall"] for x in other]))
        a, b = spotting[v]["recall"]["ci95"], spotting[f"{v}_mismatched"]["recall"]["ci95"]
        spotting[v]["above_mismatched"] = bool(a and b and separated(a, b))
        spotting[v]["minus_mismatched_paired"] = paired_recall(own, other)
    motion = {name: mean_statistics([r["motion"][name] for r in rows]) for name in rows[0]["motion"]}
    motion_ratio = {name: ratio_to(m, motion["real"]) for name, m in motion.items()}
    verdicts = {v: r2.verdict(checks, scores[v], spotting[v]["recall"], spotting[f"{v}_mismatched"]["recall"])
                for v in variants}
    if refined:
        verdicts.update({f"{v}{REFINED}": r2.verdict(refined_checks, scores[f"{v}{REFINED}"],
                                                     spotting[f"{v}{REFINED}"]["recall"],
                                                     spotting[f"{v}{REFINED}_mismatched"]["recall"])
                         for v in variants})
    overlong = {v: int(sum(bool((r["plans"].get(v) or {}).get("overlong")) for r in rows)) for v in variants}
    return dict(kind="round15-compose-eval", settings=s, clips=n, references=texts, signs=merged["signs"], indexing=INDEXING,
                shards=sorted(p["shard"] for p in partials), stage1_check=partials[0]["stage1_check"],
                comparators=comparators, rule=args.rule, rule_checks=checks, bootstrap=args.bootstrap, forms=scores,
                **(dict(refined_rule=args.refined_rule, refined_rule_checks=refined_checks, refiner=s["refiner"],
                        memorisation=memorisation_report(store, s, rows, hyps, variants)) if refined else {}),
                spotting=dict(windows=list(SPOT["windows"]), hop=SPOT["hop"], top_k=SPOT["top_k"], scores=spotting),
                motion=motion, motion_ratio_to_real=motion_ratio, verdicts=verdicts, overlong_clips=overlong,
                **merge_core_counts(partials),
                per_clip=[{k: r[k] for k in ("clip", "sign", "entries", "stage1_words", "plans")} for r in rows])


def summary(report: dict) -> None:
    def ci(d):
        return f"{d['difference']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
    print(f"BLEU on {report['clips']} clips:", flush=True)
    for name, row in report["forms"].items():
        extra = "".join(f"  -{k[6:]} {ci(v)}" for k, v in row.items() if k.startswith("minus_") and k != "minus_blank")
        print(f"  {name:34s} {row['bleu']:6.2f} [{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]{extra}", flush=True)
    print("wrist speed / real: " + json.dumps({k: (v or {}).get("wrist_speed")
                                              for k, v in report["motion_ratio_to_real"].items()}), flush=True)
    for v, d in report["verdicts"].items():
        print(f"VERDICT {v}: {d['verdict']} " + json.dumps(d["checks"]), flush=True)


def _write(store, args, payload: dict) -> None:
    data = json.dumps(payload, indent=1, default=float).encode()
    if args.output_file:
        with open(args.output_file, "xb") as stream:
            stream.write(data)
    if args.output_key:
        if store.exists(args.output_key):
            raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
        store.put(args.output_key, data)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="one shard")
    r.add_argument("--mix-key", default=EVAL_MIX, help="the evaluation mix (vocabulary, clips, stage-2 model)")
    r.add_argument("--mix-sha256", default=EVAL_MIX_SHA)
    r.add_argument("--corpora", help="the isolated corpora of the vocabulary (default: compositional."
                   "isolated_corpora of the evaluation mix, round 15's four for round10-mixed-v1)")
    r.add_argument("--clips", type=int, default=1000)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--block", type=int, default=32, help="clips per shard block and translation batch")
    r.add_argument("--seed", type=int, default=0, help="the reference derangement (the words derangement: + 1)")
    r.add_argument("--stage1-report-key", help="a stored stage-1 report: shared clips must have its words")
    r.add_argument("--checkpoint-key")
    r.add_argument("--generator-mix-key")
    r.add_argument("--generator-mix-sha256")
    r.add_argument("--model-file")
    r.add_argument("--expect-model-sha256")
    r.add_argument("--variants", default="core_crossfade")
    r.add_argument("--selection", default="stage1", choices=("stage1", "stage2"))
    r.add_argument("--timing", default="sign-fitted", choices=("asllrp", "stage1", "sign-fitted"))
    r.add_argument("--core-margin", type=float)
    r.add_argument("--total-fit", default="pad", choices=("pad", "scale", "none"))
    r.add_argument("--space-normalise", default="off", choices=("off", "sentence", "how2sign"),
                   help="one signing space per sentence (composition_space.space_normalise): torso and dominant-hand "
                        "centroid to the sentence's mean (sentence) or How2Sign train's (how2sign), excursion scaled "
                        "to How2Sign train's; needs --space-stats-key/--space-stats-sha256")
    r.add_argument("--space-stats-key", help="How2Sign train statistics (scripts/round15_space_stats.py)")
    r.add_argument("--space-stats-sha256")
    r.add_argument("--bridge", default="linear", choices=("linear", "minimum-jerk"))
    r.add_argument("--tighten-cores", action="store_true", help="cut median-position cores (entry/all) again by "
                   "the activity rule inside their window (sign-fitted timing)")
    r.add_argument("--part-temperature", default="body=1.1,left_hand=1.1,right_hand=1.1,face=1.0")
    r.add_argument("--guidance", type=float, default=1.5)
    r.add_argument("--recognizer-run", default="sign-recognizer-v2")
    r.add_argument("--device", default="cuda")
    r.add_argument("--word-workers", type=int, default=1)
    r.add_argument("--rerank", type=int, default=1, help="samples per word, the best kept by the prior's own "
                   "teacher-forced PMI (signtranslator.generation.rerank; 1: unchanged; core_crossfade only)")
    r.add_argument("--cache-dir", default=str(Path(tempfile.gettempdir()) / "sign-translator-cache"))
    r.add_argument("--refiner-checkpoint-key", help="refine every composed form with this sentence refiner "
                   "(<form>@refined; adds the /rest_only control and <variant>@recon)")
    r.add_argument("--refiner-sha256", help="pin the refiner checkpoint")
    r.add_argument("--refiner-guidance", type=float, default=1., help="the refiner's guidance against the empty source")
    r.add_argument("--refiner-temperature", help="the refiner's sampling temperature: one number for every part or "
                   "part=value,...; default: --part-temperature")
    m = sub.add_parser("merge", help="combine the shards and score")
    m.add_argument("--shard-keys", default="")
    m.add_argument("--shard-files")
    m.add_argument("--end-to-end-report", help="a pilot report of the same clips: its meaning.generated hypotheses")
    m.add_argument("--paired-with", help="another merged report on the same clips")
    m.add_argument("--rule")
    m.add_argument("--refined-rule", help="the declared rule of the refined forms (<variant>@refined); the "
                   f"design's: {REFINED_RULE}")
    m.add_argument("--bootstrap", type=int, default=1000)
    for q in (r, m):
        q.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
        q.add_argument("--output-key")
        q.add_argument("--output-file")
    args = p.parse_args(argv)
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    import warnings
    warnings.simplefilter("ignore")
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    if args.command == "run":
        variants = [v for v in args.variants.split(",") if v]
        import round15_stage2 as r2
        if not variants or any(v not in r2.VARIANTS for v in variants) or len(set(variants)) != len(variants):
            p.error(f"--variants: distinct names from {','.join(r2.VARIANTS)}")
        if any(v in r2.GENERATED for v in variants) and not args.checkpoint_key:
            p.error("--checkpoint-key is required for generated variants")
        if bool(args.generator_mix_key) != bool(args.generator_mix_sha256):
            p.error("--generator-mix-key and --generator-mix-sha256 go together")
        if args.timing != "asllrp" and args.total_fit != "pad":
            p.error("--total-fit applies to the asllrp timing only")
        if args.space_normalise != "off" and not (args.space_stats_key and args.space_stats_sha256):
            p.error("--space-normalise needs --space-stats-key and --space-stats-sha256")
        if (args.space_normalise != "off" or args.bridge != "linear") and args.timing == "stage1":
            p.error("--space-normalise and --bridge minimum-jerk apply to the asllrp and sign-fitted timings")
        if args.space_normalise != "off" and "core_chained" in variants:
            p.error("--space-normalise needs word pieces (core_crossfade, oracle_stage2)")
        if args.tighten_cores and args.timing != "sign-fitted":
            p.error("--tighten-cores applies to the sign-fitted timing (its median-position cores)")
        if args.rerank < 1 or (args.rerank > 1 and (args.timing == "stage1" or "core_chained" in variants)):
            p.error("--rerank >= 1; above 1 only for core_crossfade with the asllrp or sign-fitted timing")
        if args.timing == "stage1" and args.core_margin is not None:
            p.error("--core-margin does not apply to the stage1 timing")
        if args.refiner_sha256 and not args.refiner_checkpoint_key:
            p.error("--refiner-sha256 pins --refiner-checkpoint-key")
        if not 0 <= args.shard < args.shards or args.block < 1:
            p.error("--shard must be in 0..--shards-1 and --block >= 1")
        payload = run(args)
    else:
        if not (args.shard_keys or args.shard_files):
            p.error("give --shard-keys or --shard-files")
        payload = merge(args, store)
        summary(payload)
    _write(store, args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
