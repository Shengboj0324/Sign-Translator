"""Round 15 instrument check: the oracle composition (docs/NEXT_PHASE_PLAN.md, "Round 15 stage 1 result", 2026-10-28
19:28 UTC; declared there before it ran, report-only). Real isolated-sign clips of the round-15 words, composed the way
V1 composes generated words, scored with the same instruments: can back-translation BLEU reward word-by-word signing at
all, and if it can, are the generated words the bottleneck?

**Same inputs as the round-15 run** (``--source-report-key``, the stored stage-1 report): its manifest (``mix_key``,
``mix_sha256``), its 300 How2Sign val clips (``pilot_content_check.validation_sentences``) and their references, its
word selection (training vocabulary of ``corpora``, amendment 1: ``compositional.selection_details``), its reference
derangement (``seed``), its words derangement (``seed + 1``), its shuffle orders (``compositional_baseline.
shuffle_order(n, clip, 0)``) and its crossfade overlap. Every one of these is recomputed here and checked against the
stored report clip by clip; any difference stops the run, so the pairing is exact.

**Real clips** (``choose_record``): for each selected word, one record drawn uniformly from the word's pool, the
training vocabulary's own records (``word_pools``: the prior manifest's ``train`` partition of the isolated corpora, a
text normalising to the entry, usable continuous channels, sorted by corpus and sign), with a seed fixed per (clip,
word) (``clip_seed``: SHA-256 of the clip index, the word and the control block). A word repeated in one sentence is the
same real clip each time. Each record is loaded as the prior's training dataset loads it (``word_clip``):
``corpus_manifest.load_sample`` (unadmitted channels masked, the trimmed sign window ``continuous.window_seconds`` kept,
a mirrored record mirrored), conformed to the How2Sign layout template (``layout_adapter.conform`` with
``train_motion_tokens.layout_template``, the layout every corpus is conformed to in training and in which the generator
assembles its states), then resampled to 24 fps.

**Forms:**
- ``oracle`` (the declared form): V1 crossfade with fitted durations. Each real window is time-resampled
  (``compositional_baseline.time_scale``: first and last frames kept, linear interpolation on the manifolds) to its
  word's fitted frame count (``round15_compositional.plan_durations("crossfade", ...)`` then ``compositional.
  word_frames``: exactly V1's durations, the reference duration plus the joins' overlap), joined by
  ``compositional.concatenate`` with the round-15 overlap (4 frames);
- ``oracle_unscaled``: the same clips at their real (trimmed) durations, the same join;
- controls of each: ``/shuffled`` (the same clips in the round-15 shuffle order: content without order) and
  ``/mismatched_words`` (another sentence's words, the round-15 words derangement, with real clips drawn in their own
  seed block and, for ``oracle``, durations fitted to *this* clip's reference); the deranged-reference control is
  ``score_forms``' ``minus_mismatched`` on every form. A sentence with no selected word reads as its blank clip;
- ``real`` and ``blank`` are translated afresh (a check that this run's instrument reproduces the stored report's
  hypotheses); ``end_to_end`` and ``generated_crossfade`` (the stored report's ``crossfade``, V1) are the stored
  report's hypotheses on the same clips, scored here with the same references and bootstrap.

**Processing:** none. The generated forms' post-processing (``exposure_diagnostic.make_post``: the tokenizer's
accepted decoder gains, then 0.1 s head smoothing) corrects the tokenizer decoder's output and the generator's head
motion; it only applies to generated states, and the stored report's ``real`` form had none.

**Scores:** corpus BLEU with the pilot's bootstrap interval; paired differences (``sentence_exposure_diagnostic.
score_forms``) of each oracle form against end_to_end, generated_crossfade, its shuffled and mismatched-words controls
(and the fitted oracle against the unscaled one); ``sentence_word_spotting`` (unchanged; the round-15 windows, hop,
top-k and recogniser) on each oracle form against its mismatched control (the next sentence's words), the instrument's
``separated`` reading and the paired difference; motion statistics (``compositional_baseline.motion_rows``), wrist speed
among them, as ratios to real. The stored report's spotting and motion of end_to_end and the generated crossfade are
copied beside them.

**Readings (declared):** (i) if ``oracle`` is NOT above end_to_end on paired BLEU (95% lower bound <= 0), BLEU cannot
measure composition, and a sentence measure that can is needed before any structural round is judged; (ii) if it is
well above, the generated words are the bottleneck (explanation a). "Well above" is read here as the lower bounds against
both end_to_end and the generated crossfade above 0 (the generated crossfade differs from the oracle only in its
words); above end_to_end but not resolved from the generated crossfade is reported as neither.

Writes one JSON report to ``--output-key`` (or ``--output-file``; both write-once). No GPU: nothing is generated.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 24.
SOURCE_REPORT = "acquisition/2026-10-17-v1/youtube-asl-v1/round15/compositional-f12-step60000.json"
TEMPLATE_CORPUS = "how2sign-two-view-text-v1"
TIMINGS = ("oracle", "oracle_unscaled")
CONTROLS = ("", "/shuffled", "/mismatched_words")
STORED = {"end_to_end": "end_to_end", "generated_crossfade": "crossfade"}   # this report's name: the stored form
BLOCKS = ("matched", "mismatched_words")

READING_I = ("(i) the oracle composition is NOT above end-to-end on paired BLEU: BLEU cannot measure composition; a "
             "sentence measure that can is needed before any structural round is judged.")
READING_II = ("(ii) the oracle composition is well above end-to-end (and above the generated crossfade): the generated "
              "words are the bottleneck (explanation a); round 14 (better words) is the lever.")
READING_NEITHER = ("neither: the oracle is above end-to-end, but not resolved from the generated crossfade composition.")

CAVEATS = [
    "Real clips are MediaPipe monocular estimates of isolated signers (ASLLVD, RIT, WLASL), conformed to the How2Sign "
    "layout; their hands and arms are untracked on some frames (reported: motion hands_tracked), where generated "
    "states are valid everywhere. The crossfade keeps one side's values where only one side is valid.",
    "Signers change at every word: each word is a different person's clip, so body size, signing space and rest pose "
    "jump at the joins (the 4-frame crossfade smooths, not removes, them). The back-translator reads normalised 2D "
    "keypoints (shoulder units), which removes scale but not posture.",
    "Fitted timing compresses real signs: the trimmed windows are about 1.3-2.2 s, the fitted words about 0.5-1 s "
    "(reported: timing.compression), so the fitted oracle signs 2-3x faster than its sources. oracle_unscaled keeps "
    "the real speed at the cost of a sentence several times longer than its reference (and the translator's 20 s "
    "window: motion over_window).",
    "English order and word selection are round 15's (amendment 1): function words with a sign are kept, "
    "unmatched words (34% of tokens) are absent, so even a perfect lexicon does not sign the whole sentence.",
    "One clip per word occurrence, drawn once (seeded): the oracle's BLEU carries the clip draw's variance, which the "
    "bootstrap over sentences does not separate.",
    "end_to_end and generated_crossfade are the stored report's hypotheses (GPU run); real and blank are retranslated "
    "here, and their agreement with the stored hypotheses is reported (reproduction).",
]


# ---- pure logic --------------------------------------------------------------------------------------------------

def word_pools(manifest, corpora, partition: str = "train") -> dict[str, list[dict]]:
    """{entry: records} of the training vocabulary (``compositional.training_vocabulary``'s own filter: ``corpora``,
    ``partition``, a text, usable continuous channels, a duration), sorted by (corpus, sign, state hash) so a draw does
    not depend on the manifest's record order."""
    from signtranslator.data_engineering.corpus_manifest import usable_channels
    from signtranslator.generation.compositional import normalise
    pools: dict[str, list[dict]] = {}
    for r in manifest["records"]:
        if r.get("corpus") not in corpora or r.get("partition") != partition or not r.get("text"):
            continue
        if not usable_channels(r, "continuous") or not (r.get("continuous") or {}).get("seconds"):
            continue
        entry = " ".join(normalise(r["text"]))
        if entry:
            pools.setdefault(entry, []).append(r)
    return {w: sorted(v, key=lambda r: (r["corpus"], r["sign"], r["continuous"]["state_sha256"]))
            for w, v in sorted(pools.items())}


def clip_seed(clip: int, word: str, block: str = "matched") -> int:
    """The draw's seed, fixed per (clip, word) and control block: the first 8 bytes of a SHA-256."""
    if block not in BLOCKS:
        raise ValueError(f"block must be one of {BLOCKS}")
    digest = hashlib.sha256(f"round15-oracle|{block}|{int(clip)}|{word}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def choose_record(pools, clip: int, word: str, block: str = "matched") -> dict:
    """One record of ``word``'s pool, uniformly, by ``clip_seed(clip, word, block)``."""
    pool = pools.get(word)
    if not pool:
        raise KeyError(f"{word!r} has no training record")
    rng = np.random.default_rng(clip_seed(clip, word, block))
    return pool[int(rng.integers(len(pool)))]


def plan_oracle(selected, pools, vocab, seconds, content_perm, overlap) -> dict:
    """{(block, clip): plan or None (no word)}; a plan is {words, records, fitted (seconds), frames (fitted)} with
    the V1 durations (``round15_compositional.plan_durations``, crossfade) of this clip's reference."""
    from signtranslator.generation.compositional import word_frames
    from round15_compositional import plan_durations
    plans = {}
    for i in range(len(selected)):
        for block, words in (("matched", selected[i]), ("mismatched_words", selected[content_perm[i]])):
            if not words:
                plans[(block, i)] = None
                continue
            fitted = plan_durations("crossfade", words, vocab, seconds[i], overlap)
            plans[(block, i)] = dict(words=list(words), records=[choose_record(pools, i, w, block) for w in words],
                                     fitted=fitted, frames=[word_frames(d) for d in fitted])
    return plans


def word_clip(record, *, cache, fetch, template):
    """One real word clip as the prior's training dataset serves it (``TextMotionDataset``): ``load_sample`` (masked
    channels, the trimmed ``window_seconds``, mirroring), conformed to the layout ``template``, then at 24 fps."""
    from signtranslator.data.layout_adapter import conform
    from signtranslator.data_engineering.corpus_manifest import load_sample
    from evaluate_text_to_motion import at_rate
    state = load_sample(record, "continuous", cache=cache, fetch=fetch)
    return at_rate(conform(state, template), FPS)


def compose_clips(clips, frames, overlap: int, *, fitted: bool, order=None, sample_id: str = "oracle"):
    """The crossfade of real word clips (V1's join): each time-resampled to its fitted frame count (``fitted``) or kept
    at its real length, taken in ``order`` (default: as given)."""
    from compositional_baseline import time_scale
    from signtranslator.generation.compositional import concatenate
    parts = [time_scale(c, f, FPS) if fitted else c for c, f in zip(clips, frames)]
    order = range(len(parts)) if order is None else order
    return concatenate([parts[k] for k in order], overlap, sample_id=sample_id)


def form_pairs() -> dict:
    """``score_forms``'s paired comparisons."""
    pairs = {}
    for t in TIMINGS:
        pairs[t] = {"end_to_end": "end_to_end", "generated_crossfade": "generated_crossfade",
                    "shuffled": f"{t}/shuffled", "mismatched_words": f"{t}/mismatched_words",
                    **{f"timing_{u}": u for u in TIMINGS if u != t}}
        for c in CONTROLS[1:]:
            pairs[f"{t}{c}"] = {"end_to_end": "end_to_end"}
    pairs["generated_crossfade"] = {"end_to_end": "end_to_end"}
    return pairs


def reading(row: dict) -> dict:
    """The declared readings on the fitted oracle's BLEU row."""
    e2e, gen = row["minus_end_to_end"], row["minus_generated_crossfade"]
    if e2e["ci95"][0] <= 0:
        which, text = "i", READING_I
    elif gen["ci95"][0] > 0:
        which, text = "ii", READING_II
    else:
        which, text = "neither", READING_NEITHER
    return dict(reading=which, text=text, minus_end_to_end=e2e, minus_generated_crossfade=gen,
                minus_mismatched_words=row["minus_mismatched_words"], minus_shuffled=row["minus_shuffled"],
                minus_deranged_reference=row["minus_mismatched"])


def check_pairing(stored: dict, val, texts, selected, vocab, reference_perm, content_perm, orders, overlap) -> None:
    """Stop unless this run reads exactly the stored report's clips, words, derangements, orders and overlap."""
    problems = []
    if stored.get("references") != texts:
        problems.append("references differ")
    if stored.get("clips") != len(texts) or len(stored.get("per_clip", [])) != len(texts):
        problems.append("clip count differs")
    if stored.get("reference_derangement") != reference_perm:
        problems.append("reference derangement differs")
    if stored.get("words_derangement") != content_perm:
        problems.append("words derangement differs")
    if (stored.get("crossfade") or {}).get("frames") != overlap:
        problems.append("crossfade overlap differs")
    for i, (row, r, w, o) in enumerate(zip(stored.get("per_clip", []), val, selected, orders)):
        if row.get("sign") != r["sign"] or row.get("words") != w or row.get("shuffled_order") != o \
                or row.get("mismatched_from") != content_perm[i]:
            problems.append(f"clip {i} ({r['sign']}) differs")
            break
    typical = stored.get("typical_seconds") or {}
    off = [w for w in {x for ws in selected for x in ws} if w not in typical or abs(typical[w] - vocab[w]) > 1e-4]
    if off:
        problems.append(f"typical durations differ: {sorted(off)[:5]}")
    if problems:
        raise SystemExit("not the stored round-15 inputs: " + "; ".join(problems))


# ---- the run -----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source-report-key", default=SOURCE_REPORT, help="the stored round-15 stage-1 report")
    p.add_argument("--recognizer-run", help="the word-spotting instrument (default: the stored report's)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=6, help="torch threads (CPU)")
    p.add_argument("--fetch-threads", type=int, default=8, help="threads prefetching the word clips' states")
    p.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    p.add_argument("--cache-dir", default=str(Path(tempfile.gettempdir()) / "sign-translator-cache"))
    p.add_argument("--output-key", help="the store key of the JSON report (write-once)")
    p.add_argument("--output-file", help="a local path for the JSON report instead (write-once; local checks)")
    args = p.parse_args(argv)
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest
    from signtranslator.generation.compositional import (KEPT, crossfade_frames, selection_details,
                                                         training_vocabulary)
    from compositional_baseline import MOTION_KEYS, hypothesis_length_ratio, motion_rows, shuffle_order
    from evaluate_text_to_motion import at_rate, blanked, load_state
    from pilot_content_check import MIX_SHA, validation_sentences
    from round15_compositional import paired_recall
    from score_back_translation import load_translator, translate
    from sentence_exposure_diagnostic import derangement, mean_statistics, ratio_to, score_forms
    import sentence_word_spotting as sws
    from strict_prior_review import separated
    from train_motion_tokens import layout_template
    from train_text_to_motion import make_store
    import validate_sign_recognizer as vsr
    torch.set_num_threads(max(1, args.threads))
    clock = {"start": time.time()}
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    stored = json.loads(store.get(args.source_report_key))
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(stored["mix_key"]) + b"\n")
        manifest = load_manifest(path, expected_sha256=stored["mix_sha256"])
    corpora = tuple(stored["corpora"])
    vocab = training_vocabulary(manifest, corpora)
    pools = word_pools(manifest, corpora)
    if set(pools) != set(vocab):
        raise AssertionError("word pools and the training vocabulary disagree")
    val = validation_sentences(manifest, stored["clips"])
    texts = [r["text"] for r in val]
    n = len(texts)
    details = [selection_details(t, vocab) for t in texts]
    selected = [[r["entry"] for r in rows if r["kind"] in KEPT] for rows in details]
    seed = stored["seeds"]["reference_derangement"]
    reference_perm, content_perm = derangement(n, seed), derangement(n, stored["seeds"]["words_derangement"])
    orders = [shuffle_order(len(w), i, 0) for i, w in enumerate(selected)]
    overlap = crossfade_frames(float(stored["crossfade"]["seconds"]))   # checked against the stored frames
    check_pairing(stored, val, texts, selected, vocab, reference_perm, content_perm, orders, overlap)
    refs = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), FPS) for r in val]
    seconds = [float(s.timestamps[-1] - s.timestamps[0]) + 1 / FPS for s in refs]
    blanks = blanked(refs)
    plans = plan_oracle(selected, pools, vocab, seconds, content_perm, overlap)

    # the real word clips: prefetch every state (I/O threads), then load each as the training dataset does
    cache = ContentCache(Path(args.cache_dir), 8 * 2 ** 30)
    template = layout_template(manifest, TEMPLATE_CORPUS, cache, store.get)
    records = {r["continuous"]["state_sha256"]: r for plan in plans.values() if plan for r in plan["records"]}
    with ThreadPoolExecutor(max(1, args.fetch_threads)) as pool:
        list(pool.map(lambda r: cache.get(r["continuous"]["state_key"], r["continuous"]["state_sha256"], store.get),
                      records.values()))
    clock["fetch"] = time.time()
    print(f"{n} clips, {sum(seconds):.0f} s of reference; vocabulary {len(vocab)} entries; "
          f"{sum(len(w) for w in selected)} words selected; {len(records)} distinct real clips fetched "
          f"({clock['fetch'] - clock['start']:.0f} s)", flush=True)

    forms = {"real": refs, "blank": blanks, **{f"{t}{c}": [] for t in TIMINGS for c in CONTROLS}}
    per_clip, chosen = [], collections.Counter()
    real_seconds, fitted_seconds = [], []
    for i in range(n):
        row = dict(sign=val[i]["sign"], words=selected[i], frames_reference=len(refs[i].timestamps))
        for block, suffixes in (("matched", ("", "/shuffled")), ("mismatched_words", ("/mismatched_words",))):
            plan = plans[(block, i)]
            if plan is None:
                for t in TIMINGS:
                    for c in suffixes:
                        forms[f"{t}{c}"].append(blanks[i])
                row[block] = None
                continue
            loaded = {}
            for r in plan["records"]:
                if r["continuous"]["state_sha256"] not in loaded:
                    loaded[r["continuous"]["state_sha256"]] = word_clip(r, cache=cache, fetch=store.get,
                                                                        template=template)
            clips = [loaded[r["continuous"]["state_sha256"]] for r in plan["records"]]
            for t in TIMINGS:
                fitted = t == "oracle"
                for c in suffixes:
                    order = orders[i] if c == "/shuffled" else None
                    forms[f"{t}{c}"].append(compose_clips(clips, plan["frames"], overlap, fitted=fitted, order=order,
                                                          sample_id=f"round15-{t}{c}:{i}"))
            real = [len(c.timestamps) for c in clips]
            if block == "matched":
                chosen.update(r["corpus"] for r in plan["records"])
                real_seconds += [f / FPS for f in real]
                fitted_seconds += list(plan["fitted"])
            row[block] = dict(words=plan["words"],
                              clips=[dict(word=w, corpus=r["corpus"], sign=r["sign"], frames_real=f, frames_fitted=g,
                                          window_seconds=r["continuous"].get("window_seconds"))
                                     for w, r, f, g in zip(plan["words"], plan["records"], real, plan["frames"])],
                              fitted=[round(d, 4) for d in plan["fitted"]])
        row["frames"] = {t: len(forms[t][i].timestamps) for t in TIMINGS}
        per_clip.append(row)
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"composed {i + 1}/{n} ({time.time() - clock['start']:.0f} s)", flush=True)
    clock["compose"] = time.time()

    # back-translation: real, blank and the oracle forms here; end_to_end and the generated crossfade stored
    kp = json.loads(store.get(f"training-runs/{stored['translator_run']}/best.json"))["checkpoint_key"]
    if kp != stored["translator_key"]:
        raise SystemExit(f"the translator run now points at {kp}, not the stored report's {stored['translator_key']}")
    translator, kind = load_translator(store, kp, MIX_SHA)
    translator = translator.to(torch.device(args.device))
    translator_fps = float(getattr(getattr(translator, "config", None), "fps", FPS))
    hyps = {}
    for name, states in forms.items():
        hyps[name] = list(translate(translator, kind, states))
        print(f"translated {name} ({time.time() - clock['start']:.0f} s)", flush=True)
    for name, source in STORED.items():
        hyps[name] = list(stored["forms"][source]["hypotheses"])
        if len(hyps[name]) != n:
            raise SystemExit(f"the stored {source} hypotheses are not {n} clips")
    reproduction = {name: dict(identical=int(sum(a == b for a, b in zip(hyps[name], stored["forms"][name]["hypotheses"]))),
                               clips=n) for name in ("real", "blank")}
    clock["translation"] = time.time()
    names = ["real", "blank", "end_to_end", "generated_crossfade"] + [f"{t}{c}" for t in TIMINGS for c in CONTROLS]
    scores = score_forms({k: hyps[k] for k in names}, texts, reference_perm, boot=args.bootstrap, pairs=form_pairs())
    for name, row in scores.items():
        row["hypothesis_over_reference_length"] = hypothesis_length_ratio(hyps[name], texts)
    for name, source in STORED.items():
        reproduction[name] = dict(bleu=scores[name]["bleu"], stored_bleu=stored["forms"][source]["bleu"])
    clock["scoring"] = time.time()

    # word spotting on the oracle forms (sentence_word_spotting's functions, unchanged; the stored run's settings)
    recognizer_run = args.recognizer_run or stored["recognizer_run"]
    spot = stored["spotting"]
    recogniser, labels, best = vsr.load_model(store, recognizer_run, args.device)
    if recognizer_run == stored["recognizer_run"] and stored.get("recognizer_checkpoint") and \
            (best or {}).get("checkpoint_key") != stored["recognizer_checkpoint"]:
        raise SystemExit(f"the recogniser run now points at {(best or {}).get('checkpoint_key')}, not the stored "
                         f"report's {stored['recognizer_checkpoint']}")
    index = {g: i for i, g in enumerate(labels)}
    spot_words = [sws.sentence_labels(t, set(labels)) for t in texts]
    spot_mismatched = spot_words[1:] + spot_words[:1]                     # the next sentence's words
    spotting, spot_rows = {}, {}
    for name in TIMINGS:
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
    for name, source in (("real", "real"), ("end_to_end", "end_to_end"), ("generated_crossfade", "crossfade")):
        for suffix in ("", "_mismatched"):
            spotting[f"stored_{name}{suffix}"] = spot["scores"][f"{source}{suffix}"]
    clock["spotting"] = time.time()

    motion = {name: mean_statistics(motion_rows(forms[name], refs, translator_fps)) for name in forms if name != "blank"}
    motion_ratio = {name: ratio_to(m, motion["real"]) for name, m in motion.items()}
    for name, source in STORED.items():
        motion_ratio[f"stored_{name}"] = stored["motion_ratio_to_real"][source]
    timing = dict(real_word_seconds_median=round(float(np.median(real_seconds)), 4) if real_seconds else None,
                  fitted_word_seconds_median=round(float(np.median(fitted_seconds)), 4) if fitted_seconds else None,
                  compression_median=round(float(np.median(np.array(real_seconds) / np.array(fitted_seconds))), 4)
                  if fitted_seconds else None,
                  frames_over_reference_median={t: round(float(np.median([r["frames"][t] / r["frames_reference"]
                                                                          for r in per_clip if r["matched"]])), 4)
                                                for t in TIMINGS})
    clips_stats = dict(occurrences=int(sum(chosen.values())), corpus_share={c: round(k / max(sum(chosen.values()), 1), 4)
                                                                            for c, k in sorted(chosen.items())},
                       distinct_matched=len({c["sign"] for r in per_clip if r["matched"] for c in r["matched"]["clips"]}),
                       distinct_fetched=len(records),
                       pool_size_median=float(np.median([len(pools[w]) for ws in selected for w in ws]))
                       if any(selected) else None)
    clock["statistics"] = time.time()

    report = dict(
        round=15, check="oracle composition", declared="docs/NEXT_PHASE_PLAN.md, Round 15 stage 1 result (19:28 UTC)",
        source_report_key=args.source_report_key, mix_key=stored["mix_key"], mix_sha256=stored["mix_sha256"],
        translator_run=stored["translator_run"], translator_key=kp, recognizer_run=recognizer_run,
        recognizer_checkpoint=best.get("checkpoint_key") if isinstance(best, dict) else None, clips=n,
        corpora=list(corpora), template_corpus=TEMPLATE_CORPUS, crossfade=dict(frames=overlap),
        processing="none (the generated forms' decoder gains and head smoothing apply to generated states only)",
        clip_seed="sha256('round15-oracle|<block>|<clip>|<word>')[:8], uniform over the word's train records sorted by "
                  "(corpus, sign, state sha256); blocks matched and mismatched_words",
        seeds=dict(reference_derangement=seed, words_derangement=stored["seeds"]["words_derangement"],
                   shuffle="compositional_baseline.shuffle_order(n, clip, 0)"),
        bootstrap=args.bootstrap, beam=4, references=texts, reference_derangement=reference_perm,
        words_derangement=content_perm, reproduction=reproduction, clip_choice=clips_stats, timing=timing,
        forms=scores, reading=reading(scores["oracle"]), reading_unscaled=reading(scores["oracle_unscaled"]),
        spotting=dict(windows=list(spot["windows"]), hop=spot["hop"], top_k=spot["top_k"], labels=len(labels),
                      scores=spotting),
        motion=motion, motion_ratio_to_real=motion_ratio,
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
    def ci(d):
        return f"{d['difference']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"

    def rc(d):
        return f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]" if d and d.get("ci95") else "n/a"
    print(f"BLEU on {report['clips']} clips (form, BLEU [95% CI], minus deranged refs, minus blank, paired "
          f"comparisons, hyp/ref length):", flush=True)
    for name, row in report["forms"].items():
        extra = "".join(f"  -{k[6:]} {ci(v)}" for k, v in row.items()
                        if k.startswith("minus_") and k not in ("minus_mismatched", "minus_blank"))
        print(f"  {name:32s} {row['bleu']:6.2f} [{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]"
              f"  -deranged {ci(row['minus_mismatched'])}"
              + (f"  -blank {ci(row['minus_blank'])}" if "minus_blank" in row else "")
              + extra + f"  len {row['hypothesis_over_reference_length']}", flush=True)
    print("reproduction: " + json.dumps(report["reproduction"]), flush=True)
    print("word spotting recall (form, recall, against the next sentence's words, above):", flush=True)
    s = report["spotting"]["scores"]
    for name in [k for k in s if not k.endswith("_mismatched")]:
        print(f"  {name:32s} {rc(s[name]['recall'])}  mism {rc(s[name + '_mismatched']['recall'])}"
              + (f"  above {s[name]['above_mismatched']}  paired "
                 f"{s[name]['minus_mismatched_paired']['difference']} {s[name]['minus_mismatched_paired']['ci95']}"
                 if "above_mismatched" in s[name] else ""), flush=True)
    print("motion (" + ", ".join(motion_keys) + "):", flush=True)
    for name, m in report["motion"].items():
        print(f"  {name:32s} " + "  ".join(f"{m.get(k):7.3f}" if m.get(k) is not None else "    nan"
                                           for k in motion_keys), flush=True)
    print("wrist speed / real: " + json.dumps({k: v.get("wrist_speed") for k, v in report["motion_ratio_to_real"].items()}),
          flush=True)
    print(json.dumps(dict(clip_choice=report["clip_choice"], timing=report["timing"], seconds=report["seconds"])),
          flush=True)
    for key in ("reading", "reading_unscaled"):
        d = report[key]
        print(f"READING {key}: {d['reading']}  oracle - end_to_end {ci(d['minus_end_to_end'])}; - generated crossfade "
              f"{ci(d['minus_generated_crossfade'])}; - mismatched words {ci(d['minus_mismatched_words'])}; "
              f"- shuffled {ci(d['minus_shuffled'])}; - deranged refs {ci(d['minus_deranged_reference'])}", flush=True)
        print(f"  {d['text']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
