"""Round 4 pilot: does a prior's generated signing carry its text's content? (diagnostic; the round's verdict stays the
declared gate + strict review)

For one prior checkpoint, at one fixed sampling setting (``--part-temperature``, ``--guidance``) and the round-3e
post-processing (head smoothing 0.1 s, head-translation smoothing 0.1 s; the tokenizer's accepted decoder gains are
applied by ``load_generator``):
- **R2b-style meaning:** ``--clips`` How2Sign validation sentences (fixed seed, the three validation signer groups
  pooled) are generated at each reference clip's duration and read by the accepted 2D back-translator; BLEU with a
  bootstrap 95% interval for generated, real and blank (the same clips);
- **R5-style lexicon:** the strict review's R5 words (the recogniser's reliable glosses that occur as words in the
  How2Sign training sentences), two samples each at the generator's own duration, read by the recogniser: top-5.
Writes one JSON report; comparing checkpoints at the same setting isolates the prior.

Round 14 (docs/NEXT_PHASE_PLAN.md, "Sem-Lex un-mirrored"), both additions only, the ``lexicon`` block unchanged:
- ``--mirrored-read``: every generated word state is also read mirrored (``pose.mirror.mirror_state``; each channel's
  availability is checked to move with it), reported as ``lexicon_mirrored`` with the same fields. The recogniser
  learned most of its vocabulary from mirrored webcam video, so it reads mirrored signs better.
- ``--word-split-key``: a JSON partitioning the scored words into ``covered`` and ``uncovered`` (R5: one-word train
  records in round10-mixed-v1's isolated corpora, 108 and 42); ``lexicon_split`` gives each subset's top-5 for every
  read (``lexicon``, and ``lexicon_mirrored`` with ``--mirrored-read``).

Recogniser v3 (docs/NEXT_PHASE_PLAN.md, "Instrument finding and the recogniser v3 plan"), an addition only:
``--word-list-key`` / ``--word-list-sha256`` score exactly a pinned word list (``pinned_words``; the round-14 R5 split
``round14/r5-word-split.json`` gives the 150 R5 words), so a new instrument's own reliable glosses cannot change the
words; the report records the list's key and SHA-256 under ``word_list``. Without it the list is derived as before.

Round 17 (docs/NEXT_PHASE_PLAN.md, "Round 17 design"), additions only: with ``--plan-prior-key`` and
``--plan-tokenizer-key`` (each with its SHA-256, checked against the detail checkpoint's own pins), ``--checkpoint-key``
is a plan-conditioned detail prior and every sample is two-stage: plan prior, then detail prior, then the tokenizer's
decoder (``--plan-temperature``/``--plan-guidance`` for the plan sampler; by default the detail's). The report records
the pins under ``plan``. ``--oracle-plan`` (diagnostic) also generates the meaning half from each reference clip's own
plan: ``meaning_oracle_plan`` and the paired gap ``oracle_gap`` (oracle minus generated), the arm's halves unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

MIX = "acquisition/2026-10-17-v1/youtube-asl-v1/round3c-mixed-v1/manifest.json"
MIX_SHA = "004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd"


def validation_sentences(manifest, clips: int):
    """The R2b-style sample: ``clips`` How2Sign validation records of the three validation signer groups, fixed seed."""
    val = [r for r in manifest["records"] if r.get("corpus") == "how2sign-two-view-text-v1" and r.get("partition") == "val"
           and r.get("signer_component") in {"c01", "c03", "c05"} and r.get("text")]
    return random.Random(0).sample(val, min(clips, len(val)))


def postprocess(s):
    """The round-3e post-processing: head rotation and head translation smoothed over 0.1 s."""
    from signtranslator.generation.postprocess import smooth_rotation_channel, smooth_vector_channel
    s = smooth_rotation_channel(s, "head_rotation", .1)
    return smooth_vector_channel(s, "head_translation", .1) if "head_translation" in s.channels else s


SPLIT_SUBSETS = ("covered", "uncovered")
SIDE_CHANNEL = {"left_hand": "right_hand", "right_hand": "left_hand", "left_gaze": "right_gaze", "right_gaze": "left_gaze"}


def available_channels(state) -> list[str]:
    """The channels with at least one valid entry (what a manifest's ``available_channels`` records)."""
    return sorted(n for n, c in state.channels.items() if c.valid.any())


def check_mirrored_availability(state, mirrored) -> None:
    """Raise unless every channel's validity and observation masks in ``mirrored`` are its counterpart's in ``state``
    (left and right channels swapped, side-labelled entries swapped within a channel), so availability is mirrored
    with the values: the defect of the mirrored ASL Citizen states, whose manifests kept the unmirrored
    ``available_channels`` and so masked the dominant hand, cannot recur here."""
    from signtranslator.pose.mirror import swap_side
    for name, c in mirrored.channels.items():
        src = state.channels[SIDE_CHANNEL.get(name, name)]
        if not src.valid.any():
            if c.valid.any() or c.observed.any():
                raise ValueError(f"mirrored {name}: available, but its counterpart is not")
            continue
        labels = list(src.labels)
        order = [labels.index(swap_side(l)) if swap_side(l) in labels else labels.index(l) for l in c.labels]
        if not (np.array_equal(c.valid, src.valid[:, order]) and np.array_equal(c.observed, src.observed[:, order])):
            raise ValueError(f"mirrored {name}: availability differs from its counterpart's")
    want = sorted(SIDE_CHANNEL.get(n, n) for n in available_channels(state))
    if available_channels(mirrored) != want:
        raise ValueError(f"mirrored channels {available_channels(mirrored)} are not the mirror of {want}")


def mirrored_read(state):
    """The state as the mirrored signer, for the recogniser (sample id kept; availability checked)."""
    from signtranslator.pose.mirror import mirror_state
    out = mirror_state(state, sample_id=state.sample_id)
    check_mirrored_availability(state, out)
    return out


def top5_hits(recogniser, states, y, device):
    """Per-sample top-5 hits (0/1) of the recogniser against the labels ``y``."""
    from validate_sign_recognizer import logits_of
    return (logits_of(recogniser, states, device).topk(5, dim=1).indices == y[:, None]).any(1).float().numpy()


def word_split(payload: bytes, words) -> dict:
    """{subset: the scored words in it, in scoring order} from a split JSON; the subsets must partition ``words``."""
    split = json.loads(payload)
    if not isinstance(split, dict):
        raise ValueError("word split: expected a JSON object")
    lists = {k: split.get(k) for k in SPLIT_SUBSETS}
    for k, v in lists.items():
        if not isinstance(v, list) or not all(isinstance(w, str) for w in v) or len(set(v)) != len(v):
            raise ValueError(f"word split: {k!r} must be a list of distinct words")
    if set(lists["covered"]) & set(lists["uncovered"]):
        raise ValueError("word split: a word is both covered and uncovered")
    union = set(lists["covered"]) | set(lists["uncovered"])
    if union != set(words):
        raise ValueError(f"word split: the subsets must partition the scored words "
                         f"({len(set(words) - union)} scored words unassigned, {len(union - set(words))} not scored)")
    return {k: [w for w in words if w in set(v)] for k, v in lists.items()}


def pinned_words(payload: bytes, index, expected: int) -> list[str]:
    """The pinned word list (``--word-list-key``): a JSON list, an object with ``words`` (both in their own order), or
    a covered/uncovered split (their sorted union, the R5 order). Every word must be a label of the recogniser and
    distinct, and the list must hold ``--words`` words, so nothing is silently dropped or added."""
    data = json.loads(payload)
    if isinstance(data, list):
        words = data
    elif isinstance(data, dict) and "words" in data:
        words = data["words"]
    elif isinstance(data, dict) and all(k in data for k in SPLIT_SUBSETS):
        words = sorted(set(data["covered"]) | set(data["uncovered"]))
    else:
        raise ValueError("word list: expected a JSON list, an object with 'words', or a covered/uncovered split")
    if not all(isinstance(w, str) for w in words) or len(set(words)) != len(words):
        raise ValueError("word list: the words must be distinct strings")
    missing = [w for w in words if w not in index]
    if missing:
        raise ValueError(f"word list: {len(missing)} words are not labels of the recogniser (e.g. {missing[:5]})")
    if len(words) != expected:
        raise ValueError(f"word list: {len(words)} words, but --words is {expected}")
    return list(words)


def split_lexicon(words, hits, subsets: dict) -> dict:
    """{subset: dict(words, samples, generated_top5)} of one read; ``hits`` holds two samples per word, in order."""
    out = {}
    for k, members in subsets.items():
        rows = [2 * words.index(w) + s for w in members for s in range(2)]
        out[k] = dict(words=len(members), samples=len(rows),
                      generated_top5=round(float(hits[rows].mean()), 4) if rows else None)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", required=True)
    p.add_argument("--mix-key", default=MIX, help="the manifest the prior was trained on (its How2Sign validation and "
                   "training records are the same in every mix that includes how2sign-two-view-text-v1)")
    p.add_argument("--mix-sha256", default=MIX_SHA)
    p.add_argument("--part-temperature", default="body=1.1,left_hand=1.1,right_hand=1.1,face=1.0")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--clips", type=int, default=300)
    p.add_argument("--vocab-mix-key", help="the manifest whose How2Sign training sentences define the word list "
                   "(default: --mix-key). Round 13 amendment 1: an arm trained on a cleaned manifest is scored on the "
                   "words of the manifest its comparison reports used, so every report scores the same words")
    p.add_argument("--vocab-mix-sha256")
    p.add_argument("--meaning-from-key", help="reuse the meaning half of an earlier report of the same checkpoint and "
                   "settings (checked) instead of recomputing it; the lexicon half is computed afresh")
    p.add_argument("--words", type=int, default=150)
    p.add_argument("--word-set", choices=("r5", "dev"), default="r5",
                   help="r5: the strict review's words (the first --words reliable glosses); dev: the next --words, "
                        "disjoint from R5, for tuning sampling without touching the test words")
    p.add_argument("--device", default="cuda")
    p.add_argument("--recognizer-run", default="sign-recognizer-v2",
                   help="the lexicon instrument; its validation.json gives the reliable glosses (round 5: v3 if validated)")
    p.add_argument("--mirrored-read", action="store_true",
                   help="round 14: also read every generated word state mirrored (pose.mirror.mirror_state); adds "
                        "lexicon_mirrored, the lexicon block is unchanged")
    p.add_argument("--word-split-key", help="round 14: JSON partitioning the scored words into 'covered' and "
                   "'uncovered'; adds lexicon_split (each subset's top-5 for every read)")
    p.add_argument("--word-split-sha256", help="the split JSON's SHA-256 (checked when given)")
    p.add_argument("--word-list-key", help="recogniser v3: score exactly this pinned word list (a JSON list, an object "
                   "with 'words', or a covered/uncovered split, read as their sorted union) instead of deriving it from "
                   "the recogniser's reliable glosses, so a new instrument cannot change the R5 words; needs "
                   "--word-list-sha256")
    p.add_argument("--word-list-sha256")
    p.add_argument("--plan-prior-key", help="round 17: --checkpoint-key is a plan-conditioned detail prior; this plan "
                   "prior samples each text's plan first (plan, then detail, then the tokenizer's decoder); needs "
                   "--plan-prior-sha256, --plan-tokenizer-key and --plan-tokenizer-sha256, all checked against the "
                   "detail checkpoint")
    p.add_argument("--plan-prior-sha256")
    p.add_argument("--plan-tokenizer-key", help="round 17: the plan tokenizer both priors were trained with (pinned)")
    p.add_argument("--plan-tokenizer-sha256")
    p.add_argument("--plan-temperature", help="round 17: the plan sampler's part temperatures (default: "
                   "--part-temperature)")
    p.add_argument("--plan-guidance", type=float, help="round 17: the plan sampler's guidance (default: --guidance)")
    p.add_argument("--oracle-plan", action="store_true",
                   help="round 17 diagnostic: the meaning half is also generated from each reference clip's own plan "
                        "(the plan tokenizer's codes of the real clip, the same seeds), read as meaning_oracle_plan, with "
                        "the paired BLEU gap oracle minus generated (oracle_gap). The arm's own meaning and lexicon halves "
                        "are unchanged. An upper bound on what the plan route can carry, never an arm's result")
    p.add_argument("--oracle-only", action="store_true",
                   help="round 17 (declared sequencing): with --oracle-plan, read only the oracle-plan generations (plus "
                        "real and blank) and stop: no plan-prior generation, no lexicon. The oracle precondition "
                        "(compare_pilot_lexicon.py --oracle-precondition against R16C) is judged on this report before "
                        "the arm's pilot runs")
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    if args.oracle_only and not args.oracle_plan:
        raise SystemExit("--oracle-only needs --oracle-plan")
    plan_args = (args.plan_prior_key, args.plan_prior_sha256, args.plan_tokenizer_key, args.plan_tokenizer_sha256)
    if any(plan_args) and not all(plan_args):
        raise SystemExit("--plan-prior-key, --plan-prior-sha256, --plan-tokenizer-key and --plan-tokenizer-sha256 go "
                         "together")
    if (args.plan_temperature or args.plan_guidance is not None or args.oracle_plan) and not args.plan_prior_key:
        raise SystemExit("--plan-temperature, --plan-guidance and --oracle-plan need --plan-prior-key")
    if args.oracle_plan and args.meaning_from_key:
        raise SystemExit("--oracle-plan recomputes the meaning half; it cannot reuse --meaning-from-key")
    import warnings
    warnings.simplefilter("ignore")
    import numpy as np
    import torch
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.pose.resample import resample_motion
    from evaluate_text_to_motion import at_rate, blanked, load_generator, load_state
    from score_back_translation import load_translator, translate
    from strict_prior_review import bleu_ci, separated
    from validate_sign_recognizer import load_model, logits_of
    store = R2Store(os.environ)
    if args.word_split_sha256 and not args.word_split_key:
        raise SystemExit("--word-split-sha256 needs --word-split-key")
    if bool(args.word_list_key) != bool(args.word_list_sha256):
        raise SystemExit("--word-list-key and --word-list-sha256 go together (the list is pinned)")
    if args.word_list_key and args.word_set != "r5":
        raise SystemExit("--word-list-key pins the R5 words; --word-set dev cannot be combined with it")
    if args.word_list_key:                             # pinned before any generation
        list_payload = store.get(args.word_list_key)
        list_sha = hashlib.sha256(list_payload).hexdigest()
        if list_sha != args.word_list_sha256:
            raise SystemExit(f"--word-list-key: SHA-256 {list_sha}, expected {args.word_list_sha256}")
    if args.word_split_key:                            # round 14: fetched and checked before any generation
        split_payload = store.get(args.word_split_key)
        split_sha = hashlib.sha256(split_payload).hexdigest()
        if args.word_split_sha256 and split_sha != args.word_split_sha256:
            raise SystemExit(f"--word-split-key: SHA-256 {split_sha}, expected {args.word_split_sha256}")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    temperature = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature.split(","))}
    plan = None
    if args.plan_prior_key:                            # round 17: two-stage generation, every checkpoint pinned
        plan = dict(prior_key=args.plan_prior_key, prior_sha256=args.plan_prior_sha256,
                    tokenizer_key=args.plan_tokenizer_key, tokenizer_sha256=args.plan_tokenizer_sha256,
                    guidance=args.plan_guidance,
                    temperature=({k: float(v) for k, v in (x.split("=") for x in args.plan_temperature.split(","))}
                                 if args.plan_temperature else None))
    _, _, sample = load_generator(store, args.checkpoint_key, args.mix_sha256, args.device, temperature=temperature,
                                  **({"plan": plan} if plan else {}))
    post = postprocess
    # lexicon word list (round 14: drawn up before the meaning half, so a bad word split fails before any generation)
    validation = (None if args.word_list_key else      # a pinned list needs no reliable-gloss list
                  json.loads(store.get(f"training-runs/{args.recognizer_run}/validation.json")))
    recogniser, labels, _ = load_model(store, args.recognizer_run, args.device)
    index = {g: i for i, g in enumerate(labels)}
    vocab_manifest = manifest
    if args.vocab_mix_key:
        if not args.vocab_mix_sha256:
            raise SystemExit("--vocab-mix-key needs --vocab-mix-sha256")
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "v.json"; path.write_bytes(store.get(args.vocab_mix_key) + b"\n")
            vocab_manifest = load_manifest(path, expected_sha256=args.vocab_mix_sha256)
    train = [r for r in vocab_manifest["records"] if r.get("corpus") == "how2sign-two-view-text-v1" and r.get("partition") == "train"]
    vocab = {w for r in train for w in re.findall(r"[a-z]+", r["text"].lower())}
    if args.word_list_key:
        words = pinned_words(list_payload, index, args.words)
    else:
        words = sorted(g for g in validation["per_gloss"]["reliable_glosses"] if g.isalpha() and g.lower() in vocab and g in index)
        words = words[:args.words] if args.word_set == "r5" else words[args.words:2 * args.words]
    subsets = word_split(split_payload, words) if args.word_split_key else None
    # meaning
    extra_meaning = {}
    if args.meaning_from_key:   # round 13 amendment 1: the lexicon re-run keeps the arm's own meaning half
        earlier = json.loads(store.get(args.meaning_from_key))
        same = dict(checkpoint_key=args.checkpoint_key, clips=args.clips, guidance=args.guidance, part_temperature=temperature,
                    plan=plan)                         # round 17: None for one-stage reports
        if {k: earlier.get(k) for k in same} != same:
            raise SystemExit(f"--meaning-from-key: the earlier report used other settings: {({k: earlier.get(k) for k in same}, same)}")
        meaning, texts = earlier["meaning"], earlier["meaning"]["references"]
        if texts != [r["text"] for r in validation_sentences(manifest, args.clips)]:
            raise SystemExit("--meaning-from-key: the earlier report translated other clips")
    else:
        val = validation_sentences(manifest, args.clips)
        refs = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), 24.) for r in val]
        texts = [r["text"] for r in val]
        seconds = [float(s.timestamps[-1] - s.timestamps[0]) + 1 / 24. for s in refs]
        gens = []
        if args.oracle_plan:                           # round 17 diagnostic: each clip's own plan
            from signtranslator.generation.plan_codes import load_plan_tokenizer, oracle_plans
            plan_vq, plan_tok, _ = load_plan_tokenizer(store.get(args.plan_tokenizer_key), args.plan_tokenizer_sha256,
                                                       device=args.device)
            plans, adjusted = oracle_plans(plan_vq, plan_tok, refs, seconds, 24.)
            plan["oracle"] = dict(clips=len(plans), length_adjusted=adjusted)
            print(json.dumps({"oracle_plan": plan["oracle"]}), flush=True)
        oracle_gens = []
        for i in range(0, len(texts), 32):
            if not args.oracle_only:
                gens += [post(o) for o in sample.batch(texts[i:i + 32], 24., seconds[i:i + 32], args.guidance, 1000 + i)]
            if args.oracle_plan:
                oracle_gens += [post(o) for o in sample.batch_with_plans(texts[i:i + 32], 24., seconds[i:i + 32],
                                                                         args.guidance, 1000 + i, plans[i:i + 32])]
        kp = json.loads(store.get("training-runs/back-translator-kp2d-uwb-ft-v1/best.json"))["checkpoint_key"]
        translator, kind = load_translator(store, kp, MIX_SHA)  # the translator's own manifest (round 3c)
        translator = translator.to(torch.device(args.device))
        meaning = {}
        reads = (("real", refs), ("blank", blanked(refs))) if args.oracle_only else \
            (("generated", gens), ("real", refs), ("blank", blanked(refs)))
        for name, states in reads:
            hyps = translate(translator, kind, states)
            point, ci = bleu_ci(hyps, texts)
            meaning[name] = dict(bleu=point, ci95=ci, hypotheses=list(hyps))     # kept for the paired BLEU test (round 7)
        meaning["references"] = list(texts)
        if not args.oracle_only:
            meaning["above_blank"] = separated(meaning["generated"]["ci95"], meaning["blank"]["ci95"])
        if args.oracle_plan:                           # round 17 (review fix 4): the oracle read and the paired gap
            from compare_pilot_lexicon import paired_bleu_difference
            hyps = translate(translator, kind, oracle_gens)
            point, ci = bleu_ci(hyps, texts)
            extra_meaning["meaning_oracle_plan"] = dict(bleu=point, ci95=ci, hypotheses=list(hyps))
            # the declared D17 gate "oracle-plan detail generations: wrist speed within [0.8, 1.5] x real", measured as
            # rounds 15-16 measured wrist speed relative to real (the back-translator's normalised 2D view)
            from compositional_baseline import motion_rows
            from sentence_exposure_diagnostic import mean_statistics, ratio_to
            ratio = ratio_to(mean_statistics(motion_rows(oracle_gens, refs, 24.)),
                             mean_statistics(motion_rows(refs, refs, 24.)))
            speed = ratio.get("wrist_speed")
            extra_meaning["oracle_motion_ratio_to_real"] = ratio
            extra_meaning["oracle_wrist_speed_gate"] = dict(
                wrist_speed_over_real=speed, band=[.8, 1.5],
                passes=bool(speed is not None and .8 <= speed <= 1.5))
            if not args.oracle_only:
                gap, (low, high) = paired_bleu_difference(hyps, meaning["generated"]["hypotheses"], texts)
                extra_meaning["oracle_gap"] = dict(difference=round(gap, 3), ci95=[round(low, 3), round(high, 3)],
                                                   meaning="oracle plan minus the plan prior's plan, paired over the clips")
            print(json.dumps({"meaning_oracle_plan": dict(bleu=point, ci95=ci),
                              "oracle_gap": extra_meaning.get("oracle_gap")}), flush=True)
        print(json.dumps({"meaning": {k: {f: v for f, v in m.items() if f != "hypotheses"} if isinstance(m, dict) else m
                                      for k, m in meaning.items() if k != "references"}}), flush=True)
    if args.oracle_only:                               # round 17: the oracle read alone, before the arm's pilot
        report = dict(checkpoint_key=args.checkpoint_key, part_temperature=temperature, guidance=args.guidance,
                      clips=len(texts), meaning=meaning, oracle_only=True, plan=dict(plan), **extra_meaning)
        store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True, default=float).encode())
        return 0
    # lexicon
    states, y = [], []
    for k, g in enumerate(words):
        for s in range(2):
            states.append(post(resample_motion(sample(g.lower(), 24., None, args.guidance, 50, 7000 + 10 * k + s), 24.)))
            y.append(index[g])
    y = torch.tensor(y)
    hits = (logits_of(recogniser, states, args.device).topk(5, dim=1).indices == y[:, None]).any(1).float().numpy()
    blank_hits = (logits_of(recogniser, blanked(states), args.device).topk(5, dim=1).indices == y[:, None]).any(1).float().numpy()
    lexicon = dict(words=len(words), word_set=args.word_set, samples=len(y), generated_top5=round(float(hits.mean()), 4),
                   blank_top5=round(float(blank_hits.mean()), 4), recognizer_run=args.recognizer_run,
                   per_word={g: int(hits[2 * k] + hits[2 * k + 1]) for k, g in enumerate(words)})
    print(json.dumps({"lexicon": lexicon}), flush=True)
    extra, reads = {}, {"lexicon": hits}              # round 14: additions only; the lexicon block above is unchanged
    if args.mirrored_read:
        mirrored = [mirrored_read(st) for st in states]
        m_hits, m_blank = top5_hits(recogniser, mirrored, y, args.device), top5_hits(recogniser, blanked(mirrored), y,
                                                                                       args.device)
        extra["lexicon_mirrored"] = dict(
            words=len(words), word_set=args.word_set, samples=len(y), generated_top5=round(float(m_hits.mean()), 4),
            blank_top5=round(float(m_blank.mean()), 4), recognizer_run=args.recognizer_run,
            per_word={g: int(m_hits[2 * k] + m_hits[2 * k + 1]) for k, g in enumerate(words)},
            mirror="signtranslator.pose.mirror.mirror_state",
            available_channels=sorted({n for st in mirrored for n in available_channels(st)}))
        reads["lexicon_mirrored"] = m_hits
        print(json.dumps({"lexicon_mirrored": extra["lexicon_mirrored"]}), flush=True)
    if args.word_split_key:
        extra["lexicon_split"] = dict(key=args.word_split_key, sha256=split_sha, words=subsets,
                                      **{read: split_lexicon(words, h, subsets) for read, h in reads.items()})
        print(json.dumps({"lexicon_split": {k: v for k, v in extra["lexicon_split"].items() if k != "words"}}), flush=True)
    report = dict(checkpoint_key=args.checkpoint_key, part_temperature=temperature, guidance=args.guidance,
                  clips=len(texts), meaning=meaning, lexicon=lexicon, **extra,
                  **({"vocab_mix_key": args.vocab_mix_key} if args.vocab_mix_key else {}),
                  **({"word_list": dict(key=args.word_list_key, sha256=list_sha, words=len(words))}
                     if args.word_list_key else {}),
                  **({"meaning_from_key": args.meaning_from_key} if args.meaning_from_key else {}),
                  **({"plan": {k: v for k, v in plan.items()}} if plan else {}), **extra_meaning)
    store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True, default=float).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
