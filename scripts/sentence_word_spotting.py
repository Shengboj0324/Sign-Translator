"""Fix 5: sentence word spotting - a sentence-level content instrument stronger than back-translation BLEU.

The isolated-sign recogniser (``--recognizer-run``; labels are uppercase ASL Citizen glosses) is slid over a motion
state: windows of ``--windows`` seconds (default 0.6, 0.9, 1.2) every ``--hop`` seconds (0.1), each read for its top
``--top-k`` (5) labels. A sentence's words map to labels by upper-casing (``--vocab all``: every label; ``reliable``:
the recogniser's reliable glosses from its validation.json); words with no label are skipped, repeats count once.
- **recall**: the fraction of the sentence's mappable words found in the top-k of some window;
- **order agreement**: each detected word is placed at the centre of the window where it scores highest; over the
  pairs of detected words, the fraction placed in the sentence's order (ties half; chance 0.5; needs two detections).

Forms, on the same ``--clips`` How2Sign validation sentences as ``pilot_content_check.py`` (its selection code, seed and
signer groups): **real** clips; **blank** (the same clips with every channel removed, ``evaluate_text_to_motion.
blanked``); **generated** (with ``--checkpoint-key``: sampled exactly as the pilot samples, same part temperatures,
guidance, batches of 32, seeds 1000 + i, the reference clip's duration, round-3e post-processing). Each form is also
scored against the *next* sentence's words (``*_mismatched``, the same windows): the chance level of spotting a
sentence's words in unrelated signing, which the blank form cannot give (every blank window is the same input).
Per form: means with bootstrap 95% intervals over sentences; per sentence: the rows. The JSON report is written to
``--output-key`` only when given (otherwise printed summary only).

Validation, 2026-10-27 (CPU, real and blank only, 300 sentences, ``sign-recognizer-v2``, top-5): ``--vocab all`` (2731
labels, median 7 mappable words): recall real 0.074 [0.063, 0.086], real against the next sentence's words 0.051
[0.039, 0.063], blank 0.002; order real 0.67 (58 sentences with two detections) vs mismatched 0.43. ``--vocab
reliable`` (474 glosses, 199 sentences with a word, median 1): real 0.080 [0.054, 0.112], mismatched 0.024 [0.009,
0.041], blank 0.005. Real is above blank *and* above chance, though only just with every label; on 30 sentences the
real-vs-mismatched gap is not resolved (0.037 vs 0.040). A generated form should be read against its own mismatched
control, not only against blank.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 24.
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def sentence_labels(text: str, labels) -> list[str]:
    """The text's words that are recogniser labels (upper-cased), first occurrence each, in text order."""
    out = []
    for w in WORD.findall(text or ""):
        g = w.upper()
        if g in labels and g not in out:
            out.append(g)
    return out


def window_bounds(frames: int, fps: float, windows=(.6, .9, 1.2), hop: float = .1):
    """[(start, stop, centre seconds)] frame windows of each length every ``hop``; a clip shorter than a window length
    gets one window of its whole length for it (duplicates removed)."""
    step = max(int(round(hop * fps)), 1)
    out = []
    for w in windows:
        n = min(max(int(round(w * fps)), 1), frames)
        for s in range(0, frames - n + 1, step):
            out.append((s, s + n, (s + n / 2) / fps))
    return sorted(set(out))


def detections(top_ids, top_scores, centres, words, index):
    """{word: centre seconds of its best-scoring window} for the ``words`` found in some window's top-k.
    ``top_ids``, ``top_scores``: (W, k) label indices and scores; ``centres``: (W,)."""
    top_ids, top_scores, centres = np.asarray(top_ids), np.asarray(top_scores, dtype=float), np.asarray(centres)
    out = {}
    for w in words:
        hit = top_ids == index[w]
        if hit.any():
            rows = np.flatnonzero(hit.any(1))
            best = rows[np.argmax(np.where(hit[rows], top_scores[rows], -np.inf).max(1))]
            out[w] = float(centres[best])
    return out


def recall(words, found) -> float | None:
    return None if not words else sum(w in found for w in words) / len(words)


def order_agreement(words, found) -> float | None:
    """Fraction of detected word pairs (in text order) whose detection times are in that order; ties count half."""
    times = [found[w] for w in words if w in found]
    if len(times) < 2:
        return None
    pairs = list(itertools.combinations(times, 2))
    return sum(1. if a < b else .5 if a == b else 0. for a, b in pairs) / len(pairs)


def summarise(values, seed: int = 0):
    """{mean, ci95, n} over the sentences with a value (None skipped)."""
    from strict_prior_review import bootstrap_mean
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return dict(mean=None, ci95=None, n=0)
    return dict(mean=round(float(v.mean()), 4), ci95=bootstrap_mean(v, seed=seed), n=int(len(v)))


def window_predictions(model, state, device, *, windows, hop, k, batch: int = 256):
    """(top_ids (W, k), top_probabilities (W, k), centres (W,)) of the recogniser over ``state``'s windows."""
    import torch
    from validate_sign_recognizer import cut, logits_of
    bounds = window_bounds(len(state.timestamps), FPS, windows, hop)
    logits = logits_of(model, [cut(state, a, b) for a, b, _ in bounds], device, batch=batch)
    p, ids = torch.softmax(logits, 1).topk(k, dim=1)
    return ids.numpy(), p.numpy(), np.array([c for _, _, c in bounds])


def score_form(preds, word_lists, index):
    """Per-sentence (detections, recall, order) of each prediction against its word list."""
    rows = []
    for (ids, p, c), words in zip(preds, word_lists):
        found = detections(ids, p, c, words, index)
        rows.append(dict(found=found, recall=recall(words, found), order=order_agreement(words, found)))
    return rows


def main(argv=None) -> int:
    from pilot_content_check import MIX, MIX_SHA
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", help="a prior to sample the generated form from (omit: real and blank only)")
    p.add_argument("--mix-key", default=MIX)
    p.add_argument("--mix-sha256", default=MIX_SHA)
    p.add_argument("--part-temperature", default="body=1.1,left_hand=1.1,right_hand=1.1,face=1.0")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--clips", type=int, default=300)
    p.add_argument("--recognizer-run", default="sign-recognizer-v2")
    p.add_argument("--vocab", choices=("all", "reliable"), default="all")
    p.add_argument("--windows", default="0.6,0.9,1.2", help="window lengths, seconds")
    p.add_argument("--hop", type=float, default=.1)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-key", help="write the JSON report here (omit: print the summary only)")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from evaluate_text_to_motion import at_rate, blanked, load_generator, load_state
    from pilot_content_check import postprocess, validation_sentences
    from strict_prior_review import separated
    from validate_sign_recognizer import load_model
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    store = R2Store(os.environ)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit("refusing to overwrite an existing report")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    val = validation_sentences(manifest, args.clips)
    refs = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), FPS) for r in val]
    texts = [r["text"] for r in val]
    model, labels, best = load_model(store, args.recognizer_run, args.device)
    index = {g: i for i, g in enumerate(labels)}
    vocab = set(labels)
    if args.vocab == "reliable":
        validation = json.loads(store.get(f"training-runs/{args.recognizer_run}/validation.json"))
        vocab = set(validation["per_gloss"]["reliable_glosses"]) & vocab
    words = [sentence_labels(t, vocab) for t in texts]
    mismatched = words[1:] + words[:1]                     # each clip against the next sentence's words
    forms = {"real": refs, "blank": blanked(refs)}
    if args.checkpoint_key:
        temperature = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature.split(","))}
        _, _, sample = load_generator(store, args.checkpoint_key, args.mix_sha256, args.device, temperature=temperature)
        seconds = [float(s.timestamps[-1] - s.timestamps[0]) + 1 / FPS for s in refs]
        gens = []
        for i in range(0, len(texts), 32):
            gens += [postprocess(o) for o in sample.batch(texts[i:i + 32], FPS, seconds[i:i + 32], args.guidance, 1000 + i)]
        forms["generated"] = [at_rate(g, FPS) for g in gens]
    windows = tuple(float(x) for x in args.windows.split(","))
    scores, rows = {}, [dict(sign=r["sign"], text=t, words=w) for r, t, w in zip(val, texts, words)]
    for name, states in forms.items():
        preds = [window_predictions(model, s, args.device, windows=windows, hop=args.hop, k=args.top_k) for s in states]
        for suffix, lists in (("", words), ("_mismatched", mismatched)):
            per = score_form(preds, lists, index)
            scores[name + suffix] = dict(recall=summarise([x["recall"] for x in per]),
                                         order=summarise([x["order"] for x in per]))
            for row, x in zip(rows, per):
                row[name + suffix] = x
    sep = {}
    for a, b in (("real", "blank"), ("real", "real_mismatched"), ("generated", "blank"),
                 ("generated", "generated_mismatched")):
        if a in scores and b in scores and scores[a]["recall"]["ci95"] and scores[b]["recall"]["ci95"]:
            sep[f"{a}_above_{b}"] = separated(scores[a]["recall"]["ci95"], scores[b]["recall"]["ci95"])
    mappable = [len(w) for w in words]
    report = dict(recognizer_run=args.recognizer_run, recognizer_checkpoint=best["checkpoint_key"], vocab=args.vocab,
                  vocab_size=len(vocab), windows=list(windows), hop=args.hop, top_k=args.top_k, clips=len(texts),
                  sentences_with_words=int(sum(m > 0 for m in mappable)),
                  median_mappable_words=float(np.median(mappable)) if mappable else 0.,
                  checkpoint_key=args.checkpoint_key, mix_sha256=args.mix_sha256,
                  sampling=dict(part_temperature=args.part_temperature, guidance=args.guidance) if args.checkpoint_key
                  else None, scores=scores, recall_separated=sep)
    print(json.dumps(report), flush=True)
    if args.output_key:
        store.put(args.output_key, json.dumps(dict(report, rows=rows), indent=1, sort_keys=True).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
