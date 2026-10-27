"""Lexicon diagnostic (round 6 audit): can a prior sign words it was trained on, and at what length?

The R5 words gave about 0 top-5 in every arm, including after about three times more isolated-sign data. This asks a
narrower question on one checkpoint. For the ``--words`` most frequent one-word **training** texts (corpora
``--corpus``, training partition) whose uppercase spelling is one of the recogniser's labels, the recogniser reads:
- ``real``: up to two real training clips of the word (the reference: it should be high);
- ``generated_own``: two samples at the prior's own predicted duration (as R5 samples);
- ``generated_real_duration``: two samples at the median duration of the word's real training clips.
Durations (predicted vs real) are reported too. If training words fail as well, the fault is in learning or sampling
one-word conditioning, not in vocabulary coverage. Writes one JSON report.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", required=True)
    p.add_argument("--mix-key", required=True)
    p.add_argument("--mix-sha256", required=True)
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--words", type=int, default=60)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--part-temperature", default="body=1.1,left_hand=1.1,right_hand=1.1,face=1.0")
    p.add_argument("--recognizer-run", default="sign-recognizer-v2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--oracle-loci", action="store_true", help="also generate with the loci measured from a real training "
                   "clip of the word (state_loci, as training sees them) instead of the text-planned ones")
    p.add_argument("--part-temperature-alt", help="also generate at this per-part temperature (own duration)")
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.generation.postprocess import smooth_rotation_channel, smooth_vector_channel
    from signtranslator.pose.resample import resample_motion
    from evaluate_text_to_motion import at_rate, load_generator, load_state
    from validate_sign_recognizer import load_model, logits_of
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    temperature = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature.split(","))}
    _, _, sample = load_generator(store, args.checkpoint_key, args.mix_sha256, args.device, temperature=temperature)
    alt_sample = None
    if args.part_temperature_alt:
        alt = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature_alt.split(","))}
        _, _, alt_sample = load_generator(store, args.checkpoint_key, args.mix_sha256, args.device, temperature=alt)
    recogniser, labels, _ = load_model(store, args.recognizer_run, args.device)
    index = {g: i for i, g in enumerate(labels)}

    def post(s):
        s = smooth_rotation_channel(s, "head_rotation", .1)
        return smooth_vector_channel(s, "head_translation", .1) if "head_translation" in s.channels else s
    by_word = collections.defaultdict(list)
    for r in manifest["records"]:
        if r.get("corpus") in args.corpus and r.get("partition") == "train" and r.get("text"):
            label = r["text"].upper().replace(" ", "")
            if label in index:
                by_word[label].append(r)
    words = sorted(by_word, key=lambda w: (-len(by_word[w]), w))[:args.words]
    from signtranslator.grammar_motion.loci_conditioning import state_loci
    real_states, real_y, gen_own, gen_dur, gen_y, rows = [], [], [], [], [], []
    gen_oracle, gen_alt, nonempty = [], [], 0
    for k, w in enumerate(words):
        clips = by_word[w]
        secs = [float(c["continuous"]["seconds"]) for c in clips]
        median = float(np.median(secs))
        for c in clips[:2]:
            real_states.append(at_rate(load_state(store, c["continuous"]["state_key"], c["continuous"]["state_sha256"]), 24.))
            real_y.append(index[w])
        own = [post(resample_motion(sample(w.lower(), 24., None, args.guidance, 50, 9000 + 10 * k + s), 24.)) for s in range(2)]
        fixed = [post(resample_motion(sample(w.lower(), 24., median, args.guidance, 50, 9500 + 10 * k + s), 24.)) for s in range(2)]
        gen_own += own; gen_dur += fixed; gen_y += [index[w]] * 2
        if args.oracle_loci:
            first = load_state(store, clips[0]["continuous"]["state_key"], clips[0]["continuous"]["state_sha256"])
            slots = state_loci(first); nonempty += int(np.abs(slots).sum() > 0)
            gen_oracle += [post(resample_motion(sample(w.lower(), 24., median, args.guidance, 50, 9700 + 10 * k + s,
                                                       loci=slots), 24.)) for s in range(2)]
        if alt_sample is not None:
            gen_alt += [post(resample_motion(alt_sample(w.lower(), 24., None, args.guidance, 50, 9900 + 10 * k + s), 24.))
                        for s in range(2)]
        rows.append(dict(word=w, training_clips=len(clips), real_median_seconds=round(median, 3),
                         generated_seconds=[round(float(s.timestamps[-1] - s.timestamps[0]), 3) for s in own]))

    def top5(states, y):
        y = torch.tensor(y)
        hits = (logits_of(recogniser, states, args.device).topk(5, dim=1).indices == y[:, None]).any(1).float().numpy()
        return hits
    real_hits, own_hits, dur_hits = top5(real_states, real_y), top5(gen_own, gen_y), top5(gen_dur, gen_y)
    for k, r in enumerate(rows):
        r["generated_own_hits"] = int(own_hits[2 * k] + own_hits[2 * k + 1])
        r["generated_real_duration_hits"] = int(dur_hits[2 * k] + dur_hits[2 * k + 1])
    gen_s = [s for r in rows for s in r["generated_seconds"]]
    report = dict(checkpoint_key=args.checkpoint_key, corpora=args.corpus, words=len(words), guidance=args.guidance,
                  recognizer_run=args.recognizer_run,
                  real_top5=round(float(real_hits.mean()), 4), generated_own_top5=round(float(own_hits.mean()), 4),
                  generated_real_duration_top5=round(float(dur_hits.mean()), 4),
                  generated_oracle_loci_top5=(round(float(top5(gen_oracle, gen_y).mean()), 4) if gen_oracle else None),
                  oracle_loci_nonempty_words=(nonempty if args.oracle_loci else None),
                  generated_alt_temperature_top5=(round(float(top5(gen_alt, gen_y).mean()), 4) if gen_alt else None),
                  alt_temperature=args.part_temperature_alt,
                  real_median_seconds=round(float(np.median([r["real_median_seconds"] for r in rows])), 3),
                  generated_median_seconds=round(float(np.median(gen_s)), 3), per_word=rows)
    print(json.dumps({k: v for k, v in report.items() if k != "per_word"}), flush=True)
    store.put(args.output_key, json.dumps(report, indent=2).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
