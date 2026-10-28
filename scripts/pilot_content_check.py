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
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile

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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", required=True)
    p.add_argument("--mix-key", default=MIX, help="the manifest the prior was trained on (its How2Sign validation and "
                   "training records are the same in every mix that includes how2sign-two-view-text-v1)")
    p.add_argument("--mix-sha256", default=MIX_SHA)
    p.add_argument("--part-temperature", default="body=1.1,left_hand=1.1,right_hand=1.1,face=1.0")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--clips", type=int, default=300)
    p.add_argument("--words", type=int, default=150)
    p.add_argument("--word-set", choices=("r5", "dev"), default="r5",
                   help="r5: the strict review's words (the first --words reliable glosses); dev: the next --words, "
                        "disjoint from R5, for tuning sampling without touching the test words")
    p.add_argument("--device", default="cuda")
    p.add_argument("--recognizer-run", default="sign-recognizer-v2",
                   help="the lexicon instrument; its validation.json gives the reliable glosses (round 5: v3 if validated)")
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
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
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    temperature = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature.split(","))}
    _, _, sample = load_generator(store, args.checkpoint_key, args.mix_sha256, args.device, temperature=temperature)
    post = postprocess
    # meaning
    val = validation_sentences(manifest, args.clips)
    refs = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), 24.) for r in val]
    texts = [r["text"] for r in val]
    seconds = [float(s.timestamps[-1] - s.timestamps[0]) + 1 / 24. for s in refs]
    gens = []
    for i in range(0, len(texts), 32):
        gens += [post(o) for o in sample.batch(texts[i:i + 32], 24., seconds[i:i + 32], args.guidance, 1000 + i)]
    kp = json.loads(store.get("training-runs/back-translator-kp2d-uwb-ft-v1/best.json"))["checkpoint_key"]
    translator, kind = load_translator(store, kp, MIX_SHA)  # the translator's own manifest (round 3c)
    translator = translator.to(torch.device(args.device))
    meaning = {}
    for name, states in (("generated", gens), ("real", refs), ("blank", blanked(refs))):
        hyps = translate(translator, kind, states)
        point, ci = bleu_ci(hyps, texts)
        meaning[name] = dict(bleu=point, ci95=ci, hypotheses=list(hyps))     # kept for the paired BLEU test (round 7)
    meaning["references"] = list(texts)
    meaning["above_blank"] = separated(meaning["generated"]["ci95"], meaning["blank"]["ci95"])
    print(json.dumps({"meaning": {k: {f: v for f, v in m.items() if f != "hypotheses"} if isinstance(m, dict) else m
                                  for k, m in meaning.items() if k != "references"}}), flush=True)
    # lexicon
    validation = json.loads(store.get(f"training-runs/{args.recognizer_run}/validation.json"))
    recogniser, labels, _ = load_model(store, args.recognizer_run, args.device)
    index = {g: i for i, g in enumerate(labels)}
    train = [r for r in manifest["records"] if r.get("corpus") == "how2sign-two-view-text-v1" and r.get("partition") == "train"]
    vocab = {w for r in train for w in re.findall(r"[a-z]+", r["text"].lower())}
    words = sorted(g for g in validation["per_gloss"]["reliable_glosses"] if g.isalpha() and g.lower() in vocab and g in index)
    words = words[:args.words] if args.word_set == "r5" else words[args.words:2 * args.words]
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
    report = dict(checkpoint_key=args.checkpoint_key, part_temperature=temperature, guidance=args.guidance,
                  clips=len(texts), meaning=meaning, lexicon=lexicon)
    store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True, default=float).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
