"""Sign-level evaluation of a plan-conditioned generator with a validated sign recogniser (plan items 2 and 3).

For every gloss on the recogniser's reliable list (``validate_sign_recognizer.py``: top-1 >= 0.8 on real validation
clips) that the generator's plan vocabulary can express, generate ``--samples`` motions from the one-sign plan line
(the gloss's Sign Bank form) and ask the recogniser which sign it sees. Reported side by side through the same
recogniser, at the same frame rate:

- ``generated``: top-1 / top-5 of the intended gloss on generated motion;
- ``real``: the same on the real ASL Citizen validation clips of those glosses (the instrument's ceiling on them);
- ``blank``: the generated clips with every channel removed (what duration alone gives).

The verdict reads only relative numbers: generated between blank and real. Refuses to run unless the recogniser's
validation report says ``usable_as_instrument``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.generation.sign_recognizer import top_k_accuracy  # noqa: E402
from signtranslator.planning.plan_text import plan_to_text  # noqa: E402
from signtranslator.pose.resample import resample_motion  # noqa: E402
from evaluate_text_to_motion import blanked, load_generator  # noqa: E402
from train_text_to_motion import make_store  # noqa: E402
from validate_sign_recognizer import load_model, logits_of  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--recognizer-run", required=True)
    parser.add_argument("--recognizer-validation-key", required=True, help="validate_sign_recognizer.py report")
    parser.add_argument("--generator-key", required=True)
    parser.add_argument("--generator-manifest-sha256", required=True)
    parser.add_argument("--plan-glosses-key", required=True,
                        help="JSON list of the glosses the generator saw in training (one-sign plans use only these)")
    parser.add_argument("--citizen-manifest-key", required=True)
    parser.add_argument("--citizen-manifest-sha256", required=True)
    parser.add_argument("--signbank-to-citizen-key", default="acquisition/2026-10-17-v1/derived/signbank-v1/asl-citizen-gloss-map.json")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=1.2)
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--guidance", type=float, default=2.)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output-key", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    report = json.loads(store.get(args.recognizer_validation_key))
    if not report["verdict"]["usable_as_instrument"]:
        raise SystemExit("the recogniser failed its instrument validation; it must not score generated motion")
    model, labels, _ = load_model(store, args.recognizer_run, args.device)
    index = {g: i for i, g in enumerate(labels)}
    plan_glosses = set(json.loads(store.get(args.plan_glosses_key)))
    # Sign Bank (plan) gloss -> ASL Citizen label, exact matches only
    to_label = {}
    for g, m in json.loads(store.get(args.signbank_to_citizen_key))["mapping"].items():
        if m["method"] == "exact":
            for c in m["candidates"]:
                to_label.setdefault(c, set()).add(g)
    reliable = set(report["per_gloss"]["reliable_glosses"])
    targets = sorted((sb, next(iter(cz))) for sb, cz in to_label.items()
                     if len(cz) == 1 and next(iter(cz)) in reliable and sb in plan_glosses)[:args.limit]
    if not targets:
        raise SystemExit("no reliable gloss the generator can express")
    _, _, sample = load_generator(store, args.generator_key, args.generator_manifest_sha256, args.device)
    states, y = [], []
    for k, (signbank, citizen) in enumerate(targets):
        line = plan_to_text([signbank], [(0, 0)], [[0] * 7])
        for s in range(args.samples):
            states.append(resample_motion(sample(line, args.fps, args.seconds, args.guidance, 50, 1000 * k + s), args.fps))
            y.append(index[citizen])
    y = torch.tensor(y)
    # the ceiling on this gloss set: the same recogniser on real validation clips of exactly these glosses
    import tempfile
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest, load_sample
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.citizen_manifest_key) + b"\n")
        citizen = load_manifest(p, expected_sha256=args.citizen_manifest_sha256)
    wanted = {c for _, c in targets}
    cache = ContentCache(Path(tempfile.gettempdir()) / "sign-translator-cache", 8 * 2 ** 30)
    real = [r for r in citizen["records"] if r.get("partition") == "val" and r["gloss"]["label"] in wanted]
    real_states = [resample_motion(load_sample(r, "continuous", cache=cache, fetch=store.get), args.fps) for r in real]
    real_y = torch.tensor([index[r["gloss"]["label"]] for r in real])
    out = dict(glosses=len(targets), samples=len(y), recognizer=args.recognizer_run, generator=args.generator_key,
               generated={k: round(v, 4) for k, v in top_k_accuracy(logits_of(model, states, args.device), y).items()},
               blank={k: round(v, 4) for k, v in top_k_accuracy(logits_of(model, blanked(states), args.device), y).items()},
               real=dict(clips=len(real_y), **{k: round(v, 4) for k, v in
                                               top_k_accuracy(logits_of(model, real_states, args.device), real_y).items()}))
    out["verdict"] = dict(above_blank=out["generated"]["top1"] > out["blank"]["top1"],
                          fraction_of_real=round(out["generated"]["top1"] / max(out["real"]["top1"], 1e-9), 3))
    store.put(args.output_key, json.dumps(out, indent=1, sort_keys=True).encode())
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
