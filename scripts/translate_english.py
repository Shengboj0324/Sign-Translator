"""English -> ASL plan -> motion: the planner and a plan-conditioned generator chained (plan item 2).

1. The gloss planner (``train_gloss_planner.py``; its ``best.json`` checkpoint, EMA weights) decodes each English
   sentence to a plan: base glosses, locus slots and grammatical markers.
2. The plan is written as its plan line (``plan_text.plan_to_text``), the condition the plan-trained generator saw.
3. The generator (any checkpoint ``evaluate_text_to_motion.load_generator`` reads, trained on a plan manifest)
   samples motion for that line. Its duration is the number of signs times ``--seconds-per-sign`` (0.57 s: the
   ASLLRP mean, 3.1 h of signing over 19,510 signs) unless ``--seconds`` is given.

Writes one JSON line per sentence (English, plan, plan line, seconds) and, with ``--output-prefix``, each generated
state as ``<prefix>/<index>.npz`` in the store. ``--plan-only`` stops after step 2.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.generation.motion_to_text import WordVocabulary  # noqa: E402
from signtranslator.planning.asllrp_gloss import GlossVocabulary  # noqa: E402
from signtranslator.planning.english_gloss import (EnglishGlossConfig, EnglishGlossPlanner, encode_sources,  # noqa: E402
                                                   plan_from_generation)
from signtranslator.planning.plan_text import plan_to_text  # noqa: E402
from train_text_to_motion import make_store  # noqa: E402

SECONDS_PER_SIGN = .57


def load_planner(store, run: str, device="cpu"):
    best = json.loads(store.get(f"training-runs/{run}/best.json"))
    saved = torch.load(io.BytesIO(store.get(best["checkpoint_key"])), map_location="cpu", weights_only=False)
    model = EnglishGlossPlanner(EnglishGlossConfig.from_dict(saved["config"]))
    model.load_state_dict(saved.get("ema") or saved["model"])
    return model.to(device).eval(), WordVocabulary(saved["words"]), GlossVocabulary(saved["glosses"]), best


def plan_sentences(model, words, glosses, sentences, *, beam: int = 4, device="cpu") -> list[dict]:
    source = encode_sources(words, sentences, max_tokens=model.config.max_source).to(device)
    plans = [plan_from_generation(glosses, g) for g in model.generate(source, beam=beam)]
    for p in plans:
        p["line"] = plan_to_text(p["glosses"], p["loci"], p["markers"])
    return plans


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--planner-run", required=True)
    parser.add_argument("--text", action="append", required=True, help="an English sentence (repeatable)")
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--generator-key")
    parser.add_argument("--generator-manifest-sha256")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--seconds-per-sign", type=float, default=SECONDS_PER_SIGN)
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--guidance", type=float, default=2.)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    model, words, glosses, best = load_planner(store, args.planner_run, args.device)
    plans = plan_sentences(model, words, glosses, args.text, beam=args.beam, device=args.device)
    sample = None
    if not args.plan_only:
        if not (args.generator_key and args.generator_manifest_sha256):
            raise SystemExit("--generator-key and --generator-manifest-sha256 are required unless --plan-only")
        from evaluate_text_to_motion import load_generator
        _, _, sample = load_generator(store, args.generator_key, args.generator_manifest_sha256, args.device)
    for i, (english, plan) in enumerate(zip(args.text, plans)):
        seconds = args.seconds or max(1, len(plan["glosses"])) * args.seconds_per_sign
        row = dict(index=i, english=english, plan_line=plan["line"], glosses=plan["glosses"],
                   loci=[list(x) for x in plan["loci"]], markers=plan["markers"], seconds=round(seconds, 3),
                   planner=best["checkpoint_key"])
        if sample is not None:
            state = sample(plan["line"], args.fps, seconds, args.guidance, args.steps, args.seed + i)
            if args.output_prefix:
                with tempfile.TemporaryDirectory() as t:
                    path = Path(t) / "s.npz"
                    state.save(path)
                    row["state_key"] = f"{args.output_prefix.rstrip('/')}/{i:04d}.npz"
                    row["state_sha256"] = store.put(row["state_key"], path.read_bytes())
            row["frames"] = len(state.timestamps)
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
