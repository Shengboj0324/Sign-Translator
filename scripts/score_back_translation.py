"""Back-translate stored generated states with the frozen motion -> English model and score them by BLEU.

``evaluate_text_to_motion.py --save-generated PREFIX`` stores ``PREFIX/<guidance>/<sign>.npz``. For each group this
writes corpus BLEU-4 of the generated motion, of the real motion of the same clips (the upper reference), and of
the blank control (the real clips with every channel removed, only their durations left: what text priors alone
give). A generator's score is read between those two bounds; it is relative, never an absolute translation claim.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.generation.motion_to_text import (MotionToText, MotionToTextConfig,  # noqa: E402
                                                      WordVocabulary, corpus_bleu)
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402
from evaluate_text_to_motion import at_rate, blanked, load_state  # noqa: E402
from train_text_to_motion import make_store  # noqa: E402


def load_translator(store, key: str, manifest_sha256: str):
    """(model with EMA weights in eval mode, vocabulary), refusing one trained on another manifest."""
    saved = torch.load(io.BytesIO(store.get(key)), map_location="cpu", weights_only=False)
    if saved.get("kind") != "motion_to_text" or saved["manifest_sha256"] != manifest_sha256:
        raise SystemExit(f"{key} is not a back-translator trained on this manifest")
    model = MotionToText(MotionToTextConfig.from_dict(saved["config"]))
    model.load_state_dict(saved.get("ema") or saved["model"])
    return model.eval(), WordVocabulary(saved["vocabulary"]["words"])


@torch.no_grad()
def translate(model, vocabulary, states, *, beam: int = 4, batch: int = 32) -> list[str]:
    from signtranslator.data.multichannel import collate_multichannel
    out = []
    for i in range(0, len(states), batch):
        out += [vocabulary.decode(ids) for ids in model.generate(collate_multichannel(states[i:i + batch]), beam=beam)]
    return out


def bleu_of(model, vocabulary, states, texts, *, beam: int) -> dict:
    hypotheses = translate(model, vocabulary, states, beam=beam)
    return dict(corpus_bleu(hypotheses, texts), examples=list(zip(hypotheses[:5], texts[:5])))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--translator-key", required=True, help="checkpoint of train_motion_to_text.py (best.json)")
    parser.add_argument("--generated", required=True, nargs="+", help="prefixes written by --save-generated")
    parser.add_argument("--translator-manifest-sha256",
                        help="the manifest the translator was trained on, when it is not --manifest-sha256")
    parser.add_argument("--fps", type=float, help="resample the real references to this rate (as the evaluation did)")
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--output-key")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    by_sign = {r["sign"]: r for r in manifest["records"]}
    model, vocabulary = load_translator(store, args.translator_key, args.translator_manifest_sha256 or args.manifest_sha256)
    out = {}
    for prefix in args.generated:
        groups = {}
        for key in sorted(k for k in store.list(prefix.rstrip("/") + "/") if k.endswith(".npz")):
            groups.setdefault(key.rsplit("/", 2)[-2], []).append(key)
        out[prefix] = {}
        for tag, members in sorted(groups.items()):
            signs = [k.rsplit("/", 1)[-1][:-4] for k in members]
            generated = []
            for key in members:
                payload = store.get(key)
                with tempfile.TemporaryDirectory() as t:
                    p = Path(t) / "s.npz"; p.write_bytes(payload)
                    generated.append(MultichannelMotion.load(p, expected_sha256=hashlib.sha256(payload).hexdigest()))
            references = [load_state(store, by_sign[s]["continuous"]["state_key"], by_sign[s]["continuous"]["state_sha256"])
                          for s in signs]
            if args.fps:
                references = [at_rate(r, args.fps) for r in references]
            texts = [by_sign[s].get("english") or by_sign[s]["text"] for s in signs]    # plan records: their English
            out[prefix][tag] = dict(generated=bleu_of(model, vocabulary, generated, texts, beam=args.beam),
                                    real=bleu_of(model, vocabulary, references, texts, beam=args.beam),
                                    blank=bleu_of(model, vocabulary, blanked(references), texts, beam=args.beam))
            print(json.dumps({prefix: {tag: {k: round(v["bleu"], 2) for k, v in out[prefix][tag].items()}}}), flush=True)
    if args.output_key:
        store.put(args.output_key, json.dumps(dict(translator_key=args.translator_key, beam=args.beam, results=out),
                                              indent=2, sort_keys=True).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
