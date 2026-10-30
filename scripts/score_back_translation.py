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
    if saved.get("kind") == "state_to_text_t5":              # instrument v2: reads the conformed state itself
        from signtranslator.generation.state_to_text_t5 import StateToT5Config, build
        config = StateToT5Config.from_dict(saved["config"])
        if config.sha256() != saved["config_sha256"]:
            raise SystemExit(f"{key}: its config does not hash to its config_sha256")
        model = build(config, pretrained=False)
        model.load_state_dict(saved.get("ema") or saved["model"])
        return model.eval(), "states"
    if saved.get("kind") == "keypoints_to_text_t5":          # 2D second opinion: reads states through FK projection
        from signtranslator.generation.keypoints_to_text_t5 import KeypointsToT5Config, build
        model = build(KeypointsToT5Config.from_dict(saved["config"]), pretrained=False)
        model.load_state_dict(saved.get("ema") or saved["model"])
        return model.eval(), "keypoints"
    if saved.get("kind") not in ("motion_to_text", "motion_to_text_t5") or saved["manifest_sha256"] != manifest_sha256:
        raise SystemExit(f"{key} is not a back-translator trained on this manifest")
    if saved["kind"] == "motion_to_text_t5":                 # v2: pretrained T5 decoder, its own tokenizer
        from signtranslator.generation.motion_to_text_t5 import MotionToT5Config, build
        model = build(MotionToT5Config.from_dict(saved["config"]), pretrained=False)
        model.load_state_dict(saved.get("ema") or saved["model"])
        return model.eval(), None
    model = MotionToText(MotionToTextConfig.from_dict(saved["config"]))
    model.load_state_dict(saved.get("ema") or saved["model"])
    return model.eval(), WordVocabulary(saved["vocabulary"]["words"])


def load_translator_run(store, run: str, manifest_sha256: str):
    """(model, kind, checkpoint key, checkpoint SHA-256 or None) of a run's ``best.json`` checkpoint (instrument v2's
    ``--translator-run``). A best.json that records ``checkpoint_sha256`` (``train_state_to_text_t5.py`` writes it) is
    an instrument-v2 run: the payload is checked against that SHA-256 and loaded weights-only
    (``state_to_text_t5.load_model``). Any other run is loaded by ``load_translator`` exactly as before."""
    import hashlib
    best = json.loads(store.get(f"training-runs/{run}/best.json"))
    key = best["checkpoint_key"]
    if not best.get("checkpoint_sha256"):
        model, kind = load_translator(store, key, manifest_sha256)
        return model, kind, key, None
    from signtranslator.generation.state_to_text_t5 import load_model
    payload = store.get(key)
    if hashlib.sha256(payload).hexdigest() != best["checkpoint_sha256"]:
        raise SystemExit(f"{key}: SHA-256 differs from {run}'s best.json")
    try:
        model, _ = load_model(payload, expected_sha256=best["checkpoint_sha256"])
    except ValueError as error:
        raise SystemExit(f"{key}: {error}")
    return model, "states", key, best["checkpoint_sha256"]


@torch.no_grad()
def translate(model, vocabulary, states, *, beam: int = 4, batch: int = 32) -> list[str]:
    from signtranslator.data.multichannel import collate_multichannel
    out = []
    device = next(model.parameters()).device
    if vocabulary == "states":                               # instrument v2 (state_to_text_t5)
        from signtranslator.data.keypoint_shards import collate_keypoints
        items = [(torch.from_numpy(f), "") for f in model.featurise(states)]
        for i in range(0, len(items), batch):
            (x, valid), _ = collate_keypoints(items[i:i + batch])
            out += model.generate_text((x.to(device), valid.to(device)), beam=beam)
        return out
    if vocabulary == "keypoints":
        from signtranslator.data.keypoint_shards import collate_keypoints
        from signtranslator.data.keypoints2d import features, from_state, normalise
        items = []
        for state in states:
            points, valid = from_state(at_rate(state, model.config.fps))
            items.append((torch.from_numpy(features(normalise(points, valid, y_down=False), valid)), ""))
        for i in range(0, len(items), batch):
            (x, valid), _ = collate_keypoints(items[i:i + batch])
            out += model.generate_text((x.to(device), valid.to(device)), beam=beam)
        return out
    for i in range(0, len(states), batch):
        motion = collate_multichannel(states[i:i + batch])
        if vocabulary is None:                               # T5 back-translator decodes to text itself
            from train_text_to_motion import to_device
            out += model.generate_text(to_device((None, None, motion, None), device)[2], beam=beam)
        else:
            out += [vocabulary.decode(ids) for ids in model.generate(motion, beam=beam)]
    return out


def bleu_of(model, vocabulary, states, texts, *, beam: int) -> dict:
    hypotheses = translate(model, vocabulary, states, beam=beam)
    return dict(corpus_bleu(hypotheses, texts), examples=list(zip(hypotheses[:5], texts[:5])), hypotheses=hypotheses)


def shuffled_control(hypotheses, texts) -> dict:
    """The real clips' translations scored against other clips' sentences (rolled by half the set): what fluent,
    generic output scores without matching its clip."""
    rolled = texts[len(texts) // 2:] + texts[:len(texts) // 2]
    return dict(corpus_bleu(hypotheses, rolled))


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
    parser.add_argument("--device", default="auto", help="auto: CUDA when available")
    parser.add_argument("--output-key")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    by_sign = {r["sign"]: r for r in manifest["records"]}
    model, vocabulary = load_translator(store, args.translator_key, args.translator_manifest_sha256 or args.manifest_sha256)
    if vocabulary is None or vocabulary in ("keypoints", "states"):  # T5 translators move whole; v1 keeps its CPU path
        model = model.to(torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                                      ("cpu" if args.device == "auto" else args.device)))
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
            out[prefix][tag]["shuffled"] = shuffled_control(out[prefix][tag]["real"]["hypotheses"], texts)
            print(json.dumps({prefix: {tag: {k: round(v["bleu"], 2) for k, v in out[prefix][tag].items()}}}), flush=True)
    if args.output_key:
        store.put(args.output_key, json.dumps(dict(translator_key=args.translator_key, beam=args.beam, results=out),
                                              indent=2, sort_keys=True).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
