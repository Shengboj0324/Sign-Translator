"""Does a motion tokenizer keep signs recognisable? (round 3b tokenizer acceptance, docs/NEXT_PHASE_PLAN.md)

Round 3's domain check compared reconstruction errors with How2Sign's and passed, yet the generator could not produce
recognisable single signs. This checks the property that matters for sign level. Clips of ``--corpus`` (``--partition``,
first ``--limit`` by sign, conformed to ``--conform-to-corpus`` at ``--fps`` as training sees them) are scored by the
validated sign recogniser in two forms:
- **real**: the clips as stored;
- **round trip**: the tokenizer's quantised reconstruction, kept only on the frames the real clip observes, with the
  unit conversion's neutral fill elsewhere. Monocular clips miss many arm frames; the reconstruction there is
  invented, not observed.

Pass rule (declared): round-trip top-5 >= ``--min-ratio`` (0.9) x real top-5. The clip's gloss must be one of the
recogniser's classes. The report gives both top-1 and top-5 and the per-channel valid fractions.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.layout_adapter import conform  # noqa: E402
from signtranslator.data.multichannel import collate_multichannel  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest, load_sample  # noqa: E402
from signtranslator.generation.text_to_motion import (assemble_state, diffusion_targets, native_values,  # noqa: E402
                                                      unit_to_raw)

MASKED = ("body", "left_hand", "right_hand")


def label_of(record) -> str | None:
    gloss = record.get("gloss") or {}
    return gloss.get("label")


@torch.no_grad()
def round_trip(model, cfg, state, fps: float, plan=None):
    """The tokenizer's reconstruction of ``state``. Round 17: with ``plan`` (a ``plan_codes.PlanVQ`` over this
    tokenizer), the decoder is given the latents decoded from the plan codes instead of the quantised latents (the
    plan tokenizer's fidelity round trip)."""
    b = collate_multichannel([state])
    unit = diffusion_targets(cfg, b)
    frames = unit.shape[1]
    latents = model.encode(unit)
    if plan is None:
        quantised, _, _ = model.quantize(latents)
    else:
        quantised = plan.eval().reconstruct(latents, model.token_valid(b.frame_valid))
    rec = model.decode_latents(quantised, frames)
    def to_state(u):
        raw = unit_to_raw(cfg, u.double())
        return assemble_state(cfg, "x", native_values(cfg, {k: v[0] for k, v in raw.items()}), len(state.timestamps),
                              fps, "round-trip", "round-trip-clock")
    out, neutral = to_state(rec), to_state(unit)
    channels = dict(out.channels)
    for name in MASKED:                     # observed frames: reconstruction; unobserved: the neutral fill
        seen = state.channels[name].valid
        channels[name] = replace(out.channels[name], values=np.where(seen[..., None], out.channels[name].values,
                                                                     neutral.channels[name].values))
    return replace(out, channels=channels)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--template-manifest", help="key@sha256 of a manifest holding a --conform-to-corpus record, when "
                                                    "--manifest-key has none (a single-corpus manifest)")
    parser.add_argument("--partition", default="val")
    parser.add_argument("--conform-to-corpus", default="how2sign-two-view-text-v1")
    parser.add_argument("--tokenizer-key", required=True)
    parser.add_argument("--recognizer-run", default="sign-recognizer-v2")
    parser.add_argument("--gloss-map-key", help="Sign Bank -> ASL Citizen gloss map (validate_sign_recognizer.py's): "
                                                "ASLLVD clips count only under a unique exact mapping")
    parser.add_argument("--label-upper", action="store_true",
                        help="labels are plain English words (WLASL): match the recogniser's gloss by exact uppercase "
                             "spelling, spaces removed; no fuzzy or variant matching")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--min-ratio", type=float, default=.9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-key")
    parser.add_argument("--gains-key", help="amendment 5: fit_tokenizer_gains.py output, applied to every round trip")
    parser.add_argument("--plan-tokenizer-key", help="round 17: a plan tokenizer over --tokenizer-key; the round trip "
                        "decodes the latents reconstructed from its plan codes (the plan's fidelity gate)")
    parser.add_argument("--plan-tokenizer-sha256", help="its SHA-256 (required with --plan-tokenizer-key)")
    args = parser.parse_args(argv)
    if bool(args.plan_tokenizer_key) != bool(args.plan_tokenizer_sha256):
        raise SystemExit("--plan-tokenizer-key and --plan-tokenizer-sha256 go together")
    global label_of
    import diagnose_tokenizer as dt
    from evaluate_text_to_motion import at_rate
    from train_text_to_motion import make_store
    from validate_sign_recognizer import load_model, logits_of
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    cache = ContentCache(Path(tempfile.gettempdir()) / "sign-translator-cache", 8 * 2 ** 30)
    recogniser, labels, _ = load_model(store, args.recognizer_run, args.device)
    index = {g: i for i, g in enumerate(labels)}
    if args.gloss_map_key:                  # as in the recogniser's own cross-domain validation
        import collections
        to_citizen = collections.defaultdict(set)
        for g, m in json.loads(store.get(args.gloss_map_key))["mapping"].items():
            if m["method"] == "exact" and g in index:
                for c in m["candidates"]:
                    to_citizen[c].add(g)
        plain = label_of
        label_of = lambda r: (next(iter(to_citizen[plain(r)])) if len(to_citizen.get(plain(r), ())) == 1  # noqa: E731
                              else None)
    if args.label_upper:
        plain_upper = label_of
        label_of = lambda r: (lambda g: g if g in index else None)((plain_upper(r) or "").upper().replace(" ", ""))  # noqa: E731
    pool = manifest["records"]
    if args.template_manifest:
        key, sha = args.template_manifest.rsplit("@", 1)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "t.json"; p.write_bytes(store.get(key) + b"\n")
            pool = load_manifest(p, expected_sha256=sha)["records"]
    template = load_sample(next(r for r in pool if r.get("corpus") == args.conform_to_corpus),
                           "continuous", cache=cache, fetch=store.get)
    records = sorted((r for r in manifest["records"] if (r.get("corpus") or manifest.get("corpus_id")) == args.corpus
                      and r["partition"] == args.partition and label_of(r) in index), key=lambda r: r["sign"])
    records = records[:args.limit]
    if not records:
        raise SystemExit("no clips whose gloss the recogniser knows")
    model, _ = dt.load_tokenizer(store, args.tokenizer_key)
    cfg = model.config.motion
    plan = None
    if args.plan_tokenizer_key:
        import hashlib
        from signtranslator.generation.plan_codes import load_plan_tokenizer
        tokenizer_sha = hashlib.sha256(store.get(args.tokenizer_key)).hexdigest()   # the plan must read these bytes
        try:
            plan, _, saved_plan = load_plan_tokenizer(store.get(args.plan_tokenizer_key), args.plan_tokenizer_sha256,
                                                      tokenizer_sha256=tokenizer_sha)
        except ValueError as error:
            raise SystemExit(f"--plan-tokenizer-key: {error}")
        if saved_plan["tokenizer_key"] != args.tokenizer_key:
            raise SystemExit(f"the plan tokenizer reads {saved_plan['tokenizer_key']}, not --tokenizer-key")
    real = [at_rate(conform(load_sample(r, "continuous", cache=cache, fetch=store.get), template), args.fps) for r in records]
    trip = [round_trip(model, cfg, s, args.fps, plan=plan) for s in real]
    if args.gains_key:
        from fit_tokenizer_gains import apply_gains
        gains = json.loads(store.get(args.gains_key))["gains"]
        trip = [apply_gains(t, gains) for t in trip]
    y = torch.tensor([index[label_of(r)] for r in records])

    def acc(states):
        top = logits_of(recogniser, states, args.device).topk(5, 1).indices
        return dict(top1=round(float((top[:, 0] == y).float().mean()), 4),
                    top5=round(float((top == y[:, None]).any(1).float().mean()), 4))
    report = dict(corpus=args.corpus, partition=args.partition, clips=len(records), tokenizer_key=args.tokenizer_key,
                  **({"plan_tokenizer_key": args.plan_tokenizer_key, "plan_tokenizer_sha256": args.plan_tokenizer_sha256}
                     if args.plan_tokenizer_key else {}),
                  recognizer_run=args.recognizer_run, real=acc(real), round_trip=acc(trip), min_ratio=args.min_ratio,
                  valid_fraction={c: round(float(np.median([s.channels[c].valid.mean() for s in real])), 3) for c in MASKED})
    report["ratio"] = round(report["round_trip"]["top5"] / max(report["real"]["top5"], 1e-9), 3)
    report["pass"] = report["ratio"] >= args.min_ratio
    print(json.dumps(report), flush=True)
    if args.output_key:
        store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True).encode())
    return 0 if report["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
