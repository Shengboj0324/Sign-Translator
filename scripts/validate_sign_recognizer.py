"""Validate the isolated-sign recogniser as an instrument before it scores generated motion.

Reads the run's best checkpoint (EMA weights) and writes one report with:

1. **In-domain**: top-1/top-5 on the ASL Citizen validation signers (never the test partition).
2. **Controls on the same clips**: every channel removed (only duration left), hands only, everything but the hands.
   An instrument that reads signs must collapse without the hands and stay near chance on the blank clips; a high
   blank or no-hands score means it uses shortcuts (duration, face) and its scores on generated motion are not
   meaningful.
3. **Cross-domain**: top-1/top-5 on ASLLVD (a different studio, cameras and signers; DAI 2), training and validation
   signers only (its test signer stays unread), for the Sign Bank glosses that map *exactly* to one ASL Citizen
   label (``asl-citizen-gloss-map.json``, method ``exact``). The recogniser never saw ASLLVD.
4. **Per-gloss reliability** on the ASL Citizen validation clips: the glosses recognised with top-1 >= ``--reliable``
   over at least ``--min-clips`` clips; sign-level evaluation of generated motion is restricted to that list.

The verdict is a fixed rule (``--min-cross-domain-top1``, ``--max-blank-top1``, ``--max-no-hands-top1``), not a judgement
made after seeing the numbers.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.multichannel import collate_multichannel  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest, load_sample, mask_channels  # noqa: E402
from signtranslator.generation.sign_recognizer import SignRecognizer, SignRecognizerConfig, top_k_accuracy  # noqa: E402
from evaluate_text_to_motion import blanked  # noqa: E402
from signtranslator.pose.resample import resample_motion  # noqa: E402
from train_text_to_motion import make_store, to_device  # noqa: E402

HANDS = ("left_hand", "right_hand")


def load_model(store, run: str, device):
    best = json.loads(store.get(f"training-runs/{run}/best.json"))
    saved = torch.load(io.BytesIO(store.get(best["checkpoint_key"])), map_location="cpu", weights_only=False)
    model = SignRecognizer(SignRecognizerConfig.from_dict(saved["config"]))
    model.load_state_dict(saved.get("ema") or saved["model"])
    return model.to(device).eval(), saved["labels"], best


@torch.no_grad()
def logits_of(model, states, device, batch: int = 64) -> torch.Tensor:
    out = []
    for i in range(0, len(states), batch):
        motion = to_device((torch.zeros(0), torch.zeros(0), collate_multichannel(states[i:i + batch]), None), device)[2]
        out.append(model(motion).cpu())
    return torch.cat(out) if out else torch.zeros(0, 1)


def manifest_of(store, key, sha):
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(key) + b"\n")
        return load_manifest(p, expected_sha256=sha)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--run", required=True)
    parser.add_argument("--citizen-manifest-key", required=True)
    parser.add_argument("--citizen-manifest-sha256", required=True)
    parser.add_argument("--asllvd-manifest-key", required=True)
    parser.add_argument("--asllvd-manifest-sha256", required=True)
    parser.add_argument("--gloss-map-key", default="acquisition/2026-10-17-v1/derived/signbank-v1/asl-citizen-gloss-map.json")
    parser.add_argument("--max-seconds", type=float, default=8.)
    parser.add_argument("--reliable", type=float, default=.8)
    parser.add_argument("--min-clips", type=int, default=3)
    parser.add_argument("--min-cross-domain-top1", type=float, default=.25)
    parser.add_argument("--max-blank-top1", type=float, default=.02)
    parser.add_argument("--max-no-hands-top1", type=float, default=.15)
    parser.add_argument("--output-key", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=float, default=0., help="resample clips to this rate (must match training)")
    parser.add_argument("--cross-partitions", default="train,val",
                        help="ASLLVD partitions for the cross-domain check; only those the recogniser never trained on")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    cache = ContentCache(Path(tempfile.gettempdir()) / "sign-translator-cache", 8 * 2 ** 30)
    model, glosses, best = load_model(store, args.run, device)
    index = {g: i for i, g in enumerate(glosses)}

    def load(records):
        from concurrent.futures import ThreadPoolExecutor
        records = [r for r in records if r["continuous"].get("seconds", 0.) <= args.max_seconds]

        def one(r):
            s = load_sample(r, "continuous", cache=cache, fetch=store.get)
            return resample_motion(s, args.fps) if args.fps else s
        with ThreadPoolExecutor(16) as pool:               # R2 fetches dominate: load in parallel, keep the order
            states = list(pool.map(one, records))
        return states, torch.tensor([r["_label"] for r in records], dtype=torch.long)

    def score(states, labels):
        return {k: round(v, 4) for k, v in top_k_accuracy(logits_of(model, states, device), labels).items()} if len(labels) else None

    citizen = manifest_of(store, args.citizen_manifest_key, args.citizen_manifest_sha256)
    val = [dict(r, _label=index[r["gloss"]["label"]]) for r in citizen["records"]
           if r.get("partition") == "val" and r["gloss"]["label"] in index]
    states, labels = load(val)
    report = dict(run=args.run, checkpoint=best["checkpoint_key"], step=best["step"], classes=len(glosses), fps=args.fps or None)
    logits = logits_of(model, states, device)
    report["in_domain"] = dict(clips=len(labels), **{k: round(v, 4) for k, v in top_k_accuracy(logits, labels).items()})
    report["controls"] = dict(blank=score(blanked(states), labels),
                              hands_only=score([mask_channels(s, HANDS) for s in states], labels),
                              no_hands=score([mask_channels(s, [n for n in s.channels if n not in HANDS]) for s in states], labels))
    hits, counts = collections.Counter(), collections.Counter()
    for pred, y in zip(logits.argmax(1).tolist(), labels.tolist()):
        counts[y] += 1; hits[y] += pred == y
    reliable = sorted(glosses[y] for y, n in counts.items() if n >= args.min_clips and hits[y] / n >= args.reliable)
    report["per_gloss"] = dict(glosses_with_clips=len(counts), reliable=len(reliable), reliable_glosses=reliable,
                               rule=f"top-1 >= {args.reliable} over >= {args.min_clips} validation clips")

    mapping = json.loads(store.get(args.gloss_map_key))["mapping"]
    to_citizen = collections.defaultdict(set)
    for g, m in mapping.items():
        if m["method"] == "exact" and g in index:
            for c in m["candidates"]:
                to_citizen[c].add(g)
    cross_parts = tuple(args.cross_partitions.split(","))
    if "test" in cross_parts:
        raise SystemExit("the ASLLVD test signer stays unread here")
    asllvd = manifest_of(store, args.asllvd_manifest_key, args.asllvd_manifest_sha256)
    cross = [dict(r, _label=index[next(iter(to_citizen[r["gloss"]["label"]]))]) for r in asllvd["records"]
             if r.get("partition") in cross_parts and len(to_citizen.get(r["gloss"]["label"], ())) == 1]
    cs, cl = load(cross)
    report["cross_domain_asllvd"] = dict(clips=len(cl), glosses=len(set(cl.tolist())), partitions=list(cross_parts),
                                         **(score(cs, cl) or {}))
    c = report["cross_domain_asllvd"].get("top1", 0.)
    verdict = dict(cross_domain_ok=c >= args.min_cross_domain_top1,
                   blank_ok=(report["controls"]["blank"] or {}).get("top1", 1.) <= args.max_blank_top1,
                   no_hands_ok=(report["controls"]["no_hands"] or {}).get("top1", 1.) <= args.max_no_hands_top1)
    verdict["usable_as_instrument"] = all(verdict.values())
    report["verdict"] = verdict
    report["thresholds"] = dict(min_cross_domain_top1=args.min_cross_domain_top1, max_blank_top1=args.max_blank_top1,
                                max_no_hands_top1=args.max_no_hands_top1)
    store.put(args.output_key, json.dumps(report, indent=1, sort_keys=True).encode())
    print(json.dumps({k: v for k, v in report.items() if k != "per_gloss"} | {"reliable_glosses": len(reliable)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
