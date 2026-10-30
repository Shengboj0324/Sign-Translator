"""Round 20 sentence sets (docs/NEXT_PHASE_PLAN.md, "Round 20: composition v2", declared 2026-10-30 19:09 UTC).

- **D20 (development):** 500 How2Sign val sentences drawn with ``random.Random(20)`` from the clips of
  ``pilot_content_check.validation_sentences``'s pool (How2Sign val, signer groups c01 / c03 / c05, a text; c00 and
  its signer are never read) outside the 1,000 (``validation_sentences(mix, 1000)``) and the 300
  (``validation_sentences(mix, 300)``), excluding every sentence whose normalised text (``compositional.normalise``,
  words joined by one space) equals the text of one of the 1,000.
- **C20 (confirmation):** a further 1,000 drawn with the same generator from what remains, disjoint from D20 in clips
  and, so that no D20 sentence is confirmed on its own text, in normalised texts. Untouched until an M1 claim.

Only manifest metadata is read (no state is loaded). Both lists are written once under ``--prefix`` (``d20.json``,
``c20.json``; an existing key stops the run before anything is written) with the manifest key and hash, the rule and
the counts; their SHA-256s are printed for the plan and the pod scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

EVAL_MIX = "acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
PREFIX = "acquisition/2026-10-17-v1/youtube-asl-v1/round20"
RULE = ("How2Sign val, signer groups c01/c03/c05 (validation_sentences' pool), outside validation_sentences(mix, 1000) "
        "and validation_sentences(mix, 300), no normalised text of the 1,000; D20 = random.Random(20).sample(rest, 500); "
        "C20 = the same generator's sample(rest minus D20's clips and texts, 1000)")


def build(manifest, *, dev: int = 500, confirm: int = 1000) -> tuple[list, list, dict]:
    from pilot_content_check import validation_sentences
    from round20_lib import draw_sets, text_key, validation_pool
    pool = validation_pool(manifest)
    thousand, three = validation_sentences(manifest, 1000), validation_sentences(manifest, 300)
    excluded_signs = {r["sign"] for r in thousand} | {r["sign"] for r in three}
    d, c, counts = draw_sets(pool, excluded_signs, {text_key(r["text"]) for r in thousand}, dev=dev, confirm=confirm)
    counts.update(thousand=len(thousand), three_hundred=len(three),
                  three_hundred_outside_thousand=len({r["sign"] for r in three} - {r["sign"] for r in thousand}),
                  dev_texts_in_three_hundred=int(sum(text_key(r["text"]) in {text_key(x["text"]) for x in three}
                                                     for r in d)))
    return d, c, counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    p.add_argument("--mix-key", default=EVAL_MIX)
    p.add_argument("--mix-sha256", default=EVAL_MIX_SHA)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--dev", type=int, default=500)
    p.add_argument("--confirm", type=int, default=1000)
    args = p.parse_args(argv)
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from train_text_to_motion import make_store
    from round20_lib import set_payload
    store = make_store(args.store)
    keys = {name: f"{args.prefix}/{name.lower()}.json" for name in ("D20", "C20")}
    for k in keys.values():
        if store.exists(k):
            raise SystemExit(f"{k} exists: written once, never overwritten")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    d, c, counts = build(manifest, dev=args.dev, confirm=args.confirm)
    out = {}
    for name, records in (("D20", d), ("C20", c)):
        payload = json.dumps(set_payload(name, records, manifest_key=args.mix_key, manifest_sha256=args.mix_sha256,
                                         rule=RULE, counts=counts), indent=1).encode()
        store.put(keys[name], payload)
        out[name] = dict(key=keys[name], sha256=hashlib.sha256(payload).hexdigest(), sentences=len(records))
    print(json.dumps(dict(counts=counts, sets=out), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
