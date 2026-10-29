"""How2Sign train signing-space statistics for the composition's space normalisation
(``signtranslator.generation.composition_space``; docs/NEXT_PHASE_PLAN.md, "Next lever after the oracle finding",
option 2). Computed once, written once, pinned by its SHA-256 in every run that uses it.

**Clips:** How2Sign *train* records of the evaluation mix only (``corpus == how2sign-two-view-text-v1`` and ``partition
== train``; the manifest's val, test and held-out partitions are never read, and How2Sign's signer 9, the final-test
signer, is in none of the train records: ``recognizer_corpora.check_loadable`` stops the run on a record of
``signer_component`` c00 (signer 9), an unknown component or a source signer_id of 9), with usable
continuous channels, sorted by sign id, ``--clips`` of them drawn with ``random.Random(--seed)``. Each state is read at
24 fps as the evaluation reads states (``evaluate_text_to_motion.load_state``, ``at_rate``).

**Statistics** (``composition_space.clip_statistics`` per clip, ``space_statistics`` over them): the median RMS angle of
the dominant wrist's direction from the shoulder about its clip mean (the excursion target), the normalised mean
direction per dominant side, the chordal-mean torso orientation, and the clip and side counts.

The output (``--output-file`` and/or ``--output-key``, write-once) holds the statistics, the mix key and SHA-256, the
sample's sign ids and their SHA-256, and the code's settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

CORPUS = "how2sign-two-view-text-v1"
EVAL_MIX = "acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"


def train_sample(manifest, clips: int, seed: int) -> list[dict]:
    """The How2Sign train records the statistics read (module docstring)."""
    from signtranslator.data_engineering.corpus_manifest import usable_channels
    from signtranslator.generation.recognizer_corpora import check_loadable
    pool = []
    for r in manifest["records"]:
        if r.get("corpus") != CORPUS or r.get("partition") != "train":
            continue
        try:                                    # the repository's guard: signer_component c00 (signer 9), an unknown
            check_loadable(r)                   # component, or a source signer_id of 9 is never read
        except RuntimeError as e:
            raise SystemExit(f"How2Sign signer 9 guard: {e}") from None
        if usable_channels(r, "continuous"):
            pool.append(r)
    pool.sort(key=lambda r: r["sign"])
    return random.Random(seed).sample(pool, min(clips, len(pool)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mix-key", default=EVAL_MIX)
    p.add_argument("--mix-sha256", default=EVAL_MIX_SHA)
    p.add_argument("--clips", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--store", default="r2")
    p.add_argument("--output-key")
    p.add_argument("--output-file")
    args = p.parse_args(argv)
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    import warnings
    warnings.simplefilter("ignore")
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.generation.composition_space import clip_statistics, space_statistics
    from evaluate_text_to_motion import at_rate, load_state
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    sample = train_sample(manifest, args.clips, args.seed)
    if any(r.get("partition") != "train" for r in sample):
        raise AssertionError("a non-train record in the sample")
    rows = []
    for k, r in enumerate(sample):
        state = at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), 24.)
        rows.append(clip_statistics(state))
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(sample)} clips", flush=True)
    signs = [r["sign"] for r in sample]
    report = dict(kind="round15-space-stats", statistics=space_statistics(rows), mix_key=args.mix_key,
                  mix_sha256=args.mix_sha256, corpus=CORPUS, partition="train", clips_requested=args.clips,
                  seed=args.seed, signs=signs, signs_sha256=hashlib.sha256("\n".join(signs).encode()).hexdigest(),
                  skipped=int(sum(r is None for r in rows)))
    data = json.dumps(report, indent=1, sort_keys=True).encode()
    if args.output_file:
        with open(args.output_file, "xb") as stream:
            stream.write(data)
    if args.output_key:
        if store.exists(args.output_key):
            raise SystemExit(f"{args.output_key} appeared during the run: refusing to overwrite")
        store.put(args.output_key, data)
    print(json.dumps(dict(sha256=hashlib.sha256(data).hexdigest(), **{k: v for k, v in report["statistics"].items()})),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
