"""Instrument check of the multi-signer gate: does real signing pass it?

Round 3's union gate found no candidate passing on every validation signer, with head and face rates failing in
opposite directions: too still against signer 3 (c03), too active against signer 11 (c05). This scores **real**
How2Sign validation clips of one signer as if they were generated, against another signer's real clips, with the
gate's own checks (``exit_gate.score_setting``, thresholds unchanged):

- ``X vs Y`` (X != Y): one real signer's style judged by another's. Where real signing fails a check, the check
  measures signer idiosyncrasy, and a generator conditioned on text alone cannot pass it on both signers at once.
- ``X vs the others pooled``: one real signer against an equal-size hash sample of the other signers' clips.
- ``X half A vs X half B``: the same signer split in two (deterministic by sign id), the within-signer baseline.

Clips are the ones each group's evaluation used (the signs stored under ``--generated-template``). Speed, variation
and marker ratios are unpaired aggregate statistics, so unpaired clip sets are valid inputs. Per-clip pairing
measures (geodesic error) are not used by the gate and are ignored here.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from exit_gate import ARTICULATED, score_setting  # noqa: E402
from score_postprocessing_grid import score  # noqa: E402
from evaluate_text_to_motion import at_rate, load_state  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--generated-template", required=True, help="prefix with {group}; its signs are the clips")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--output-key")
    args = parser.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    by_sign = {r["sign"]: r for r in manifest["records"]}
    groups = args.groups.split(",")
    clips = {}
    for g in groups:
        prefix = args.generated_template.format(group=g).rstrip("/") + "/"
        keys = [k for k in store.list(prefix) if k.endswith(".npz")]
        tag = sorted({k.rsplit("/", 2)[-2] for k in keys})[0]
        signs = sorted(k.rsplit("/", 1)[-1][:-4] for k in keys if k.rsplit("/", 2)[-2] == tag)
        records = [by_sign[s] for s in signs]
        states = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), args.fps)
                  for r in records]
        clips[g] = (states, records)
        print(json.dumps(dict(group=g, clips=len(states))), flush=True)

    def judged(gen, refs) -> dict:
        n = min(len(gen[0]), len(refs[0]))
        result = score(gen[0][:n], refs[0][:n], gen[1][:n])
        row = score_setting("real", "real", result, ARTICULATED)
        return dict(passed=row["passed"], total=row["total"], clips=n,
                    failing={k: round(float(v), 3) for k, (v, ok) in row["checks"].items() if not ok},
                    checks={k: round(float(v), 3) for k, (v, ok) in row["checks"].items()})
    out = dict(groups=groups, cross={}, within={})
    for x, y in itertools.permutations(groups, 2):
        out["cross"][f"{x}_vs_{y}"] = judged(clips[x], clips[y])
        r = out["cross"][f"{x}_vs_{y}"]
        print(json.dumps({f"{x} judged by {y}": f"{r['passed']}/{r['total']}", "failing": r["failing"]}), flush=True)
    out["pooled_others"] = {}
    for x in groups:                                  # one real signer against the other signers pooled
        others = [y for y in groups if y != x]
        pool = (sum((clips[y][0] for y in others), []), sum((clips[y][1] for y in others), []))
        n = min(len(clips[x][0]), len(pool[0]))
        idx = sorted(range(len(pool[0])), key=lambda i: hashlib.sha256(pool[1][i]["sign"].encode()).hexdigest())[:n]
        out["pooled_others"][x] = judged(clips[x], ([pool[0][i] for i in idx], [pool[1][i] for i in idx]))
        r = out["pooled_others"][x]
        print(json.dumps({f"{x} judged by the others pooled": f"{r['passed']}/{r['total']}", "failing": r["failing"]}),
              flush=True)
    for g in groups:
        states, records = clips[g]
        half = [int(hashlib.sha256(r["sign"].encode()).hexdigest(), 16) % 2 for r in records]
        a = ([s for s, h in zip(states, half) if h == 0], [r for r, h in zip(records, half) if h == 0])
        b = ([s for s, h in zip(states, half) if h == 1], [r for r, h in zip(records, half) if h == 1])
        out["within"][g] = judged(a, b)
        r = out["within"][g]
        print(json.dumps({f"{g} half A judged by half B": f"{r['passed']}/{r['total']}", "failing": r["failing"]}),
              flush=True)
    if args.output_key:
        store.put(args.output_key, json.dumps(out, indent=2, sort_keys=True).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
