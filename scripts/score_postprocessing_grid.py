"""Round 3: post-processing candidates of one generator setting, scored on every validation signer at once.

For a sweep setting ``--run`` evaluated per signer (``evaluate_text_to_motion.py --signer-component G
--save-generated <generated-template with run and group>``), this loads each group's stored states and their
references **once**. Then, for every head-smoothing value in ``--head-smoothing``, it scores two variants:
- ``<run>-hs<h>``: head smoothing only;
- ``<run>-hs<h>-fc``: head smoothing plus a face calibration estimated on the **pooled** validation signers
  (all groups together, per guidance) and applied unchanged to each group, so no signer gets a calibration fitted to
  itself.

Each variant is written per group as ``<evaluations>/<run>-hs<h>[-fc]-<group>.json`` in the evaluator's schema, so
``multi_signer_gate.py`` reads it like any run. Results are keyed by the evaluator's setting names (``motion_tokens_v2
guidance X``), and the pooled calibration is stored with every result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.generation.motion_metrics import compare  # noqa: E402
from signtranslator.generation.postprocess import (calibrate_face, damp_rotation_channel,  # noqa: E402
                                                   estimate_face_calibration, smooth_rotation_channel)
from signtranslator.generation.signing_space import location_errors  # noqa: E402
from signtranslator.grammar_motion.evaluation import grammar_comparison  # noqa: E402
from score_saved_generated import CHANNELS, load_generated  # noqa: E402
from evaluate_text_to_motion import at_rate, load_state  # noqa: E402

LABEL = "motion_tokens_v2"


def setting_name(tag: str) -> str:
    """``guidance-2`` -> the evaluator's ``motion_tokens_v2 guidance 2``."""
    return f"{LABEL} guidance {tag.split('-', 1)[1]}" if tag.startswith("guidance-") else tag


def length_slices(generated, references, records) -> dict:
    """R4's measure (strict_prior_review.py): clips split at sentence-length terciles; per slice the body, hand and head
    speed and variation ratios. Round 3c gates on it, so the selection cannot pick what R4 then fails."""
    import numpy as np
    from exit_gate import ARTICULATED
    lengths = np.array([len(r["text"].split()) for r in records])
    cuts = np.percentile(lengths, [100 / 3, 200 / 3])
    out = {}
    for name, mask in (("short", lengths <= cuts[0]), ("medium", (lengths > cuts[0]) & (lengths <= cuts[1])),
                       ("long", lengths > cuts[1])):
        idx = np.nonzero(mask)[0]
        if len(idx):
            res = compare([generated[i] for i in idx], [references[i] for i in idx], ARTICULATED)
            out[name] = {f"{c}_{k}": round(float(res[c][k]), 3) for c in ARTICULATED if c in res
                         for k in ("speed_ratio", "variation_ratio")}
    return out


def smoothness(generated, references) -> dict:
    """R7's measure (smoothness_check.py): per channel, median generated >= 6 Hz amplitude over the real median."""
    import numpy as np
    from signtranslator.generation.motion_metrics import high_frequency_rms
    from smoothness_check import CHANNELS as R7_CHANNELS
    out = {}
    for c in R7_CHANNELS:
        g = [v for v in (high_frequency_rms(s, c) for s in generated) if v is not None]
        r = [v for v in (high_frequency_rms(s, c) for s in references) if v is not None]
        out[c] = round(float(np.median(g) / max(np.median(r), 1e-12)), 3) if g and r else None
    return out


def score(generated, references, records, *, round3c: bool = False, pooled: bool = False) -> dict:
    channels = tuple(c for c in CHANNELS if c in generated[0].channels and generated[0].channels[c].valid.any())
    result = compare(generated, references, channels)
    if round3c:
        result["smoothness"] = smoothness(generated, references)
        if pooled:
            result["length_slices"] = length_slices(generated, references, records)
    if all("LeftForearm" in r.channels["body"].labels for r in references):
        result["signing_space"] = location_errors(generated, references)
    result["grammar"] = grammar_comparison(generated, references, [r["text"] for r in records])
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--generated-template", required=True, help="prefix with {run} and {group}")
    parser.add_argument("--evaluations", required=True, help="store prefix the reports go under")
    parser.add_argument("--head-smoothing", default="0,0.1,0.2,0.3")
    parser.add_argument("--head-damping", default="1", help="head excursion factors (postprocess.damp_rotation_channel); "
                                                           "1 = none. Variants with a factor below 1 are named -hd<a>")
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--round3c", action="store_true",
                        help="round 3c (declared 2026-10-25): head translation smoothed with the head-rotation window "
                             "(at least 0.1 s); R7 smoothness scored in every group and R4's length slices on the pooled "
                             "group, both gated by exit_gate.score_setting")
    parser.add_argument("--pool-group", help="round 3b: also score every variant on all groups' clips pooled, written "
                                             "as group <this name> (the population envelope's input)")
    args = parser.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    by_sign = {r["sign"]: r for r in manifest["records"]}
    groups = args.groups.split(",")
    data = {}                                    # group -> tag -> (states, references, records)
    for g in groups:
        data[g] = {}
        for tag, pairs in load_generated(store, args.generated_template.format(run=args.run, group=g)).items():
            records = [by_sign[sign] for sign, _ in pairs]
            refs = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), args.fps)
                    for r in records]
            data[g][tag] = ([s for _, s in pairs], refs, records)
    missing = [g for g in groups if not data[g]]
    if missing:   # empty reports would make the gate drop this setting silently
        raise SystemExit(f"{args.run}: no stored generations for group(s) {', '.join(missing)}")
    tags = set.intersection(*(set(d) for d in data.values()))
    if not tags:
        raise SystemExit(f"{args.run}: no guidance setting stored for every group")
    import itertools
    for h, a in itertools.product((float(x) for x in args.head_smoothing.split(",")),
                                  [float(x) for x in args.head_damping.split(",")]):
        def post(s):
            s = smooth_rotation_channel(s, "head_rotation", h) if h > 0 else s
            s = damp_rotation_channel(s, "head_rotation", a) if a != 1 else s
            if args.round3c and "head_translation" in s.channels:
                from signtranslator.generation.postprocess import smooth_vector_channel
                s = smooth_vector_channel(s, "head_translation", max(h, .1))
            return s
        smoothed = {g: {tag: ([post(s) for s in data[g][tag][0]],) + data[g][tag][1:] for tag in tags} for g in groups}
        calibration = {tag: estimate_face_calibration([s for g in groups for s in smoothed[g][tag][0]],
                                                      [r for g in groups for r in smoothed[g][tag][1]])
                       for tag in tags}
        for fc in (False, True):
            variant = f"{args.run}-hs{h:g}" + (f"-hd{a:g}" if a != 1 else "") + ("-fc" if fc else "")
            for g in groups + ([args.pool_group] if args.pool_group else []):
                results = {}
                for tag in sorted(tags):
                    if g == args.pool_group:
                        states, refs, records = (sum((list(smoothed[h][tag][i]) for h in groups), []) for i in range(3))
                    else:
                        states, refs, records = smoothed[g][tag]
                    if fc:
                        states = [calibrate_face(s, calibration[tag]) for s in states]
                    results[setting_name(tag)] = score(states, refs, records, round3c=args.round3c,
                                                       pooled=g == args.pool_group)
                    if fc:
                        results[setting_name(tag)]["face_calibration"] = dict(calibration[tag], pooled_groups=groups)
                store.put(f"{args.evaluations.rstrip('/')}/{variant}-{g}.json", json.dumps(dict(
                    schema_version=1, run=variant, group=g, head_smoothing=h, head_damping=a, face_calibrated=fc,
                    generated=args.generated_template.format(run=args.run, group=g), manifest_sha256=args.manifest_sha256,
                    results=results), indent=2, sort_keys=True, default=float).encode())
                print(json.dumps({f"{variant}-{g}": {k: round(v["frechet"], 1) for k, v in results.items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
