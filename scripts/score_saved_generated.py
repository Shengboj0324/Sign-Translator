"""Re-score stored generated states (``evaluate_text_to_motion.py --save-generated``) after post-processing.

Post-processing settings (``--head-smoothing``) need no regeneration: this reads the stored
states and the references of the same clips, applies the setting, and writes an evaluation
report in ``evaluate_text_to_motion.py``'s schema (per-channel accuracy and articulation,
Fréchet distance, signing-space errors, grammar comparison), so ``exit_gate.py`` can read it
beside the generated runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.generation.motion_metrics import compare  # noqa: E402
from signtranslator.generation.postprocess import (calibrate_face, estimate_face_calibration,  # noqa: E402
                                                   smooth_rotation_channel)
from signtranslator.generation.signing_space import location_errors  # noqa: E402
from signtranslator.grammar_motion.evaluation import grammar_comparison  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402
from evaluate_text_to_motion import at_rate, load_state  # noqa: E402
from train_text_to_motion import make_store  # noqa: E402

CHANNELS = ("body", "left_hand", "right_hand", "head_rotation", "root_translation", "head_translation", "left_gaze",
            "right_gaze")


def load_generated(store, prefix: str) -> dict:
    """tag -> list of (sign, state) under ``prefix/<tag>/<sign>.npz``."""
    groups = {}
    for key in sorted(k for k in store.list(prefix.rstrip("/") + "/") if k.endswith(".npz")):
        payload = store.get(key)
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "s.npz"; path.write_bytes(payload)
            state = MultichannelMotion.load(path, expected_sha256=hashlib.sha256(payload).hexdigest())
        tag, sign = key.rsplit("/", 2)[-2], key.rsplit("/", 1)[-1][:-4]
        groups.setdefault(tag, []).append((sign, state))
    return groups


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--generated", required=True, help="a --save-generated prefix")
    parser.add_argument("--head-smoothing", type=float, default=0.)
    parser.add_argument("--fps", type=float, help="resample the references to this rate (as the evaluation did)")
    parser.add_argument("--face-calibrate", action="store_true",
                        help="per-blendshape face calibration (postprocess.estimate_face_calibration) estimated on these "
                             "clips and applied; stored with the result")
    parser.add_argument("--output-key", required=True)
    args = parser.parse_args(argv)
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    by_sign = {r["sign"]: r for r in manifest["records"]}
    results = {}
    for tag, pairs in load_generated(store, args.generated).items():
        generated = [smooth_rotation_channel(s, "head_rotation", args.head_smoothing) if args.head_smoothing > 0 else s
                     for _, s in pairs]
        records = [by_sign[sign] for sign, _ in pairs]
        references = [load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]) for r in records]
        if args.fps:
            references = [at_rate(r, args.fps) for r in references]
        calibration = None
        if args.face_calibrate:
            calibration = estimate_face_calibration(generated, references)
            generated = [calibrate_face(s, calibration) for s in generated]
        channels = tuple(c for c in CHANNELS if c in generated[0].channels and generated[0].channels[c].valid.any())
        name = f"{tag} head-smoothing {args.head_smoothing:g}" + (" face-calibrated" if calibration else "")
        results[name] = compare(generated, references, channels)
        if calibration:
            results[name]["face_calibration"] = calibration
        if all("LeftForearm" in r.channels["body"].labels for r in references):
            results[name]["signing_space"] = location_errors(generated, references)
        results[name]["grammar"] = grammar_comparison(generated, references, [r["text"] for r in records])
        print(json.dumps({name: {k: results[name][k] for k in ("frechet", "head_rotation") if k in results[name]}},
                         default=float), flush=True)
    store.put(args.output_key, json.dumps(dict(schema_version=1, generated=args.generated,
                                               head_smoothing=args.head_smoothing, face_calibrated=args.face_calibrate,
                                               manifest_sha256=args.manifest_sha256,
                                               results=results), indent=2, sort_keys=True, default=float).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
