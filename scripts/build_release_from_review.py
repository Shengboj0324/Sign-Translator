"""Pinned release spec of the generator setting selected by the exit gate and confirmed by the strict review.

Building is refused unless the strict review (``scripts/strict_prior_review.py``) passed: a failed review is a failed
review, not a release. The setting is the one the review judged, from its ``selected`` block:
- guidance;
- per-part temperatures;
- head smoothing;
- the face calibration it applied, inlined so the release pins it.

The checkpoint is pinned by SHA-256, and the planner and training provenance come from ``--base-release``. The
gate's failing checks are listed under ``known_limits`` for the model card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def part_temperature(spec: str) -> dict:
    return {k: float(v) for k, v in (item.split("=") for item in spec.split(","))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--strict-key", required=True)
    parser.add_argument("--gate-key", required=True)
    parser.add_argument("--checkpoint-key", required=True, help="the reviewed generator checkpoint")
    parser.add_argument("--manifest-sha256", required=True, help="the generator's training manifest")
    parser.add_argument("--base-release", type=Path, required=True, help="planner and provenance come from here")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from signtranslator.serving.release import GeneratorSpec, Pinned, Release
    from train_text_to_motion import make_store
    store = make_store(args.store)
    review = json.loads(store.get(args.strict_key))
    if not review.get("pass"):
        failed = [k for k, r in review.get("rules", {}).items() if not r.get("pass")]
        print(f"strict review did not pass (failing: {', '.join(failed) or 'verdict missing'}): no release built")
        return 4
    gate = json.loads(store.get(args.gate_key))
    sel = review["selected"]
    if (sel["run"], sel["setting"]) != (gate["selected"]["run"], gate["selected"]["setting"]):
        raise SystemExit("the strict review judged a different setting than the gate selected")
    base = Release.from_dict(json.loads(args.base_release.read_text()))
    checkpoint = Pinned(args.checkpoint_key, hashlib.sha256(store.get(args.checkpoint_key)).hexdigest())
    generator = GeneratorSpec(checkpoint, args.manifest_sha256, condition="english", guidance=float(sel["guidance"]),
                              part_temperature=part_temperature(sel["part_temperature"]),
                              head_smoothing=float(sel["head_smoothing"]), head_damping=float(sel.get("head_damping", 1.)),
                              face_calibration=review.get("face_calibration"))
    release = Release(name=base.name, version=args.version, generator=generator, provenance=base.provenance,
                      planner=base.planner, fps=base.fps, seconds_per_sign=base.seconds_per_sign,
                      max_text_chars=base.max_text_chars)
    data = release.to_dict()
    data["selection"] = dict(gate_key=args.gate_key, strict_key=args.strict_key, run=sel["run"], setting=sel["setting"],
                             known_limits=gate["selected"].get("failing", []))
    args.output.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
    print(f"release {release.version}: {sel['run']} / {sel['setting']}, sha256 {release.sha256()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
