"""Round 19 smoke-recipe check (plan "Round 19", 14:16 UTC, and its amendment 1, 15:14 UTC): an arm's smoke ``run.json``
must be R16C's smoke ``run.json`` but for the run name and the manifest. Arm R19w may also add exactly one corpus
weight, ``youtube-asl-v1=0.5767``; nothing else. The model config SHA must be R16C's, and the trainer's options added
since R16C (round 17's plan flags) must sit at their defaults. Exit 0 when the recipes match, else 1 with every
difference listed.

    python scripts/round19_recipe_check.py --arm r19w \\
        --control training-runs/tokens-rvq-d0b-prior-r16c-control-smoke/run.json \\
        --candidate training-runs/tokens-rvq-d0b-prior-r19w-refresh-weighted-smoke/run.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

YOUTUBE_WEIGHT = "youtube-asl-v1=0.5767"         # amendment 1: 99,887 / 173,202 trainer-visible YouTube records
EXTRA_WEIGHTS = {"r19": [], "r19w": [YOUTUBE_WEIGHT]}
MIX = "round19-mixed-v1"
ALLOWED = {"run", "manifest_key", "manifest_sha256"}
NEW_DEFAULTS = dict(plan_tokenizer_key=None, plan_tokenizer_sha256=None, plan_tokenizer_manifest_sha256=None,
                    plan_dropout=0., plan_corruption=0., plan_corruption_max=.5, plan_corruption_source="prior",
                    plan_corruption_prior_key=None, plan_corruption_prior_sha256=None, plan_memory=True,
                    init_text_key=None, init_text_sha256=None)


def differences(control: dict, candidate: dict, arm: str) -> list[str]:
    """Every way ``candidate``'s recipe differs from ``control``'s beyond what arm ``arm`` may change."""
    a, b = control["args"], candidate["args"]
    problems = []
    for k in a:
        if k in ALLOWED:
            continue
        want = a[k] + EXTRA_WEIGHTS[arm] if k == "corpus_weight" else a[k]
        if b.get(k, object()) != want:
            problems.append(f"args.{k}: {want!r} expected, {b.get(k, '<absent>')!r} found")
    problems += [f"new args.{k} = {b[k]!r}" for k in set(b) - set(a)
                 if k not in NEW_DEFAULTS or b[k] != NEW_DEFAULTS[k]]
    if control.get("config_sha256") != candidate.get("config_sha256"):
        problems.append(f"config_sha256 {control.get('config_sha256')} -> {candidate.get('config_sha256')}")
    if (candidate.get("manifest_key") or "").rsplit("/", 2)[-2:-1] != [MIX] or control.get("kind") != candidate.get("kind"):
        problems.append(f"manifest {candidate.get('manifest_key')} / kind {candidate.get('kind')}")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=sorted(EXTRA_WEIGHTS), required=True)
    p.add_argument("--control", required=True, help="R16C's smoke run.json (local path or store key)")
    p.add_argument("--candidate", required=True, help="the arm's smoke run.json (local path or store key)")
    args = p.parse_args(argv)
    store = None
    if not (Path(args.control).exists() and Path(args.candidate).exists()):
        from signtranslator.data.text_motion_dataset import R2Store
        store = R2Store(os.environ)
    load = lambda s: json.loads(Path(s).read_bytes() if Path(s).exists() else store.get(s))  # noqa: E731
    control, candidate = load(args.control), load(args.candidate)
    problems = differences(control, candidate, args.arm)
    print(f"{args.arm} smoke recipe vs R16C smoke: config_sha256 {candidate.get('config_sha256')}; train records "
          f"{control.get('train_records')} -> {candidate.get('train_records')}; {len(problems)} differences")
    for line in problems:
        print("  " + line)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
