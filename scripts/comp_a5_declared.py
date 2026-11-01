"""Post-S2 A5's declaration, apart from A1 to A4's: ``comp_a5_declared.py COMMIT`` exits 0 only when ``runpod/fleet/comp_a5_declared.json`` names COMMIT, A5's rule text and the digest of
the running code with A5's job file (``comp_a_declared.check``); ``comp_a5_declared.py record COMMIT`` prints that record at declaration time. Nothing is declared by this script."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import comp_a_declared as dc  # noqa: E402
from comp_a5_read import JOB_FILE, RULE  # noqa: E402

RECORD = "runpod/fleet/comp_a5_declared.json"
RULES = {"a5": RULE}


def check(commit: str, root: Path = ROOT) -> str:
    return dc.check(commit, "a5", root, record_path=RECORD, job=JOB_FILE, rules=RULES)


def make_record(commit: str) -> dict:
    return dc.make_record(commit, job=JOB_FILE, rules=RULES)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "record":
        print(json.dumps(make_record(argv[1]), indent=1, sort_keys=True), flush=True)
        return 0
    if len(argv) != 1:
        raise SystemExit("usage: comp_a5_declared.py COMMIT, or comp_a5_declared.py record COMMIT")
    print(f"declared {argv[0]} a5 by {check(argv[0])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
