"""Post-S2 A6's checks at run time. ``env``: this pod's python, torch, transformers and tokenizers are A5's pinned ones. ``same-pod KEY``: a stored base report came from this code and this pod's environment, GPU included, so a draw's arms
never mix. ``started KEY``: write-once marker that an arm of an attempt began; an arm that began and did not finish is never rerun under that attempt. ``a5-done``: A5's summary is stored, so A6's draws reveal no per-draw number of A5's read."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import comp_a_lib as lib  # noqa: E402
from comp_a5_guard import check_environment  # noqa: E402
from comp_a6_read import JOB_FILE  # noqa: E402

A5_SUMMARY = "acquisition/2026-10-17-v1/youtube-asl-v1/round20/comp-a5/a5-summary.json"


def check_same_pod(report: dict) -> None:
    stored = ((report.get("settings") or {}).get("round20") or {}).get("provenance")
    here = lib.provenance_row(JOB_FILE)
    if not stored or stored != here:
        raise SystemExit(f"base came from other code or another environment: {stored} against this pod's {here}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from train_text_to_motion import make_store
    if argv == ["env"]:
        check_environment()
    elif argv[:1] == ["same-pod"] and len(argv) == 2:
        check_same_pod(json.loads(make_store("r2").get(argv[1])))
    elif argv[:1] == ["started"] and len(argv) == 2:
        store = make_store("r2")
        if store.exists(argv[1]):
            raise SystemExit(f"{argv[1]} exists")
        store.put(argv[1], b"started")
    elif argv == ["a5-done"]:
        if not make_store("r2").exists(A5_SUMMARY):
            raise SystemExit(f"{A5_SUMMARY} is not stored: A6's draws wait for A5's summary")
    else:
        raise SystemExit("usage: comp_a6_guard.py env | same-pod KEY | started KEY | a5-done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
