"""Post-S2 A5's checks at run time. ``env``: this pod's python, torch, transformers and tokenizers are the pinned ones. ``same-pod KEY``: a stored margin0 report came from this code and this pod's
environment, GPU included, so a draw's two arms never mix (a mix happened in A3's stored read). ``started KEY``: write-once marker that an arm of an attempt began; an arm that began and did not
finish is never rerun under that attempt, because a failure reruns both arms under the next one."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import comp_a_lib as lib  # noqa: E402
from comp_a5_read import JOB_FILE, PINNED  # noqa: E402


def check_environment() -> None:
    from signtranslator.eval.provenance import environment
    env = environment()
    wrong = {k: env.get(k) for k, v in PINNED.items() if env.get(k) != v}
    if wrong:
        raise SystemExit(f"this pod's packages are not the pinned {PINNED}: {wrong}")


def check_same_pod(report: dict) -> None:
    stored = ((report.get("settings") or {}).get("round20") or {}).get("provenance")
    here = lib.provenance_row(JOB_FILE)
    if not stored or stored != here:
        raise SystemExit(f"margin0 came from other code or another environment: {stored} against this pod's {here}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from train_text_to_motion import make_store
    if argv[:1] == ["env"] and len(argv) == 1:
        check_environment()
    elif argv[:1] == ["same-pod"] and len(argv) == 2:
        check_same_pod(json.loads(make_store("r2").get(argv[1])))
    elif argv[:1] == ["started"] and len(argv) == 2:
        store = make_store("r2")
        if store.exists(argv[1]):
            raise SystemExit(f"{argv[1]} exists")
        store.put(argv[1], b"started")
    else:
        raise SystemExit("usage: comp_a5_guard.py env | same-pod KEY | started KEY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
