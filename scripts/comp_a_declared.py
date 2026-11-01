"""Post-S2 workstream A: is this full read declared? ``comp_a_declared.py COMMIT TEST`` exits 0 only when COMMIT is a real ancestor commit of the
code that runs and its ``docs/NEXT_PHASE_PLAN.md`` holds this test's draft rule text (``comp_a_read.DRAFT_RULES``; whitespace aside, because the plan
wraps lines).

With a git checkout, ``git merge-base --is-ancestor`` and ``git show`` decide. A pod's bundle (``git archive``, no ``.git``) cannot look up a commit, so it
needs ``runpod/fleet/comp_a_declared.json``, committed after the declaration: ``{"commit": ..., "rules": {test: sha256 of the normalised text}}``. The
bundle's own plan must hold the text, and COMMIT must be the file's. A repository test holds that file to the ancestor and the text whenever it exists.
No file and no git: stop. Nothing is declared by this script.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

PLAN = "docs/NEXT_PHASE_PLAN.md"
RECORD = "runpod/fleet/comp_a_declared.json"


def normal(text: str) -> str:
    return " ".join(text.split())


def rule_hash(text: str) -> str:
    return hashlib.sha256(normal(text).encode()).hexdigest()


def holds_rule(plan_text: str, rule_text: str) -> bool:
    return normal(rule_text) in normal(plan_text)


def git(root: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def check(commit: str, test: str, root: Path = ROOT) -> str:
    """Why this is declared, or SystemExit."""
    from comp_a_read import DRAFT_RULES
    if test not in DRAFT_RULES:
        raise SystemExit(f"{test}: no draft rule")
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit or ""):
        raise SystemExit(f"{commit!r} is not a commit")
    rule = DRAFT_RULES[test]
    if (root / ".git").exists():
        full = git(root, "rev-parse", "--verify", f"{commit}^{{commit}}")
        if full.returncode:
            raise SystemExit(f"{commit} is not a commit of this repository")
        if git(root, "merge-base", "--is-ancestor", full.stdout.strip(), "HEAD").returncode:
            raise SystemExit(f"{commit} is not an ancestor of this code")
        shown = git(root, "show", f"{full.stdout.strip()}:{PLAN}")
        if shown.returncode or not holds_rule(shown.stdout, rule):
            raise SystemExit(f"{commit}'s plan does not hold the rule of {test}")
        return "git"
    record = root / RECORD
    if not record.exists():
        raise SystemExit(f"no .git and no {RECORD}: this bundle cannot show a declaration")
    payload = json.loads(record.read_text())
    if payload.get("commit") != commit:
        raise SystemExit(f"{RECORD} names {payload.get('commit')}, not {commit}")
    if payload.get("rules", {}).get(test) != rule_hash(rule):
        raise SystemExit(f"{RECORD} does not hold this rule of {test}")
    if not holds_rule((root / PLAN).read_text(), rule):
        raise SystemExit(f"this bundle's plan does not hold the rule of {test}")
    return "record"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit("usage: comp_a_declared.py COMMIT TEST")
    print(f"declared {argv[0]} {argv[1]} by {check(argv[0], argv[1])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
