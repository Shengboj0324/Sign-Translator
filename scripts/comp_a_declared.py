"""Post-S2 workstream A: is this full read declared, for this code? ``comp_a_declared.py COMMIT TEST`` exits 0 only when
``runpod/fleet/comp_a_declared.json`` (committed after the declaration) names COMMIT, this test's rule text (``comp_a_read.DRAFT_RULES``; whitespace aside,
because the plan wraps lines), and the digest of the code that is running (``provenance.code_digest``, with this job's shell file). That holds with or
without git. With a git checkout COMMIT must also be a real ancestor of HEAD, and its ``docs/NEXT_PHASE_PLAN.md`` must hold the text. A pod's bundle
(``git archive``, no ``.git``) cannot look up a commit, so there the bundle's own plan must hold it, and a repository test holds the record to the commit.

``comp_a_declared.py record COMMIT`` prints the record to commit, at declaration time. The digest covers scripts, the package and the job file, not this
record or the plan, so committing the record does not move it; any later code change does, and the read stops until a new declaration records it.
Nothing is declared by this script.
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

from signtranslator.eval.provenance import code_digest  # noqa: E402

PLAN = "docs/NEXT_PHASE_PLAN.md"
RECORD = "runpod/fleet/comp_a_declared.json"
JOB = "runpod/fleet/comp_a.sh"


def normal(text: str) -> str:
    return " ".join(text.split())


def rule_hash(text: str) -> str:
    return hashlib.sha256(normal(text).encode()).hexdigest()


def holds_rule(plan_text: str, rule_text: str) -> bool:
    return normal(rule_text) in normal(plan_text)


def git(root: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def make_record(commit: str, *, job: str = JOB, rules=None) -> dict:
    """The record of a declaration at COMMIT, for this code."""
    from comp_a_read import DRAFT_RULES
    return dict(commit=commit, code_digest=code_digest(extra_files=(job,)), rules={t: rule_hash(r) for t, r in (rules or DRAFT_RULES).items()})


def check(commit: str, test: str, root: Path = ROOT, *, record_path: str = RECORD, job: str = JOB, rules=None) -> str:
    """How this is declared, or SystemExit."""
    from comp_a_read import DRAFT_RULES
    DRAFT_RULES = rules or DRAFT_RULES
    if test not in DRAFT_RULES:
        raise SystemExit(f"{test}: no draft rule")
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit or ""):
        raise SystemExit(f"{commit!r} is not a commit")
    rule = DRAFT_RULES[test]
    record = root / record_path
    if not record.exists():
        raise SystemExit(f"no {record_path}: nothing shows a declaration for this code")
    payload = json.loads(record.read_text())
    if payload.get("commit") != commit:
        raise SystemExit(f"{record_path} names {payload.get('commit')}, not {commit}")
    if payload.get("rules", {}).get(test) != rule_hash(rule):
        raise SystemExit(f"{record_path} does not hold this rule of {test}")
    digest = code_digest(extra_files=(job,))
    if payload.get("code_digest") != digest:
        raise SystemExit(f"{record_path} was declared for code {payload.get('code_digest')}; this code is {digest}")
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
    if not holds_rule((root / PLAN).read_text(), rule):
        raise SystemExit(f"this bundle's plan does not hold the rule of {test}")
    return "record"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit("usage: comp_a_declared.py COMMIT TEST, or comp_a_declared.py record COMMIT")
    if argv[0] == "record":
        print(json.dumps(make_record(argv[1]), indent=1, sort_keys=True), flush=True)
        return 0
    print(f"declared {argv[0]} {argv[1]} by {check(argv[0], argv[1])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
