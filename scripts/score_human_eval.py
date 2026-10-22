"""Score returned review sheets against the private key (plan item 3b; docs/HUMAN_EVALUATION_PROTOCOL.md).

Every sheet is validated (offered columns unchanged, answers in range, one hashed reviewer key per sheet); rows with
problems are reported and left out, as are reviewers who answered sheets of two groups (they saw sentences twice).
The report gives per system: four-way comprehension choice (Wilson 95% interval, chance 25%), sentence-type / negation /
topic recovery, naturalness (bootstrap interval), indicative BLEU of the free translations; each system against
``--reference`` paired by utterance (sign-flip permutation p); and Fleiss' kappa between reviewers of the same group.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.eval_framework.human_study import key_sha256, merge_answers, parse_sheet, score_study  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True, help="key/key.json from build_human_eval_packet.py")
    parser.add_argument("--sheet", type=Path, action="append", required=True, help="a returned sheet (repeatable)")
    parser.add_argument("--reference", default="reference", help="system label the others are paired against")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    key = json.loads(args.key.read_text())
    recorded = key.pop("key_sha256", None)
    if recorded is not None and recorded != key_sha256(key):
        raise SystemExit("the key does not match its recorded hash")
    sheets, problems = [], []
    for path in args.sheet:
        answers, issues = parse_sheet(path.read_text(encoding="utf-8-sig"), key)
        sheets.append(answers)
        problems += [dict(sheet=path.name, **i) for i in issues]
    answers, merge_problems = merge_answers(sheets, key)
    report = score_study(answers, key, reference=args.reference if args.reference in key["systems"] else None)
    report.update(problems=problems + merge_problems, sheets=len(args.sheet), key_sha256=recorded,
                  partition=key.get("partition"), manifest_sha256=key.get("manifest_sha256"))
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({s: dict(choice=v["comprehension_choice"], naturalness=v.get("naturalness"))
                      for s, v in report["systems"].items()}))
    if problems or merge_problems:
        print(f"{len(problems) + len(merge_problems)} problems: see {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
