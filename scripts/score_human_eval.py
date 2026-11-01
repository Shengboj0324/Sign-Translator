"""Score returned review sheets against the private key (plan item 3b; docs/HUMAN_EVALUATION_PROTOCOL.md).

Every sheet is validated (offered columns unchanged, answers in range, one hashed reviewer key per sheet); rows with
problems are reported and left out, as are reviewers who answered sheets of two groups (they saw sentences twice).
The report gives per system: four-way comprehension choice (Wilson 95% interval, chance 25%), sentence-type / negation /
topic recovery, naturalness (bootstrap interval), indicative BLEU of the free translations; each system against
``--reference`` paired by utterance (sign-flip permutation p); and Fleiss' kappa between reviewers of the same group.

A two-stage key (How2Sign read): ``--sheet`` takes the stage 1 sheets and ``--stage2-sheet`` the stage 2 sheets, with
``--stage1-receipts`` from ``build_human_eval_packet.py how2sign-stage2``; a stage 2 answer without a matching stage 1 hash
is refused.

``--bar`` adds the release-bar read (``report["bar"]``, ``human_study.bar_read``): the declared terms only
(docs/HUMAN_EVALUATION_PROTOCOL.md section 6, docs/NEXT_PHASE_PLAN.md "Rater read terms defined" and "interval method and
sheet order changed"), a pass or fail per condition and per system, given the private ``--roster``. When the read cannot
be made it writes why and exits 2. Wrist speed, timing and unseen signers are read elsewhere, so it never allows a release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.eval_framework.human_study import (BarRefused, bar_read, key_sha256, merge_answers,  # noqa: E402
                                                       parse_roster, parse_sheet, read_two_stage, score_study)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True, help="key/key.json from build_human_eval_packet.py")
    parser.add_argument("--sheet", type=Path, action="append", required=True, help="a returned sheet (repeatable)")
    parser.add_argument("--reference", default="reference", help="system label the others are paired against")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bar", action="store_true", help="also read the release-bar terms (the key needs a negated "
                        "share)")
    parser.add_argument("--roster", type=Path, help="--bar: private JSON {reviewer key: group}")
    parser.add_argument("--stage2-sheet", type=Path, action="append", default=[], help="a returned stage 2 sheet")
    parser.add_argument("--stage1-receipts", type=Path, help="two-stage key: JSON {reviewer key: stage 1 hash}")
    parser.add_argument("--bootstrap", type=int, default=2000, help="--bar: replicates of the bootstrap reported beside "
                        "method D")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    key = json.loads(args.key.read_text())
    recorded = key.pop("key_sha256", None)
    if recorded is not None and recorded != key_sha256(key):
        raise SystemExit("the key does not match its recorded hash")
    if args.bar and not args.roster:
        raise SystemExit("--bar needs the private --roster")
    sheets, problems = [], []
    if key.get("stages") == 2:
        if not args.stage1_receipts:
            raise SystemExit("a two-stage study needs --stage1-receipts")
        answers, problems = read_two_stage([p.read_bytes() for p in args.sheet], [p.read_bytes() for p in args.stage2_sheet],
                                           key, json.loads(args.stage1_receipts.read_text()))
        sheets.append(answers)
    else:
        if args.bar:
            raise SystemExit("--bar reads a two-stage study: this key has one stage")
        for path in args.sheet:
            answers, issues = parse_sheet(path.read_text(encoding="utf-8-sig"), key)
            sheets.append(answers)
            problems += [dict(sheet=path.name, **i) for i in issues]
    answers, merge_problems = merge_answers(sheets, key)
    report = score_study(answers, key, reference=args.reference if args.reference in key["systems"] else None)
    report.update(problems=problems + merge_problems, sheets=len(args.sheet), key_sha256=recorded,
                  partition=key.get("partition"), manifest_sha256=key.get("manifest_sha256"))
    refused = None
    if args.bar:
        try:
            report["bar"] = bar_read(answers, key, reference=args.reference, roster=parse_roster(args.roster.read_text()),
                                     problems=problems + merge_problems, boot=args.bootstrap, seed=args.bootstrap_seed)
        except BarRefused as e:
            report["bar"] = e.details
            refused = str(e)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({s: dict(choice=v["comprehension_choice"], naturalness=v.get("naturalness"))
                      for s, v in report["systems"].items()}))
    if refused:
        print(f"the bar read was refused: {refused}", file=sys.stderr)
        return 2
    if args.bar:
        print(json.dumps({s: dict(passes_rater_terms=v["passes_rater_terms"],
                                  conditions={c: d["result"] for c, d in v["conditions"].items()})
                          for s, v in report["bar"]["systems"].items() if "conditions" in v}))
    if problems or merge_problems:
        print(f"{len(problems) + len(merge_problems)} problems: see {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
