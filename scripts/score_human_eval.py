"""Score returned review sheets against the private key (plan item 3b; docs/HUMAN_EVALUATION_PROTOCOL.md).

Every sheet is validated (offered columns unchanged, answers in range, one hashed reviewer key per sheet); rows with
problems are reported and left out, as are reviewers who answered sheets of two groups (they saw sentences twice).
The report gives per system: four-way comprehension choice (Wilson 95% interval, chance 25%), sentence-type / negation /
topic recovery, naturalness (bootstrap interval), indicative BLEU of the free translations; each system against
``--reference`` paired by utterance (sign-flip permutation p); and Fleiss' kappa between reviewers of the same group.

A two-stage key (How2Sign read): ``--sheet`` takes the stage 1 sheets and ``--stage2-sheet`` the stage 2 sheets, with
``--stage1-receipts`` from ``build_human_eval_packet.py how2sign-stage2``; a stage 2 answer without a matching stage 1 hash
is refused. Every read of such a key, a pilot's included, needs the private ``--roster``: ``report["coverage"]`` gives designed
rows against returned rows per system and per group, every viewable row without a choice is listed in ``report["problems"]``, and
under 95% the run writes its report and exits 2, unless ``--allow-low-coverage``, which never touches ``--bar``'s own gate.
``report["stage1"]`` keeps what came back in stage 1 whether or not stage 2 did. Sentences left out of the draw for having no
selected word are in ``report["excluded_no_selected_word"]``.

A key holding the rater pilot's word-format arms adds ``report["word_format"]`` (``human_study.word_format_read``): paired
contrasts of composition, composition_shuffled, oracle and the reference, report only. ``--bar`` refuses such a key.

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
from signtranslator.eval_framework.human_study import (BarRefused, SheetError, bar_read, check_roster, decode_sheet,  # noqa: E402
                                                       key_sha256, load_roster, merge_answers, parse_sheet, people_problems,
                                                       read_stages, score_study, stage1_summary, study_coverage, word_format_read)


def main(argv=None) -> int:
    try:
        return run(argv)
    except SheetError as error:
        raise SystemExit(str(error))


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True, help="key/key.json from build_human_eval_packet.py")
    parser.add_argument("--sheet", type=Path, action="append", required=True, help="a returned sheet (repeatable)")
    parser.add_argument("--reference", default="reference", help="system label the others are paired against")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bar", action="store_true", help="also read the release-bar terms (the key needs a negated "
                        "share)")
    parser.add_argument("--roster", type=Path, help="two-stage key, or --bar: private JSON {reviewer key: group}")
    parser.add_argument("--allow-low-coverage", action="store_true", help="two-stage key: exit 0 under 95%% designed-row coverage, and "
                        "say so in the report; --bar's own gate is not touched")
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
    sheets, problems, stage1, roster = [], [], None, None
    if args.roster:
        roster = load_roster(args.roster)
    if key.get("stages") == 2:
        if not args.stage1_receipts:
            raise SystemExit("a two-stage study needs --stage1-receipts")
        if roster is None:
            raise SystemExit("a two-stage study needs the private --roster, a pilot's too: coverage is read against it")
        issues = check_roster(roster, key)
        if issues:
            raise SystemExit(f"{args.roster}: {'; '.join(issues)}")
        read = read_stages([p.read_bytes() for p in args.sheet], [p.read_bytes() for p in args.stage2_sheet], key,
                           json.loads(args.stage1_receipts.read_text()))
        problems, stage1 = read["problems"], read["stage1"]
        sheets.append(read["answers"])
    else:
        if args.bar:
            raise SystemExit("--bar reads a two-stage study: this key has one stage")
        for path in args.sheet:
            answers, issues = parse_sheet(decode_sheet(path.read_bytes()), key)
            sheets.append(answers)
            problems += [dict(sheet=path.name, **i) for i in issues]
    answers, merge_problems = merge_answers(sheets, key)
    counted, people = answers, []
    if key.get("stages") == 2:
        counted, people = people_problems(answers, key, roster)
        stage1, stage1_people = people_problems(stage1, key, roster)
        named = {p["reviewer_sha256"] for p in people}
        people += [p for p in stage1_people if p["reviewer_sha256"] not in named]
    report = score_study(counted, key, reference=args.reference if args.reference in key["systems"] else None)
    report.update(problems=problems + merge_problems + people, sheets=len(args.sheet), key_sha256=recorded,
                  partition=key.get("partition"), manifest_sha256=key.get("manifest_sha256"),
                  excluded_no_selected_word=key.get("excluded_no_selected_word"), avatar_acceptance=key.get("avatar_acceptance"),
                  negation_read_first=key.get("negation_read_first"))
    short = None
    if key.get("stages") == 2:
        report["coverage"] = study_coverage(counted, key, roster, problems + merge_problems)
        report["stage1"] = stage1_summary(stage1, key)
        short = report["coverage"]["short"]
        report["coverage"]["allowed_below_threshold"] = bool(short and args.allow_low_coverage)
    if key.get("word_format"):
        report.update(report_only_systems=key["report_only_systems"], word_format=word_format_read(counted, key, stage1=stage1))
    refused = None
    if args.bar:
        try:
            report["bar"] = bar_read(answers, key, reference=args.reference, roster=roster,
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
    listed = problems + merge_problems + people
    if listed:
        print(f"{len(listed)} problems: see {args.output}", file=sys.stderr)
    if people:
        print(f"{len(people)} raters were not counted: see {args.output}", file=sys.stderr)
        return 2
    if short and not args.allow_low_coverage:
        print(f"designed-row coverage is under 95%: {short}. The report is written. Pass --allow-low-coverage to read it anyway",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
