"""Post-S2 workstream A's shared reader (reports/post_s2_plan.md sections 1 and 3): one arm's merged report, read on its stored
hypotheses and rows. No model loads. D20 only; report-only; write-once. Nothing adopts from it.

Each form gets: raw BLEU and its gap share (form - blank) / (real - blank), both with sentence-bootstrap intervals; likelihood
minus blank and its gap share; v2 spotting; wrist speed, spread, hands and length against real; loop share and word echo. A form
at predicted length is read against ``blank@predicted``, and its true-length blank is reported beside it. ``real`` stays the true
clip in every share, as there is no predicted-length real.

Each comparison (``--comparator KEY:FORM``) adds ``round20_compare.compare_forms`` (refused unless the reports are paired) and the
paired differences of share, loop and echo. ``instrument_check`` counts identical ``real`` and ``blank`` hypotheses across the two
runs. Device or an unpinned ``transformers`` can move them, so a difference is flagged and never hidden.

``draft_rule`` evaluates the draft text in ``DRAFT_RULES`` with ``declared`` false unless ``--declared-commit`` names the plan
commit that declared it. It holds no "adopted" field, and this script never writes one. ``gate`` exits 0 only when an A3 oracle
read has both paired points above 0 against the same-timing r10 oracle (the job's stop before generating new entries).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import comp_a_lib as lib  # noqa: E402

PREDICTED = "@predicted"
TESTS = ("a1", "a3-oracle", "a3-gen", "a4")
DRAFT_RULES = dict(
    a1=("Against a9@predicted, on D20: paired BLEU and paired likelihood minus blank@predicted both above 0; v2 spotting's paired "
        "lower bound above -0.02; wrist at most 1.25x real; spread at least 0.8x a9@predicted's. Control: real D20 with the same arm "
        "scales must not lose more than 10 points of real's gap to blank; if it does the arm fails however composition reads. The "
        "margin-0 arm scales nothing, so it has no control. Selected on D20, confirmed nowhere: C20 is closed."),
    **{"a3-oracle": ("Against the r10 oracle at the same predicted timing and margin 0: paired BLEU and paired likelihood minus "
                     "blank@predicted both above 0. a1 is reported beside it and does not decide, because a1's timing differs."),
       "a3-gen": ("Against the margin-0 a9 arm at predicted timing: paired BLEU and paired likelihood minus blank@predicted both "
                  "above 0; v2 spotting's paired lower bound above -0.02; wrist no more than 1.10x that arm's, because added words "
                  "must not be bought with speed; hands tracked at least 0.9x that arm's."),
       "a4": ("Against a1 at a1's own settings: the paired BLEU lower bound above 0 and paired likelihood above 0 reopen section 1's "
              "tokenizer bound (word source still has room), and the bound is then re-stated. Otherwise the bound stands. Nothing is "
              "adopted: this is a ceiling read.")})


def blank_of(form: str, forms) -> str:
    return f"blank{PREDICTED}" if PREDICTED in form and f"blank{PREDICTED}" in forms else "blank"


def check_arm(report: dict, *, smoke: bool) -> None:
    """D20, report-only, and all of D20 unless this is a smoke."""
    s = report.get("settings") or {}
    sentences = (s.get("round20") or {}).get("sentences") or {}
    if sentences.get("sha256") != lib.D20_SHA or sentences.get("key") != lib.D20_KEY:
        raise SystemExit(f"this report read {sentences.get('key')}: this reader reads D20 only")
    if not report.get("report_only"):
        raise SystemExit("this report is not report-only")
    if not smoke and report.get("clips") != 500:
        raise SystemExit(f"this report read {report.get('clips')} clips, not all of D20")
    if smoke and report.get("clips", 99) > 5:
        raise SystemExit("a smoke reads at most 5 clips")


def _share_samples(stats, real, blank, boot: int, seed: int = 0):
    from signtranslator.eval.s0_metrics import bleu_from_statistics
    n = len(stats)
    idx = np.random.default_rng(seed).integers(0, n, (boot, n))
    f, r, b = (bleu_from_statistics(x[idx].sum(1)) for x in (stats, real, blank))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (f - b) / (r - b)
    return out[np.isfinite(out)]


def share_row(stats, real, blank, *, boot: int) -> dict:
    """(form - blank) / (real - blank) on the same sentences, with the sentence bootstrap's interval."""
    from signtranslator.eval.s0_metrics import bleu_from_statistics
    f, r, b = (float(bleu_from_statistics(x.sum(0))) for x in (stats, real, blank))
    point = (f - b) / (r - b) if r != b else None
    samples = _share_samples(stats, real, blank, boot)
    ci = [round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)] if len(samples) else None
    return dict(share=None if point is None else round(point, 4), ci95=ci, bleu=round(f, 4), real=round(r, 4), blank=round(b, 4))


def loop_echo(report: dict, form: str, *, boot: int) -> dict:
    from round20_lib import paired_mean
    hyps = report["forms"][form]["hypotheses"]
    words = [p["stage1_words"] for p in report["per_clip"]]
    refs = report["references"]
    rows, blank_rows, real_rows = (lib.echo_rows(report["forms"][f]["hypotheses"], words, refs) for f in (form, "blank", "real"))
    rate = lib.echo_share(rows)
    blank, real = lib.echo_share(blank_rows), lib.echo_share(real_rows)
    lift = None if None in (rate, blank, real) or real == blank else round((rate - blank) / (real - blank), 4)
    out = dict(loop_share=paired_mean(lib.loop_flags(hyps), boot=boot), word_echo=None if rate is None else round(rate, 4),
               echo_lift_over_blank_relative_to_real=lift, echo_content_words=sum(b for _, b in rows))
    extra = ((report.get("round20") or {}).get("vocabulary_extra") or {}).get("new_entries")
    if extra:
        out["new_entry_echo"] = lib.new_entry_echo(hyps, words, refs, extra)
    return out


def form_read(report: dict, form: str, *, boot: int) -> dict:
    """Everything one form reports (module docstring)."""
    from round20_lib import paired_mean
    from signtranslator.eval.s0_metrics import bleu_statistics, length_ratio
    refs = report["references"]
    forms = report["forms"]
    blank = blank_of(form, forms)
    stats = bleu_statistics(forms[form]["hypotheses"], refs)
    real, blank_stats = bleu_statistics(forms["real"]["hypotheses"], refs), bleu_statistics(forms[blank]["hypotheses"], refs)
    row = forms[form]
    out = dict(form=form, blank=blank, bleu=row["bleu"], ci95=row["ci95"], gap_share=share_row(stats, real, blank_stats, boot=boot),
               length_ratio=length_ratio(stats), **loop_echo(report, form, boot=boot))
    if blank != "blank":
        out["gap_share_true_length_blank"] = share_row(stats, real, bleu_statistics(forms["blank"]["hypotheses"], refs), boot=boot)
    ll = (report.get("likelihood") or {}).get("per_form") or {}
    if form in ll and blank in ll and form != blank:
        a, b, r = ll[form], ll[blank], ll.get("real")
        minus = paired_mean(a, b, boot=boot)
        mean_of = lambda x: float(np.mean([v for v in x if v is not None]))          # noqa: E731
        out["likelihood_minus_blank"] = minus
        if r:
            gap = mean_of(r) - mean_of(b)
            out["likelihood_gap_share"] = round((mean_of(a) - mean_of(b)) / gap, 4) if gap else None
    spot = ((report.get("spotting") or {}).get("scores") or {}).get(form)
    if spot:
        out["spotting"] = dict(recall=spot["recall"], order=spot.get("order"), minus_mismatched=spot.get("minus_mismatched_paired"))
    ratio = (report.get("motion_ratio_to_real") or {}).get(form) or {}
    out["motion_to_real"] = {k: ratio.get(k) for k in ("wrist_speed", "wrist_spread", "hands_tracked", "seconds_over_reference")}
    return out


def instrument_check(a: dict, b: dict) -> dict:
    out = {}
    for form in ("real", "blank"):
        x, y = a["forms"][form]["hypotheses"], b["forms"][form]["hypotheses"]
        out[f"{form}_identical"] = int(sum(p == q for p, q in zip(x, y)))
    out["clips"] = len(a["references"])
    out["moved"] = out["real_identical"] != out["clips"] or out["blank_identical"] != out["clips"]
    return out


def comparison(arm: dict, form: str, other: dict, other_form: str, *, boot: int, shards_a=None, shards_b=None) -> dict:
    """compare_forms plus the paired share, loop and echo differences (arm minus comparator)."""
    from round20_compare import compare_forms
    from round20_lib import paired_mean
    from signtranslator.eval.s0_metrics import bleu_from_statistics, bleu_statistics
    out = compare_forms(arm, form, other, other_form, boot=boot, shards_a=shards_a, shards_b=shards_b)
    refs = arm["references"]
    sa, sb = bleu_statistics(arm["forms"][form]["hypotheses"], refs), bleu_statistics(other["forms"][other_form]["hypotheses"], refs)
    real = bleu_statistics(arm["forms"]["real"]["hypotheses"], refs)
    blank = bleu_statistics(arm["forms"][blank_of(form, arm["forms"])]["hypotheses"], refs)
    gap = float(bleu_from_statistics(real.sum(0)) - bleu_from_statistics(blank.sum(0)))
    idx = np.random.default_rng(0).integers(0, len(refs), (boot, len(refs)))
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = ((bleu_from_statistics(sa[idx].sum(1)) - bleu_from_statistics(sb[idx].sum(1))) /
                   (bleu_from_statistics(real[idx].sum(1)) - bleu_from_statistics(blank[idx].sum(1))))
    samples = samples[np.isfinite(samples)]
    out["share_points"] = dict(difference=round(float(out["bleu"]["difference"]) / gap, 4) if gap else None,
                               ci95=[round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)]
                               if len(samples) else None)
    out["loop_difference"] = paired_mean(lib.loop_flags(arm["forms"][form]["hypotheses"]),
                                         lib.loop_flags(other["forms"][other_form]["hypotheses"]), boot=boot)
    words = [p["stage1_words"] for p in arm["per_clip"]]
    ea, eb = lib.echo_rows(arm["forms"][form]["hypotheses"], words, refs), lib.echo_rows(other["forms"][other_form]["hypotheses"], words, refs)
    out["echo_difference"] = paired_mean([x / y if y else None for x, y in ea], [x / y if y else None for x, y in eb], boot=boot)
    out["instrument_check"] = instrument_check(arm, other)
    return out


def control_read(report: dict, *, boot: int) -> dict | None:
    """A1's control: real against real with the arm's scales, in points of real's gap to blank. None without the form."""
    if lib.REAL_SCALED not in report["forms"]:
        return None
    from signtranslator.eval.s0_metrics import bleu_from_statistics, bleu_statistics
    from round20_lib import paired_mean
    refs = report["references"]
    real, scaled, blank = (bleu_statistics(report["forms"][f]["hypotheses"], refs) for f in ("real", lib.REAL_SCALED, "blank"))
    gap = float(bleu_from_statistics(real.sum(0)) - bleu_from_statistics(blank.sum(0)))
    loss = float(bleu_from_statistics(real.sum(0)) - bleu_from_statistics(scaled.sum(0)))
    idx = np.random.default_rng(0).integers(0, len(refs), (boot, len(refs)))
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = ((bleu_from_statistics(real[idx].sum(1)) - bleu_from_statistics(scaled[idx].sum(1))) /
                   (bleu_from_statistics(real[idx].sum(1)) - bleu_from_statistics(blank[idx].sum(1))))
    samples = samples[np.isfinite(samples)]
    ll = (report.get("likelihood") or {}).get("per_form") or {}
    return dict(scales=(report["settings"]["round20"] or {}).get("arm_scale"), bleu_loss=round(loss, 4),
                loss_share_of_gap=round(loss / gap, 4) if gap else None,
                loss_ci95=[round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)] if len(samples) else None,
                paired=report["forms"][lib.REAL_SCALED].get("minus_real"),
                likelihood_paired=paired_mean(ll[lib.REAL_SCALED], ll["real"], boot=boot) if lib.REAL_SCALED in ll else None,
                spotting_paired=(report.get("spotting") or {}).get("scores", {}).get(lib.REAL_SCALED, {}).get("minus_real_paired"),
                motion_to_real=report["motion_ratio_to_real"].get(lib.REAL_SCALED))


def _points(c: dict) -> tuple[float | None, float | None]:
    ll = (c.get("likelihood_minus_blank") or {}).get("difference")
    return c["bleu"]["difference"], (ll or {}).get("difference")


def _spot_lower(c: dict):
    guard = c.get("spotting_guard") or {}
    return guard["difference"]["ci95"][0] if guard.get("available") else None


def _above(value, bar) -> bool:
    return value is not None and value > bar


def draft_rule(test: str, *, comparison_row, arm: dict | None = None, form: str | None = None, other: dict | None = None,
               other_form: str | None = None, control=None, declared_commit: str | None = None) -> dict:
    """The draft text's checks on one comparison. ``declared`` says whether a plan commit declared it; nothing here adopts."""
    bleu, ll = _points(comparison_row)
    checks = dict(paired_bleu_above_0=_above(bleu, 0.), paired_likelihood_above_0=_above(ll, 0.))
    if test == "a4":
        checks = dict(paired_bleu_lower_bound_above_0=_above(comparison_row["bleu"]["ci95"][0], 0.), paired_likelihood_above_0=_above(ll, 0.))
    motion = lambda r, f: (r.get("motion_ratio_to_real") or {}).get(f) or {}           # noqa: E731
    if test in ("a1", "a3-gen"):
        checks["spotting_lower_bound_above_minus_0.02"] = _above(_spot_lower(comparison_row), -.02)
        mine, theirs = motion(arm, form), motion(other, other_form)
        if test == "a1":
            wrist, spread = mine.get("wrist_speed"), mine.get("wrist_spread")
            checks["wrist_at_most_1.25"] = wrist is not None and wrist <= 1.25
            checks["spread_at_least_0.8_of_comparator"] = spread is not None and theirs.get("wrist_spread") is not None and \
                spread >= .8 * theirs["wrist_spread"]
            if control is not None:
                share = control["loss_share_of_gap"]
                checks["real_control_loses_at_most_10_points"] = share is not None and share <= .10
        else:
            wrist, hands = mine.get("wrist_speed"), mine.get("hands_tracked")
            checks["wrist_at_most_1.10_of_comparator"] = wrist is not None and theirs.get("wrist_speed") is not None and \
                wrist <= 1.1 * theirs["wrist_speed"]
            checks["hands_at_least_0.9_of_comparator"] = hands is not None and theirs.get("hands_tracked") is not None and \
                hands >= .9 * theirs["hands_tracked"]
    return dict(declared=bool(declared_commit), declared_commit=declared_commit, text=DRAFT_RULES[test], checks=checks,
                all_true=all(checks.values()),
                note="evaluated for reading only; this field adopts nothing" if declared_commit else "DRAFT, NOT DECLARED: adopts nothing")


def read_report(test: str, arm: dict, *, comparators=(), boot: int = 1000, smoke: bool = False, shards=None,
                declared_commit: str | None = None) -> dict:
    """``comparators``: [(label, report, form, shards)]; the first is the rule's comparator."""
    if test not in TESTS:
        raise SystemExit(f"--test: one of {TESTS}")
    check_arm(arm, smoke=smoke)
    variant = arm["settings"]["variants"][0]
    forms = [f for f in arm["forms"]]
    main = f"{variant}{PREDICTED}" if f"{variant}{PREDICTED}" in forms else variant
    out = dict(kind="comp-a-read", report_only=True, declared=bool(declared_commit), declared_commit=declared_commit, test=test,
               main_form=main, smoke=smoke, clips=arm["clips"], set=dict(key=lib.D20_KEY, sha256=lib.D20_SHA),
               reads={f: form_read(arm, f, boot=boot) for f in forms}, round20=dict(
                   vocabulary_extra=(arm.get("round20") or {}).get("vocabulary_extra"),
                   oracle_choice=(arm.get("round20") or {}).get("oracle_choice"),
                   per_variant=(arm.get("round20") or {}).get("per_variant")))
    control = control_read(arm, boot=boot)
    if control:
        out["control"] = control
    out["comparisons"] = {}
    for k, (label, other, other_form, other_shards) in enumerate(comparators):
        if smoke:
            raise SystemExit("a smoke's clips are not a stored report's: no comparator")
        row = comparison(arm, main, other, other_form, boot=boot, shards_a=shards, shards_b=other_shards)
        out["comparisons"][label] = row
        if k == 0:
            out["draft_rule"] = draft_rule(test, comparison_row=row, arm=arm, form=main, other=other, other_form=other_form,
                                           control=control, declared_commit=declared_commit)
    return out


def oracle_gate(read: dict) -> bool:
    """A3's stop before generation: an oracle arm read with both paired points above 0 against its first comparator."""
    if read.get("kind") != "comp-a-read" or read.get("test") != "a3-oracle" or read.get("smoke") or "draft_rule" not in read:
        return False
    checks = read["draft_rule"]["checks"]
    return bool(checks.get("paired_bleu_above_0") and checks.get("paired_likelihood_above_0"))


def main(argv=None) -> int:
    from round20_compare import report_shards
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("read")
    r.add_argument("--test", required=True, choices=TESTS)
    r.add_argument("--arm", required=True, help="the arm's merged report")
    r.add_argument("--comparator", action="append", default=[], help="LABEL=KEY:FORM; the first is the draft rule's")
    r.add_argument("--declared-commit", help="the plan commit that declared this read's rule")
    r.add_argument("--bootstrap", type=int, default=1000)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--output-key")
    r.add_argument("--output-file")
    g = sub.add_parser("gate")
    g.add_argument("--read", required=True)
    for q in (r, g):
        q.add_argument("--store", default="r2")
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.command == "gate":
        passed = oracle_gate(json.loads(store.get(args.read)))
        print(json.dumps(dict(read=args.read, gate=passed)), flush=True)
        return 0 if passed else 1
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    arm = json.loads(store.get(args.arm))
    comparators = []
    for spec in args.comparator:
        label, _, rest = spec.partition("=")
        key, _, form = rest.rpartition(":")
        if not (label and key and form):
            p.error(f"--comparator {spec!r}: LABEL=KEY:FORM")
        other = json.loads(store.get(key))
        comparators.append((label, other, form, report_shards(store, key, other, None)))
    out = read_report(args.test, arm, comparators=comparators, boot=args.bootstrap, smoke=args.smoke,
                      shards=report_shards(store, args.arm, arm, None), declared_commit=args.declared_commit)
    out["arm"] = dict(key=args.arm, sha256=hashlib.sha256(store.get(args.arm)).hexdigest())
    data = json.dumps(out, indent=1, default=float).encode()
    if args.output_file:
        with open(args.output_file, "xb") as stream:
            stream.write(data)
    if args.output_key:
        store.put(args.output_key, data)
    print(f"read {args.arm}: draft rule " + json.dumps((out.get("draft_rule") or {}).get("checks")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
