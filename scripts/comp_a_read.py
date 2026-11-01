"""Post-S2 workstream A's shared reader (reports/post_s2_plan.md sections 1 and 3): one arm's merged report, read on its stored
hypotheses and rows. No model loads. D20 only; report-only; write-once. Nothing adopts from it.

Each form gets: raw BLEU and its gap share (form - blank) / (real - blank), both with sentence-bootstrap intervals; likelihood minus blank and
its gap share; v2 spotting; wrist speed, spread, hands and length against real, per side where the run stored them; two-handed contact; loop
share and word echo. A form at predicted length is read against ``blank@predicted`` for BLEU, likelihood and echo lift alike, and its
true-length blank is reported beside it. ``real`` stays the true clip in every share, as there is no predicted-length real.

Each comparison (``--comparator LABEL=KEY:FORM:DECLARED``, DECLARED the settings the two may differ in, ``+``-joined, or ``none``) adds
``round20_compare.compare_forms``, refused unless the reports are paired and differ only as declared, and the paired differences of share, loop
and echo. ``instrument_check`` counts identical ``real`` and ``blank`` hypotheses across the two runs: device or an unpinned ``transformers`` can
move them, so a difference is flagged and never hidden.

**One primary arm per phase is pre-registered** (``PRIMARY``). Only it gets ``draft_rule``, always on its *first* comparator, and it holds no
"adopted" field: this script never writes one. Every other arm is ``comparator`` or ``sensitivity`` and has no rule. A scaled arm without its
real control stops the read. ``gate`` exits 0 only when the A3 oracle's primary arm has every check true.

``decompose`` is the clip-choice read: chosen minus sample0 is the choice alone and sample0 minus a1 is repetition alone, each with a bootstrap
that also resamples the entries, because one clip serves every sentence holding its entry; the fixed draws give the read's variance over clips.
A chosen read that does not rise is reported ``not informative``, never as evidence that word source has no room.
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
PRIMARY = {"a1": "hold", "a3-oracle": "top10", "a3-gen": "top10", "a4": "n16"}
COMPARATOR_ARMS = {"a1": ("margin0",), "a3-oracle": ("base",), "a3-gen": (), "a4": ("sample0",)}
A9_FORM = "a9_predicted_margin01"          # the A2 file's table of a10's a9@predicted, per side
DRAFT_RULES = dict(
    a1=("Primary arm: hold. The scaled arms are report-only sensitivities and margin0 is the comparator. Against a1-margin0 on D20 at predicted "
        "timing: paired BLEU and paired likelihood minus blank@predicted both above 0; v2 spotting's paired lower bound above -0.02; "
        "wrist per side and pooled no further from 1.0x real than a9@predicted's and at least 0.8x real; spread no further from 1.0x real than "
        "a9@predicted's; two-handed contact at least 0.9x margin0's. These motion checks are a no-regression floor and not the release bar. The read "
        "states separately whether pooled wrist lies within 0.8-1.25x; hold's does not. a9@predicted is the second comparator, read beside it. A scaled "
        "arm with no real control stops its read."),
    **{"a3-oracle": ("Primary arm: top10; top30 is report-only. Against the r10 oracle at the same predicted timing and margin 0: paired BLEU and paired "
                     "likelihood minus blank@predicted both above 0; wrist no further from 1.0x real than a9@predicted's (known before any read, first 300 "
                     "clips: base 1.32x, top10 1.54x, top30 1.64x); hands at least 0.9x the base's; the new entries' echo above the base's echo of the same "
                     "entries on the same sentences; gap share against blank@predicted above a1's own, which favours predicted arms by about 3 points "
                     "(a9@predicted 23.2 against 19.9). BLEU can rise with speed alone, so the echo check is what credits coverage. A failure does not close "
                     "coverage: every new top-10 word comes from ASLLRP continuous segments, a route that read lower as a whole (a2 1.879 against a1 2.219), "
                     "and source and coverage are not separated here. Fingerspelled single-clip tokens stay out of any primary arm or claim. Part of the arm: "
                     "ASLLRP segment pieces widened to at least the list's 0.2 s, every n't token signed as not, and the pooled annotated fraction as the "
                     "new entries' core rule."),
       "a3-gen": ("Primary arm: top10; top30 is report-only. Against the margin-0 a9 arm at predicted timing: paired BLEU and paired likelihood minus "
                  "blank@predicted both above 0; v2 spotting's paired lower bound above -0.02; wrist no further from 1.0x real than a9@predicted's; hands at "
                  "least 0.9x that arm's. New entries generated at the same 0.2 s floor as the oracle's pieces."),
       "a4": ("Primary arm: n16. Read in the decomposition: n16's entry-and-sentence bootstrap lower bound against sample0 above 0, paired likelihood above "
              "0, and n16's BLEU above each of sample0 and draw1 to draw4. Repetition is the mean of those five draws minus a1. All of these reopen section "
              "1's tokenizer bound, which is then re-stated. Any other result is not informative, and the bound stands without support from this read. This "
              "is no ceiling, and nothing is adopted.")})
KNOWN_AT_WRITING = dict(
    a1=("a10 had been read by the review: a9@predicted's spread 2.04x real. The review's motion arithmetic was known: composition over scaled real about "
        "1.46-1.55x, the hold's optimistic bound of left path -39% and pooled wrist 1.55x to 1.23x. A2 and the local hold rebuild (motion only) were known. "
        "Five-clip smokes of margin0 and left0.5 had been run. No D20 read of hold existed. Known from stored motion before any read: hold's left 1.37x, right "
        "1.24x, pooled 1.29x, spread 1.71x, left height -0.74, contact unchanged by construction. The motion checks therefore pass and are not evidence; the "
        "1.25x bar was replaced knowing 1.29x. A pass means hold reads no worse than margin0 with less motion. It meets no release bar."),
    **{"a3-oracle": ("a1's D20 read and its 19% share were known. Text coverage on D20 was known (reports/comp_a3_tokens.json), and the wrist figures the rule states. "
                     "A five-clip smoke of top10 had been run. No full read of any A3 arm existed."),
       "a3-gen": ("As a3-oracle, and a9's and a10's reads were known. No read of any A3 arm existed."),
       "a4": ("a1's and a4's D20 reads were known, and a v3d word read on 50 entries. No D20 read of any clip-choice arm existed.")})
JOB_WRITTEN = ("margin0", "base", "sample0")   # a comparator from this same job: its code and environment must be the arm's


def blank_of(form: str, forms) -> str:
    return f"blank{PREDICTED}" if PREDICTED in form and f"blank{PREDICTED}" in forms else "blank"


def role_of(test: str, arm_name: str) -> str:
    if PRIMARY[test] == arm_name:
        return "primary"
    return "comparator" if arm_name in COMPARATOR_ARMS[test] else "sensitivity"


def check_arm_name(test: str, name: str, settings: dict) -> None:
    """The arm's name must be what its settings say it is: a name selects the rule."""
    r20 = settings.get("round20") or {}
    if test == "a1":
        scales = {"left0.5": dict(left=.5, right=1.), "both0.7": dict(left=.7, right=.7)}
        ok = (name == "margin0" and bool(r20.get("comp_a_read")) and not r20.get("arm_scale") and not r20.get("hold_left")) or \
             (name == "hold" and bool(r20.get("hold_left")) and not r20.get("arm_scale")) or (name in scales and r20.get("arm_scale") == scales[name])
    elif test in ("a3-oracle", "a3-gen"):
        top = (r20.get("vocabulary_extra") or {}).get("top")
        ok = (test == "a3-oracle" and name == "base" and bool(r20.get("comp_a_read")) and top is None) or \
             (name in ("top10", "top30") and top == int(name[3:]) and bool(r20.get("word_cache_extra")) == (test == "a3-gen"))
    else:
        pick = (r20.get("oracle_choice") or {}).get("pick")
        ok = pick == ("16" if name == "n16" else name) and (name in ("n16", "sample0") or name.startswith("draw"))
    if not ok:
        raise SystemExit(f"--arm-name {name} is not what this report's settings say for {test}")


def check_provenance(arm: dict, other: dict, label: str) -> None:
    """A comparator this job wrote must be the same code and environment as the arm. Stored a10 and a1 stay exempt, under ``instrument_check``."""
    if label not in JOB_WRITTEN:
        return
    a, b = (((r.get("settings") or {}).get("round20") or {}).get("provenance") for r in (arm, other))
    if not a or not b or a != b:
        raise SystemExit(f"comparator {label} came from other code or another environment: {b} against {a}")


def release_band(report: dict, form: str) -> dict:
    """Stated apart from any rule: the release bar's pooled wrist band."""
    wrist = ((report.get("motion_ratio_to_real") or {}).get(form) or {}).get("wrist_speed")
    return dict(pooled_wrist=wrist, within_0_8_to_1_25=wrist is not None and .8 <= wrist <= 1.25)


def check_arm(report: dict, *, smoke: bool) -> None:
    """D20, report-only, and all of D20 unless this is a smoke; a scaled arm has its real control."""
    s = report.get("settings") or {}
    r20 = s.get("round20") or {}
    sentences = r20.get("sentences") or {}
    if sentences.get("sha256") != lib.D20_SHA or sentences.get("key") != lib.D20_KEY:
        raise SystemExit(f"this report read {sentences.get('key')}: this reader reads D20 only")
    if not report.get("report_only"):
        raise SystemExit("this report is not report-only")
    if not smoke and report.get("clips") != 500:
        raise SystemExit(f"this report read {report.get('clips')} clips, not all of D20")
    if smoke and report.get("clips", 99) > 5:
        raise SystemExit("a smoke reads at most 5 clips")
    if r20.get("arm_scale") and lib.REAL_SCALED not in report["forms"]:
        raise SystemExit("a scaled arm with no real control is not read")


def share_row(stats, real, blank, *, boot: int) -> dict:
    """(form - blank) / (real - blank) on the same sentences, with the sentence bootstrap's interval."""
    from signtranslator.eval.s0_metrics import bleu_from_statistics
    f, r, b = (float(bleu_from_statistics(x.sum(0))) for x in (stats, real, blank))
    point = (f - b) / (r - b) if r != b else None
    n = len(stats)
    idx = np.random.default_rng(0).integers(0, n, (boot, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (bleu_from_statistics(stats[idx].sum(1)) - bleu_from_statistics(blank[idx].sum(1))) / \
            (bleu_from_statistics(real[idx].sum(1)) - bleu_from_statistics(blank[idx].sum(1)))
    s = s[np.isfinite(s)]
    ci = [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)] if len(s) else None
    return dict(share=None if point is None else round(point, 4), ci95=ci, bleu=round(f, 4), real=round(r, 4), blank=round(b, 4))


def loop_echo(report: dict, form: str, *, boot: int) -> dict:
    """Loop share and word echo; the lift is against the blank at the form's own length."""
    from round20_lib import paired_mean
    hyps = report["forms"][form]["hypotheses"]
    words = [p["stage1_words"] for p in report["per_clip"]]
    refs = report["references"]
    blank_name = blank_of(form, report["forms"])
    rows, blank_rows, real_rows = (lib.echo_rows(report["forms"][f]["hypotheses"], words, refs) for f in (form, blank_name, "real"))
    rate = lib.echo_share(rows)
    blank, real = lib.echo_share(blank_rows), lib.echo_share(real_rows)
    lift = None if None in (rate, blank, real) or real == blank else round((rate - blank) / (real - blank), 4)
    out = dict(loop_share=paired_mean(lib.loop_flags(hyps), boot=boot), word_echo=None if rate is None else round(rate, 4),
               echo_lift_over_blank_relative_to_real=lift, echo_lift_blank=blank_name, echo_content_words=sum(b for _, b in rows))
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
        mean_of = lambda x: float(np.mean([v for v in x if v is not None]))          # noqa: E731
        out["likelihood_minus_blank"] = paired_mean(a, b, boot=boot)
        if r:
            gap = mean_of(r) - mean_of(b)
            out["likelihood_gap_share"] = round((mean_of(a) - mean_of(b)) / gap, 4) if gap else None
    spot = ((report.get("spotting") or {}).get("scores") or {}).get(form)
    if spot:
        out["spotting"] = dict(recall=spot["recall"], order=spot.get("order"), minus_mismatched=spot.get("minus_mismatched_paired"))
    ratio = (report.get("motion_ratio_to_real") or {}).get(form) or {}
    out["motion_to_real"] = {k: ratio.get(k) for k in ("wrist_speed", "wrist_spread", "hands_tracked", "seconds_over_reference")}
    if form in (report.get("comp_a_motion") or {}):
        out["per_side"] = report["comp_a_motion"][form]
    return out


def instrument_check(a: dict, b: dict) -> dict:
    out = {}
    for form in ("real", "blank"):
        x, y = a["forms"][form]["hypotheses"], b["forms"][form]["hypotheses"]
        out[f"{form}_identical"] = int(sum(p == q for p, q in zip(x, y)))
    out["clips"] = len(a["references"])
    out["moved"] = out["real_identical"] != out["clips"] or out["blank_identical"] != out["clips"]
    return out


def clustered_difference(stats_a, stats_b, entries, *, boot: int, seed: int = 0) -> dict:
    """a - b's corpus BLEU with a bootstrap over sentences and entries together: each sentence's weight is its own Poisson(1) draw times the mean
    of its entries' (Owen's pigeonhole bootstrap), so sentences sharing a clip move together. ``entries``: per sentence, the entries it signs.
    ``stats_a`` a list: the mean of those forms' BLEU, on the same weights."""
    many = stats_a if isinstance(stats_a, list) else [stats_a]
    from signtranslator.eval.s0_metrics import bleu_from_statistics
    names = sorted({e for row in entries for e in row})
    where = {e: k for k, e in enumerate(names)}
    rng = np.random.default_rng(seed)
    n = len(entries)
    index = [np.array([where[e] for e in row], dtype=int) for row in entries]
    diffs = np.empty(boot)
    for b in range(boot):
        w = rng.poisson(1., n).astype(float)
        v = rng.poisson(1., len(names)).astype(float)
        weight = w * np.array([v[i].mean() if len(i) else 1. for i in index])
        diffs[b] = np.mean([bleu_from_statistics((weight[:, None] * s).sum(0)) for s in many]) - bleu_from_statistics((weight[:, None] * stats_b).sum(0))
    point = float(np.mean([bleu_from_statistics(s.sum(0)) for s in many]) - bleu_from_statistics(stats_b.sum(0)))
    return dict(difference=round(point, 4), ci95=[round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
                entries=len(names), bootstrap="sentences and entries")


def comparison(arm: dict, form: str, other: dict, other_form: str, *, boot: int, shards_a=None, shards_b=None, declared=()) -> dict:
    """compare_forms plus the paired share, loop and echo differences (arm minus comparator), on the arm's gap."""
    from round20_compare import compare_forms
    from round20_lib import paired_mean
    from signtranslator.eval.s0_metrics import bleu_from_statistics, bleu_statistics
    out = compare_forms(arm, form, other, other_form, boot=boot, shards_a=shards_a, shards_b=shards_b, declared=declared)
    out["declared_differences"] = list(declared)
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


def control_read(report: dict, form: str, *, boot: int) -> dict | None:
    """The scaled arm's control: real against real with the arm's scales, in points of real's gap to blank, with likelihood and spotting beside
    it, per side, and composition over the scaled real (scaling can pass a wrist bar only by shrinking). None without the form."""
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
    ratio = report.get("motion_ratio_to_real") or {}
    wrist, scaled_wrist = (ratio.get(form) or {}).get("wrist_speed"), (ratio.get(lib.REAL_SCALED) or {}).get("wrist_speed")
    return dict(scales=(report["settings"]["round20"] or {}).get("arm_scale"), bleu_loss=round(loss, 4),
                loss_share_of_gap=round(loss / gap, 4) if gap else None,
                loss_ci95=[round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)] if len(samples) else None,
                paired=report["forms"][lib.REAL_SCALED].get("minus_real"),
                likelihood_paired=paired_mean(ll[lib.REAL_SCALED], ll["real"], boot=boot) if lib.REAL_SCALED in ll else None,
                spotting_paired=(report.get("spotting") or {}).get("scores", {}).get(lib.REAL_SCALED, {}).get("minus_real_paired"),
                motion_to_real=ratio.get(lib.REAL_SCALED), per_side=(report.get("comp_a_motion") or {}).get(lib.REAL_SCALED),
                composition_over_scaled_real=round(wrist / scaled_wrist, 4) if wrist and scaled_wrist else None,
                real_below_band=scaled_wrist is not None and scaled_wrist < .8,
                note="scaling is report-only: it passes a wrist bar only by shrinking the motion")


def _points(c: dict) -> tuple[float | None, float | None]:
    ll = (c.get("likelihood_minus_blank") or {}).get("difference")
    return c["bleu"]["difference"], (ll or {}).get("difference")


def _spot_lower(c: dict):
    guard = c.get("spotting_guard") or {}
    return guard["difference"]["ci95"][0] if guard.get("available") else None


def _above(value, bar) -> bool:
    return value is not None and value > bar


def no_further(value, reference) -> bool:
    """``value`` no further from 1.0x real than ``reference``."""
    return value is not None and reference is not None and abs(value - 1.) <= abs(reference - 1.)


def a9_reference(a9_sides: dict | None, other: dict | None = None, other_form: str | None = None) -> dict:
    """a9@predicted's figures: per side from the A2 file, pooled from a10's report where the read has it, else from the A2 file, which must agree with the
    report."""
    motion = ((other or {}).get("motion_ratio_to_real") or {}).get(other_form) or {}
    table = (a9_sides or {}).get("forms", {}).get(A9_FORM) or {}
    pooled = table.get("pooled_wrist_ratio")
    if pooled is not None and motion.get("wrist_speed") is not None and abs(pooled - motion["wrist_speed"]) > .01:
        raise SystemExit(f"the A2 file's a9@predicted wrist {pooled} is not its report's {motion['wrist_speed']}")
    return dict(wrist=motion.get("wrist_speed", pooled), spread=motion.get("wrist_spread", table.get("pooled_spread_ratio")),
                left=(table.get("ratio_to_real") or {}).get("left"), right=(table.get("ratio_to_real") or {}).get("right"))


def draft_rule(test: str, *, row, arm=None, form=None, first=None, first_form=None, a9=None, gap_share=None, a1_share=None, declared_commit=None,
               new_echo=None, base_echo=None) -> dict:
    """The draft text's checks on the first comparison ``row``. ``declared`` says whether a plan commit declared it; nothing here adopts."""
    bleu, ll = _points(row)
    motion = lambda r, f: (r.get("motion_ratio_to_real") or {}).get(f) or {}           # noqa: E731
    checks = dict(paired_bleu_above_0=_above(bleu, 0.), paired_likelihood_above_0=_above(ll, 0.))
    mine, theirs = motion(arm or {}, form), motion(first or {}, first_form)
    if test in ("a1", "a3-gen"):
        checks["spotting_lower_bound_above_minus_0.02"] = _above(_spot_lower(row), -.02)
    if test == "a1":
        a9 = a9 or {}
        side = ((arm or {}).get("comp_a_motion") or {}).get(form) or {}
        base = ((first or {}).get("comp_a_motion") or {}).get(first_form) or {}
        values = dict(pooled=mine.get("wrist_speed"), left=side.get("left_speed_over_real"), right=side.get("right_speed_over_real"))
        refs = dict(pooled=a9.get("wrist"), left=a9.get("left"), right=a9.get("right"))
        for name, value in values.items():
            checks[f"wrist_{name}_no_further_than_a9"] = no_further(value, refs[name])
            checks[f"wrist_{name}_at_least_0.8"] = value is not None and value >= .8
        checks["spread_no_further_than_a9"] = no_further(mine.get("wrist_spread"), a9.get("spread"))
        a, b = side.get("contact_share"), base.get("contact_share")
        checks["contact_at_least_0.9_of_margin0"] = a is not None and b is not None and a >= .9 * b
    if test in ("a3-oracle", "a3-gen"):
        hands = mine.get("hands_tracked")
        checks["wrist_no_further_than_a9"] = no_further(mine.get("wrist_speed"), (a9 or {}).get("wrist"))
        checks["hands_at_least_0.9_of_comparator"] = hands is not None and theirs.get("hands_tracked") is not None and hands >= .9 * theirs["hands_tracked"]
    if test == "a3-oracle":
        checks["new_entries_echo_above_the_base"] = _above(new_echo, base_echo) if base_echo is not None else False
        checks["gap_share_above_a1"] = gap_share is not None and a1_share is not None and gap_share > a1_share
    return dict(declared=bool(declared_commit), declared_commit=declared_commit, text=DRAFT_RULES[test], known_when_written=KNOWN_AT_WRITING[test],
                checks=checks, all_true=all(checks.values()),
                note="evaluated for reading only; this field adopts nothing" if declared_commit else "DRAFT, NOT DECLARED: adopts nothing")


def main_form_of(arm: dict) -> str:
    variant = arm["settings"]["variants"][0]
    return f"{variant}{PREDICTED}" if f"{variant}{PREDICTED}" in arm["forms"] else variant


def a1_share_of(other: dict, form: str, *, boot: int) -> float | None:
    """a1's own gap share, on a1's own real and blank."""
    from signtranslator.eval.s0_metrics import bleu_statistics
    refs = other["references"]
    stats = {f: bleu_statistics(other["forms"][f]["hypotheses"], refs) for f in (form, "real", "blank")}
    return share_row(stats[form], stats["real"], stats["blank"], boot=boot)["share"]


def read_report(test: str, arm: dict, *, arm_name: str, comparators=(), boot: int = 1000, smoke: bool = False, shards=None, declared_commit: str | None = None,
                a9_sides: dict | None = None) -> dict:
    """``comparators``: [(label, report, form, shards, declared)]. The first is the rule's comparator, and only a primary arm has a rule."""
    if test not in TESTS:
        raise SystemExit(f"--test: one of {TESTS}")
    check_arm(arm, smoke=smoke)
    check_arm_name(test, arm_name, arm.get("settings") or {})
    main = main_form_of(arm)
    role = role_of(test, arm_name)
    if role == "primary" and not comparators and not smoke:
        raise SystemExit("the primary arm has no comparator: nothing can be read against its rule")
    extra_info = (arm.get("round20") or {}).get("vocabulary_extra") or {}
    if role == "primary" and extra_info.get("fingerspelled_single_clips"):
        raise SystemExit(f"fingerspelled single-clip tokens stay out of a primary arm: {extra_info['fingerspelled_single_clips']}")
    out = dict(kind="comp-a-read", report_only=True, declared=bool(declared_commit), declared_commit=declared_commit, test=test, arm_name=arm_name, role=role,
               main_form=main, smoke=smoke, clips=arm["clips"], set=dict(key=lib.D20_KEY, sha256=lib.D20_SHA),
               provenance=((arm.get("settings") or {}).get("round20") or {}).get("provenance"), release_band=release_band(arm, main),
               reads={f: form_read(arm, f, boot=boot) for f in arm["forms"]}, hold_left=arm.get("hold_left"), round20=dict(
                   vocabulary_extra=(arm.get("round20") or {}).get("vocabulary_extra"),
                   oracle_choice=(arm.get("round20") or {}).get("oracle_choice"),
                   per_variant=(arm.get("round20") or {}).get("per_variant")))
    control = control_read(arm, main, boot=boot)
    if control:
        out["control"] = control
    out["comparisons"] = {}
    if comparators and smoke:
        raise SystemExit("a smoke's clips are not a stored report's: no comparator")
    from signtranslator.eval.s0_metrics import bleu_statistics
    words = [p["stage1_words"] for p in arm["per_clip"]]
    for label, other, other_form, other_shards, declared in comparators:
        check_provenance(arm, other, label)
        row = comparison(arm, main, other, other_form, boot=boot, shards_a=shards, shards_b=other_shards, declared=declared)
        if test == "a4":
            refs = arm["references"]
            row["clustered_bleu"] = clustered_difference(bleu_statistics(arm["forms"][main]["hypotheses"], refs),
                                                         bleu_statistics(other["forms"][other_form]["hypotheses"], refs), words, boot=boot)
        out["comparisons"][label] = row
    if role == "primary" and comparators and test != "a4":             # A4's rule is read in the decomposition, with its draws
        label, first, first_form, _, _ = comparators[0]
        a9 = None
        if a9_sides is None:
            raise SystemExit(f"{test}'s rule needs the A2 file's a9@predicted table")
        if test == "a1":
            second = next((c for c in comparators if c[0] == "a9_predicted"), None)
            if second is None:
                raise SystemExit("A1's rule needs its a9_predicted comparator")
            a9 = a9_reference(a9_sides, second[1], second[2])
        else:
            a9 = a9_reference(a9_sides)
        a1 = next((c for c in comparators if c[0] == "a1"), None)
        share, a1_share, new_echo, base_echo = None, None, None, None
        if test == "a3-oracle":
            if a1 is None:
                raise SystemExit("the A3 oracle's rule needs its a1 comparator")
            share, a1_share = out["reads"][main]["gap_share"]["share"], a1_share_of(a1[1], a1[2], boot=boot)
            entries = extra_info.get("new_entries") or ()
            new_echo = lib.new_entry_echo(arm["forms"][main]["hypotheses"], words, arm["references"], entries)
            base_echo = lib.new_entry_echo(first["forms"][first_form]["hypotheses"], words, arm["references"], entries)   # the same entries on the same sentences
        out["draft_rule"] = draft_rule(test, row=out["comparisons"][label], arm=arm, form=main, first=first, first_form=first_form, a9=a9, gap_share=share,
                                       a1_share=a1_share, declared_commit=declared_commit, new_echo=new_echo, base_echo=base_echo)
        out["draft_rule"]["comparator"] = label
    return out


def oracle_gate(read: dict) -> bool:
    """A3's stop before generation: the oracle's primary arm with every check true."""
    if read.get("kind") != "comp-a-read" or read.get("test") != "a3-oracle" or read.get("smoke") or read.get("role") != "primary" or "draft_rule" not in read:
        return False
    return bool(read["draft_rule"]["all_true"])


def decompose(chosen: dict, sample0: dict, draws: list, a1: dict, *, boot: int = 1000, declared_commit: str | None = None) -> dict:
    """The clip-choice read's decomposition (module docstring). Every report's main form is oracle_stage2."""
    from round20_compare import check_pairable
    from round20_lib import paired_mean
    from signtranslator.eval.s0_metrics import bleu_from_statistics, bleu_statistics
    form = "oracle_stage2"
    for r in (chosen, sample0, *draws):
        check_arm(r, smoke=False)
    for r in (sample0, *draws):
        check_pairable(chosen, r, ("oracle_choice",))
        check_provenance(chosen, r, "sample0")
    check_pairable(sample0, a1, ("oracle_choice",))
    if len(draws) != 4:
        raise SystemExit("the decomposition needs sample0 and draw1 to draw4")
    for name, r in zip(("n16", "sample0", "draw1", "draw2", "draw3", "draw4"), (chosen, sample0, *draws)):
        check_arm_name("a4", name, r.get("settings") or {})
    refs = chosen["references"]
    words = [p["stage1_words"] for p in chosen["per_clip"]]
    stat = lambda r: bleu_statistics(r["forms"][form]["hypotheses"], refs)            # noqa: E731
    ll = lambda r: r["likelihood"]["per_form"][form]                                  # noqa: E731
    choice = dict(clustered_bleu=clustered_difference(stat(chosen), stat(sample0), words, boot=boot), likelihood=paired_mean(ll(chosen), ll(sample0), boot=boot))
    five = [sample0, *draws]
    repetition = dict(clustered_bleu=clustered_difference([stat(r) for r in five], stat(a1), words, boot=boot))
    bleus = [float(bleu_from_statistics(stat(r).sum(0))) for r in five]
    chosen_bleu = float(bleu_from_statistics(stat(chosen).sum(0)))
    checks = dict(clustered_bleu_lower_bound_above_0=choice["clustered_bleu"]["ci95"][0] > 0, paired_likelihood_above_0=_above((choice["likelihood"] or {}).get("difference"), 0.),
                  **{f"n16_above_{name}": chosen_bleu > x for name, x in zip(("sample0", "draw1", "draw2", "draw3", "draw4"), bleus)})
    rises = all(checks.values())
    rule = dict(declared=bool(declared_commit), declared_commit=declared_commit, text=DRAFT_RULES["a4"], known_when_written=KNOWN_AT_WRITING["a4"], checks=checks, all_true=rises,
                note="evaluated for reading only; this field adopts nothing" if declared_commit else "DRAFT, NOT DECLARED: adopts nothing")
    return dict(kind="comp-a4-decomposition", report_only=True, choice_chosen_minus_sample0=choice, repetition_mean_of_five_draws_minus_a1=repetition,
                draw_bleu=[round(x, 4) for x in bleus], draw_bleu_sd=round(float(np.std(bleus, ddof=1)), 4), draft_rule=rule,
                provenance=lib.provenance_row(),
                reading="choice rises: section 1's tokenizer bound is to be re-stated" if rises else
                "not informative: this does not show that word source has no room")


def _comparators(store, specs, parser):
    from round20_compare import report_shards
    out = []
    for spec in specs:
        label, _, rest = spec.partition("=")
        parts = rest.rsplit(":", 2)
        if not label or len(parts) != 3 or not all(parts):
            parser.error(f"--comparator {spec!r}: LABEL=KEY:FORM:DECLARED")
        key, form, declared = parts
        other = json.loads(store.get(key))
        out.append((label, other, form, report_shards(store, key, other, None), () if declared == "none" else tuple(declared.split("+"))))
    return out


def main(argv=None) -> int:
    from round20_compare import report_shards
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("read")
    r.add_argument("--test", required=True, choices=TESTS)
    r.add_argument("--arm", required=True, help="the arm's merged report")
    r.add_argument("--arm-name", required=True, help="the arm's name in its phase (PRIMARY names the one with a rule)")
    r.add_argument("--comparator", action="append", default=[], help="LABEL=KEY:FORM:DECLARED, DECLARED +-joined settings or none; the first is the rule's")
    r.add_argument("--a9-sides-file", help="A1's rule: reports/comp_a2_left_arm.json")
    r.add_argument("--a9-sides-sha256")
    r.add_argument("--declared-commit", help="the plan commit that declared this read's rule")
    r.add_argument("--bootstrap", type=int, default=1000)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--output-key")
    r.add_argument("--output-file")
    g = sub.add_parser("gate")
    g.add_argument("--read", required=True)
    d = sub.add_parser("decompose")
    d.add_argument("--chosen", required=True)
    d.add_argument("--sample0", required=True)
    d.add_argument("--draw", action="append", default=[])
    d.add_argument("--a1", required=True)
    d.add_argument("--declared-commit")
    d.add_argument("--bootstrap", type=int, default=1000)
    d.add_argument("--output-key", required=True)
    for q in (r, g, d):
        q.add_argument("--store", default="r2")
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.command == "gate":
        passed = oracle_gate(json.loads(store.get(args.read)))
        print(json.dumps(dict(read=args.read, gate=passed)), flush=True)
        return 0 if passed else 1
    if args.command == "decompose":
        if store.exists(args.output_key):
            raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
        out = decompose(json.loads(store.get(args.chosen)), json.loads(store.get(args.sample0)), [json.loads(store.get(k)) for k in args.draw],
                        json.loads(store.get(args.a1)), boot=args.bootstrap, declared_commit=args.declared_commit)
        store.put(args.output_key, json.dumps(out, indent=1, default=float).encode())
        print(out["reading"], flush=True)
        return 0
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    a9_sides = None
    if args.a9_sides_file:
        raw = Path(args.a9_sides_file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != args.a9_sides_sha256:
            raise SystemExit(f"{args.a9_sides_file} does not match --a9-sides-sha256")
        a9_sides = json.loads(raw)
    arm = json.loads(store.get(args.arm))
    out = read_report(args.test, arm, arm_name=args.arm_name, comparators=_comparators(store, args.comparator, p), boot=args.bootstrap, smoke=args.smoke,
                      shards=report_shards(store, args.arm, arm, None), declared_commit=args.declared_commit, a9_sides=a9_sides)
    out["arm"] = dict(key=args.arm, sha256=hashlib.sha256(store.get(args.arm)).hexdigest())
    out["read_provenance"] = lib.provenance_row()
    data = json.dumps(out, indent=1, default=float).encode()
    if args.output_file:
        with open(args.output_file, "xb") as stream:
            stream.write(data)
    if args.output_key:
        store.put(args.output_key, data)
    print(f"read {args.arm} ({out['role']}): draft rule " + json.dumps((out.get("draft_rule") or {}).get("checks")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
