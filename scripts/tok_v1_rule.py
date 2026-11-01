"""Reads a merged V1 report and applies the declared rule. **The plan's text decides, not this script**; any difference from it is a bug here. ``tok_v1.py`` holds no rule, and the job never calls this. It reads a local file only.

**Scope.** D20, instrument v1, one digest and environment; likelihood minus blank primary.

**Void all** on a failed canary, path check or ``check``, or mixed digests. **Void Generated and Sensitivity** if composed-check fails or export settings differ from the pins.

**Cap.** c = (rt_nearest − blank) / (v0_nearest_fill − blank). Kept: c ≥ 0.75 and lower bound ≥ 0.5. Lost: c ≤ 0.25 and upper bound ≤ 0.5. Else partial. rt_post_fill − rt_nearest is reported: its lower bound above 0 means the input mismatch matters.

**Generated.** The share of a9@predicted's hand frames within the flag of decoded-neutral, minus decoded real on tracked frames, in the same tercile. Shown: lower bound > 0 and point ≥ 0.05 in the mid and high terciles. Absent: upper bound < 0.02 in the mid and high terciles, and the flag catches
at least half of D20's decoded-neutral. Else unresolved.

**Sensitivity.** a9p_replaced − a9p_control. Rise: lower bound > 0 and point ≥ 0.015 nats. No rise: upper bound < 0.015 and at least 1% flagged. Else unresolved.

**Hypothesis.** Supported: Shown and Rise. Refuted: Absent, or Shown and No rise. Else open.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

STATUS = "declared in plan; the plan's text decides"
HIGH, LOW, BOUND = .75, .25, .5
SHOWN_POINT, ABSENT_UPPER, CATCH_FLOOR = .05, .02, .5
RISE_NATS, FLAGGED_FLOOR = .015, .01
DECIDING_TERCILES = ("mid", "high")


def check_nonzero(report: dict) -> bool:
    """``tok_v1 check``'s own verdict on this report, so the two cannot differ."""
    import tok_v1
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "report.json"
        path.write_text(json.dumps(report))
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                return tok_v1.run_check(argparse.Namespace(report=str(path))) != 0
            except (KeyError, TypeError):
                return True


def void_reasons(report: dict) -> list[str]:
    reasons, checks = [], report.get("checks") or {}
    canary, path_check = checks.get("canary") or {}, checks.get("path_check") or {}
    if canary.get("reproduced") is not True:
        reasons.append("canary not reproduced")
    if path_check.get("reproduced") is not True:
        reasons.append("path check not passed")
    if check_nonzero(report):
        reasons.append("check nonzero")
    if not report.get("code_digest") or not report.get("environment"):
        reasons.append("no code digest or environment")
    digest = report.get("code_digest")
    if (canary and canary.get("code_digest") != digest) or (path_check and path_check.get("code_digest") != digest) or (canary and canary.get("environment") != report.get("environment")):
        reasons.append("mixed digests or environments")
    if (report.get("neutral") or {}).get("code_digest") != digest:
        reasons.append("mixed digests or environments")
    return reasons


def cap(share: dict | None, post_fill: dict | None) -> dict:
    out = dict(share=share, rt_post_fill_minus_rt_nearest=post_fill, input_mismatch_matters=bool(post_fill and post_fill.get("ci95") and post_fill["ci95"][0] > 0))
    if not share or share.get("point") is None or share.get("ci95") is None:
        return dict(out, outcome="partial")
    point, interval = share["point"], share["ci95"]
    if point >= HIGH and interval[0] >= BOUND:
        return dict(out, outcome="kept")
    if point <= LOW and interval[1] <= BOUND:
        return dict(out, outcome="lost")
    return dict(out, outcome="partial")


def generated(terciles: dict | None, control: dict | None) -> dict:
    rows = [(terciles or {}).get(t) for t in DECIDING_TERCILES]
    catch = (control or {}).get("catch_rate")
    out = dict(excess=terciles, catch_rate=catch)
    if any(r is None or r.get("ci95") is None for r in rows):
        return dict(out, outcome="unresolved")
    if all(r["ci95"][0] > 0 and r["difference"] >= SHOWN_POINT for r in rows):
        return dict(out, outcome="shown")
    if all(r["ci95"][1] < ABSENT_UPPER for r in rows) and catch is not None and catch >= CATCH_FLOOR:
        return dict(out, outcome="absent")
    return dict(out, outcome="unresolved")


def sensitivity(block: dict | None, flagged_share) -> dict:
    out = dict(difference=block, flagged_share=flagged_share)
    if not block or block.get("ci95") is None:
        return dict(out, outcome="unresolved")
    point, interval = block["point"], block["ci95"]
    if interval[0] > 0 and point >= RISE_NATS:
        return dict(out, outcome="rise")
    if interval[1] < RISE_NATS and flagged_share is not None and flagged_share >= FLAGGED_FLOOR:
        return dict(out, outcome="no_rise")
    return dict(out, outcome="unresolved")


def hypothesis(gen: str, sens: str) -> str:
    if gen == "shown" and sens == "rise":
        return "supported"
    if gen == "absent" or (gen == "shown" and sens == "no_rise"):
        return "refuted"
    return "open"


def read_rule(report: dict) -> dict:
    if report.get("declared_read") is not True:
        raise SystemExit("a smoke's report: no rule applies")
    out = dict(status=STATUS, code_digest=report.get("code_digest"))
    reasons = void_reasons(report)
    if reasons:
        return dict(out, void=True, reasons=reasons, cap=dict(outcome="void"), generated=dict(outcome="void"), sensitivity=dict(outcome="void"), hypothesis="void")
    reads = report["reads"]["likelihood_minus_blank"]
    neutral = report["neutral"]
    composed = report["checks"].get("composed_check") or {}
    usable = composed.get("reproduced") is True and composed.get("settings_match") is True
    out.update(void=False, reasons=[], cap=cap(reads["cap_share_of_nearest_fill"], reads.get("rt_post_fill_minus_rt_nearest")))
    if not usable:
        return dict(out, generated=dict(outcome="void", reason="composed-check failed or export settings differ"), sensitivity=dict(outcome="void", reason="composed-check failed or export settings differ"), hypothesis="open")
    gen = generated(((neutral.get("generated") or {}).get("a9_predicted") or {}).get("minus_decoded_real"), neutral.get("positive_control"))
    sens = sensitivity(reads.get("a9p_replaced_minus_a9p_control"), (report.get("sensitivity_counts") or {}).get("flagged_share"))
    return dict(out, generated=gen, sensitivity=sens, hypothesis=hypothesis(gen["outcome"], sens["outcome"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    print(json.dumps(read_rule(json.loads(Path(args.report).read_text())), indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
