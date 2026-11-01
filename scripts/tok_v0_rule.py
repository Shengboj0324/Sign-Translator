"""Reads a merged V0 report and applies the rule the coordinator gave for the declaration. **The plan's declaration decides, not this script**; any difference from it is a bug here. It reads a local
report file and nothing else, and the job never calls it. ``tok_v0.py`` holds no rule. Likelihood minus blank is primary.

**Void all:** canary or path check not passed, ``check`` nonzero, mixed or missing digest or environment. A smoke's report is refused. **Void Q2 and Q3** if real − identity's lower bound is at most 0.

**Q1,** A = v0_blank_real_marks − v0_blank_other_marks and B = v0_other_motion − v0_other_motion_other_marks, in nats. Shown if either has a lower bound above 0 and a point of at least 0.015; not detected if
both upper bounds are below 0.015; else unresolved.

**Q2,** p2 = (v0_other_marks − identity) / (real − identity). Yes: p2 at least 0.75 with a lower bound of at least 0.5. No: p2 at most 0.25 with an upper bound of at most 0.5. Else partial.

**Q3,** p3 = (v0_nearest_fill − identity) / (real − identity). Fill: p3 at least 0.75 with a lower bound of at least 0.5. Mark: p3 at most 0.25 with an upper bound of at most 0.5. Else both.
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

STATUS = "for the declaration in the plan; the plan's text decides"
Q1_NATS = .015
HIGH, LOW, BOUND = .75, .25, .5


def check_nonzero(report: dict) -> bool:
    """``tok_v0 check``'s own verdict on this report, so the two cannot differ."""
    import tok_v0
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "report.json"
        path.write_text(json.dumps(report))
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                return tok_v0.run_check(argparse.Namespace(report=str(path))) != 0
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
    if canary and (canary.get("code_digest") != report.get("code_digest") or canary.get("environment") != report.get("environment")):
        reasons.append("mixed digest or environment")
    if path_check and path_check.get("code_digest") != report.get("code_digest"):
        reasons.append("mixed digest or environment")
    if not report.get("reads"):
        reasons.append("the report has no reads block")
    return reasons


def q1(reads: dict) -> dict:
    a, b = reads["a"], reads["b"]
    shown = [n for n, d in (("a", a), ("b", b)) if d["ci95"] is not None and d["ci95"][0] > 0 and d["point"] >= Q1_NATS]
    quiet = all(d["ci95"] is not None and d["ci95"][1] < Q1_NATS for d in (a, b))
    return dict(outcome="shown" if shown else "not_detected" if quiet else "unresolved", shown_by=shown, a=a, b=b)


def position(p: dict, high: str, low: str, between: str) -> str:
    point, interval = p["point"], p["ci95"]
    if point is None or interval is None:
        return between
    if point >= HIGH and interval[0] >= BOUND:
        return high
    if point <= LOW and interval[1] <= BOUND:
        return low
    return between


def read_rule(report: dict) -> dict:
    if report.get("declared_read") is not True:
        raise SystemExit("a smoke's report: no rule applies")
    out = dict(status=STATUS, code_digest=report.get("code_digest"))
    reasons = void_reasons(report)
    if reasons:
        return dict(out, void=True, reasons=reasons, q1=dict(outcome="void"), q2=dict(outcome="void"), q3=dict(outcome="void"))
    reads = report["reads"]["likelihood_minus_blank"]
    gap = reads["real_minus_identity"]
    out.update(void=False, reasons=[], q1=q1(reads), real_minus_identity=gap)
    if gap["ci95"] is None or gap["ci95"][0] <= 0:
        return dict(out, q2=dict(outcome="void", reason="real - identity's lower bound is at most 0"), q3=dict(outcome="void", reason="real - identity's lower bound is at most 0"))
    return dict(out, q2=dict(outcome=position(reads["p2"], "yes", "no", "partial"), p2=reads["p2"]), q3=dict(outcome=position(reads["p3"], "fill", "mark", "both"), p3=reads["p3"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    print(json.dumps(read_rule(json.loads(Path(args.report).read_text())), indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
