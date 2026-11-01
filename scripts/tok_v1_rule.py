"""Reads a merged V1 report and applies the DRAFT rule (for review; the plan's declaration decides, not this script). ``tok_v1.py`` holds no rule, and the job never calls this. It reads a local file only.

**Void all:** canary or path check not passed, ``check`` nonzero, mixed or missing digest or environment. A smoke's report is refused.

**Cap,** on likelihood minus blank: c = (rt_nearest − blank) / (v0_nearest_fill − blank). Kept: c at least 0.75 with a lower bound of at least 0.5. Lost: c at most 0.25 with an upper bound of at most 0.5. Else partial.

**Generated,** a9@predicted's neutral share at 5° minus real's on tracked frames, clip by clip. Shown: lower bound above 0, point at least 0.05, and the point above 0 at 2° and at 10° too. Absent: upper bound below 0.02. Else unresolved.
**Alignment,** a9's word clips at 5°: neutral where training was untracked minus where tracked, lower bound above 0.

**Sensitivity,** a9p_replaced − a9p on likelihood. Void if composed-check did not reproduce. Rise: lower bound above 0 and point at least 0.015 nats. No rise: upper bound below 0.015. Else unresolved. Never a score.

**The hypothesis** (generated hands carry trained neutral fill, and v1 reads it badly). Supported: generated shown, aligned, a rise. Refuted for generated signing: generated absent, or no rise with generated shown. Otherwise open.
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

STATUS = "draft, not declared; the plan's text decides"
HIGH, LOW, BOUND = .75, .25, .5
SHOWN_POINT, ABSENT_UPPER, RISE_NATS = .05, .02, .015
FLAG, NEAR, FAR = "5.0", "2.0", "10.0"


def check_nonzero(report: dict) -> bool:
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
    if canary and (canary.get("code_digest") != report.get("code_digest") or canary.get("environment") != report.get("environment")):
        reasons.append("mixed digest or environment")
    if path_check and path_check.get("code_digest") != report.get("code_digest"):
        reasons.append("mixed digest or environment")
    return reasons


def cap(block: dict | None) -> dict:
    if not block or not block.get("point") and block.get("point") != 0 or block.get("ci95") is None:
        return dict(outcome="partial", share=block)
    point, interval = block["point"], block["ci95"]
    if point >= HIGH and interval[0] >= BOUND:
        return dict(outcome="kept", share=block)
    if point <= LOW and interval[1] <= BOUND:
        return dict(outcome="lost", share=block)
    return dict(outcome="partial", share=block)


def generated(differences: dict) -> dict:
    main, near, far = differences[FLAG], differences[NEAR], differences[FAR]
    if main is None:
        return dict(outcome="unresolved", excess=None)
    interval = main["ci95"]
    if interval[0] > 0 and main["difference"] >= SHOWN_POINT and near is not None and far is not None and near["difference"] > 0 and far["difference"] > 0:
        return dict(outcome="shown", excess=main)
    if interval[1] < ABSENT_UPPER:
        return dict(outcome="absent", excess=main)
    return dict(outcome="unresolved", excess=main)


def aligned(block: dict | None) -> dict:
    row = ((block or {}).get("thresholds") or {}).get(FLAG) or {}
    return dict(aligned=bool(row.get("ci95")) and row["ci95"][0] > 0, row=row or None)


def sensitivity(block: dict | None, composed: dict | None) -> dict:
    if not (composed or {}).get("reproduced"):
        return dict(outcome="void", reason="a9p did not reproduce the stored a10 read")
    if not block or block.get("ci95") is None:
        return dict(outcome="unresolved", difference=block)
    point, interval = block["point"], block["ci95"]
    if interval[0] > 0 and point >= RISE_NATS:
        return dict(outcome="rise", difference=block)
    if interval[1] < RISE_NATS:
        return dict(outcome="no_rise", difference=block)
    return dict(outcome="unresolved", difference=block)


def hypothesis(gen: str, is_aligned: bool, sens: str) -> str:
    if gen == "shown" and is_aligned and sens == "rise":
        return "supported"
    if gen == "absent" or (gen == "shown" and sens == "no_rise"):
        return "refuted_for_generated_signing"
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
    gen = generated(neutral["a9_predicted_minus_real_tracked"])
    align = aligned(neutral.get("word_alignment", {}).get("a9_words"))
    sens = sensitivity(reads["a9p_replaced_minus_a9p"], report["checks"].get("composed_check"))
    return dict(out, void=False, reasons=[], cap=cap(reads["cap_share_of_nearest_fill"]), generated=gen, alignment=align, sensitivity=sens, hypothesis=hypothesis(gen["outcome"], align["aligned"], sens["outcome"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    print(json.dumps(read_rule(json.loads(Path(args.report).read_text())), indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
