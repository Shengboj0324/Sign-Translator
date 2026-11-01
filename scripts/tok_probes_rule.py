"""Reads a merged tokenizer-probes report and applies the rule declared in docs/NEXT_PHASE_PLAN.md, "Tokenizer probes and seam ablation declared (2026-11-01,
04:36 UTC)". **The plan's text decides, not this script.** It reads a local report file and nothing else, and the job never calls it. Any difference from the plan
is a bug here.

**Void, all reads.** Canary not reproduced; ``check`` nonzero; no code digest; more than one digest or environment (the merge refuses a mix, and the canary's block must
name the report's); ``seam_cval45`` ``bridged_frame_share`` outside [0.40, 0.47] or ``frames_per_bridge`` outside [7, 9]. A smoke's report is not a read: refused.

**B1,** on corpus BLEU, D = identity − round_trip and p = (no_quant − round_trip) / D. Void if D's lower bound is at most 0. Autoencoder limits: p ≤ 0.25, identity −
no_quant's lower bound above 0, no_quant − round_trip's upper bound at least 0. Quantiser limits: p ≥ 0.75, no_quant − round_trip's lower bound above 0. Otherwise
unresolved. **Agreement:** the same rule on likelihood minus blank and on the three ``_masked`` forms (corpus BLEU); landing on the opposite branch makes the outcome
unresolved, and these reads cannot create a branch. The plan does not name the masked forms' likelihood, so that read is shown and cannot change the outcome.

**C0,** ``seam_cval45``'s BLEU share of real's gap. Fall shown: upper bound below 0.85. Seams ruled out: lower bound at least 0.85, ``seam_cmax45`` and ``seam_crnd45``
BLEU shares at least 0.85 at point, ``seam_cval45``'s likelihood share at least 0.85 at point. Otherwise unresolved.

**Reported, not deciding.** The loop effect per form (``paired.real.loop_share``'s lower bound above 0 and the form's point at least 41.3%), the healing effect (cmin − cval),
the squeeze forms, ``filled_cuts``, the word-count forms.
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
P_AUTOENCODER, P_QUANTISER = .25, .75
SEAM_BAR = .85
BRIDGED_SHARE, FRAMES_PER_BRIDGE = (.40, .47), (7., 9.)
LOOP_POINT = .413                     # midway between real 28.2 and a1 54.4, by eval.loop_share
OPPOSITE = dict(autoencoder="quantiser", quantiser="autoencoder")
SQUEEZE = ("seam_csq196_min", "seam_csq196_rnd")
LOOP_FORMS = ("seam_cval45", "seam_cmin45", "seam_cmin30", "seam_cmax45", "seam_crnd45", "seam_crnd30", "seam_csq196_min", "seam_csq196_rnd")
ACTIONS = dict(autoencoder="V3-D is eligible for section 9's 200-clip test, nothing more",
               quantiser="no training by itself",
               unresolved="no tokenizer training on B1's evidence",
               fall_shown="L1 goes to its 200-clip test, nothing else",
               seams_ruled_out="L1 does not start. May be stated only as: linear bridges between continuous real poses do not limit v1. It does not clear composition's stitched seams",
               c0_unresolved="L1 neither starts nor closes on C0, goes to the owner, and gets no more D20 forms")


def check_nonzero(report: dict) -> bool:
    """``tok_probes check``'s own verdict on this report, so the two cannot differ."""
    import tok_probes
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "report.json"
        path.write_text(json.dumps(report))
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                return tok_probes.run_check(argparse.Namespace(report=str(path))) != 0
            except (KeyError, TypeError):     # a report without the fields check reads
                return True


def void_reasons(report: dict) -> list[str]:
    reasons = []
    checks = report.get("checks") or {}
    canary = checks.get("canary") or {}
    if canary.get("reproduced") is not True:
        reasons.append("canary not reproduced")
    if check_nonzero(report):
        reasons.append("check nonzero")
    if not report.get("code_digest"):
        reasons.append("no code digest")
    if not report.get("environment"):
        reasons.append("no environment")
    if canary and (canary.get("code_digest") != report.get("code_digest") or canary.get("environment") != report.get("environment")):
        reasons.append("more than one digest or environment")
    seam = ((report.get("forms") or {}).get("seam_cval45") or {}).get("seam") or {}
    share, frames = seam.get("bridged_frame_share"), seam.get("frames_per_bridge")
    if share is None or not BRIDGED_SHARE[0] <= share <= BRIDGED_SHARE[1]:
        reasons.append(f"seam_cval45 bridged_frame_share {share} outside {list(BRIDGED_SHARE)}")
    if frames is None or not FRAMES_PER_BRIDGE[0] <= frames <= FRAMES_PER_BRIDGE[1]:
        reasons.append(f"seam_cval45 frames_per_bridge {frames} outside {list(FRAMES_PER_BRIDGE)}")
    return reasons


def b1_branch(metric: dict | None) -> dict:
    """The rule on one metric of one set of forms."""
    if metric is None:
        return dict(branch="void", reason="its forms are not in this report")
    p = metric["p"]
    if metric["identity_minus_round_trip"]["ci95"][0] <= 0 or p is None:
        return dict(branch="void", reason="D's lower bound is at most 0", p=p)
    gain = metric["no_quant_minus_round_trip"]["ci95"]
    if p <= P_AUTOENCODER and metric["identity_minus_no_quant"]["ci95"][0] > 0 and gain[1] >= 0:
        return dict(branch="autoencoder", p=p)
    if p >= P_QUANTISER and gain[0] > 0:
        return dict(branch="quantiser", p=p)
    return dict(branch="unresolved", p=p, below_round_trip=gain[1] < 0)


def b1_outcome(blocks: dict) -> dict:
    unmasked, masked = blocks.get("unmasked") or {}, blocks.get("masked") or {}
    primary = b1_branch(unmasked.get("bleu"))
    agreement = dict(likelihood=b1_branch(unmasked.get("likelihood_minus_blank")), masked=b1_branch(masked.get("bleu")))
    outcome, downgraded = primary["branch"], []
    if outcome in OPPOSITE:
        downgraded = [name for name, read in agreement.items() if read["branch"] == OPPOSITE[outcome]]
        if downgraded:
            outcome = "unresolved"
    return dict(outcome=outcome, primary=primary, agreement=agreement, downgraded_by=downgraded, shown_only=dict(masked_likelihood=b1_branch(masked.get("likelihood_minus_blank"))),
                action=ACTIONS.get(outcome))


def _at_least(value, bar) -> bool:
    return value is not None and value >= bar


def loop_effect(form: dict) -> dict:
    lower, point = form["paired"]["real"]["loop_share"]["ci95"][0], form["loop_share"]["mean"]
    return dict(shown=bool(lower > 0 and point >= LOOP_POINT), lower_bound=lower, point=point)


def reported(forms: dict, sensitivity_only) -> dict:
    primary, heal = forms["seam_cval45"], forms.get("seam_cmin45")
    against = primary.get("comparators", {}).get("seam_cmin45", {}).get("bleu")
    return dict(loop_effect={n: loop_effect(forms[n]) for n in LOOP_FORMS if n in forms},
                healing_effect=dict(cmin_share=heal["share_of_gap"]["bleu"], cval_share=primary["share_of_gap"]["bleu"],
                                    cmin_minus_cval_bleu=dict(difference=-against["difference"], ci95=[-against["ci95"][1], -against["ci95"][0]]) if against else None) if heal else None,
                squeeze={n: forms[n]["share_of_gap"]["bleu"] for n in SQUEEZE if n in forms}, filled_cuts=primary["seam"]["filled_cuts"],
                bridged_frame_share=primary["seam"]["bridged_frame_share"], frames_per_bridge=primary["seam"]["frames_per_bridge"], likelihood_share=primary["share_of_gap"]["likelihood"],
                word_count_forms={n: forms[n]["share_of_gap"]["bleu"] for n in sensitivity_only if n in forms})


def c0_outcome(forms: dict) -> dict:
    primary = forms["seam_cval45"]["share_of_gap"]
    share, interval = primary["bleu"], primary["bleu_ci95"]
    others = {n: (forms.get(n) or {}).get("share_of_gap", {}).get("bleu") for n in ("seam_cmax45", "seam_crnd45")}
    conditions = dict(lower_bound=interval is not None and interval[0] >= SEAM_BAR, **{n: _at_least(v, SEAM_BAR) for n, v in others.items()},
                      likelihood=_at_least(primary["likelihood"], SEAM_BAR))
    if interval is not None and interval[1] < SEAM_BAR:
        outcome = "fall_shown"
    elif all(conditions.values()):
        outcome = "seams_ruled_out"
    else:
        outcome = "unresolved"
    return dict(outcome=outcome, share=share, ci95=interval, shares=others, conditions=conditions, action=ACTIONS["c0_unresolved" if outcome == "unresolved" else outcome])


def read_rule(report: dict) -> dict:
    if report.get("declared_read") is not True:
        raise SystemExit("a smoke's report: no rule applies")
    out = dict(status=STATUS, code_digest=report.get("code_digest"))
    reasons = void_reasons(report)
    if reasons:
        return dict(out, void=True, reasons=reasons, b1=dict(outcome="void"), c0=dict(outcome="void"))
    forms = report["forms"]
    return dict(out, void=False, reasons=[], b1=b1_outcome(report["b1"]), c0=c0_outcome(forms), reported=reported(forms, report.get("sensitivity_only") or []))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    print(json.dumps(read_rule(json.loads(Path(args.report).read_text())), indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
