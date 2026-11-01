"""Reads a merged tokenizer-probes report and applies the DRAFT rule (not declared; it is for review). Separate from ``tok_probes.py``, which stays
report_only and holds no threshold, and from the job, which never calls this.

**B1,** primary form ``no_quant``, on BLEU. D = identity − round_trip, p = (no_quant − round_trip) / D. D's lower bound at or below 0: void. p at most 0.25 with
identity − no_quant's lower bound above 0: the decoder limits. p at least 0.75 with no_quant − round_trip's lower bound above 0: the quantiser limits. Otherwise
unresolved.

**Seam,** primary form ``seam_cval45``, share of real's gap on BLEU. Upper bound below 0.85: fall shown. Point at least 0.85: no fall shown. Otherwise unresolved.
A keep does not clear composition's seams, since the bridge endpoints come from one continuous motion, not stitched words. A fall may overstate, since windows can
sit on holds or moves. ``seam_cmax45``'s upper bound below 0.85 under a keep marks the keep as weak. Loop share against real is reported beside the outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

P_DECODER, P_QUANTISER, SEAM_KEEP = .25, .75, .85
STATUS = "draft, not declared"


def b1_outcome(block: dict | None) -> dict:
    if block is None:
        return dict(outcome="void", reason="B1's forms are not all in this report")
    bleu = block["bleu"]
    p = bleu["p"]
    if bleu["identity_minus_round_trip"]["ci95"][0] <= 0 or p is None:
        return dict(outcome="void", reason="identity - round_trip is not resolved above 0", p=p)
    if p <= P_DECODER and bleu["identity_minus_no_quant"]["ci95"][0] > 0:
        return dict(outcome="decoder", p=p)
    if p >= P_QUANTISER and bleu["no_quant_minus_round_trip"]["ci95"][0] > 0:
        return dict(outcome="quantiser", p=p)
    return dict(outcome="unresolved", p=p)


def seam_outcome(forms: dict) -> dict:
    primary = forms["seam_cval45"]
    share, interval = primary["share_of_gap"]["bleu"], primary["share_of_gap"]["bleu_ci95"]
    loop = primary["paired"]["real"]["loop_share"]
    out = dict(share=share, ci95=interval, likelihood_share=primary["share_of_gap"]["likelihood"], loop_share_minus_real=loop,
               loops_rise=loop["ci95"][0] > 0, bridged_frame_share=primary["seam"]["bridged_frame_share"], frames_per_bridge=primary["seam"]["frames_per_bridge"],
               filled_cuts=primary["seam"]["filled_cuts"])
    if interval[1] < SEAM_KEEP:
        out["outcome"] = "fall_shown"
    elif share >= SEAM_KEEP:
        out["outcome"] = "no_fall_shown"
        out["weak_keep"] = forms["seam_cmax45"]["share_of_gap"]["bleu_ci95"][1] < SEAM_KEEP
    else:
        out["outcome"] = "unresolved"
    return out


def read_rule(report: dict) -> dict:
    if report.get("report_only") is not True:
        raise SystemExit("not a report_only tokenizer-probes report")
    if not report.get("declared_read"):
        raise SystemExit("a smoke's report: no rule applies")
    if not (report.get("checks", {}).get("canary") or {}).get("reproduced"):
        raise SystemExit("no passed canary: the read is void")
    return dict(status=STATUS, code_digest=report["code_digest"], b1=b1_outcome(report["b1"]["unmasked"]), seam=seam_outcome(report["forms"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    print(json.dumps(read_rule(json.loads(Path(args.report).read_text())), indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
