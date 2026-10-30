"""Round 18 verdict (docs/NEXT_PHASE_PLAN.md "Round 18: the UWB 4x-data test", declared 2026-10-30 13:03 UTC), from
the two paired comparisons ``compare_pilot_lexicon.py`` writes for the arm's pilot:

- against R16C (``round16/pilot-r16c.json``, run with ``--per-subset``) and
- against W14b step 30000 (``v3read/pilot-w14b-step30000-v3d.json``).

**Rule (all must hold):** paired BLEU lower bound > 0 against R16C; paired BLEU lower bound > 0 against W14b; v3
lexicon lower bound > -0.05 against R16C. End-to-end against blank (the arm pilot's own blank read, 1.859 in every
round-16 report) is reported beside it, not gated.

Not evaluable (exit 2) when a comparison lacks the paired BLEU (no per-clip hypotheses), names another arm or control
than expected, or was read by another instrument than recogniser v3d; or when the arm pilot is not the declared
command's output (1,000 clips, the pinned word list, recogniser v3d, the expected checkpoint). Exit 0 PASS, 4 FAIL.

    python scripts/round18_verdict.py --arm-pilot pilot-r18.json --vs-r16c compare-r18-vs-r16c.json \\
        --vs-w14b compare-r18-vs-w14b.json --out verdict-r18.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

O = "acquisition/2026-10-17-v1/youtube-asl-v1"
ARM_RUN = "tokens-rvq-d0b-prior-r18-uwb"
ARM_CHECKPOINT = f"training-runs/{ARM_RUN}/checkpoints/step-00010000.pt"
ARM_PILOT = f"{O}/round18/pilot-r18.json"
R16C_PILOT = f"{O}/round16/pilot-r16c.json"
W14B_PILOT = f"{O}/v3read/pilot-w14b-step30000-v3d.json"
INSTRUMENT = "sign-recognizer-v3d"
WORD_LIST_SHA = "676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a"
CLIPS = 1000
LEXICON_FLOOR = -.05


class NotEvaluable(Exception):
    pass


def check_pilot(pilot: dict, checkpoint_key: str) -> None:
    problems = []
    if pilot.get("checkpoint_key") != checkpoint_key:
        problems.append(f"checkpoint {pilot.get('checkpoint_key')!r}, expected {checkpoint_key!r}")
    if pilot.get("clips") != CLIPS:
        problems.append(f"{pilot.get('clips')} clips, expected {CLIPS}")
    if (pilot.get("word_list") or {}).get("sha256") != WORD_LIST_SHA:
        problems.append("word list is not the pinned round14/r5-word-split.json")
    if (pilot.get("lexicon") or {}).get("recognizer_run") != INSTRUMENT:
        problems.append(f"lexicon read by {(pilot.get('lexicon') or {}).get('recognizer_run')!r}, not {INSTRUMENT}")
    if "hypotheses" not in ((pilot.get("meaning") or {}).get("generated") or {}):
        problems.append("no per-clip hypotheses (the paired BLEU needs them)")
    if problems:
        raise NotEvaluable("arm pilot: " + "; ".join(problems))


def check_comparison(name: str, cmp: dict, arm_key: str, control_key: str) -> None:
    problems = []
    if cmp.get("arm_key") != arm_key:
        problems.append(f"arm {cmp.get('arm_key')!r}, expected {arm_key!r}")
    if cmp.get("control_key") != control_key:
        problems.append(f"control {cmp.get('control_key')!r}, expected {control_key!r}")
    if cmp.get("instrument") != INSTRUMENT:
        problems.append(f"instrument {cmp.get('instrument')!r}, not {INSTRUMENT}")
    if not cmp.get("bleu_paired") or len(cmp["bleu_paired"].get("ci95") or ()) != 2:
        problems.append("no paired BLEU difference")
    if len(cmp.get("ci95") or ()) != 2:
        problems.append("no lexicon interval")
    if problems:
        raise NotEvaluable(f"{name}: " + "; ".join(problems))


def verdict(pilot: dict, vs_r16c: dict, vs_w14b: dict, *, arm_key: str = ARM_PILOT,
            checkpoint_key: str = ARM_CHECKPOINT) -> dict:
    check_pilot(pilot, checkpoint_key)
    check_comparison("vs R16C", vs_r16c, arm_key, R16C_PILOT)
    check_comparison("vs W14b", vs_w14b, arm_key, W14B_PILOT)
    rows = [
        dict(row="paired BLEU vs R16C", difference=vs_r16c["bleu_paired"]["difference"],
             ci95=vs_r16c["bleu_paired"]["ci95"], bar="lower bound > 0", passes=vs_r16c["bleu_paired"]["ci95"][0] > 0),
        dict(row="paired BLEU vs W14b", difference=vs_w14b["bleu_paired"]["difference"],
             ci95=vs_w14b["bleu_paired"]["ci95"], bar="lower bound > 0", passes=vs_w14b["bleu_paired"]["ci95"][0] > 0),
        dict(row="v3 lexicon vs R16C", difference=vs_r16c["difference"], ci95=vs_r16c["ci95"],
             bar=f"lower bound > {LEXICON_FLOOR}", passes=vs_r16c["ci95"][0] > LEXICON_FLOOR),
    ]
    meaning = pilot["meaning"]
    blank = (meaning.get("blank") or {}).get("bleu")
    return dict(
        verdict="PASS" if all(r["passes"] for r in rows) else "FAIL", rows=rows,
        rule="docs/NEXT_PHASE_PLAN.md, Round 18 (declared 2026-10-30 13:03 UTC): all three rows must pass",
        arm=dict(pilot_key=arm_key, checkpoint_key=pilot["checkpoint_key"], bleu=meaning["generated"]["bleu"],
                 bleu_ci95=meaning["generated"].get("ci95"), lexicon_top5=pilot["lexicon"]["generated_top5"]),
        controls=dict(r16c=dict(pilot_key=R16C_PILOT, **vs_r16c["control"]),
                      w14b=dict(pilot_key=W14B_PILOT, **vs_w14b["control"])),
        reported=dict(blank_bleu=blank, real_bleu=(meaning.get("real") or {}).get("bleu"),
                      arm_minus_blank=None if blank is None else round(meaning["generated"]["bleu"] - blank, 3),
                      arm_above_blank=meaning.get("above_blank"), lexicon_vs_w14b=dict(
                          difference=vs_w14b["difference"], ci95=vs_w14b["ci95"]),
                      per_subset_vs_r16c=vs_r16c.get("per_subset")),
        next=("a declared longer run on the same mix" if all(r["passes"] for r in rows) else
              "more data at this scale does not fix end-to-end; the composition path (M1 2.55) is the focus"))


def load(spec: str, store) -> dict:
    return json.loads(Path(spec).read_bytes() if Path(spec).exists() else store.get(spec))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm-pilot", required=True, help="local path or store key of the arm's pilot report")
    p.add_argument("--vs-r16c", required=True, help="compare_pilot_lexicon.py output, arm vs R16C")
    p.add_argument("--vs-w14b", required=True, help="compare_pilot_lexicon.py output, arm vs W14b")
    p.add_argument("--arm-key", default=ARM_PILOT, help="the arm pilot's store key the comparisons must name")
    p.add_argument("--checkpoint-key", default=ARM_CHECKPOINT)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    store = None
    if not all(Path(s).exists() for s in (args.arm_pilot, args.vs_r16c, args.vs_w14b)):
        from signtranslator.data.text_motion_dataset import R2Store
        store = R2Store(os.environ)
    try:
        out = verdict(load(args.arm_pilot, store), load(args.vs_r16c, store), load(args.vs_w14b, store),
                      arm_key=args.arm_key, checkpoint_key=args.checkpoint_key)
    except NotEvaluable as error:
        out = dict(verdict="NOT EVALUABLE", reason=str(error))
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out))
        return 2
    Path(args.out).write_text(json.dumps(out, indent=1))
    for r in out["rows"]:
        print(f"{r['row']:<22} {r['difference']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  {r['bar']:<20} "
              f"{'pass' if r['passes'] else 'FAIL'}")
    rep = out["reported"]
    print(f"arm BLEU {out['arm']['bleu']} (blank {rep['blank_bleu']}, real {rep['real_bleu']}); "
          f"lexicon {out['arm']['lexicon_top5']}")
    print(f"R18-VERDICT {out['verdict']}")
    return 0 if out["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
