"""Test-set protection for the declared evaluation chain (prepared for F12's step-60000 chain, 2026-10-28; to be
declared as an amendment before that chain runs).

R6, the strict review's test-set confirmation, and everything else that reads test data (the cross-signer baseline's
test envelope, the strict review's ``test`` and ``heldout_official`` evaluations, R7 on the test states) run **only**
if every validation rule passed:

- the exit gate (``multi_signer_gate.py``): its selected candidate passes every check on every validation group
  (``all_pass`` true and no failing check);
- the strict review's validation rules R1, R2, R2b, R3, R4, R5, from a ``strict_prior_review.py --validation-only``
  report of that same selection (not a ``--quick`` dry run);
- R7 on the validation states (``smoothness_check.py``).

Every rule must be present and its ``pass`` must be exactly ``true``: a missing rule, a missing report, ``null`` or
any non-boolean value blocks (fail closed). Otherwise the chain stops before reading any test data and records
``NOT_RUN``.

``decide`` is used twice: by the chain (this script's CLI, before any test-reading stage) and again inside
``strict_prior_review.py --r6-from-validation-key``, which refuses to evaluate the test partition unless it allows.

CLI: ``decide`` writes the decision to ``--output-key`` and exits 0 (R6 may run) or 3 (blocked); ``verdict`` writes the
chain's final verdict, with R6 and R7-test recorded as not run when the guard blocked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

VALIDATION_RULES = ("R1_retrieval_above_blank", "R2_back_translation_above_blank",
                    "R2b_2d_back_translation_above_blank", "R3_no_copying", "R4_length_slices", "R5_sign_level_words")
NOT_RUN = "R6 not run: validation rules failed (test partition left untouched)"
BLOCKED = 3


def _passed(value) -> bool:
    """Only a literal boolean ``True`` passes (``1``, ``"true"`` or ``None`` do not)."""
    return value is True


def gate_passed(gate: dict | None) -> tuple[bool, list[str]]:
    """Whether the gate's selected candidate passed every check, with the reasons it did not."""
    if not isinstance(gate, dict) or not isinstance(gate.get("selected"), dict):
        return False, ["exit gate report missing or without a selection"]
    sel = gate["selected"]
    rows = [r for r in gate.get("rows") or [] if r.get("run") == sel.get("run") and r.get("setting") == sel.get("setting")]
    reasons = []
    if len(rows) != 1:
        reasons.append(f"exit gate: the selected candidate has {len(rows)} rows (expected 1)")
    elif not _passed(rows[0].get("all_pass")):
        reasons.append(f"exit gate: the selected candidate passed {rows[0].get('passed')}/{rows[0].get('total')} checks")
    if sel.get("failing") != []:
        reasons.append(f"exit gate: failing checks {sel.get('failing')}")
    return not reasons, reasons


def r7_provenance(review: dict) -> tuple[list[str], dict] | None:
    """(sorted generated prefixes, post-processing) that smoothness_check.py's validation report of the review's
    selection must show: every group's ``<derived>/generated/<run>-<group>/<tag>`` with the selection's head smoothing
    and damping and the review's head-translation smoothing. None if the review does not record them."""
    sel, derived, groups = review.get("selected") or {}, review.get("derived"), review.get("groups")
    if not derived or not groups or not all(k in sel for k in ("generated_run", "tag", "head_smoothing", "head_damping")):
        return None
    if not isinstance(review.get("translation_smoothing"), (int, float)):
        return None
    prefixes = sorted(f"{derived}/generated/{sel['generated_run']}-{g}/{sel['tag']}" for g in groups)
    return prefixes, dict(head_smoothing=float(sel["head_smoothing"]), head_damping=float(sel["head_damping"]),
                          translation_smoothing=float(review["translation_smoothing"]))


def r7_test_provenance(review: dict) -> tuple[list[str], dict] | None:
    """The same for R7 on R6's saved test states (already post-processed, so judged without further processing)."""
    sel, saved = review.get("selected") or {}, review.get("heldout_save_generated")
    if not saved or "tag" not in sel:
        return None
    return [f"{saved.rstrip('/')}/test/{sel['tag']}"], dict(head_smoothing=0., head_damping=1., translation_smoothing=0.)


def decide(gate: dict | None, review: dict | None, r7_validation: dict | None) -> dict:
    """The guard: ``run_r6`` is True only if the gate, every validation rule of the strict review and R7 on the
    validation states passed, for the same selected setting."""
    ok_gate, reasons = gate_passed(gate)
    rules = {"exit_gate": ok_gate}
    if not isinstance(review, dict):
        reasons.append("strict review report missing")
        review = {}
    if not _passed(review.get("validation_only")):
        reasons.append("the strict review report is not a --validation-only report")
    if review.get("quick_dry_run") is not False:
        reasons.append("the strict review report is a --quick dry run (never a verdict)")
    sel = (gate or {}).get("selected") or {}
    reviewed = review.get("selected") or {}
    if (reviewed.get("run"), reviewed.get("setting")) != (sel.get("run"), sel.get("setting")) or not sel.get("run"):
        reasons.append(f"the strict review judged {reviewed.get('run')} / {reviewed.get('setting')}, the gate selected "
                       f"{sel.get('run')} / {sel.get('setting')}")
    review_rules = review.get("rules") or {}
    for rule in VALIDATION_RULES:
        entry = review_rules.get(rule)
        ok = isinstance(entry, dict) and _passed(entry.get("pass"))
        rules[rule] = ok
        if not ok:
            reasons.append(f"{rule}: " + ("missing" if not isinstance(entry, dict) else f"pass = {entry.get('pass')!r}"))
    ok_r7 = isinstance(r7_validation, dict) and _passed(r7_validation.get("pass"))
    if not ok_r7:
        reasons.append("R7 (validation states): " + ("report missing" if not isinstance(r7_validation, dict)
                                                     else f"pass = {r7_validation.get('pass')!r}"))
    else:                       # it must be the report of this selection's states and post-processing, not a stale one
        expected = r7_provenance(review)
        found = (sorted(r7_validation.get("groups") or {}), r7_validation.get("post_processing"))
        if expected is None or found != expected:
            ok_r7 = False
            reasons.append(f"R7 (validation states): the report judged {found}, the selection needs {expected}")
    rules["R7_smoothness_validation"] = ok_r7
    run = not reasons
    return dict(run_r6=run, rules=rules, reasons=reasons, selected=dict(run=sel.get("run"), setting=sel.get("setting")),
                message="R6 allowed: the exit gate and every validation rule passed" if run else NOT_RUN)


def verdict(review: dict, guard: dict, r7: dict[str, dict | None]) -> dict:
    """The chain's verdict: every rule R1-R7 must pass; R6 and R7-test are recorded as not run when blocked."""
    rules = {k: (v.get("pass") if isinstance(v, dict) else None) for k, v in (review.get("rules") or {}).items()}
    if not guard["run_r6"]:
        rules["R6_test_confirmation"] = "not run"
    def judged(report, expected) -> bool:     # passed, and is the report of this selection's states (not a stale one)
        return (isinstance(report, dict) and _passed(report.get("pass")) and expected is not None
                and (sorted(report.get("groups") or {}), report.get("post_processing")) == expected)
    r7_val = judged(r7.get("validation"), r7_provenance(review))
    r7_test = judged(r7.get("test"), r7_test_provenance(review)) if guard["run_r6"] else "not run"
    rules["R7_smoothness"] = r7_val and r7_test is True
    passed = (guard["run_r6"] and all(_passed(rules.get(k)) for k in VALIDATION_RULES)
              and _passed(rules.get("R6_test_confirmation")) and r7_val and r7_test is True
              and _passed(review.get("pass")) and _passed(guard["rules"].get("exit_gate"))
              and guard["selected"]["run"] == (review.get("selected") or {}).get("run")
              and guard["selected"]["setting"] == (review.get("selected") or {}).get("setting"))
    return dict(rules=rules, exit_gate=guard["rules"]["exit_gate"], review_pass=review.get("pass"),
                r7_validation=r7_val, r7_test=r7_test, test_set_guard=dict(run_r6=guard["run_r6"], message=guard["message"],
                                                                         reasons=guard["reasons"]),
                **({} if guard["run_r6"] else {"r6": NOT_RUN}), **{"pass": bool(passed)})


def _get(store, key: str | None):
    if not key or not store.exists(key):
        return None
    return json.loads(store.get(key))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("decide", "verdict"))
    parser.add_argument("--store", default="r2")
    parser.add_argument("--gate-key", required=True)
    parser.add_argument("--review-key", required=True, help="decide: the --validation-only report; verdict: the final "
                                                            "report (merged with R6 when it ran, else the validation one)")
    parser.add_argument("--r7-validation-key", required=True)
    parser.add_argument("--r7-test-key", help="verdict: R7 on the test states (only read when the guard allowed R6)")
    parser.add_argument("--guard-key", help="verdict: the decide output")
    parser.add_argument("--output-key", required=True)
    args = parser.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.command == "decide":
        out = decide(_get(store, args.gate_key), _get(store, args.review_key), _get(store, args.r7_validation_key))
        store.put(args.output_key, json.dumps(out, indent=2, sort_keys=True).encode())
        print(json.dumps(out))
        return 0 if out["run_r6"] else BLOCKED
    guard = _get(store, args.guard_key)
    if guard is None:
        raise SystemExit("verdict: the guard decision is missing")
    review = _get(store, args.review_key)
    if review is None:
        raise SystemExit("verdict: the strict review report is missing")
    r7 = dict(validation=_get(store, args.r7_validation_key), test=_get(store, args.r7_test_key) if guard["run_r6"] else None)
    out = verdict(review, guard, r7)
    store.put(args.output_key, json.dumps(out, indent=2, sort_keys=True).encode())
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
