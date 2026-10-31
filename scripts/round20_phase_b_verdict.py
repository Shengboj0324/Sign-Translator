"""Round 20 Phase B's verdict (docs/NEXT_PHASE_PLAN.md, "Round 20: composition v2", "Phase B", amendments 1-3). Reads stored
reports only; nothing generates, nothing is tuned. Pod job: ``runpod/fleet/round20_phase_b.sh``.

``verdict``, ``--set val1000``: the stacked arm's merged ``round15_compose_eval.py`` report against the stored W14b
composition (2.306). **Each item of the declared rule is judged and stored separately; all must hold:**

- ``bleu_vs_w14b_composition``: paired BLEU lower bound > 0 (``forms.core_crossfade.minus_paired_core_crossfade``, from the
  merge's ``--paired-with``);
- ``bleu_vs_other_sentence_words``: paired BLEU lower bound > 0 against its own other-sentence words
  (``minus_mismatched_words``);
- ``bleu_vs_own_end_to_end``: paired BLEU lower bound > 0 against its own end-to-end (``minus_end_to_end``, from the merge's
  ``--end-to-end-report``);
- ``v2_spotting_vs_w14b_composition``: recogniser v2's recall, paired per sentence, lower bound > -0.02, from both reports'
  shards (merged reports keep no per-clip spotting; ``round20_compare.spotting_guard``, 1,000 resamples);
- ``wrist_speed_reported``: the arm's wrist speed over real's, against the comparator's 2.64x. Holds if it is reported.

**Its own end-to-end** decodes as the arm's words do. With gains: the prior's stored 1,000-clip pilot, W14b
``round14b/pilot-w14b-step30000.json`` (the stored comparator's own end-to-end), R16 ``round16/pilot-r16.json`` (step 10000, A5's
and a8's checkpoint). With ``--no-gains 1``: that pilot's command rerun with ``--no-decoder-gains`` (the pod job's e2e phase),
because removing gains raises BLEU and the gained pilot would be a lenient comparator; a gained pilot is refused for such an arm,
and a pilot without gains for a gained one. The rerun must match the stored pilot (``--stored-pilot-key``) on checkpoint, clips,
guidance, temperatures and references. The merged report's ``comparators.end_to_end.checkpoint_key`` must be the arm's prior. The
item carries a note either way.

**Reported, never part of the rule:** next to BLEU, the length ratio (hypothesis over reference words, as S0) and the
likelihood minus blank; the point estimate against M1's 2.55 on this set (M1 needs C20 as well). No ``@predicted`` form: it
is reported only for an A3 arm, and no A3 arm was adopted. **The paired likelihood against W14b composition is not
available:** the stored comparator ran without ``--likelihood``, so it holds none at any level; the rule does not ask for it.

**Refused, nothing stored** (exit 1): another comparator or pilot (hash, or wrong prior), other clips, references,
derangement seed or translator (``round20_compare.check_pairable``); other selected words per clip; any setting of the arm that differs from the comparator's
apart from the adopted options; adopted options that are not the arm named by ``--prior``, ``--no-gains``, ``--best-of``.

**Incomplete, nothing stored** (exit 2): a rule item or a reported read cannot be read from the reports. The verdict is
printed with what is missing, and no write-once key is taken by a report that needs repairing. A stored verdict is complete:
rule failures are stored like passes (exit 0).

**One arm on a set:** the verdict is refused when another arm's report or shard is stored on the set (``other_arms``), unless the
arm's report records a ``declared_replacement``, which the verdict records too.

``--set c20`` (only after ``c20-gate``, never run so far): the same arm on the confirmation set. No comparator or pilot read
that set, so the only rule of the set applies: the point estimate >= 2.55. ``m1_claim`` holds if both sets hold it.

``c20-gate --verdict-key``: exit 0 only if the stored val1000 verdict's rule passed, its point estimate is >= 2.55, its arm is the
arm asked for, and no other arm on the 1,000 stands beside it without a declared replacement; else exit 1, and the pod job reads
nothing of C20.

``one-arm``: exit 0 only if no other arm is stored on the set (or a replacement is declared where one is). ``stack-check``: the
stored D20 read of a combination, from the reports: it must be that combination, and have both point estimates above 0 against
each ``--against-key`` (``round20_compare.compare_forms``'s ``both_improve``); with ``--lowpass 1`` its ``@lowpass`` against itself
too (amendment 4, rules 3 and 4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FORM = "core_crossfade"
PAIRED = f"paired:{FORM}"
M1_THRESHOLD = 2.55
SPOTTING_FLOOR = -.02
W14B_WRIST_SPEED = 2.64            # the plan's read of the comparator (motion_ratio_to_real: 2.6448)
SCORER = "sign-recognizer-v3d"
LOWPASS_HZ = 6.                    # amendment 4: A6 enters only as the stack's main form (LOWPASS=1)
# every round-20 option the arm leaves as the comparator had it (--likelihood reads, the adopted options are below)
ROUND20_OFF = dict(vocabulary_mix=None, semlex_core="fraction", activity_margin=None, oracle_source="isolated",
                   gloss_manifest=None, retime="reference", tempo=1., bridge_velocity=None,
                   predicted_duration=False, semlex_oracle_tracked=False, lowpass_report_hz=None)
NOT_PREDICTED = "not reported: @predicted applies to an A3 arm and no A3 arm was adopted"
NO_PAIRED_LIKELIHOOD = "the stored comparator ran without --likelihood: no likelihood of W14b composition exists at any level"


ARM_FILE = re.compile(r"^((?:w14b|r16)-(?:gains|nogains)-n(?:0|4|8|16)(?:-lp6)?-(val1000|c20))(?:-shard\d+)?\.json$")
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


def other_arms(keys, out: str, sentence_set: str, report: str) -> list[str]:
    """The arms other than ``report`` with a merged report or a shard directly under ``out`` on this set."""
    found = set()
    prefix = out.rstrip("/") + "/"
    for key in keys:
        match = ARM_FILE.match(key[len(prefix):]) if key.startswith(prefix) else None
        if match and match[2] == sentence_set and match[1] != report:
            found.add(match[1])
    return sorted(found)


def one_arm(others: list[str], declared, report: str) -> None:
    """Stop unless ``report`` is the set's only arm, or a replacement is declared where there is one to replace."""
    if declared and not COMMIT.match(str(declared)):
        raise SystemExit(f"{declared!r} is not a plan commit")
    if others and not declared:
        raise SystemExit(f"{report}: this set already holds {others}, and no amendment declares a replacement")
    if declared and not others:
        raise SystemExit(f"{report}: a replacement ({declared}) is declared, but this set holds no other arm")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def lower_bound_item(row, threshold: float, source: str) -> dict:
    """A rule item from a stored paired difference: the lower bound of its interval must be above ``threshold``."""
    try:
        difference, (low, high) = float(row["difference"]), row["ci95"]
        low, high = float(low), float(high)
    except (KeyError, TypeError, ValueError):
        return dict(available=False, passed=None, source=source, threshold=threshold,
                    reason=f"{source}: no paired difference with an interval in the report")
    return dict(available=True, passed=bool(low > threshold), source=source, threshold=threshold, difference=difference,
                ci95=[low, high], lower_bound=low)


def settings_differences(arm: dict, comparator: dict) -> list[str]:
    """The settings of ``round15_compose_eval.SETTINGS`` that differ, apart from the two that carry the adopted options:
    ``checkpoint_key`` and ``round20`` (``arm_errors`` reads those). ``rerank`` is 1 when a report predates it."""
    from round15_compose_eval import SETTINGS
    out = []
    for key in SETTINGS:
        if key in ("checkpoint_key", "round20"):
            continue
        x, y = arm.get(key), comparator.get(key)
        if key == "rerank":
            x, y = 1 if x is None else x, 1 if y is None else y
        if x != y:
            out.append(f"settings.{key}: {x!r} against the comparator's {y!r}")
    return out


def arm_errors(settings: dict, *, prior_checkpoint: str, no_gains: bool, best_of: int, sentences,
               lowpass: bool = False) -> list[str]:
    """What in the arm's settings is not the named arm: the prior, the gains, the cache, nothing else on, likelihood on."""
    r = settings.get("round20") or {}
    out = [f"settings.round20.{k}: {r.get(k)!r}, not {v!r}" for k, v in ROUND20_OFF.items() if r.get(k, v) != v]
    if r.get("lowpass_hz") != (LOWPASS_HZ if lowpass else None):
        out.append(f"settings.round20.lowpass_hz: {r.get('lowpass_hz')!r}, not {LOWPASS_HZ if lowpass else None!r}")
    if not r.get("likelihood"):
        out.append("settings.round20.likelihood is not on")
    if r.get("sentences") != sentences and (not sentences or (r.get("sentences") or {}).get("sha256") != sentences["sha256"]
                                            or (r.get("sentences") or {}).get("key") != sentences["key"]):
        out.append(f"settings.round20.sentences: {r.get('sentences')!r}, not {sentences!r}")
    cache = r.get("word_cache")
    if best_of:
        if not cache:
            return out + [f"--best-of {best_of}: the report has no word cache"]
        out += [f"word_cache.{k}: {cache.get(k)!r}, not {v!r}" for k, v in (
            ("n", best_of), ("checkpoint_key", prior_checkpoint), ("scorer", SCORER),
            ("no_decoder_gains", True if no_gains else None)) if cache.get(k) != v]
        if settings.get("checkpoint_key") is not None:
            out.append("a cached arm generated words as well (settings.checkpoint_key is set)")
        if r.get("no_decoder_gains"):
            out.append("a cached arm's gains are its cache's: settings.round20.no_decoder_gains is set")
    else:
        if cache:
            out.append("--best-of 0: the report used a word cache")
        if settings.get("checkpoint_key") != prior_checkpoint:
            out.append(f"settings.checkpoint_key: {settings.get('checkpoint_key')!r}, not {prior_checkpoint!r}")
        if bool(r.get("no_decoder_gains")) != no_gains:
            out.append(f"settings.round20.no_decoder_gains: {r.get('no_decoder_gains')!r}, not {no_gains!r}")
    return out


def read_errors(settings: dict, *, prior_checkpoint: str, no_gains: bool, best_of: int, sentences_sha256: str) -> list[str]:
    """What in a stored D20 read is not the named combination (its report options, A6's among them, are its own)."""
    r = settings.get("round20") or {}
    out = [] if (r.get("sentences") or {}).get("sha256") == sentences_sha256 else ["it did not read D20"]
    out += [f"settings.round20.{k}: {r.get(k)!r}" for k in ("vocabulary_mix", "lowpass_hz") if r.get(k) is not None]
    if r.get("retime") != "reference":
        out.append(f"settings.round20.retime: {r.get('retime')!r}")
    cache = r.get("word_cache")
    if best_of:
        if not cache:
            return out + [f"best of {best_of}: the read has no word cache"]
        out += [f"word_cache.{k}: {cache.get(k)!r}, not {v!r}" for k, v in (
            ("n", best_of), ("checkpoint_key", prior_checkpoint), ("no_decoder_gains", True if no_gains else None))
            if cache.get(k) != v]
    else:
        if cache:
            out.append("best of 0: the read used a word cache")
        if settings.get("checkpoint_key") != prior_checkpoint:
            out.append(f"settings.checkpoint_key: {settings.get('checkpoint_key')!r}")
        if bool(r.get("no_decoder_gains")) != no_gains:
            out.append(f"settings.round20.no_decoder_gains: {r.get('no_decoder_gains')!r}")
    return out


def stack_check(read: dict, against: dict, *, prior_checkpoint: str, no_gains: bool, best_of: int, lowpass: bool,
                sentences_sha256: str) -> dict:
    """The stored D20 reads behind a combination (module docstring); SystemExit when they do not support it."""
    from round20_compare import compare_forms
    wrong = read_errors(read.get("settings") or {}, prior_checkpoint=prior_checkpoint, no_gains=no_gains, best_of=best_of,
                        sentences_sha256=sentences_sha256)
    if wrong:
        raise SystemExit("the stored D20 read is not this combination: " + "; ".join(wrong))
    if not against:
        raise SystemExit("no read to compare this combination with")

    def estimates(c):
        ll = (c.get("likelihood_minus_blank") or {}).get("difference") or {}
        return dict(bleu=c["bleu"]["difference"], likelihood=ll.get("difference"), both_improve=c["both_improve"])
    out = {key: estimates(compare_forms(read, FORM, other, FORM, boot=100)) for key, other in against.items()}
    misses = [key for key, row in out.items() if not row["both_improve"]]
    if misses:
        raise SystemExit(f"both point estimates are not above 0 against {misses}: {out}")
    if lowpass:
        if not (no_gains and best_of):
            raise SystemExit("A6 enters only inside the stack (amendment 4, rule 4)")
        form = f"{FORM}@lowpass"
        out[form] = estimates(compare_forms(read, form, read, FORM, boot=100))
        if not out[form]["both_improve"]:
            raise SystemExit(f"the stack's @lowpass does not have both point estimates above 0 against the stack: {out[form]}")
    return out


def exact_bleu(report: dict) -> float:
    """The arm's corpus BLEU unrounded (the stored 3 decimals could round 2.5496 up to 2.55); it must round to the stored."""
    from sentence_exposure_diagnostic import bleu_from_stats, bleu_stats
    row = report["forms"][FORM]
    point = float(bleu_from_stats(bleu_stats(row["hypotheses"], report["references"]).sum(0)))
    if round(point, 3) != row["bleu"]:
        raise SystemExit(f"the report's BLEU {row['bleu']} is not its hypotheses' ({point:.4f})")
    return point


def m1_read(point: float) -> dict:
    return dict(threshold=M1_THRESHOLD, point_estimate=round(point, 4), met=bool(point >= M1_THRESHOLD))


def length_ratios(report: dict) -> dict:
    from sentence_exposure_diagnostic import bleu_stats
    from signtranslator.eval.s0_metrics import length_ratio
    return {name: length_ratio(bleu_stats(report["forms"][name]["hypotheses"], report["references"]))
            for name in (FORM, "real", "blank") if name in report["forms"]}


def next_to_bleu(arm: dict) -> dict:
    """The arm's BLEU with the two reads that sit next to it; ``None`` where the report has no likelihood."""
    row = arm["forms"][FORM]
    forms = (arm.get("likelihood") or {}).get("forms") or {}
    return dict(bleu=row["bleu"], ci95=row.get("ci95"), length_ratio=length_ratios(arm),
                likelihood_minus_blank=(forms.get(FORM) or {}).get("minus_blank"),
                real_likelihood_minus_blank=(forms.get("real") or {}).get("minus_blank"))


def wrist_item(arm: dict, comparator: dict | None) -> dict:
    ratio = ((arm.get("motion_ratio_to_real") or {}).get(FORM) or {}).get("wrist_speed")
    stored = (((comparator or {}).get("motion_ratio_to_real") or {}).get(FORM) or {}).get("wrist_speed")
    if ratio is None:
        return dict(available=False, passed=None, reason=f"motion_ratio_to_real.{FORM}.wrist_speed is not in the report")
    return dict(available=True, passed=True, ratio_to_real=ratio, against=W14B_WRIST_SPEED, comparator_ratio_to_real=stored)


def check_arm_form(arm: dict) -> None:
    if FORM not in (arm.get("forms") or {}) or not arm.get("references"):
        raise SystemExit(f"the arm's report has no {FORM} form or no references")
    if arm.get("clips") != len(arm["references"]):
        raise SystemExit("the arm's report is not one complete run (clips against references)")


def words_of(report: dict) -> list:
    return [(r.get("sign"), r.get("stage1_words"), r.get("entries")) for r in report.get("per_clip") or ()]


def judge_val1000(arm: dict, comparator: dict, pilot: dict, *, shards_arm, shards_comparator, prior_checkpoint: str,
                  no_gains: bool, best_of: int, boot: int = 1000, lowpass: bool = False, stored_pilot: dict | None = None) -> dict:
    """The verdict of the 1,000 from the loaded reports (module docstring); SystemExit when they are not paired."""
    from round20_compare import check_pairable, spotting_guard
    check_arm_form(arm)
    if FORM not in (comparator.get("forms") or {}):
        raise SystemExit(f"the comparator has no {FORM} form")
    check_pairable(arm, comparator)
    wrong = settings_differences(arm["settings"], comparator["settings"])
    wrong += arm_errors(arm["settings"], prior_checkpoint=prior_checkpoint, no_gains=no_gains, best_of=best_of, sentences=None,
                        lowpass=lowpass)
    if wrong:
        raise SystemExit("not the declared arm with the comparator's settings: " + "; ".join(wrong))
    if arm.get("stage1_check") != comparator.get("stage1_check") or words_of(arm) != words_of(comparator):
        raise SystemExit("the arm's selected words are not the comparator's (stage-1 check or per-clip words)")
    comparators = arm.get("comparators") or {}
    if PAIRED not in arm["forms"] or arm["forms"][PAIRED].get("hypotheses") != comparator["forms"][FORM]["hypotheses"]:
        raise SystemExit("the merge was not --paired-with this comparator")
    e2e = comparators.get("end_to_end")
    meaning = pilot.get("meaning") or {}
    if not e2e or "end_to_end" not in arm["forms"] or \
            arm["forms"]["end_to_end"].get("hypotheses") != (meaning.get("generated") or {}).get("hypotheses"):
        raise SystemExit("the merge's --end-to-end-report is not this pilot")
    if pilot.get("checkpoint_key") != prior_checkpoint or e2e.get("checkpoint_key") != prior_checkpoint:
        raise SystemExit(f"the end-to-end pilot is {pilot.get('checkpoint_key')!r}, not the arm's prior {prior_checkpoint!r}")
    if meaning.get("references") != arm["references"]:
        raise SystemExit("the end-to-end pilot's references are not the arm's")
    if pilot.get("first_clips"):
        raise SystemExit("the end-to-end pilot is a smoke (first_clips)")
    if bool(pilot.get("no_decoder_gains")) != no_gains:
        raise SystemExit("the arm's words have no decoder gains, the end-to-end pilot has them: a lenient comparator" if no_gains
                         else "the arm's words have decoder gains, the end-to-end pilot does not")
    same_pilot = None
    if no_gains:
        if stored_pilot is None:
            raise SystemExit("an end-to-end pilot without gains needs the stored pilot it reran (--stored-pilot-key)")
        differ = [k for k in ("checkpoint_key", "clips", "guidance", "part_temperature") if pilot.get(k) != stored_pilot.get(k)]
        if meaning["references"] != (stored_pilot.get("meaning") or {}).get("references"):
            differ.append("references")
        if differ:
            raise SystemExit(f"the end-to-end pilot without gains differs from the stored pilot in {differ}")
        same_pilot = {name: sum(a == b for a, b in zip(meaning[name]["hypotheses"], stored_pilot["meaning"][name]["hypotheses"]))
                      for name in ("real", "blank") if name in meaning and name in stored_pilot["meaning"]}
    row = arm["forms"][FORM]
    guard = spotting_guard(arm, FORM, comparator, FORM, shards_arm, shards_comparator, boot=boot)
    if guard["available"]:
        spotting = lower_bound_item(guard["difference"], SPOTTING_FLOOR, "the shards' per-clip spotting")
        spotting.update(arm_recall=guard["a"], comparator_recall=guard["b"], recogniser=guard["recogniser"],
                        word_source=guard["word_source"])
    else:
        spotting = dict(available=False, passed=None, threshold=SPOTTING_FLOOR, reason=guard["reason"])
    own = lower_bound_item(row.get("minus_end_to_end"), 0., "minus_end_to_end")
    own["note"] = ("against the prior's end-to-end regenerated WITHOUT gains (the stored pilot's seeds and clips), as this arm's "
                   "words decode" if no_gains else "against the prior's stored end-to-end pilot, with gains, as this arm's words")
    items = dict(
        bleu_vs_w14b_composition=lower_bound_item(row.get(f"minus_paired_{FORM}"), 0., f"minus_paired_{FORM}"),
        bleu_vs_other_sentence_words=lower_bound_item(row.get("minus_mismatched_words"), 0., "minus_mismatched_words"),
        bleu_vs_own_end_to_end=own,
        v2_spotting_vs_w14b_composition=spotting,
        wrist_speed_reported=wrist_item(arm, comparator))
    reads = next_to_bleu(arm)
    missing = [f"{k}: {v['reason']}" for k, v in items.items() if not v["available"]]
    if reads["likelihood_minus_blank"] is None:
        missing.append("likelihood minus blank: the report has none")
    paired_ll = (((arm.get("likelihood") or {}).get("forms") or {}).get(FORM) or {}).get(f"minus_paired_{FORM}")
    real = sum(a == b for a, b in zip(arm["forms"]["real"]["hypotheses"], comparator["forms"]["real"]["hypotheses"])) \
        if "real" in arm["forms"] and "real" in comparator["forms"] else None
    return dict(
        kind="round20-phase-b-verdict", set="val1000", clips=arm["clips"], complete=not missing, missing=missing,
        rule_passed=bool(not missing and all(v["passed"] for v in items.values())), items=items, next_to_bleu=reads,
        m1=dict(m1_read(exact_bleu(arm)), note="the condition on this set; M1 also needs >= 2.55 on C20"),
        predicted=NOT_PREDICTED,
        comparator=dict(bleu=comparator["forms"][FORM]["bleu"], length_ratio=length_ratios(comparator),
                        paired_likelihood=(dict(available=True, difference=paired_ll) if paired_ll
                                           else dict(available=False, reason=NO_PAIRED_LIKELIHOOD)),
                        real_hypotheses_identical=real,
                        pairing=("the same clips, references and derangement seed; a fresh arm has the comparator's word "
                                 "seeds, and its codes only if this device reproduces them; a cached arm's seeds are per "
                                 "entry, so only the sentences are shared")),
        end_to_end=dict(checkpoint_key=pilot["checkpoint_key"], bleu=meaning["generated"].get("bleu"),
                        real_identical=e2e.get("real_identical"), blank_identical=e2e.get("blank_identical"),
                        no_decoder_gains=no_gains, **({"identical_to_stored_pilot": same_pilot} if no_gains else {}),
                        **({"note": "the pilot and the comparator are not low-passed; this arm's composed forms and controls "
                                    "are"} if lowpass else {})))


def judge_c20(arm: dict, val_verdict: dict, *, prior_checkpoint: str, no_gains: bool, best_of: int, sentences: dict,
              comparator: dict, lowpass: bool = False) -> dict:
    """The confirmation set: M1's second condition, and the reads beside it. No comparator or pilot read this set."""
    check_arm_form(arm)
    gate_c20(val_verdict, expected_arm=dict(prior_checkpoint_key=prior_checkpoint, no_decoder_gains=no_gains, best_of=best_of,
                                            lowpass=lowpass))
    wrong = settings_differences(arm["settings"], comparator["settings"])
    wrong += arm_errors(arm["settings"], prior_checkpoint=prior_checkpoint, no_gains=no_gains, best_of=best_of,
                        sentences=sentences, lowpass=lowpass)
    if wrong:
        raise SystemExit("not the declared arm with the comparator's settings: " + "; ".join(wrong))
    reads = next_to_bleu(arm)
    items = dict(bleu_vs_other_sentence_words=lower_bound_item(arm["forms"][FORM].get("minus_mismatched_words"), 0.,
                                                               "minus_mismatched_words"),
                 wrist_speed_reported=wrist_item(arm, comparator))
    missing = [f"{k}: {v['reason']}" for k, v in items.items() if not v["available"]]
    if reads["likelihood_minus_blank"] is None:
        missing.append("likelihood minus blank: the report has none")
    m1 = m1_read(exact_bleu(arm))
    return dict(kind="round20-phase-b-verdict", set="c20", clips=arm["clips"], complete=not missing, missing=missing,
                items=items, next_to_bleu=reads, m1=m1, predicted=NOT_PREDICTED,
                m1_claim=bool(not missing and m1["met"] and val_verdict["m1"]["met"]),
                val1000_point_estimate=val_verdict["m1"]["point_estimate"])


def gate_c20(verdict: dict, *, expected_arm: dict) -> None:
    """C20 stays untouched unless the stored val1000 verdict is this arm's, and its rule passed with a point estimate >= 2.55."""
    if verdict.get("kind") != "round20-phase-b-verdict" or verdict.get("set") != "val1000" or not verdict.get("complete"):
        raise SystemExit("C20 refused: this is not a complete stored val1000 verdict")
    arm = verdict.get("arm") or {}
    other = {k: (arm.get(k), v) for k, v in expected_arm.items() if arm.get(k) != v}
    if other:
        raise SystemExit(f"C20 refused: the verdict is another arm's: {other}")
    if verdict.get("rule_passed") is not True:
        raise SystemExit("C20 refused: the val1000 verdict's rule did not pass")
    m1 = verdict.get("m1") or {}
    if m1.get("met") is not True or not float(m1.get("point_estimate", 0.)) >= M1_THRESHOLD:
        raise SystemExit(f"C20 refused: the val1000 point estimate {m1.get('point_estimate')} is below {M1_THRESHOLD}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--arm-key", required=True, help="the arm's merged report; its shards are <key stem>-shard<s>.json")
    v.add_argument("--comparator-key", required=True, help="W14b composition; its shards are <key stem>-shard<s>.json")
    v.add_argument("--comparator-sha256", required=True)
    v.add_argument("--end-to-end-key", help="val1000: the prior's stored 1,000-clip pilot")
    v.add_argument("--end-to-end-sha256")
    v.add_argument("--stored-pilot-key", help="val1000, --no-gains 1: the stored gained pilot that --end-to-end-key reran")
    v.add_argument("--stored-pilot-sha256")
    v.add_argument("--out", required=True, help="the prefix holding the set's arms")
    v.add_argument("--report", required=True, help="this arm's name on the set (the report's file name without .json)")
    v.add_argument("--prior", required=True, choices=("w14b", "r16"))
    v.add_argument("--prior-checkpoint-key", required=True)
    v.add_argument("--no-gains", required=True, type=int, choices=(0, 1))
    v.add_argument("--best-of", required=True, type=int, choices=(0, 4, 8, 16))
    v.add_argument("--lowpass", type=int, choices=(0, 1), default=0, help="the arm's main form is low-passed at 6 Hz")
    v.add_argument("--set", required=True, choices=("val1000", "c20"), dest="sentence_set")
    v.add_argument("--val1000-verdict-key", help="c20: the arm's stored val1000 verdict")
    v.add_argument("--sentences-key", help="c20: the set the report must have read")
    v.add_argument("--sentences-sha256")
    v.add_argument("--bootstrap", type=int, default=1000)
    v.add_argument("--output-key")
    v.add_argument("--output-file")
    g = sub.add_parser("c20-gate")
    g.add_argument("--verdict-key", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--report", required=True, help="this arm's name on the 1,000")
    g.add_argument("--prior", required=True, choices=("w14b", "r16"))
    g.add_argument("--prior-checkpoint-key", required=True)
    g.add_argument("--no-gains", required=True, type=int, choices=(0, 1))
    g.add_argument("--best-of", required=True, type=int, choices=(0, 4, 8, 16))
    g.add_argument("--lowpass", required=True, type=int, choices=(0, 1))
    o = sub.add_parser("one-arm")
    o.add_argument("--out", required=True)
    o.add_argument("--set", required=True, choices=("val1000", "c20"), dest="sentence_set")
    o.add_argument("--report", required=True)
    o.add_argument("--declared-replacement")
    c = sub.add_parser("stack-check")
    c.add_argument("--read-key", required=True)
    c.add_argument("--against-key", action="append", default=[])
    c.add_argument("--prior-checkpoint-key", required=True)
    c.add_argument("--no-gains", required=True, type=int, choices=(0, 1))
    c.add_argument("--best-of", required=True, type=int, choices=(0, 4, 8, 16))
    c.add_argument("--lowpass", required=True, type=int, choices=(0, 1))
    c.add_argument("--sentences-sha256", required=True)
    for q in (v, g, o, c):
        q.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.command == "c20-gate":
        if not store.exists(args.verdict_key):
            raise SystemExit(f"C20 refused: no stored verdict {args.verdict_key}")
        verdict = json.loads(store.get(args.verdict_key))
        gate_c20(verdict, expected_arm=dict(prior=args.prior, prior_checkpoint_key=args.prior_checkpoint_key,
                                            no_decoder_gains=bool(args.no_gains), best_of=args.best_of,
                                            lowpass=bool(args.lowpass)))
        others = other_arms(store.list(f"{args.out}/"), args.out, "val1000", args.report)
        if others and not verdict.get("declared_replacement"):
            raise SystemExit(f"C20 refused: the 1,000 also hold {others}, and no amendment declares this arm their replacement")
        print(f"C20 gate open: {args.verdict_key}", flush=True)
        return 0
    if args.command == "one-arm":
        others = other_arms(store.list(f"{args.out}/"), args.out, args.sentence_set, args.report)
        one_arm(others, args.declared_replacement, args.report)
        print(f"one arm: {args.report}" + (f"; DECLARED REPLACEMENT {args.declared_replacement} of {others}" if others else ""),
              flush=True)
        return 0
    if args.command == "stack-check":
        keys = [args.read_key, *args.against_key]
        missing = [k for k in keys if not store.exists(k)]
        if missing:
            raise SystemExit(f"unread combination: {missing} are not stored")
        print(json.dumps(stack_check(json.loads(store.get(args.read_key)),
                                     {k: json.loads(store.get(k)) for k in args.against_key},
                                     prior_checkpoint=args.prior_checkpoint_key, no_gains=bool(args.no_gains),
                                     best_of=args.best_of, lowpass=bool(args.lowpass),
                                     sentences_sha256=args.sentences_sha256), indent=1), flush=True)
        return 0
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    from round20_compare import report_shards
    arm_payload, comparator_payload = store.get(args.arm_key), store.get(args.comparator_key)
    if sha(comparator_payload) != args.comparator_sha256:
        raise SystemExit(f"{args.comparator_key} does not match --comparator-sha256")
    arm, comparator = json.loads(arm_payload), json.loads(comparator_payload)
    others = other_arms(store.list(f"{args.out}/"), args.out, args.sentence_set, args.report)
    declared = arm.get("declared_replacement")
    one_arm(others, declared, args.report)
    if args.sentence_set == "val1000":
        if not (args.end_to_end_key and args.end_to_end_sha256):
            p.error("val1000: give --end-to-end-key and --end-to-end-sha256")
        pilot_payload = store.get(args.end_to_end_key)
        if sha(pilot_payload) != args.end_to_end_sha256:
            raise SystemExit(f"{args.end_to_end_key} does not match --end-to-end-sha256")
        stored_pilot = None
        if args.stored_pilot_key:
            payload = store.get(args.stored_pilot_key)
            if sha(payload) != args.stored_pilot_sha256:
                raise SystemExit(f"{args.stored_pilot_key} does not match --stored-pilot-sha256")
            stored_pilot = json.loads(payload)
        out = judge_val1000(arm, comparator, json.loads(pilot_payload),
                            shards_arm=report_shards(store, args.arm_key, arm, None),
                            shards_comparator=report_shards(store, args.comparator_key, comparator, None),
                            prior_checkpoint=args.prior_checkpoint_key, no_gains=bool(args.no_gains),
                            best_of=args.best_of, boot=args.bootstrap, lowpass=bool(args.lowpass), stored_pilot=stored_pilot)
        out["end_to_end"].update(key=args.end_to_end_key, sha256=args.end_to_end_sha256)
    else:
        if not (args.val1000_verdict_key and args.sentences_key and args.sentences_sha256):
            p.error("c20: give --val1000-verdict-key, --sentences-key and --sentences-sha256")
        if not store.exists(args.val1000_verdict_key):
            raise SystemExit(f"C20 refused: no stored verdict {args.val1000_verdict_key}")
        out = judge_c20(arm, json.loads(store.get(args.val1000_verdict_key)), prior_checkpoint=args.prior_checkpoint_key,
                        no_gains=bool(args.no_gains), best_of=args.best_of, comparator=comparator,
                        lowpass=bool(args.lowpass),
                        sentences=dict(key=args.sentences_key, sha256=args.sentences_sha256))
        out["val1000_verdict_key"] = args.val1000_verdict_key
    out.update(declared_replacement=declared, other_arms=others)
    out["arm"] = dict(key=args.arm_key, sha256=sha(arm_payload), prior=args.prior,
                      prior_checkpoint_key=args.prior_checkpoint_key, no_decoder_gains=bool(args.no_gains),
                      best_of=args.best_of, lowpass=bool(args.lowpass))
    out.setdefault("comparator", {}).update(key=args.comparator_key, sha256=args.comparator_sha256)
    data = json.dumps(out, indent=1).encode()
    print(data.decode(), flush=True)
    if not out["complete"]:
        print("INCOMPLETE, nothing stored: " + "; ".join(out["missing"]), file=sys.stderr, flush=True)
        return 2
    if args.output_file:
        with open(args.output_file, "xb") as stream:
            stream.write(data)
    if args.output_key:
        if store.exists(args.output_key):
            raise SystemExit(f"{args.output_key} appeared during the run: refusing to overwrite")
        store.put(args.output_key, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
