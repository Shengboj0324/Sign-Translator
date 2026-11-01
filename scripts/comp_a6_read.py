"""Post-S2 A6: a narrower left-arm hold and a better text-only length, each against A3 generated top10, paired within word draws, on D20 at predicted timing. Report only: nothing adopts from it,
and nothing here reads a form until the job runs it. Motion and lengths came first, from stored caches and metadata (``reports/comp_a6_motion.json``, ``reports/comp_a6_lengths.json``).

**Arms, on the same words:** ``base`` (top10 at today's length, the comparator, from this same job), ``hold-x`` (exclusion: an entry with at least 25% two-handed records keeps its left arm, as does every new entry
of the list), ``hold-xr`` (exclusion plus rest: the held arm goes down to a pose from ASLLRP train one-handed signs and back, minimum jerk under the speed cap), and ``length`` (the fit signer-balanced on c02 + c04).
One hold arm is the pre-registered primary (``PRIMARY``); the other is read beside it and decides nothing.

**Word draws.** The rule reads all eight of A5's stored draws (``DRAWS``), paired within each, with a t interval over draws and a two-way bootstrap; fewer is ``incomplete``, never a smaller test. A single read on A3's stored words
(``SCREEN``) is possible before those exist. It reads one draw of words with no draw variance, so it decides nothing and can change no arm, threshold or primary.

**Order of the metrics:** likelihood minus its own-length blank, then v2 spotting, then BLEU over draws. A shorter form gains BLEU without content, so the length arm's BLEU is read net of its blank's own change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import comp_a5_read as a5  # noqa: E402
import comp_a_lib as lib  # noqa: E402
import comp_a_read as rd  # noqa: E402

DRAWS = a5.DRAWS
SCREEN = "a3"
JOB_FILE = "runpod/fleet/comp_a6.sh,runpod/fleet/comp_a6_chain.sh"     # the digest covers the job and its chain
FORM = a5.FORMS["predicted"]
BLANK = "blank@predicted"
BASE = "base"
HOLD_ARMS = ("hold-x", "hold-xr")
LENGTH = "length"
ARMS = (BASE, *HOLD_ARMS, LENGTH)
PRIMARY_HOLD = "hold-xr"
EXCLUDE = .25
HOLD_SHA = "1b7c24a81aa75c58c523096595b40df537fcd4a49094b88b1eb19686c17440d7"
REST_SHA = "f75c2a29870ed3b414a76fdf676713971dad5dba9e8dd9aa4a647f0bfdedd3de"
A3_CACHE_SHA = "507ee9cc8f436107863d203ec844e46114d55d7a053883f030d901b07c7f137b"
A3_EXTRA_SHA = "2d614f416abf2d7558c36b7ba117d6042aac81be31373f31878654dab2dfff5c"
BALANCED_FIT = (-.3577, .3445, 18289)
DECLARED = {"hold-x": ("hold_left",), "hold-xr": ("hold_left",), LENGTH: ("duration_balanced",)}
LIKELIHOOD_FLOOR = -.01
SPOTTING_FLOOR = -.02
BLEU_FLOOR = -.15
HANDS_SHARE = .9
CONTACT_SHARE = .9
LENGTH_BAND = (.95, 1.10)
RULES = dict(
    hold=("Primary arm: PRIMARY. Against A3 generated top10 at today's length, paired within each of A5's eight stored word draws, on D20 at predicted timing, report only. Passed only if all hold: mean paired likelihood minus blank@predicted, its 95% lower bound "
          "above -0.01 on both the two-way bootstrap over sentences and draws and the t interval over the eight draws; v2 spotting's 95% lower bound above -0.02 on both; BLEU over draws, the t interval's lower bound above -0.15 (about 5 points of D20's gap); pooled wrist and left "
          "wrist each closer to 1.0x real than top10's; mean hands at least 0.9x top10's; two-handed contact at least 0.9x top10's. These are no-loss bars: a pass means the arm reads no worse than top10 with less motion, and not that it reads better. Anything else is not passed, and "
          "the failed check is reported. A true effect can still fail this rule. The other hold arm is reported and decides nothing, whatever it reads. A single read on A3's stored words decides nothing and can change no arm, threshold or primary. Fewer than eight finished draws is "
          "incomplete, not a smaller test; no draw is replaced, added or dropped after any result. A job failure before all of a draw's arms are stored reruns them all on the same words and nothing else; a draw with all arms stored is final. Each arm's hash goes in the ledger when "
          "its draw ends, and no per-draw number is read until the summary. No draw of this read starts before A5's summary is stored. D20 chose top10 and A5's draws are its own, so D20's sentences are not fresh. A pass is no M1 claim, no adoption and no release bar."),
    length=("Primary arm: length, the fit signer-balanced on How2Sign train's c02 + c04 (seconds = -0.3577 + 0.3445 x tokens), chosen on train alone and never fitted on D20's lengths. Against A3 generated top10 at today's length, paired within each of A5's eight stored word draws, on D20, "
            "report only. Passed only if all hold: pooled predicted over true length within 0.95-1.10x; mean paired likelihood minus each form's own-length blank, its 95% lower bound above -0.01 on both the two-way bootstrap over sentences and draws and the t interval over the eight "
            "draws; v2 spotting's 95% lower bound above -0.02 on both; BLEU over draws net of the blank's own change between the two lengths, the t interval's lower bound above -0.15 (about 5 points of D20's gap); mean hands at least 0.9x top10's. These are no-loss bars: a pass means a "
            "nearer length costs no content, and not that it adds any. The shorter length raises wrist speed by construction: wrist is reported beside it, decides nothing, and moves away from the release bar. Stretching lowers speed by construction too, so today's lower speed was no "
            "progress either. Anything else is not passed, and the failed check is reported. A true effect can still fail this rule. A single read on A3's stored words decides nothing and can change no arm, threshold or primary. Fewer than eight finished draws is incomplete, not a smaller "
            "test; no draw is replaced, added or dropped after any result. A job failure before all of a draw's arms are stored reruns them all on the same words and nothing else; a draw with all arms stored is final. Each arm's hash goes in the ledger when its draw ends, and no per-draw "
            "number is read until the summary. No draw of this read starts before A5's summary is stored. D20 chose top10 and A5's draws are its own, so D20's sentences are not fresh. A pass is no M1 claim, no adoption and no release bar."))
RULES["hold"] = RULES["hold"].replace("PRIMARY", PRIMARY_HOLD)
KNOWN_AT_WRITING = ("A1's hold had failed its rule on D20: likelihood -0.0276, spotting -0.0405, replacing the left arm on 46.6% of frames, 226 of its held uses from entries with at least 25% two-handed records. A3 generated top10's stored read was known: BLEU +0.343 "
                    "[-0.018, +0.718], likelihood +0.0128, spotting +0.0149, one word draw, on two GPUs. A4's real-clip draws were known (SD 0.237). V1 was known: the tokenizer's cap kept at 4.124 on D20 without neutral fill. A5's declaration was known; its summary was not stored and no draw of "
                    "it was read here. Known from stored motion before any read of these arms, top10's words on A3's caches, rebuilt to 0.0 of the stored wrist speeds on 499 clips: pooled wrist top10 1.774x, plain hold 1.544x, exclusion 1.595x, exclusion plus rest 1.666x, length 2.019x; left wrist "
                    "2.228x, 1.665x, 1.791x, 1.964x, 2.532x; left height on annotated one-handed words -1.459, -0.907, -0.979, -1.107, -1.458 against the review's ASLLRP -1.08; share of frames replaced 0, 42.7%, 37.6%, 37.6%, 0. So both hold arms' motion checks pass and are not evidence, and neither "
                    "meets the 1.25x bar. Exclusion plus rest was made primary after those figures and before any read: it answers both causes A1 left open, erased content and a static pose, at the cost of more travel than exclusion alone. Known about lengths: today's fit runs 1.198x true on the "
                    "clips with words and 1.206x over all of D20's metadata; the balanced line had read 1.08x earlier, in S2's probe design. The candidate was chosen on train alone by a rule written before the word model's D20 ratio was computed, and that choice went to the line, on the rule's tie to fewer parameters: train error 0.2479 against the word model's 0.2450, within the 0.005 tie. D20's metadata "
                    "was read after a first choice: pooled 1.078x and median per-clip error 0.25 for the line, 1.072x and 0.235 for the word model, which reads slightly better on both sets and was not chosen. The first train read split folds by clip instead of video. That was fixed after D20's ratios "
                    "had been read; the fits and the choice did not change, only the cross-validated errors. The bounds -0.01, -0.02 and -0.15 were set with all of this known. S0 had shown a faster form gains BLEU without content, so length's BLEU is read net of its blank. Five-clip smokes of these arms had not been run when this was written.")


def settings_of(report: dict) -> dict:
    return (report.get("settings") or {}).get("round20") or {}


def check_arm_name(name: str, report: dict) -> None:
    """The arm's name must be what its settings say: a name selects the rule."""
    r20 = settings_of(report)
    top = (r20.get("vocabulary_extra") or {}).get("top") == 10 and bool(r20.get("word_cache_extra")) and not r20.get("arm_scale") and not r20.get("oracle_choice")
    hold, balanced = r20.get("hold_left"), r20.get("duration_balanced")
    want = {BASE: not hold and not balanced,
            "hold-x": hold == dict(sha256=HOLD_SHA, exclude=EXCLUDE) and not balanced,
            "hold-xr": hold == dict(sha256=HOLD_SHA, exclude=EXCLUDE, rest_sha256=REST_SHA) and not balanced,
            LENGTH: not hold and balanced is True}
    if name not in want or not (top and want[name]):
        raise SystemExit(f"{name} is not what this report's settings say")
    model = (report.get("round20") or {}).get("duration_model") or {}
    fit = (model.get("intercept"), model.get("slope"), model.get("records"))
    if name == LENGTH and (fit != BALANCED_FIT or not model.get("signer_balanced")):
        raise SystemExit(f"length: its fit is {model}, not the committed balanced one")
    if name != LENGTH and model.get("signer_balanced"):
        raise SystemExit(f"{name}: a balanced fit on an arm at today's length")


def check_draw(k, arms: dict) -> None:
    """One draw's four reports: D20, report-only, all of it, what their names say, paired, the same words, and one pod's code and environment."""
    from round20_compare import check_pairable
    missing = [a for a in ARMS if a not in arms]
    if missing:
        raise SystemExit(f"{k}: no {missing}")
    for name in ARMS:
        rd.check_arm(arms[name], smoke=False)
        check_arm_name(name, arms[name])
        if name != BASE:
            check_pairable(arms[name], arms[BASE], DECLARED[name])
        r20 = settings_of(arms[name])
        draw = (r20.get("word_cache") or {}).get("draw")
        if k == SCREEN:
            if draw is not None or (r20["word_cache"].get("sha256"), r20["word_cache_extra"].get("sha256")) != (A3_CACHE_SHA, A3_EXTRA_SHA):
                raise SystemExit(f"{name}: the screen reads A3's stored words")
        elif draw != k or (r20["word_cache_extra"] or {}).get("sha256") is None:
            raise SystemExit(f"draw {k} {name}: its word cache is draw {draw}")
    first = settings_of(arms[BASE]).get("provenance")
    if not first or any(settings_of(arms[a]).get("provenance") != first for a in ARMS):
        raise SystemExit(f"{k}: its arms came from other code or another environment")


def check_set(draws: dict) -> None:
    """All eight of A5's draws, or the screen alone; one digest and packages; no two draws sharing words; the same sentences. The GPU may differ between draws, never within one."""
    keys = tuple(sorted(draws, key=str))
    if keys != (SCREEN,) and tuple(sorted(k for k in draws if k != SCREEN)) != DRAWS or (SCREEN in draws and len(draws) > 1):
        raise SystemExit(f"incomplete: draws {list(keys)}, the rule names {list(DRAWS)}; fewer than eight finished draws is no smaller test")
    for k, arms in draws.items():
        check_draw(k, arms)
    bases = [arms[BASE] for arms in draws.values()]
    prov = settings_of(bases[0])["provenance"]
    packages = {x: prov["environment"].get(x) for x in a5.PINNED}
    for report in bases:
        other = settings_of(report)["provenance"]
        if other["code_digest"] != prov["code_digest"] or {x: other["environment"].get(x) for x in packages} != packages:
            raise SystemExit("two draws came from other code or packages")
        if report["references"] != bases[0]["references"] or report["signs"] != bases[0]["signs"]:
            raise SystemExit("the draws read other sentences")
    if len({settings_of(b)["word_cache"]["sha256"] for b in bases}) != len(bases) or len({settings_of(b)["word_cache_extra"]["sha256"] for b in bases}) != len(bases):
        raise SystemExit("two draws share a word cache: they are one draw")


def net_of_blank(pairs: dict) -> dict:
    """Paired BLEU minus the same difference on the two reports' own blanks, per draw. The hold arms share top10's blank; the length arm's blank has its length."""
    order = sorted(pairs, key=str)
    values = [(pairs[k][1]["forms"][FORM]["bleu"] - pairs[k][0]["forms"][FORM]["bleu"]) - (pairs[k][1]["forms"][BLANK]["bleu"] - pairs[k][0]["forms"][BLANK]["bleu"]) for k in order]
    row = dict(per_draw=[round(float(x), 4) for x in values], mean=round(float(np.mean(values)), 4), t_ci95=None)
    if len(values) > 1:
        row["t_ci95"] = a5.t_interval(values)
    return row


def side_motion(report: dict) -> dict:
    return (report.get("comp_a_motion") or {}).get(FORM) or {}


def mean_of(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(values)), 4) if values else None


def arm_summary(draws: dict, shards: dict, name: str, *, boot: int) -> dict:
    """Every number of one arm against base across the draws. One draw has no draw variance: its t interval and SD are None."""
    pairs = {k: (arms[BASE], arms[name]) for k, arms in draws.items()}
    pair_shards = {k: (shards[k][BASE], shards[k][name]) for k in draws}
    order = sorted(pairs, key=str)
    if len(order) > 1:
        row = a5.form_summary(pairs, pair_shards, FORM, boot=boot)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)         # the SD and t interval of one draw, set to None below
            row = a5.form_summary(pairs, pair_shards, FORM, boot=boot)
        for metric in ("bleu", "likelihood_minus_blank", "spotting"):
            row[metric].update(t_ci95=None, between_draw_sd=None)
    motion = row.pop("motion")
    row["motion"] = dict(arm_wrist=motion["top10_wrist"], base_wrist=motion["margin0_wrist"], arm_hands=motion["top10_hands"], base_hands=motion["margin0_hands"],
                         arm_left=mean_of([side_motion(draws[k][name]) for k in order], "left_speed_over_real"), base_left=mean_of([side_motion(draws[k][BASE]) for k in order], "left_speed_over_real"),
                         arm_right=mean_of([side_motion(draws[k][name]) for k in order], "right_speed_over_real"), base_right=mean_of([side_motion(draws[k][BASE]) for k in order], "right_speed_over_real"),
                         arm_contact=mean_of([side_motion(draws[k][name]) for k in order], "contact_share"), base_contact=mean_of([side_motion(draws[k][BASE]) for k in order], "contact_share"))
    row["base_bleu"], row["arm_bleu"] = row.pop("margin0_bleu"), row.pop("top10_bleu")
    row["gap_share"] = dict(base=row["gap_share"]["margin0"], arm=row["gap_share"]["top10"])
    row["bleu_net_of_blank"] = net_of_blank(pairs)
    row["length_over_true"] = [((draws[k][name].get("predicted_timing") or {}).get("forms") or {}).get(FORM, {}).get("seconds_over_reference") for k in order]
    return row


def _lower(row, interval) -> float | None:
    v = row.get(interval)
    return None if v is None else v[0]


def _above(value, floor) -> bool:
    return value is not None and value > floor


def closer_to_real(value, base) -> bool:
    """Strictly nearer 1.0x real than the base, an unknown number failing."""
    return value is not None and base is not None and abs(value - 1.) < abs(base - 1.)


def content_checks(row: dict) -> dict:
    ll, spot = row["likelihood_minus_blank"], row["spotting"]
    return dict(likelihood_two_way_lower_bound_above_minus_0_01=_above(_lower(ll, "two_way_ci95"), LIKELIHOOD_FLOOR), likelihood_t_lower_bound_above_minus_0_01=_above(_lower(ll, "t_ci95"), LIKELIHOOD_FLOOR),
                spotting_two_way_lower_bound_above_minus_0_02=_above(_lower(spot, "two_way_ci95"), SPOTTING_FLOOR), spotting_t_lower_bound_above_minus_0_02=_above(_lower(spot, "t_ci95"), SPOTTING_FLOOR))


def at_least(value, base, share) -> bool:
    return value is not None and base is not None and value >= share * base


def decide_hold(row: dict) -> dict:
    m = row["motion"]
    checks = dict(content_checks(row), bleu_t_lower_bound_above_minus_0_15=_above(_lower(row["bleu"], "t_ci95"), BLEU_FLOOR), pooled_wrist_closer_to_real_than_top10=closer_to_real(m["arm_wrist"], m["base_wrist"]),
                  left_wrist_closer_to_real_than_top10=closer_to_real(m["arm_left"], m["base_left"]), hands_at_least_0_9_of_top10=at_least(m["arm_hands"], m["base_hands"], HANDS_SHARE),
                  contact_at_least_0_9_of_top10=at_least(m["arm_contact"], m["base_contact"], CONTACT_SHARE))
    return dict(checks=checks, outcome="passed" if all(checks.values()) else "not_passed")


def pooled_length(row: dict) -> float | None:
    values = [(x or {}).get("pooled") for x in row["length_over_true"]]
    return None if not values or any(v is None for v in values) else round(float(np.mean(values)), 4)


def decide_length(row: dict) -> dict:
    m, ratio = row["motion"], pooled_length(row)
    checks = dict(pooled_length_within_0_95_to_1_10=ratio is not None and LENGTH_BAND[0] <= ratio <= LENGTH_BAND[1], **content_checks(row),
                  bleu_net_of_blank_t_lower_bound_above_minus_0_15=_above((row["bleu_net_of_blank"]["t_ci95"] or [None])[0], BLEU_FLOOR), hands_at_least_0_9_of_top10=at_least(m["arm_hands"], m["base_hands"], HANDS_SHARE))
    return dict(checks=checks, outcome="passed" if all(checks.values()) else "not_passed")


def read(draws: dict, shards: dict, *, boot: int = 1000, declared_commit: str | None = None, job_file: str = JOB_FILE, attempts: dict | None = None) -> dict:
    check_set(draws)
    screen = SCREEN in draws
    arms = {name: arm_summary(draws, shards, name, boot=boot) for name in (*HOLD_ARMS, LENGTH)}
    note = "evaluated for reading only; this field adopts nothing" if declared_commit else "DRAFT, NOT DECLARED: adopts nothing"
    if screen:
        rule = dict(hold=dict(arm=PRIMARY_HOLD, outcome="screen_decides_nothing", checks=None, text=RULES["hold"]), length=dict(arm=LENGTH, outcome="screen_decides_nothing", checks=None, text=RULES["length"]))
    else:
        rule = dict(hold=dict(arm=PRIMARY_HOLD, text=RULES["hold"], **decide_hold(arms[PRIMARY_HOLD])), length=dict(arm=LENGTH, text=RULES["length"], **decide_length(arms[LENGTH])))
    return dict(kind="comp-a6-read", report_only=True, mode="screen" if screen else "draws", declared=bool(declared_commit), declared_commit=declared_commit, draws=sorted(draws, key=str), set=dict(key=lib.D20_KEY, sha256=lib.D20_SHA),
                arms=arms, rule=rule, known_when_written=KNOWN_AT_WRITING, note=note, reported_only=[a for a in HOLD_ARMS if a != PRIMARY_HOLD], attempts=attempts or {}, provenance=lib.provenance_row(job_file))


def main(argv=None) -> int:
    from round20_compare import report_shards
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--draw", action="append", default=[], help=f"K=KEY,KEY,KEY,KEY in the order {','.join(ARMS)}; K is 1 to 8, or {SCREEN} alone")
    p.add_argument("--declared-commit")
    p.add_argument("--attempts", action="append", default=[], help="K=A:arms;A:arms: every attempt of draw K and the arms it stored")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--store", default="r2")
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    draws, shards = {}, {}
    for spec in args.draw:
        k, _, keys = spec.partition("=")
        parts = keys.split(",")
        name = SCREEN if k == SCREEN else int(k) if k.isdigit() else None
        if name is None or len(parts) != len(ARMS) or not all(parts) or name in draws:
            p.error(f"--draw {spec!r}: K={','.join(a.upper() for a in ARMS)}, once each")
        draws[name] = {a: json.loads(store.get(key)) for a, key in zip(ARMS, parts)}
        shards[name] = {a: report_shards(store, key, draws[name][a], None) for a, key in zip(ARMS, parts)}
    attempts = {spec.partition("=")[0]: spec.partition("=")[2] for spec in args.attempts}
    out = read(draws, shards, boot=args.bootstrap, declared_commit=args.declared_commit, attempts=attempts)
    store.put(args.output_key, json.dumps(out, indent=1, default=float).encode())
    print(f"A6 {out['mode']}: hold {out['rule']['hold']['outcome']}, length {out['rule']['length']['outcome']}", flush=True)
    return 0


__all__ = ["read", "decide_hold", "decide_length", "check_set", "RULES", "KNOWN_AT_WRITING"]

if __name__ == "__main__":
    raise SystemExit(main())
