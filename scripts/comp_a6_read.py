"""Post-S2 A6: a 2x2 of hold-x and the fitted length on A3 generated top10, paired within A5's eight stored word draws, on D20. Report only: nothing adopts from it, and nothing here reads a form until the job runs it. Motion and lengths came
first, from stored caches and metadata (``reports/comp_a6_motion.json``, ``reports/comp_a6_lengths.json``).

**Arms, on the same words:** ``base`` (top10 at today's length), ``hold-x`` (an entry with at least 25% two-handed records keeps its left arm, as does every new entry of the list), ``length`` (the fit signer-balanced on c02 + c04) and ``hold-x-length``
(both). The hold rule reads ``hold-x-length`` against ``length``, with ``hold-x`` against ``base`` beside it deciding nothing; the length rule reads ``length`` against ``base``.

**Word draws.** The rules read all eight of A5's stored draws, paired within each, with a t interval over draws and a two-way bootstrap; fewer is ``incomplete``, never a smaller test. A read on A3's stored words (``SCREEN``) adds ``hold-xr`` (exclusion plus a
rest pose, longest ramps), report only, and has no rule: one draw of words has no draw variance, so it can change no arm, threshold or primary.

**Order of the metrics:** likelihood minus its own-length blank, then v2 spotting, then BLEU over draws. A shorter form gains BLEU without content, so the length rule reads BLEU net of its blank's own change. With A5's summary stored, each draw's ``base`` is
compared with A5's top10 hypotheses: a free instrument check that decides nothing.
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
HOLD = "hold-x"
LENGTH = "length"
BOTH = "hold-x-length"
HOLD_REST = "hold-xr"
ARMS = (BASE, HOLD, LENGTH, BOTH)
SCREEN_ARMS = (*ARMS, HOLD_REST)
COMPARATOR = {HOLD: BASE, LENGTH: BASE, BOTH: LENGTH, HOLD_REST: BASE}
EXCLUDE = .25
HOLD_SHA = "1b7c24a81aa75c58c523096595b40df537fcd4a49094b88b1eb19686c17440d7"
REST_SHA = "f75c2a29870ed3b414a76fdf676713971dad5dba9e8dd9aa4a647f0bfdedd3de"
A3_CACHE_SHA = "507ee9cc8f436107863d203ec844e46114d55d7a053883f030d901b07c7f137b"
A3_EXTRA_SHA = "2d614f416abf2d7558c36b7ba117d6042aac81be31373f31878654dab2dfff5c"
BALANCED_FIT = (-.3577, .3445, 18289)
DECLARED = {HOLD: ("hold_left",), LENGTH: ("duration_balanced",), BOTH: ("hold_left",), HOLD_REST: ("hold_left",)}
LIKELIHOOD_TWO_WAY_FLOOR = -.02
LIKELIHOOD_T_FLOOR = -.01
SPOTTING_FLOOR = -.02
BLEU_FLOOR = -.15
HANDS_SHARE = .9
LENGTH_BAND = (.95, 1.10)
A5_GOES_ON = ("confirmed", "direction_only_not_confirmed")
RULES = dict(
    hold=("Primary arm: hold-x at the fitted length, against top10 at the fitted length, paired within each of A5's eight stored word draws, on D20, report only. hold-x keeps the left arm on entries with at least 25% two-handed records and on the list's new entries. hold-x at "
          "today's length is read beside it and decides nothing. Passed only if all hold: mean paired likelihood minus each form's own-length blank, its 95% lower bound above -0.02 on the two-way bootstrap and above -0.01 on the t interval over the eight draws; v2 spotting's 95% lower "
          "bound above -0.02 on both; BLEU over draws, the t interval's lower bound above -0.15; mean hands at least 0.9x its comparator's. These are small-loss bars: a pass means any loss is small, possibly up to about 60% of A3's stored likelihood and spotting gains and 44% of its BLEU "
          "gain, with less motion, and not that the arm reads better. Known before any read, and not evidence: pooled wrist 2.019x to 1.814x, contact unchanged by construction. The form stays far outside the 0.8-1.25x band, so a pass is no progress against the release bar. Anything else "
          "is not passed, the failed check is reported, and not passed is not evidence of a loss. A6's draws run only if A5's summary reads confirmed or direction only."),
    length=("Primary arm: length, the fit signer-balanced on How2Sign train's c02 + c04 (seconds = -0.3577 + 0.3445 x tokens), chosen by a train-only rule after its D20 ratio of about 1.08 was known, and never fitted on D20's lengths. Against top10 at today's length, paired within each "
            "of A5's eight stored word draws, on D20, report only. Passed only if all hold: pooled predicted over true length within 0.95-1.10x (known: 1.078, with median per clip 1.147, both reported); likelihood and spotting as the hold rule; BLEU net of the blank's own change, the t "
            "interval's lower bound above -0.15; mean hands at least 0.9x top10's. Today's length runs 1.20x true, which lowers wrist by construction; the fitted length removes that, so wrist rises from 1.774x to 2.019x. Wrist is reported, decides nothing, and this rise is neither a loss nor "
            "progress. A pass means a nearer length costs little content, not that it adds any."))
A5_SUMMARY_AT_DECLARATION = "not stored when this was written"     # the slot: if A6 is declared after A5's summary, its outcome goes here, and the digest moves with it
KNOWN_AT_WRITING = ("A1's hold had failed its rule on D20: likelihood -0.0276, spotting -0.0405, replacing the left arm on 46.6% of frames, 226 of its held uses from entries with at least 25% two-handed records. A3 generated top10's stored read was known: BLEU +0.343 [-0.018, +0.718], "
                    "likelihood +0.0128, spotting +0.0149, one word draw, on two GPUs. A4's real-clip draws were known (SD 0.237). V1 was known: the tokenizer's cap kept at 4.124 on D20 without neutral fill. A5's declaration was known, its digest 08b9a150; no draw of it was read here. A5's summary at "
                    "declaration: " + A5_SUMMARY_AT_DECLARATION + ". Known from stored motion before any read, top10's words on A3's caches, rebuilt to 0.0 of the stored wrist speeds on 499 clips, pooled wrist: top10 1.774x, length 2.019x, hold-x 1.595x, hold-x + length 1.814x; left wrist 2.228x, 2.532x, "
                    "1.791x, 2.032x; share of frames replaced 0, 0, 37.6%, 37.7%; contact unchanged by construction. The first draft of this read made exclusion plus rest primary, with shortest ramps, after those motion figures and before any read. A review of commit 37c8d18 replaced it with this 2x2 and "
                    "these rule texts, knowing all of the above: shortest ramps left 90% of stretches above real's p90 speed, and hold-x + length has 1 cap-bound stretch, peak 1.376 against the cap's 1.354. On the longest ramps hold-xr, report only and on A3's words alone, reads pooled 1.648x, left 1.918x, left height -1.099, no cap-bound stretch, peak 1.351, and 48% of its rest stretches above real's p90. The small-loss bars and the 60% and 44% in the text were set by that review with A3's stored gains "
                    "known. Known about lengths: today's fit runs 1.198x true on the clips with words and 1.206x over all of D20's metadata; the balanced line had read about 1.08x earlier, in S2's probe design, before the train-only rule was written, so that rule was not blind to it. By that rule the choice "
                    "went to the line, on its tie to fewer parameters: train error 0.2479 against the word model's 0.2450, within the 0.005 tie. D20's metadata then read pooled 1.078x, median per clip 1.147, median per-clip error 0.25 for the line, and 1.072x, 1.133, 0.235 for the word model, which reads "
                    "slightly better on both sets and was not chosen. The first train read split folds by clip instead of video. That was fixed after D20's ratios had been read; the fits and the choice did not change, only the cross-validated errors. S0 had shown a faster form gains BLEU without content. "
                    "Five-clip smokes of these arms had not been run when this was written.")


def settings_of(report: dict) -> dict:
    return (report.get("settings") or {}).get("round20") or {}


def check_arm_name(name: str, report: dict) -> None:
    """The arm's name must be what its settings say: a name selects the rule."""
    r20 = settings_of(report)
    top = (r20.get("vocabulary_extra") or {}).get("top") == 10 and bool(r20.get("word_cache_extra")) and not r20.get("arm_scale") and not r20.get("oracle_choice")
    hold, balanced = r20.get("hold_left"), r20.get("duration_balanced")
    x = dict(sha256=HOLD_SHA, exclude=EXCLUDE)
    want = {BASE: not hold and not balanced, HOLD: hold == x and not balanced, LENGTH: not hold and balanced is True, BOTH: hold == x and balanced is True,
            HOLD_REST: hold == dict(x, rest_sha256=REST_SHA) and not balanced}
    if name not in want or not (top and want[name]):
        raise SystemExit(f"{name} is not what this report's settings say")
    model = (report.get("round20") or {}).get("duration_model") or {}
    fit = (model.get("intercept"), model.get("slope"), model.get("records"))
    fitted = name in (LENGTH, BOTH)
    if fitted and (fit != BALANCED_FIT or not model.get("signer_balanced")):
        raise SystemExit(f"{name}: its fit is {model}, not the committed balanced one")
    if not fitted and model.get("signer_balanced"):
        raise SystemExit(f"{name}: a balanced fit on an arm at today's length")


def arms_of(k) -> tuple:
    return SCREEN_ARMS if k == SCREEN else ARMS


def check_draw(k, arms: dict) -> None:
    """One draw's reports: D20, report-only, all of it, what their names say, paired, the same words, and one pod's code and environment."""
    from round20_compare import check_pairable
    names = arms_of(k)
    if set(arms) != set(names):
        raise SystemExit(f"{k}: its arms are {sorted(arms)}, not {list(names)}")
    for name in names:
        rd.check_arm(arms[name], smoke=False)
        check_arm_name(name, arms[name])
        if name != BASE:
            check_pairable(arms[name], arms[COMPARATOR[name]], DECLARED[name])
        r20 = settings_of(arms[name])
        draw = (r20.get("word_cache") or {}).get("draw")
        if k == SCREEN:
            if draw is not None or (r20["word_cache"].get("sha256"), r20["word_cache_extra"].get("sha256")) != (A3_CACHE_SHA, A3_EXTRA_SHA):
                raise SystemExit(f"{name}: the screen reads A3's stored words")
        elif draw != k or (r20["word_cache_extra"] or {}).get("sha256") is None:
            raise SystemExit(f"draw {k} {name}: its word cache is draw {draw}")
    first = settings_of(arms[BASE]).get("provenance")
    if not first or any(settings_of(arms[a]).get("provenance") != first for a in names):
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
    """Every number of one arm against its comparator across the draws. One draw has no draw variance: its t interval and SD are None."""
    other = COMPARATOR[name]
    pairs = {k: (arms[other], arms[name]) for k, arms in draws.items()}
    pair_shards = {k: (shards[k][other], shards[k][name]) for k in draws}
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
    sides = lambda arm, key: mean_of([side_motion(draws[k][arm]) for k in order], key)         # noqa: E731
    row["motion"] = dict(arm_wrist=motion["top10_wrist"], comparator_wrist=motion["margin0_wrist"], arm_hands=motion["top10_hands"], comparator_hands=motion["margin0_hands"],
                         arm_left=sides(name, "left_speed_over_real"), comparator_left=sides(other, "left_speed_over_real"), arm_right=sides(name, "right_speed_over_real"), comparator_right=sides(other, "right_speed_over_real"),
                         arm_contact=sides(name, "contact_share"), comparator_contact=sides(other, "contact_share"))
    row["comparator"], row["comparator_bleu"], row["arm_bleu"] = other, row.pop("margin0_bleu"), row.pop("top10_bleu")
    row["gap_share"] = dict(comparator=row["gap_share"]["margin0"], arm=row["gap_share"]["top10"])
    row["bleu_net_of_blank"] = net_of_blank(pairs)
    row["length_over_true"] = [((draws[k][name].get("predicted_timing") or {}).get("forms") or {}).get(FORM, {}).get("seconds_over_reference") for k in order]
    hold = [(draws[k][name].get("hold_left") or {}) for k in order]
    row["cap_bound"], row["hold_peak"] = sum(h.get("cap_bound", 0) for h in hold), max((h.get("peak", 0.) for h in hold), default=0.)
    return row


def _lower(row, interval) -> float | None:
    v = row.get(interval)
    return None if v is None else v[0]


def _above(value, floor) -> bool:
    return value is not None and value > floor


def content_checks(row: dict) -> dict:
    ll, spot = row["likelihood_minus_blank"], row["spotting"]
    return dict(likelihood_two_way_lower_bound_above_minus_0_02=_above(_lower(ll, "two_way_ci95"), LIKELIHOOD_TWO_WAY_FLOOR), likelihood_t_lower_bound_above_minus_0_01=_above(_lower(ll, "t_ci95"), LIKELIHOOD_T_FLOOR),
                spotting_two_way_lower_bound_above_minus_0_02=_above(_lower(spot, "two_way_ci95"), SPOTTING_FLOOR), spotting_t_lower_bound_above_minus_0_02=_above(_lower(spot, "t_ci95"), SPOTTING_FLOOR))


def at_least(value, base, share) -> bool:
    return value is not None and base is not None and value >= share * base


def hands_check(row: dict) -> bool:
    return at_least(row["motion"]["arm_hands"], row["motion"]["comparator_hands"], HANDS_SHARE)


def decide_hold(row: dict) -> dict:
    checks = dict(content_checks(row), bleu_t_lower_bound_above_minus_0_15=_above(_lower(row["bleu"], "t_ci95"), BLEU_FLOOR), hands_at_least_0_9_of_comparator=hands_check(row))
    return dict(checks=checks, outcome="passed" if all(checks.values()) else "not_passed")


def pooled_length(row: dict) -> float | None:
    values = [(x or {}).get("pooled") for x in row["length_over_true"]]
    return None if not values or any(v is None for v in values) else round(float(np.mean(values)), 4)


def decide_length(row: dict) -> dict:
    ratio = pooled_length(row)
    checks = dict(pooled_length_within_0_95_to_1_10=ratio is not None and LENGTH_BAND[0] <= ratio <= LENGTH_BAND[1], **content_checks(row),
                  bleu_net_of_blank_t_lower_bound_above_minus_0_15=_above((row["bleu_net_of_blank"]["t_ci95"] or [None])[0], BLEU_FLOOR), hands_at_least_0_9_of_comparator=hands_check(row))
    return dict(checks=checks, outcome="passed" if all(checks.values()) else "not_passed")


def a5_instrument_check(draws: dict, a5_top10: dict) -> dict | None:
    """Each draw's base against A5's top10 on the same words: how many hypotheses agree. Free, and it decides nothing. Disagreement can come from another GPU, and is reported."""
    if not a5_top10:
        return None
    out = {}
    for k, other in a5_top10.items():
        base = draws[k][BASE]
        if base["references"] != other["references"]:
            raise SystemExit(f"draw {k}: A5's top10 read other sentences")
        same = lambda form: int(sum(p == q for p, q in zip(base["forms"][form]["hypotheses"], other["forms"][form]["hypotheses"])))   # noqa: E731
        out[k] = dict(form_identical=same(FORM), real_identical=same("real"), blank_identical=same(BLANK), clips=len(base["references"]),
                      same_gpu=settings_of(base)["provenance"]["environment"].get("gpu") == settings_of(other)["provenance"]["environment"].get("gpu"))
    return dict(note="reported; decides nothing", draws=out)


def read(draws: dict, shards: dict, *, boot: int = 1000, declared_commit: str | None = None, job_file: str = JOB_FILE, attempts: dict | None = None, a5_top10: dict | None = None) -> dict:
    check_set(draws)
    screen = SCREEN in draws
    arms = {name: arm_summary(draws, shards, name, boot=boot) for name in (SCREEN_ARMS if screen else ARMS) if name != BASE}
    note = "evaluated for reading only; this field adopts nothing" if declared_commit else "DRAFT, NOT DECLARED: adopts nothing"
    if screen:
        rule = dict(hold=dict(arm=BOTH, comparator=LENGTH, outcome="screen_decides_nothing", checks=None, text=RULES["hold"]), length=dict(arm=LENGTH, comparator=BASE, outcome="screen_decides_nothing", checks=None, text=RULES["length"]))
    else:
        rule = dict(hold=dict(arm=BOTH, comparator=LENGTH, text=RULES["hold"], **decide_hold(arms[BOTH])), length=dict(arm=LENGTH, comparator=BASE, text=RULES["length"], **decide_length(arms[LENGTH])))
    return dict(kind="comp-a6-read", report_only=True, mode="screen" if screen else "draws", declared=bool(declared_commit), declared_commit=declared_commit, draws=sorted(draws, key=str), set=dict(key=lib.D20_KEY, sha256=lib.D20_SHA),
                arms=arms, rule=rule, known_when_written=KNOWN_AT_WRITING, note=note, reported_only=[HOLD, HOLD_REST] if screen else [HOLD], attempts=attempts or {}, a5_instrument_check=a5_instrument_check(draws, a5_top10 or {}), provenance=lib.provenance_row(job_file))


def main(argv=None) -> int:
    from round20_compare import report_shards
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--draw", action="append", default=[], help=f"K=KEY,... in the order {','.join(ARMS)}; K is 1 to 8, or {SCREEN} alone, with {HOLD_REST} last")
    p.add_argument("--declared-commit")
    p.add_argument("--attempts", action="append", default=[], help="K=A:arms;A:arms: every attempt of draw K and the arms it stored")
    p.add_argument("--a5-top10", action="append", default=[], help="K=KEY: A5's top10 of draw K, compared with base: the free instrument check")
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
        names = arms_of(name) if name is not None else ()
        if name is None or len(parts) != len(names) or not all(parts) or name in draws:
            p.error(f"--draw {spec!r}: K={','.join(a.upper() for a in (names or ARMS))}, once each")
        draws[name] = {a: json.loads(store.get(key)) for a, key in zip(names, parts)}
        shards[name] = {a: report_shards(store, key, draws[name][a], None) for a, key in zip(names, parts)}
    attempts = {spec.partition("=")[0]: spec.partition("=")[2] for spec in args.attempts}
    a5_top10 = {}
    for spec in args.a5_top10:
        k, _, key = spec.partition("=")
        if not k.isdigit() or int(k) not in draws or not key:
            p.error(f"--a5-top10 {spec!r}: K=KEY, a draw of this read")
        a5_top10[int(k)] = json.loads(store.get(key))
    out = read(draws, shards, boot=args.bootstrap, declared_commit=args.declared_commit, attempts=attempts, a5_top10=a5_top10)
    store.put(args.output_key, json.dumps(out, indent=1, default=float).encode())
    print(f"A6 {out['mode']}: hold {out['rule']['hold']['outcome']}, length {out['rule']['length']['outcome']}", flush=True)
    return 0


__all__ = ["read", "decide_hold", "decide_length", "check_set", "RULES", "KNOWN_AT_WRITING"]

if __name__ == "__main__":
    raise SystemExit(main())
