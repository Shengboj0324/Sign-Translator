"""Post-S2 A6: a 2x2 of hold-x and the fitted length on A3 generated top10, paired within A5's eight stored word draws, on D20. Report only: nothing adopts from it, and nothing here reads a form until the job runs it. Motion and lengths came
first, from stored caches and metadata (``reports/comp_a6_motion.json``, ``reports/comp_a6_lengths.json``).

**Arms, on the same words:** ``base`` (top10 at today's length), ``hold-x`` (an entry with at least 25% two-handed records keeps its left arm, as does every new entry of the list), ``length`` (the fit signer-balanced on c02 + c04) and ``hold-x-length``
(both). The hold rule reads ``hold-x-length`` against ``length``, with ``hold-x`` against ``base`` beside it deciding nothing; the length rule reads ``length`` against ``base``.

**Word draws.** The rules read all eight of A5's stored draws, paired within each, with a t interval over draws and a two-way bootstrap; fewer is ``incomplete``, never a smaller test. A read on A3's stored words (``SCREEN``) adds ``hold-xr`` (exclusion plus a
rest pose, longest ramps), report only, and has no rule: one draw of words has no draw variance, so it can change no threshold or primary. It does gate the hold arms (``hold_gate``): hold-x and hold-x-length run on the draws only if the screen's point estimates of hold-x minus
base sit above the hold floors on likelihood and spotting. Otherwise the draws run base and length, the length rule still reads, and the hold rule reads ``HOLD_NOT_RUN``.

**Order of the metrics:** likelihood minus its own-length blank, then v2 spotting, then BLEU over draws. A shorter form reads BLEU on the lower of raw and net of its blank's own change. Each draw's ``base`` is compared with A5's top10 hypotheses, and the screen's with A3's stored top10: on the
same GPU and pinned thread count at least 98% must match, or the draw stops.
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
GATE_CLOSED_ARMS = (BASE, LENGTH)
HOLD_NOT_RUN = "not run, screen failed"
COMPARATOR = {HOLD: BASE, LENGTH: BASE, BOTH: LENGTH, HOLD_REST: BASE}
EXCLUDE = .25
HOLD_SHA = "1b7c24a81aa75c58c523096595b40df537fcd4a49094b88b1eb19686c17440d7"
REST_SHA = "f75c2a29870ed3b414a76fdf676713971dad5dba9e8dd9aa4a647f0bfdedd3de"
A3_CACHE_SHA = "507ee9cc8f436107863d203ec844e46114d55d7a053883f030d901b07c7f137b"
A3_EXTRA_SHA = "2d614f416abf2d7558c36b7ba117d6042aac81be31373f31878654dab2dfff5c"
BALANCED_FIT = (-.3577, .3445, 18289)
DECLARED = {HOLD: ("hold_left",), LENGTH: ("duration_balanced",), BOTH: ("hold_left",), HOLD_REST: ("hold_left",)}
A3_GAINS = dict(bleu=.343, likelihood=.0128, spotting=.0149)
A1_LOSS = dict(likelihood=-.0276, spotting=-.0405)
SHARES = dict(bleu=.44, likelihood=.6, spotting=.6)
A5_SUMMARY_AT_DECLARATION = None   # the slot: slot_from_summary's dict once A5's summary exists (comp_a6_declared.py slot FILE). A record may not be declared while this is None
PATH_READ_AT_DECLARATION = None    # the slot: branch path-read's result, in words, if it lands before declaration
PRIOR_REVIEW_POWER = None          # the slot: the prior review's power figures, in words, as it gave them
HANDS_SHARE = .9
TODAY_FIT = (-.083, .3691, 18289)
A5_GOES_ON = ("confirmed", "direction_only_not_confirmed")
MATCH_RATE = .98
PINNED_THREADS = 8
METRICS = dict(bleu="bleu", likelihood="likelihood_minus_blank", spotting="spotting")
WORD_PATH = "one-word weight 0.27% two-view, sampling share 8.8%, sentence-token weight 27.2%"


def slot_from_summary(summary: dict, sha256: str) -> dict:
    """What a declaration fixes from A5's summary: its outcome, and per metric at predicted timing the mean paired gain, the t interval's lower bound and the paired SD."""
    row = summary["forms"]["predicted"]
    return dict(sha256=sha256, outcome=summary["rule"]["outcome"], **{k: row[v]["mean"] for k, v in METRICS.items()}, t_lower={k: row[v]["t_ci95"][0] for k, v in METRICS.items()}, paired_sd={k: row[v]["between_draw_sd"] for k, v in METRICS.items()})


def no_room(slot: dict | None) -> dict:
    """{metric: why}: A5's mean gain at or below 0, or its lower 95% t bound over the eight draws at or below 0: no gain was established to protect. Nothing is judged this way until the slot is filled."""
    out = {}
    for k in SHARES:
        if slot is None:
            continue
        if slot[k] <= 0:
            out[k] = f"A5's mean gain is {slot[k]}: no gain was established to protect"
        elif slot["t_lower"][k] <= 0:
            out[k] = f"A5's lower 95% t bound is {slot['t_lower'][k]}: no gain was established to protect"
    return out


def floors(slot: dict | None = None) -> dict:
    """The bars: a declared share of the mean paired gains, of A5's once its summary fills the slot and of A3's stored read until then. A metric with no room has no bar: None."""
    gains = slot or A3_GAINS
    room = no_room(slot)
    return {k: None if k in room else -round(share * float(gains[k]), 4) for k, share in SHARES.items()}


def needed_share(loss: float, floor: float | None) -> float | None:
    """The share of a loss to remove for its point estimate to reach ``floor``. No bar, no share."""
    if floor is None:
        return None
    return round(float((abs(loss) - abs(floor)) / abs(loss)), 4) if loss < 0 and abs(loss) > abs(floor) else 0.


FLOORS = floors(A5_SUMMARY_AT_DECLARATION)
NO_ROOM = no_room(A5_SUMMARY_AT_DECLARATION)
LIKELIHOOD_FLOOR, SPOTTING_FLOOR, BLEU_FLOOR = FLOORS["likelihood"], FLOORS["spotting"], FLOORS["bleu"]
BARS_AGAINST = "A5's summary" if A5_SUMMARY_AT_DECLARATION else "A3's stored read, until A5's summary fills the slot at declaration"
NEEDED = {k: needed_share(A1_LOSS[k], FLOORS[k]) for k in A1_LOSS}


def bar(floor) -> str:
    return "above no bar (no_room)" if floor is None else f"above {floor:.4f}"


def needed_text() -> str:
    return " and ".join("no_room" if v is None else f"{v:.0%}" for v in NEEDED.values())


SINGLE_STEP = ("Each pass is a single step: the hold rule reads hold-x at the fitted length and the length rule reads the fitted length against today's, so passing both does not show that the two together lose little; both together against top10 at today's length and the "
               "difference in differences are reported and decide nothing. ")
NOT_FRESH = "D20 chose top10 and A5's draws read its sentences, so D20's sentences are not fresh. "
GATE = "A6's full runs, the screen among them, run only if A5's summary reads confirmed or direction only."
ROOM_RULE = ("For each metric, if A5's mean gain is at or below 0, or its lower 95% t bound over the eight draws is at or below 0, that metric's outcome is no_room: no gain was established to protect. A rule with any no_room metric does not pass and prints no_room with the reason; "
             "the hold gate closes; the arms still report. ")
INSTRUMENT = ("Each draw's top10 at today's length must match A5's top10 for that draw, and the screen's A3's stored top10: on the same GPU type and pinned thread count at least 98% of its hypotheses, for the form, for real and for blank@predicted. Below that, the draw stops and later "
              "attempts of that draw are refused until it is declared. The draws need at least one same-GPU draw, else the read is incomplete; the count and rate of every draw are printed. ")
REDECLARE = "Any reader bug, or any code change after the screen or a draw, means re-declaring under a new OUT named in the job, with what was stored disclosed; no key is scoped by digest automatically. "
HOLD_GATE = ("The gate: the screen on A3's stored words reads hold-x against top10 at today's length, report only, and hold-x and hold-x at the fitted length run on the eight draws only if the screen's point estimates of likelihood and spotting sit above the two floors above. Otherwise "
             "the draws run top10 and length only, the length rule still reads, and this rule reads not run, screen failed, which is printed. Known from A1's stored rows before this gate was set: the 34% of clips holding the excluded uses hold 39% of A1's likelihood loss and 40% of its spotting "
             "loss, and excluding those uses is estimated to remove 3-15% against the " + needed_text() + " these floors need, so the screen may well stop the hold arms; that estimate is no read of hold-x. ")
RULES = dict(
    hold=("Primary arm: hold-x at the fitted length, against top10 at the fitted length, paired within each of A5's eight stored word draws, on D20, report only. hold-x keeps the left arm on entries with at least 25% two-handed records, on unannotated entries and on the list's new "
          "entries; the 25% cut was set after A1's D20 failure. hold-x at today's length is read beside it and decides nothing. Passed only if all hold: mean paired likelihood minus each form's own-length blank, its 95% lower bound " + bar(LIKELIHOOD_FLOOR) + " on both the two-way "
          "bootstrap and the t interval over the eight draws; v2 spotting's 95% lower bound " + bar(SPOTTING_FLOOR) + " on both; BLEU over draws, the t interval's lower bound " + bar(BLEU_FLOOR) + "; mean hands at least 0.9x its comparator's. These are small-loss bars, "
          "set as shares of the mean paired gains of " + BARS_AGAINST + ": 44% of BLEU's, 60% of likelihood's and of spotting's. " + ROOM_RULE + "A pass means any loss is small, possibly up to those shares, with less motion, and not that the arm reads better. Known before any read, and not evidence: "
          "pooled wrist 2.019x to 1.814x, contact unchanged by construction. The form stays far outside the 0.8-1.25x band, so a pass is no progress against the release bar. " + HOLD_GATE + SINGLE_STEP + INSTRUMENT + REDECLARE + NOT_FRESH + "Anything else is not passed, the failed check is reported, and not passed is not "
          "evidence of a loss. " + GATE),
    length=("Primary arm: length, the fit signer-balanced on How2Sign train's c02 + c04 (seconds = -0.3577 + 0.3445 x tokens), chosen by a train-only rule after its D20 ratio of about 1.08 was known, and never fitted on D20's lengths. Against top10 at today's length, paired within each "
            "of A5's eight stored word draws, on D20, report only. Passed only if all hold: likelihood and spotting as the hold rule; BLEU, the t interval's lower bound " + bar(BLEU_FLOOR) + " on the lower of raw BLEU and BLEU net of the blank's own change; mean hands at least 0.9x "
            "top10's. " + ROOM_RULE + "Pooled predicted over true length is reported and decides nothing: known on the report's state frames, 1.071 against today's 1.198, median per clip 1.128 against 1.287. Today's length runs 1.20x true, which lowers wrist by construction; the fitted length removes "
            "that, so wrist rises from 1.774x to 2.019x. Wrist is reported, decides nothing, and this rise is neither a loss nor progress. A pass means a nearer length costs little content, not that it adds any. This rule reads whether or not the screen lets the hold arms run. " + SINGLE_STEP + INSTRUMENT + REDECLARE + NOT_FRESH + GATE))


def a5_known(slot) -> str:
    """What was known of A5's summary: its slot, in full, once filled."""
    if not slot:
        return "not stored when this was written"
    return (f"sha256 {slot['sha256']}, reading {slot['outcome']}, mean gains BLEU {slot['bleu']}, likelihood {slot['likelihood']}, spotting {slot['spotting']}, lower t bounds {slot['t_lower']}, paired SDs {slot['paired_sd']}; no_room {no_room(slot) or 'on no metric'}; the power table is in "
            "reports/comp_a6_hold_concentration.json")


KNOWN_AT_WRITING = ("A1's hold had failed its rule on D20: likelihood -0.0276, spotting -0.0405, replacing the left arm on 46.6% of frames, 226 of its held uses from entries with at least 25% two-handed records. The 25% cut was set after that failure, and hold-x's motivation depends on two-view being the "
                    "right reference: the same clips read through the monocular path show more left-arm motion and spread (reports/wrist_by_signer.json), and the word clips' training is " + WORD_PATH + "; the decoder's share is not measured and no generated clip's amplitude was tested. Path read (branch path-read): "
                    + (PATH_READ_AT_DECLARATION or "not landed when this was written") + ". A1's loss does not concentrate in the sentences holding the excluded uses: 34% of clips hold 39% of the likelihood loss and 40% of the spotting loss, and the estimates of what excluding them removes are 3-15% against the "
                    + needed_text() + " the floors need (reports/comp_a6_hold_concentration.json, a9's words, the ceiling not a strict bound). Prior review's power figures: " + (PRIOR_REVIEW_POWER or "not supplied to this branch") + ". A3 generated top10's stored read was known: BLEU +0.343 [-0.018, +0.718], likelihood +0.0128, "
                    "spotting +0.0149, one word draw, on two GPUs. A4's real-clip draws were known (SD 0.237). V1 was known: the tokenizer's cap kept at 4.124 on D20 without neutral fill. A5's declaration was known, its digest 08b9a150; no draw of it was read here. A5's summary at declaration: "
                    + a5_known(A5_SUMMARY_AT_DECLARATION) + ". While exploring, the builder printed one metadata row (seconds and frames) of How2Sign's test-split extraction report; no state was read; that split holds c00. Known from stored motion before any read, top10's words on A3's caches, rebuilt to 0.0 of the stored wrist speeds on "
                    "499 clips, pooled wrist: top10 1.774x, length 2.019x, hold-x 1.595x, hold-x + length 1.814x; left wrist 2.228x, 2.532x, 1.791x, 2.032x; share of frames replaced 0, 0, 37.6%, 37.7%; contact unchanged by construction. The first draft of this read made exclusion plus rest primary, with "
                    "shortest ramps, after those motion figures and before any read. A review of commit 37c8d18 replaced it with this 2x2 knowing all of the above: shortest ramps left 90% of stretches above real's p90 speed, and hold-x + length has 1 cap-bound stretch, peak 1.376 against the cap's 1.354. "
                    "On the longest ramps hold-xr, report only and on A3's words alone, reads pooled 1.648x, left 1.918x, left height -1.099, no cap-bound stretch, peak 1.351, and 48% of its rest stretches above real's p90. A review of commit fbe9b29 found the earlier bars looser than their stated shares and set "
                    "them as declared shares of the mean gains, restated against A5's mean at declaration, with A3's and A5's gains known. A review of commit 11e310c added no_room, the 98% stop, and re-declaration in place of a reader amendment, before A5 read. Known about lengths: today's fit runs 1.198x true on the report's state frames (1.206x on metadata seconds); the balanced line had read about 1.08x earlier, in S2's probe "
                    "design, before the train-only rule was written, so that rule was not blind to it. By that rule the choice went to the line, on its tie to fewer parameters: train error 0.2479 against the word model's 0.2450, within the 0.005 tie. D20 then read, on state frames, pooled 1.071x, median per "
                    "clip 1.128, median per-clip error 0.245 for the line, and 1.065x, 1.123, 0.230 for the word model, which reads slightly better on both sets and was not chosen. The first train read split folds by clip instead of video. That was fixed after D20's ratios had been read; the fits and the "
                    "choice did not change, only the cross-validated errors. S0 had shown a faster form gains BLEU without content. The screen's gate on the hold arms was set after the concentration result above, with both it and A1's failure known, on point estimates of one word draw on D20's own "
                    "sentences; it was accepted that the screen waits for A5's summary. Five-clip smokes of these arms had not been run when this was written; running them before declaration is recommended.")


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
    if not fitted and (fit != TODAY_FIT or model.get("signer_balanced")):
        raise SystemExit(f"{name}: its fit is {model}, not today's")


def arms_of(k, gate_open: bool = True) -> tuple:
    return SCREEN_ARMS if k == SCREEN else ARMS if gate_open else GATE_CLOSED_ARMS


def check_draw(k, arms: dict, *, smoke: bool = False, gate_open: bool = True) -> None:
    """One draw's reports: D20, report-only, all of it, what their names say, paired, the same words, and one pod's code and environment."""
    from round20_compare import check_pairable
    names = arms_of(k, gate_open)
    if set(arms) != set(names):
        raise SystemExit(f"{k}: its arms are {sorted(arms)}, not {list(names)}")
    for name in names:
        rd.check_arm(arms[name], smoke=smoke)
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


def check_set(draws: dict, *, smoke: bool = False, gate_open: bool = True) -> None:
    """All eight of A5's draws, or the screen alone; one digest and packages; no two draws sharing words; the same sentences. The GPU may differ between draws, never within one."""
    keys = tuple(sorted(draws, key=str))
    if keys != (SCREEN,) and tuple(sorted(k for k in draws if k != SCREEN)) != DRAWS or (SCREEN in draws and len(draws) > 1):
        raise SystemExit(f"incomplete: draws {list(keys)}, the rule names {list(DRAWS)}; fewer than eight finished draws is no smaller test")
    if smoke and SCREEN not in draws:
        raise SystemExit("a smoke reads A3's words alone")
    for k, arms in draws.items():
        check_draw(k, arms, smoke=smoke, gate_open=gate_open)
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


def arm_summary(draws: dict, shards: dict, name: str, *, boot: int, other: str | None = None) -> dict:
    """Every number of one arm against its comparator across the draws. One draw has no draw variance: its t interval and SD are None."""
    other = other or COMPARATOR[name]
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
    hold = [(draws[k][name].get("hold_left_predicted") or {}) for k in order]         # the predicted form alone, like the motion figures in KNOWN_AT_WRITING
    row["cap_bound_per_draw"], row["hold_peak_per_draw"] = [h.get("cap_bound") for h in hold], [h.get("peak") for h in hold]
    mismatched = lambda arm: [((draws[k][arm].get("spotting") or {}).get("scores") or {}).get(FORM, {}).get("minus_mismatched_paired") for k in order]    # noqa: E731
    row["own_minus_mismatched"] = dict(arm=mismatched(name), comparator=mismatched(other))
    return row


def interaction(both_on_length: dict, hold_on_base: dict) -> dict:
    """The difference in differences per metric and draw: what hold-x does at the fitted length minus what it does at today's. Report only."""
    out = {}
    for metric in ("bleu", "likelihood_minus_blank", "spotting"):
        values = [a - b for a, b in zip(both_on_length[metric]["per_draw"], hold_on_base[metric]["per_draw"])]
        out[metric] = dict(per_draw=[round(float(x), 4) for x in values], mean=round(float(np.mean(values)), 4), t_ci95=a5.t_interval(values) if len(values) > 1 else None)
    return out


def _lower(row, interval) -> float | None:
    v = row.get(interval)
    return None if v is None else v[0]


def _above(value, floor) -> bool:
    """Strictly above a bar. An unknown number, or a metric with no bar, fails."""
    return value is not None and floor is not None and value > floor


def room_outcome(checks: dict) -> dict:
    """A rule's outcome: any no_room metric comes first, and prints its reason."""
    if NO_ROOM:
        return dict(checks=checks, outcome="no_room", no_room=dict(NO_ROOM))
    return dict(checks=checks, outcome="passed" if all(checks.values()) else "not_passed")


def content_checks(row: dict) -> dict:
    ll, spot = row["likelihood_minus_blank"], row["spotting"]
    return dict(likelihood_two_way_lower_bound_above_floor=_above(_lower(ll, "two_way_ci95"), LIKELIHOOD_FLOOR), likelihood_t_lower_bound_above_floor=_above(_lower(ll, "t_ci95"), LIKELIHOOD_FLOOR),
                spotting_two_way_lower_bound_above_floor=_above(_lower(spot, "two_way_ci95"), SPOTTING_FLOOR), spotting_t_lower_bound_above_floor=_above(_lower(spot, "t_ci95"), SPOTTING_FLOOR))


def at_least(value, base, share) -> bool:
    return value is not None and base is not None and value >= share * base


def hands_check(row: dict) -> bool:
    return at_least(row["motion"]["arm_hands"], row["motion"]["comparator_hands"], HANDS_SHARE)


def decide_hold(row: dict) -> dict:
    checks = dict(content_checks(row), bleu_t_lower_bound_above_floor=_above(_lower(row["bleu"], "t_ci95"), BLEU_FLOOR), hands_at_least_0_9_of_comparator=hands_check(row))
    return room_outcome(checks)


def lower_of_raw_and_net(row: dict) -> float | None:
    """The lower of raw BLEU's and net BLEU's t lower bounds; unknown if either is."""
    raw, net = _lower(row["bleu"], "t_ci95"), (row["bleu_net_of_blank"]["t_ci95"] or [None])[0]
    return None if raw is None or net is None else min(raw, net)


def decide_length(row: dict) -> dict:
    checks = dict(content_checks(row), bleu_lower_of_raw_and_net_t_lower_bound_above_floor=_above(lower_of_raw_and_net(row), BLEU_FLOOR), hands_at_least_0_9_of_comparator=hands_check(row))
    return room_outcome(checks)


def a5_instrument_check(draws: dict, a5_top10: dict) -> dict | None:
    """Each draw's base against A5's top10 on the same words (the screen's against A3's stored top10): counts and rates per draw. On the same GPU below ``MATCH_RATE`` the job has already stopped, so the read refuses; another GPU is reported.
    The draws need at least one same-GPU draw."""
    if not a5_top10:
        return None
    out = {}
    for k, other in a5_top10.items():
        base = draws[k][BASE]
        if base["references"] != other["references"]:
            raise SystemExit(f"draw {k}: A5's top10 read other sentences")
        for cache in ("word_cache", "word_cache_extra"):
            if (settings_of(base).get(cache) or {}).get("sha256") != (settings_of(other).get(cache) or {}).get("sha256"):
                raise SystemExit(f"draw {k}: base and A5's top10 read other words ({cache})")
        same = lambda form: int(sum(p == q for p, q in zip(base["forms"][form]["hypotheses"], other["forms"][form]["hypotheses"])))   # noqa: E731
        clips = len(base["references"])
        row = dict(form_identical=same(FORM), real_identical=same("real"), blank_identical=same(BLANK), clips=clips, same_gpu=settings_of(base)["provenance"]["environment"].get("gpu") == settings_of(other)["provenance"]["environment"].get("gpu"))
        row["rate"] = round(min(row["form_identical"], row["real_identical"], row["blank_identical"]) / clips, 4)
        out[k] = row
        if row["same_gpu"] and row["rate"] < MATCH_RATE:
            raise SystemExit(f"draw {k}: base matches A5's top10 at {row['rate']} on the same GPU, below {MATCH_RATE}: the draw stops until it is declared under a new OUT")
    if SCREEN not in out and not any(r["same_gpu"] for r in out.values()):
        raise SystemExit("incomplete: no draw ran on A5's GPU, so nothing tests the instrument")
    return dict(note="at least 98% on the same GPU, or the draw stops; other GPUs are reported", draws=out)


REDECLARE_NOW = "re-declare under a new OUT named in the job, with what was stored disclosed"


def check_digest(draws: dict, declared_commit: str | None, job_file: str) -> None:
    """A declared read runs on the declared code: every arm's digest is this reader's."""
    if not declared_commit:
        return
    here = lib.provenance_row(job_file)["code_digest"]
    other = sorted({settings_of(r)["provenance"]["code_digest"] for arms in draws.values() for r in arms.values()} - {here})
    if other:
        raise SystemExit(f"this reader's code is {here}, its arms' {other}: {REDECLARE_NOW}")


def gate_of(likelihood, spotting) -> dict:
    """The hold arms' gate: both point estimates of hold-x minus base above their floors, an unknown number closing it."""
    passed = not NO_ROOM and _above(likelihood, LIKELIHOOD_FLOOR) and _above(spotting, SPOTTING_FLOOR)
    return dict(passed=passed, likelihood=likelihood, spotting=spotting, likelihood_floor=LIKELIHOOD_FLOOR, spotting_floor=SPOTTING_FLOOR, no_room=dict(NO_ROOM), hold_arms="run on the draws" if passed else HOLD_NOT_RUN)


def hold_gate(screen: dict, declared_commit: str | None, *, job_file: str = JOB_FILE) -> dict:
    """The gate a stored screen sets for the draws: the screen must be a declared screen of this declaration, on this code and these floors."""
    if screen.get("mode") != "screen":
        raise SystemExit(f"the hold gate reads a screen's summary, not {screen.get('mode')}")
    if not screen.get("declared") or not declared_commit or screen.get("declared_commit") != declared_commit:
        raise SystemExit(f"the hold gate needs a declared screen of this declaration: the screen says {screen.get('declared_commit')}, this read {declared_commit}")
    if (screen.get("provenance") or {}).get("code_digest") != lib.provenance_row(job_file)["code_digest"]:
        raise SystemExit(f"the screen's summary came from other code: {REDECLARE_NOW}")
    if {k: (screen.get("floors") or {}).get(k) for k in FLOORS} != FLOORS:
        raise SystemExit(f"the screen's floors {screen.get('floors')} are not this code's {FLOORS}")
    row = screen["arms"][HOLD]
    return gate_of(row["likelihood_minus_blank"]["mean"], row["spotting"]["mean"])


def read(draws: dict, shards: dict, *, boot: int = 1000, declared_commit: str | None = None, job_file: str = JOB_FILE, attempts: dict | None = None, a5_top10: dict | None = None, smoke: bool = False, screen_summary: dict | None = None, screen_sha256: str | None = None) -> dict:
    screen = SCREEN in draws
    if not screen and (screen_summary is None or not screen_sha256):
        raise SystemExit("the draws need the screen's summary and its hash: it gates the hold arms")
    gate = None if screen else hold_gate(screen_summary, declared_commit, job_file=job_file)
    gate_open = screen or gate["passed"]
    check_set(draws, smoke=smoke, gate_open=gate_open)
    check_digest(draws, declared_commit, job_file)
    if not screen and set(a5_top10 or ()) != set(DRAWS):
        raise SystemExit(f"the draws need all eight of A5's top10 reports: {sorted(a5_top10 or ())}")
    arms = {name: arm_summary(draws, shards, name, boot=boot) for name in (SCREEN_ARMS if screen else ARMS if gate_open else GATE_CLOSED_ARMS) if name != BASE}
    both_on_base = arm_summary(draws, shards, BOTH, boot=boot, other=BASE) if gate_open else None
    if screen and not smoke:
        gate = gate_of(arms[HOLD]["likelihood_minus_blank"]["mean"], arms[HOLD]["spotting"]["mean"])
    note = "evaluated for reading only; this field adopts nothing" if declared_commit else "DRAFT, NOT DECLARED: adopts nothing"
    if screen:
        rule = dict(hold=dict(arm=BOTH, comparator=LENGTH, outcome="smoke_decides_nothing" if smoke else "screen_decides_nothing", checks=None, text=RULES["hold"]), length=dict(arm=LENGTH, comparator=BASE, outcome="smoke_decides_nothing" if smoke else "screen_decides_nothing", checks=None, text=RULES["length"]))
    else:
        hold_rule = decide_hold(arms[BOTH]) if gate_open else dict(checks=None, outcome="no_room", no_room=dict(NO_ROOM)) if NO_ROOM else dict(checks=None, outcome=HOLD_NOT_RUN)
        rule = dict(hold=dict(arm=BOTH, comparator=LENGTH, text=RULES["hold"], **hold_rule), length=dict(arm=LENGTH, comparator=BASE, text=RULES["length"], **decide_length(arms[LENGTH])))
    return dict(kind="comp-a6-read", report_only=True, mode="smoke" if smoke else "screen" if screen else "draws", declared=bool(declared_commit), declared_commit=declared_commit, draws=sorted(draws, key=str), set=dict(key=lib.D20_KEY, sha256=lib.D20_SHA),
                arms=arms, hold_gate=gate, both_on_base=both_on_base, difference_in_differences=interaction(arms[BOTH], arms[HOLD]) if gate_open else None, report_only_rows="both_on_base and difference_in_differences decide nothing", floors=dict(FLOORS, against=BARS_AGAINST), screen_sha256=screen_sha256, rule=rule, known_when_written=KNOWN_AT_WRITING, note=note, reported_only=[HOLD, HOLD_REST] if screen else [HOLD] if gate_open else [], attempts=attempts or {}, a5_instrument_check=a5_instrument_check(draws, a5_top10 or {}), provenance=lib.provenance_row(job_file))


def main(argv=None) -> int:
    from round20_compare import report_shards
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--draw", action="append", default=[], help=f"K=KEY,... in the order {','.join(ARMS)}; K is 1 to 8, or {SCREEN} alone, with {HOLD_REST} last")
    p.add_argument("--declared-commit")
    p.add_argument("--attempts", action="append", default=[], help="K=A:arms;A:arms: every attempt of draw K and the arms it stored")
    p.add_argument("--smoke", action="store_true", help="a smoke's reports, A3's words: every check and summary runs, nothing decides and no number is printed")
    p.add_argument("--screen-key", help="the draws: the screen's stored summary, which gates the hold arms")
    p.add_argument("--screen-sha256", help="the hash of --screen-key: it goes in the draws' report")
    p.add_argument("--a5-top10", action="append", default=[], help="K=KEY: A5's top10 of draw K, or A3's stored top10 for K=a3, compared with base")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--store", default="r2")
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    screen_summary = None
    if args.screen_key:
        raw_screen = store.get(args.screen_key)
        if hashlib.sha256(raw_screen).hexdigest() != args.screen_sha256:
            raise SystemExit(f"{args.screen_key} does not match --screen-sha256")
        screen_summary = json.loads(raw_screen)
    gate_open = hold_gate(screen_summary, args.declared_commit)["passed"] if screen_summary else True
    draws, shards = {}, {}
    for spec in args.draw:
        k, _, keys = spec.partition("=")
        parts = keys.split(",")
        name = SCREEN if k == SCREEN else int(k) if k.isdigit() else None
        names = arms_of(name, gate_open) if name is not None else ()
        if name is None or len(parts) != len(names) or not all(parts) or name in draws:
            p.error(f"--draw {spec!r}: K={','.join(a.upper() for a in (names or ARMS))}, once each")
        draws[name] = {a: json.loads(store.get(key)) for a, key in zip(names, parts)}
        shards[name] = {a: report_shards(store, key, draws[name][a], None) for a, key in zip(names, parts)}
    attempts = {spec.partition("=")[0]: spec.partition("=")[2] for spec in args.attempts}
    a5_top10 = {}
    for spec in args.a5_top10:
        k, _, key = spec.partition("=")
        name = SCREEN if k == SCREEN else int(k) if k.isdigit() else None
        if name not in draws or not key:
            p.error(f"--a5-top10 {spec!r}: K=KEY, a draw of this read")
        a5_top10[name] = json.loads(store.get(key))
    out = read(draws, shards, boot=args.bootstrap, declared_commit=args.declared_commit, attempts=attempts, a5_top10=a5_top10, smoke=args.smoke, screen_summary=screen_summary, screen_sha256=args.screen_sha256)
    store.put(args.output_key, json.dumps(out, indent=1, default=float).encode())
    print(f"A6 {out['mode']}: hold {out['rule']['hold']['outcome']}, length {out['rule']['length']['outcome']}", flush=True)
    for reason in NO_ROOM.items() if not args.smoke else ():
        print(f"A6 no_room {reason[0]}: {reason[1]}", flush=True)
    for k, row in ((out.get("a5_instrument_check") or {}).get("draws") or {}).items():
        print(f"COMP-A6-MATCH draw={k} identical={row['form_identical']}/{row['clips']} rate={row['rate']} same_gpu={row['same_gpu']}", flush=True)
    return 0


__all__ = ["read", "decide_hold", "decide_length", "check_set", "RULES", "KNOWN_AT_WRITING"]

if __name__ == "__main__":
    raise SystemExit(main())
