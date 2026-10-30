"""The declared UWB lifter gate, row by row: value, bar, pass / fail / not-evaluable, and the overall verdict.

Declared in docs/NEXT_PHASE_PLAN.md, "UWB lifter: declared selection and gate" (2026-10-30, 03:34 UTC), amendment 1
(03:59 UTC), owner declaration (04:02 UTC), visibility selection result ``inside`` (04:20 UTC). Inputs:

- ``--gate-report``: ``uwb_lifter_gate.py --out`` (``summary`` with the primary method ``lifter[<rule>]``);
- ``--how2sign-val``: ``uwb_how2sign_val_medians.py`` (the bundle's How2Sign-val median per gate group, by the gate's
  code path; the bundle's training report holds per-side local-rotation errors, not the gate's groups);
- ``--downstream``: ``uwb_downstream_check.py`` (the back-translator row);
- ``--outside-frame``: ``uwb_outside_frame_accuracy.py`` (the visibility selection report: chest-up forearm survival
  of the selected rule against MediaPipe's own keep decision on the How2Sign chest-up crops, per side).

Each input is a local path, a store key or ``KEY@SHA256``; a missing input leaves its rows **not-evaluable**.

**Rows** (bars exactly as declared):

| Check | Row(s) |
|---|---|
| Accuracy, medians | upper arms <= 12, forearms <= 18, wrists <= 15, fingers <= 16 deg (primary method, common frames); each CI upper bound <= bar + 1 |
| Accuracy, p90 | upper arms <= 25, forearms <= 40 |
| Relative | each group's YouTube median <= 1.6x the How2Sign-val median (same bundle, primary method and candidate rules as the gate run; measured on >= 200 values from >= 5 videos; else not-evaluable) |
| Global offset | own median - rotated median <= 3 deg, upper arms and forearms |
| Strata | no stratum with >= 200 pairs worse than a group's bar by more than 3 deg; gating strata: hips visible / not, fps about 30 / not, frame-count difference 0 / -1. Stream-mismatch strata (and frame-count difference "other") are reported, not gated |
| Validity | hand-valid agreement >= 0.88 per side; forearm agreement where our chest is valid (both hips: the hip-visible frames) >= 0.85 per side; UWB-only valid <= 10% per side (the larger of the hand and forearm rates); chest-up forearm survival >= 0.8x MediaPipe's per side (a side is measured only if MediaPipe keeps >= 200 outside-frame frames from >= 5 videos) |
| Timing | mirror-better <= 3% of clips; abs(lag) <= 1 on >= 90% of arm-aligned pairs |
| Downstream | recomputed from the downstream report's reads and differences: ratio >= 0.9 and lifted - blank lower bound > 0 (provided ours - blank > 0), pooled **and** on UWB's dev partition (>= 200 pairs from >= 50 videos); <= 10% refused pairs; same gate report, bundle, rule and 2,000 resamples |
| Setup | the gate ran the selected rule and the pinned bundle, 5 pairs per video, 2,000 resamples, the 70 development videos excluded |

Diagnostics only (reported, never gating): input p99 abs z (reference 4) and lag drift.

**Overall**: NOT-EVALUABLE if any proposal_check boolean disagrees with its recomputation; else FAIL if any gating row
fails; PASS only if every gating row passes; otherwise NOT-EVALUABLE.

**Rounding**: values are compared unrounded wherever the inputs carry them (agreements, UWB-only rates and the mirror
share from their counts, the lag share from the report's pairs, the downstream BLEUs). Medians, p90s and their
intervals exist only as the gate summary's 2-dp values (``uwb_lifter_gate.stat``) and chest-up survivals as 4-dp
values: those rows are marked ``rounded_source`` and compared as given, so a value exactly at the bar passes.

**Cross-check**: every boolean of the gate report's ``proposal_check`` is recomputed here by its own definition and
compared; a different value is a **disagreement**. Where the declared gate differs from the proposal's definition
(strata scope; drift, which the proposal checks and the declaration makes a diagnostic) both readings are listed.

**Gate v2** (``--gate-version 2``; plan "UWB lifter v2 and gate v2: declared", 2026-10-30 11:12 UTC; the default mode is
gate v1, unchanged). Every gate v1 bar is kept; the accuracy, relative, offset and strata rows read ``lifter[inside]``
on the common frames, the validity rows read ``lifter[head]`` (the rule the head becomes; ``inside``'s are reported).
New rows: setup (inside, the gate v1 candidate rules, head as a secondary method, the declared v2 bundle, 12 pairs per
video, 2,000 resamples, 70 development videos excluded, lag curves, the declared split); **population** (every gated
video held out, the split disjoint by channel and video, the pairs index and the bundle's training report on this
split); **v1 reference** (the xy2 bundle on the identical sample: gating as a setup row, not-evaluable unless present and
identical: same evaluated pairs and every gate setting; its medians reported); **How2Sign non-inferiority** (per
group, v2's How2Sign-val median <= v1's + 0.5 deg); **truth anchor** (full-frame two-view forearm median <= v1's,
read unrounded from v1's pinned selection report: 28.24 left / 29.07 right); **chest-up survival of head** (tight crops
of signers 1 and 2, out-of-frame forearms: >= 0.8x MediaPipe where it keeps >= 200 frames from >= 5 videos; else head
must keep more than inside, admit >= 1 measured outside bin (otherwise not-evaluable) and be admissible (otherwise
fail)); **timing** redefined (mirror unchanged; pooled lag per video with >= 3 arm-aligned stream-match pairs within 1
for >= 90% of videos, and no video whose pooled lag lies beyond 1 gaining > 2 deg over lag 0; v1's value reported);
**downstream** on ``lifter[head]`` states (the conversion rule). Reported: channel-clustered 95% intervals beside the
by-video ones, head-kept accuracy, held-out minus lifter-val gaps, diagnostics. Amendments 1 (11:51 UTC) and 2 (12:03
UTC). Only proposal_check keys whose rows keep their gate v1 definition on the same method are cross-checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

BUNDLE_SHA256 = "02a90e4ff333202d556bee62b12216ffca85761e45dafb011144fcbc1bd82dde"
SELECTED_RULE = "inside"
GROUPS = ("upper_arms", "forearms", "wrists", "fingers")
BARS = dict(median=dict(upper_arms=12., forearms=18., wrists=15., fingers=16.), ci_slack=1.,
            p90=dict(upper_arms=25., forearms=40.), relative=1.6, rotation_gain=3., stratum_pairs=200,
            stratum_slack=3., hand_agreement=.88, forearm_agreement=.85, uwb_only=.10, chest_up_survival=.80,
            mirror_share=.03, lag_within_1=.90, downstream_ratio=.9, input_z=4., drift_limit=1., drift_share=.95)
GATING_STRATA = {"hips": ("visible", "not_visible"), "fps30": ("yes", "no"), "fc_diff": ("0", "-1")}
SAMPLE = dict(pairs_per_video=5, bootstrap=2000, excluded_videos=70)
MIN_VALUES, MIN_VIDEOS = 200, 5            # a How2Sign-val median on fewer is unmeasured (amendment 1's rule)
PASS, FAIL, NA = "pass", "fail", "not-evaluable"


def load(store, spec: str | None):
    """(JSON, SHA-256) of a local path, store key or KEY@SHA256; (None, None) without a spec."""
    if not spec:
        return None, None
    from uwb_downstream_check import read_spec
    payload, sha = read_spec(store, spec)
    return json.loads(payload), sha


def row(check: str, rid: str, value, bar, status: str, *, gating: bool = True, proposal_key: str | None = None,
        detail=None, rounded: str | None = None, proposal_value=..., note: str | None = None) -> dict:
    """One row. ``rounded``: the source only provides a rounded value (named precision), compared as given (a value
    exactly at the bar passes); ``proposal_value``: the proposal_check reading when it differs in source from
    ``status`` (rounded vs unrounded); ``note``: a remark printed with the row."""
    out = dict(check=check, id=rid, value=value, bar=bar, status=status, gating=gating, proposal_key=proposal_key,
               detail=detail, rounded_source=rounded, note=note)
    if proposal_value is not ...:
        out["proposal_value"] = proposal_value
    return out


def as_bool(status: str):
    return None if status == NA else status == PASS


ROUNDED_2DP = "gate summary, 2 dp (uwb_lifter_gate.stat)"


def at_most(value, bar) -> str:
    return NA if value is None else PASS if value <= bar else FAIL


def at_least(value, bar) -> str:
    return NA if value is None else PASS if value >= bar else FAIL


def r2(v):
    """Display only: values are compared unrounded."""
    return None if v is None else round(float(v), 4)


# ----------------------------------------------------------------------------------------------------------------------

def accuracy_rows(e: dict) -> list[dict]:
    out = []
    for g, bar in BARS["median"].items():
        s = e["common"].get(g, {})
        median = s.get("median") if s.get("n") else None
        out.append(row("Accuracy, medians", f"median.{g}<={bar}", median, bar, at_most(median, bar),
                       proposal_key=f"median.{g}<={bar}", detail=dict(n=s.get("n")), rounded=ROUNDED_2DP))
        ci = s.get("median_ci95_by_video")
        limit = bar + BARS["ci_slack"]
        out.append(row("Accuracy, medians", f"median_ci_upper.{g}<={limit}", ci and ci[1], limit,
                       at_most(ci and ci[1], limit), proposal_key=f"median_ci_upper.{g}<={limit}", detail=dict(ci95=ci),
                       rounded=ROUNDED_2DP))
    for g, bar in BARS["p90"].items():
        s = e["common"].get(g, {})
        p90 = s.get("p90") if s.get("n") else None
        out.append(row("Accuracy, p90", f"p90.{g}<={bar}", p90, bar, at_most(p90, bar), proposal_key=f"p90.{g}<={bar}",
                       rounded=ROUNDED_2DP))
    return out


def candidate_rules(spec) -> set:
    rules = spec.split(",") if isinstance(spec, str) else list(spec or ())
    return {r for r in rules if r}


def relative_rows(e: dict, h2s: dict | None, gate_sha: str | None, primary: str | None = None,
                  candidates=None) -> list[dict]:
    """The denominator must come from the same bundle, the same primary method and the same candidate rules (they
    define the common frames) as the gate run."""
    out = []
    reason = None
    if h2s is None:
        reason = "no How2Sign-val medians given (uwb_how2sign_val_medians.py)"
    elif h2s.get("regressors_sha256") != gate_sha:
        reason = f"How2Sign-val medians are of bundle {h2s.get('regressors_sha256')}, the gate ran {gate_sha}"
    elif h2s["summary"].get("primary") != primary:
        reason = f"How2Sign-val primary method {h2s['summary'].get('primary')}, the gate's is {primary}"
    elif candidate_rules(h2s.get("candidate_visibility")) != candidate_rules(candidates):
        reason = (f"How2Sign-val candidate rules {sorted(candidate_rules(h2s.get('candidate_visibility')))} differ from "
                  f"the gate run's {sorted(candidate_rules(candidates))} (they define the common frames)")
    primary = None if h2s is None else h2s["summary"]["primary"]
    for g in GROUPS:
        yt = e["common"].get(g, {})
        ref = {} if reason else h2s["summary"]["methods"].get(primary, {}).get("common", {}).get(g, {})
        why = reason
        if not why and (ref.get("n", 0) < MIN_VALUES or ref.get("videos", 0) < MIN_VIDEOS):
            why = f"How2Sign-val {g}: {ref.get('n', 0)} values from {ref.get('videos', 0)} videos (unmeasured)"
        value = None
        if not why and yt.get("n") and ref.get("median"):
            value = yt["median"] / ref["median"]
        out.append(row("Relative to How2Sign validation", f"relative.{g}<={BARS['relative']}x", value,
                       BARS["relative"], at_most(value, BARS["relative"]), rounded="ratio of two 2-dp medians",
                       detail=dict(youtube_median=yt.get("median"), how2sign_val_median=ref.get("median"),
                                   how2sign_val_n=ref.get("n"), how2sign_val_videos=ref.get("videos"),
                                   how2sign_val_ci95=ref.get("median_ci95_by_video"), reason=why)))
    return out


def offset_rows(e: dict) -> list[dict]:
    out = []
    for g in ("upper_arms", "forearms"):
        a, b = e["own"].get(g, {}), e["rotated"].get(g, {})
        gain = a["median"] - b["median"] if a.get("n") and b.get("n") else None
        key = f"rotation_gain.{g}<={BARS['rotation_gain']}"
        out.append(row("Global offset", key, gain, BARS["rotation_gain"], at_most(gain, BARS["rotation_gain"]),
                       proposal_key=key, detail=dict(own=a.get("median"), rotated=b.get("median")),
                       rounded="difference of two 2-dp medians"))
    return out


def stratum_failures(strata: dict, kinds: dict | None) -> tuple[list[str], list[dict]]:
    """(failures, per-stratum rows): strata with >= 200 pairs whose group median exceeds the bar by more than 3 deg;
    ``kinds`` restricts to {kind: values} (None: every stratum, the proposal's reading)."""
    failures, rows = [], []
    for kind, values in strata.items():
        for value, s in values.items():
            gating = kinds is not None and value in kinds.get(kind, ())
            worst = []
            for g, bar in BARS["median"].items():
                m = s["common"].get(g, {}).get("median")
                if m is not None and m > bar + BARS["stratum_slack"]:
                    worst.append(f"{kind}={value}:{g}={m}")
            measured = s["pairs"] >= BARS["stratum_pairs"]
            if measured and (kinds is None or gating):
                failures += worst
            rows.append(dict(kind=kind, value=value, pairs=s["pairs"], gating=bool(gating), measured=measured,
                             medians={g: s["common"].get(g, {}).get("median") for g in GROUPS}, over=worst))
    return failures, rows


def strata_rows(summary: dict) -> list[dict]:
    strata = summary.get("strata", {})
    if not strata:
        return [row("Strata", "strata_within_3deg_of_bar", None, f"bar + {BARS['stratum_slack']}", NA,
                    proposal_key="strata_within_3deg_of_bar")]
    failures, table = stratum_failures(strata, GATING_STRATA)
    out = [row("Strata", "strata_within_3deg_of_bar", failures or "none over", f"bar + {BARS['stratum_slack']} "
               f"(strata with >= {BARS['stratum_pairs']} pairs)", FAIL if failures else PASS,
               proposal_key="strata_within_3deg_of_bar", detail=dict(failures=failures))]
    for s in table:
        status = (FAIL if s["over"] else PASS) if s["measured"] else NA
        out.append(row("Strata", f"stratum.{s['kind']}={s['value']}", s["medians"], f"bar + {BARS['stratum_slack']}",
                       status, gating=False, detail=dict(pairs=s["pairs"], gating_stratum=s["gating"],
                                                         over=s["over"], note=None if s["measured"] else
                                                         f"fewer than {BARS['stratum_pairs']} pairs: not judged")))
    return out


def share_of(cells: dict, keys) -> float | None:
    """Unrounded share of the confusion cells ``keys`` (the gate's summaries round to 4 dp)."""
    total = sum(cells.get(k, 0) for k in ("both", "ours_only", "uwb_only", "neither"))
    return sum(cells.get(k, 0) for k in keys) / total if total else None


def validity_rows(e: dict, outside: dict | None, gate_sha: str | None, rule: str | None) -> list[dict]:
    """Agreements and UWB-only rates unrounded from the confusion counts; ``proposal_value`` reads the rounded rates
    as proposal_check does."""
    return validity_core_rows(e) + chest_up_rows(outside, gate_sha, rule)


def validity_core_rows(e: dict) -> list[dict]:
    """Hand agreement, forearm agreement where our chest is valid, and UWB-only rate, per side."""
    out = []
    for side in ("Left", "Right"):
        h = e["hand_valid"].get(f"all.{side}", {})
        key = f"hand_agreement.{side}>={BARS['hand_agreement']}"
        value = share_of(h, ("both", "neither")) if "both" in h else h.get("agreement")
        out.append(row("Validity", key, value, BARS["hand_agreement"], at_least(value, BARS["hand_agreement"]),
                       proposal_key=key, proposal_value=as_bool(at_least(h.get("agreement"), BARS["hand_agreement"])),
                       rounded=None if "both" in h else "gate summary, 4 dp",
                       detail=dict(frames=h.get("frames"), kappa=h.get("kappa"), rounded=h.get("agreement"))))
    for side in ("Left", "Right"):
        f = e["forearm_valid_where_our_chest_is"].get(side, {})
        key = f"forearm_agreement_hips_visible.{side}>={BARS['forearm_agreement']}"
        value = share_of(f, ("both", "neither")) if "both" in f else f.get("agreement")
        out.append(row("Validity", key, value, BARS["forearm_agreement"], at_least(value, BARS["forearm_agreement"]),
                       proposal_key=key,
                       proposal_value=as_bool(at_least(f.get("agreement"), BARS["forearm_agreement"])),
                       rounded=None if "both" in f else "gate summary, 4 dp",
                       detail=dict(frames=f.get("frames"), rounded=f.get("agreement"),
                                   note="frames where our chest is valid (both hips)")))
    for side in ("Left", "Right"):
        h, f = e["hand_valid"].get(f"all.{side}", {}), e["forearm_valid_where_our_chest_is"].get(side, {})
        rounded = [x.get("uwb_only_rate") for x in (h, f) if x.get("uwb_only_rate") is not None]
        exact = [share_of(x, ("uwb_only",)) if "both" in x else x.get("uwb_only_rate") for x in (h, f)
                 if x.get("uwb_only_rate") is not None]
        key = f"uwb_only_rate.{side}<={BARS['uwb_only']}"
        value = max(exact) if exact else None
        out.append(row("Validity", key, value, BARS["uwb_only"], at_most(value, BARS["uwb_only"]), proposal_key=key,
                       proposal_value=as_bool(at_most(max(rounded) if rounded else None, BARS["uwb_only"])),
                       detail=dict(hand=h.get("uwb_only_rate"), forearm=f.get("uwb_only_rate"))))
    return out


OUTSIDE_BINS = ("0-0.1", "0.1-0.3", "0.3-inf")


def chest_up_rows(outside: dict | None, gate_sha: str | None, rule: str | None) -> list[dict]:
    """Per side: the selected rule's forearm survival on the How2Sign chest-up crops (every bin) over MediaPipe's own
    keep decision (Tasks visibility >= 0.5 on shoulder, elbow and wrist), from the visibility selection report.

    **Measurability** (coordinator ruling on the review of d2591fa): a side is measured only if MediaPipe keeps >= 200
    frames from >= 5 videos on that side's outside-frame subset (bins 0-0.1, 0.1-0.3, 0.3-inf of the crops); otherwise
    that side is not-evaluable, never pass. Kept frames are each bin's frames x survival (survival is rounded to 4 dp
    in the report, so the count is rounded to the nearest frame); videos are the largest per-bin count of videos with
    a kept, scored frame (a lower bound of their union: conservative). A side where ``inside`` keeps no outside-frame
    frame fails by construction: the rule drops every point outside the image."""
    reason = None
    if outside is None:
        reason = "no visibility selection report given (uwb_outside_frame_accuracy.py)"
    elif outside.get("regressors_sha256") != gate_sha:
        reason = f"selection report of bundle {outside.get('regressors_sha256')}, the gate ran {gate_sha}"
    elif (outside.get("selection") or {}).get("chosen") != rule:
        reason = f"the selection chose {(outside.get('selection') or {}).get('chosen')}, the gate ran {rule}"
    elif rule not in outside.get("by_decision", {}) or "mediapipe" not in outside.get("by_decision", {}):
        reason = f"rule {rule} or MediaPipe not measured in the selection report"
    out = []
    for side in ("Left", "Right"):
        value, detail, note = None, dict(reason=reason), None
        if not reason:
            cells = outside["by_decision"][rule][side]["Forearm"]
            mp_cells = outside["by_decision"]["mediapipe"][side]["Forearm"]
            kept, mp = cells["all"], mp_cells["all"]
            mp_frames = sum(round((mp_cells.get(b, {}).get("frames") or 0) * (mp_cells.get(b, {}).get("survival") or 0))
                            for b in OUTSIDE_BINS)
            mp_videos = max([(mp_cells.get(b, {}).get("kept") or {}).get("videos", 0) for b in OUTSIDE_BINS] or [0])
            rule_outside = sum(round((cells.get(b, {}).get("frames") or 0) * (cells.get(b, {}).get("survival") or 0))
                               for b in OUTSIDE_BINS)
            weighted = {d: (((outside.get("uwb_weighted_survival") or {}).get(d) or {}).get(side) or {}).get("Forearm")
                        for d in (rule, "mediapipe")}
            measured = mp_frames >= MIN_VALUES and mp_videos >= MIN_VIDEOS
            if not measured:
                detail["reason"] = (f"unmeasured: MediaPipe keeps {mp_frames} outside-frame frames from >= {mp_videos} "
                                    f"videos (needs {MIN_VALUES} from {MIN_VIDEOS})")
            elif not mp.get("survival"):
                detail["reason"] = "MediaPipe keeps no chest-up forearm frame: the ratio is undefined"
            else:
                value = kept["survival"] / mp["survival"]
            if rule == "inside" and rule_outside == 0 and measured:
                note = "fails by construction: 'inside' drops every point outside the image"
            detail.update(rule_survival=kept.get("survival"), mediapipe_survival=mp.get("survival"),
                          frames=kept.get("frames"), mediapipe_outside_frames=mp_frames,
                          mediapipe_outside_videos_lower_bound=mp_videos, rule_outside_frames=rule_outside,
                          uwb_weighted=weighted, note="How2Sign val signers' chest-up crops (Sem-Lex framings and "
                                                      "tight crops), every distance bin")
        out.append(row("Validity", f"chest_up_forearm_survival.{side}>={BARS['chest_up_survival']}x_mediapipe", value,
                       BARS["chest_up_survival"], at_least(value, BARS["chest_up_survival"]), detail=detail,
                       rounded="ratio of two 4-dp survivals (uwb_outside_frame_accuracy)", note=note))
    return out


def timing_rows(e: dict, summary: dict, pairs: list | None = None) -> list[dict]:
    """Mirror share from its counts; the lag share recomputed unrounded from the report's pairs when they are there
    (the summary rounds it to 4 dp)."""
    mc = e["mirror_check"]
    share = mc["mirror_better"] / mc["clips"] if mc["clips"] else None
    key = f"mirror_better_share<={BARS['mirror_share']}"
    w = summary["alignment"]["within_1_share"]
    aligned = [p for p in (pairs or []) if p.get("status") == "evaluated"
               and (p.get("alignment") or {}).get("method") == "arm-2d"]
    exact = sum(abs(p["alignment"]["lag"]) <= 1 for p in aligned) / len(aligned) if aligned else None
    key2 = f"lag_within_1_share>={BARS['lag_within_1']}"
    value = exact if exact is not None else w
    return [row("Timing", key, share, BARS["mirror_share"], at_most(share, BARS["mirror_share"]), proposal_key=key,
                detail=dict(clips=mc["clips"], mirror_better=mc["mirror_better"])),
            row("Timing", key2, value, BARS["lag_within_1"], at_least(value, BARS["lag_within_1"]), proposal_key=key2,
                proposal_value=as_bool(at_least(w, BARS["lag_within_1"])),
                rounded=None if exact is not None else "gate summary, 4 dp",
                detail=dict(methods=summary["alignment"]["methods"], summary_rounded=w,
                            from_pairs=len(aligned) or None))]


def downstream_rows(downstream: dict | None, gate_sha: str | None, report: dict | None = None,
                    expected_visibility: str | None = None) -> list[dict]:
    """The row recomputed from the downstream report's reads and differences (its recorded status is never trusted):
    pooled and UWB-dev reads, the dev size rule and the refusal rule (``uwb_downstream_check.combined_verdict``);
    not-evaluable when it read another gate report, bundle or rule, or another bootstrap count."""
    import uwb_downstream_check as ds
    rid, bar = "downstream_back_translator", (f">= {BARS['downstream_ratio']} x ours and lifted - blank lower bound > 0 "
                                              f"(ours - blank > 0), pooled and on UWB dev")

    def na(reason):
        return [row("Downstream", rid, None, bar, NA, detail=dict(reason=reason))]
    if downstream is None:
        return na("no downstream report given (uwb_downstream_check.py)")
    report = report or {}
    checks = [("gate report", downstream.get("gate_report_sha256"), gate_sha),
              ("bundle", downstream.get("regressors_sha256"), report.get("regressors_sha256")),
              ("visibility rule", downstream.get("visibility"),
               expected_visibility or (report.get("settings") or {}).get("visibility")),
              ("bootstrap resamples", downstream.get("bootstrap"), SAMPLE["bootstrap"])]
    for name, theirs, ours in checks:
        if theirs != ours:
            return na(f"the downstream report's {name} is {theirs}, expected {ours}")
    sample = downstream.get("sample") or {}
    if not downstream.get("reads") or not downstream.get("differences") or not sample:
        return na("the downstream report lacks reads, differences or its sample counts")
    pooled = ds.verdict(dict(reads=downstream["reads"], differences=downstream["differences"]))
    dev = downstream.get("dev")
    dev_v = ds.verdict(dev) if dev and dev.get("reads") and dev.get("differences") else None
    combined = ds.combined_verdict(pooled, dev_v, dev_pairs=(dev or {}).get("pairs", 0),
                                   dev_videos=(dev or {}).get("videos_scored", 0), refused=sample.get("refused", 0),
                                   sampled=sample.get("union", sample.get("sampled", 0)))
    recorded = (downstream.get("verdict") or {}).get("status")
    status = {"pass": PASS, "fail": FAIL}.get(combined["status"], NA)
    return [row("Downstream", rid, pooled.get("ratio_lifted_over_ours"), bar, status,
                detail=dict(recomputed=combined, pooled=pooled, dev=dev_v, recorded_status=recorded,
                            recorded_differs=recorded != combined["status"], reads=downstream.get("reads"),
                            differences=downstream.get("differences"), sample=sample))]


def setup_rows(report: dict, rule: str | None) -> list[dict]:
    s = report.get("settings", {})
    out = [row("Setup", "visibility_is_selected_rule", s.get("visibility"), SELECTED_RULE,
               PASS if s.get("visibility") == SELECTED_RULE else FAIL),
           row("Setup", "bundle_is_xy2", report.get("regressors_sha256"), BUNDLE_SHA256,
               PASS if report.get("regressors_sha256") == BUNDLE_SHA256 else FAIL),
           row("Setup", "pairs_per_video", s.get("pairs_per_video"), SAMPLE["pairs_per_video"],
               PASS if s.get("pairs_per_video") == SAMPLE["pairs_per_video"] and not s.get("videos") else FAIL,
               detail=dict(videos_limit=s.get("videos"))),
           row("Setup", "bootstrap_resamples", s.get("bootstrap"), SAMPLE["bootstrap"],
               PASS if s.get("bootstrap") == SAMPLE["bootstrap"] else FAIL),
           row("Setup", "development_videos_excluded", len(report.get("excluded_videos", [])), SAMPLE["excluded_videos"],
               PASS if len(report.get("excluded_videos", [])) == SAMPLE["excluded_videos"] else FAIL)]
    return out


def diagnostic_rows(summary: dict) -> list[dict]:
    z = [v["max_p99_abs_z"] for v in summary.get("inputs", {}).values()]
    drifts = list(summary.get("lag_drift_frames_per_minute", {}).values())
    share = r2(np.mean(np.abs(drifts) < BARS["drift_limit"])) if drifts else None
    key = f"drift_below_1_share>={BARS['drift_share']}"
    return [row("Diagnostics", "input_p99_abs_z", max(z) if z else None, f"reference {BARS['input_z']}",
                "reported", gating=False),
            row("Diagnostics", key, share, BARS["drift_share"], "reported", gating=False, proposal_key=key,
                detail=dict(checked_videos=len(drifts), note="drift is a diagnostic in the declaration"))]


def build_rows(report: dict, *, h2s: dict | None = None, downstream: dict | None = None, outside: dict | None = None,
               gate_sha: str | None = None) -> list[dict]:
    summary = report["summary"]
    primary = summary["primary"]
    if primary not in summary["methods"]:
        raise SystemExit(f"the primary method {primary} is not in the gate summary")
    e = summary["methods"][primary]
    settings = report.get("settings", {})
    rule = settings.get("visibility")
    bundle = report.get("regressors_sha256")
    return (setup_rows(report, rule) + accuracy_rows(e)
            + relative_rows(e, h2s, bundle, primary, settings.get("candidate_visibility")) + offset_rows(e)
            + strata_rows(summary) + validity_rows(e, outside, bundle, rule) + timing_rows(e, summary, report.get("pairs"))
            + downstream_rows(downstream, gate_sha, report) + diagnostic_rows(summary))


def overall(rows: list[dict], check: dict | None = None) -> str:
    """FAIL if a gating row fails, PASS if every gating row passes, else NOT-EVALUABLE; any disagreement with
    proposal_check makes the whole verdict NOT-EVALUABLE (the inputs are not consistent)."""
    if check and check.get("disagreements"):
        return "NOT-EVALUABLE"
    gating = [r["status"] for r in rows if r["gating"]]
    if FAIL in gating:
        return "FAIL"
    return "PASS" if gating and all(s == PASS for s in gating) else "NOT-EVALUABLE"


def proposal_reading(rows: list[dict], summary: dict) -> dict:
    """Each proposal_check boolean recomputed here by the proposal's own definition (True / False / None): rounded
    sources where proposal_check reads them, every stratum, and drift."""
    out = {}
    for r in rows:
        if r["proposal_key"] and r["gating"]:
            out[r["proposal_key"]] = r["proposal_value"] if "proposal_value" in r else as_bool(r["status"])
    strata = summary.get("strata", {})
    failures, _ = stratum_failures(strata, None)
    out["strata_within_3deg_of_bar"] = not failures
    drifts = list(summary.get("lag_drift_frames_per_minute", {}).values())
    out[f"drift_below_1_share>={BARS['drift_share']}"] = (
        None if not drifts else bool(np.mean(np.abs(drifts) < BARS["drift_limit"]) >= BARS["drift_share"]))
    return out


def cross_check(rows: list[dict], summary: dict) -> dict:
    """Disagreements between proposal_check's booleans and their recomputation here, and the declared rows whose
    reading differs from the proposal's (scope, gating or rounding)."""
    proposal = summary.get("proposal_check", {})
    mine = proposal_reading(rows, summary)
    compared, disagreements, missing = 0, [], []
    for key, value in proposal.items():
        if not (isinstance(value, bool) or value is None) or key == "strata_failures":
            continue
        if key not in mine:
            missing.append(key)
            continue
        compared += 1
        if mine[key] != value:
            disagreements.append(dict(key=key, proposal_check=value, recomputed=mine[key]))
    declared = {r["id"]: r for r in rows}
    differences = []
    strata_declared = declared.get("strata_within_3deg_of_bar", {}).get("status")
    if strata_declared is not None and proposal.get("strata_within_3deg_of_bar") is not None and \
            (strata_declared == PASS) != proposal["strata_within_3deg_of_bar"]:
        differences.append(dict(key="strata_within_3deg_of_bar", proposal_check=proposal["strata_within_3deg_of_bar"],
                                declared=strata_declared, why="the proposal judges every stratum (stream match, "
                                "frame-count difference 'other'); the declaration gates hips, fps 30 and frame-count "
                                "difference 0 / -1 and reports stream mismatch"))
    drift_key = f"drift_below_1_share>={BARS['drift_share']}"
    if proposal.get(drift_key) is False:
        differences.append(dict(key=drift_key, proposal_check=False, declared="diagnostic (not gating)",
                                why="the declaration lists drift under diagnostics only"))
    for r in rows:
        if "proposal_value" in r and r["gating"] and r["proposal_value"] != as_bool(r["status"]):
            differences.append(dict(key=r["proposal_key"], proposal_check=r["proposal_value"], declared=r["status"],
                                    why="the declared row compares the unrounded value; proposal_check the rounded one"))
    return dict(compared=compared, disagreements=disagreements, not_recomputed=missing,
                declared_differences=differences)


def markdown(rows: list[dict], verdict: str, check: dict) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        if isinstance(v, dict):
            return ", ".join(f"{k} {fmt(x)}" for k, x in v.items())
        if isinstance(v, list):
            return "; ".join(str(x) for x in v)
        return "—" if v is None else str(v)
    lines = [f"**Overall: {verdict}**", "", "| Check | Row | Value | Bar | Status | Gating | Note |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        notes = [n for n in (r.get("note"), (r.get("detail") or {}).get("reason") if isinstance(r.get("detail"), dict)
                             else None, r.get("rounded_source") and f"rounded source: {r['rounded_source']}") if n]
        lines.append(f"| {r['check']} | `{r['id']}` | {fmt(r['value'])} | {fmt(r['bar'])} | {r['status']} | "
                     f"{'yes' if r['gating'] else 'no'} | {'; '.join(notes)} |")
    lines += ["", f"Cross-check against proposal_check: {check['compared']} booleans compared, "
                  f"{len(check['disagreements'])} disagreements, not recomputed: "
                  f"{', '.join(check['not_recomputed']) or 'none'}."]
    for d in check["disagreements"]:
        lines.append(f"- DISAGREEMENT `{d['key']}`: proposal_check {d['proposal_check']}, recomputed {d['recomputed']}")
    for d in check["declared_differences"]:
        lines.append(f"- declared differs from proposal `{d['key']}`: proposal {d['proposal_check']}, declared "
                     f"{d['declared']} ({d['why']})")
    lines += ["", "Values are compared unrounded wherever the inputs carry them; a row marked 'rounded source' compares "
                  "the value as the gate's summary rounds it (a value exactly at the bar then passes)."]
    return "\n".join(lines) + "\n"


# ======================================================================================================================
# Gate v2 (docs/NEXT_PHASE_PLAN.md, "UWB lifter v2 and gate v2: declared", 2026-10-30 11:12 UTC)

V2_DECLARATION = ("docs/NEXT_PHASE_PLAN.md: UWB lifter v2 and gate v2: declared (2026-10-30, 11:12 UTC); every gate v1 "
                  "bar unchanged (declared selection and gate 03:34 UTC, amendment 1 03:59, owner 04:02)")
V2 = dict(pairs_per_video=12, candidates="learned,inside,margin:0.1,margin:0.25", secondary="head",
          h2s_noninferiority=.5, truth_anchor_declared=dict(Left=28.2, Right=29.1), pooled_lag_share=.90,
          pooled_lag_min_pairs=3, pooled_lag_bound=1, pooled_lag_gain=2., chest_up_survival=.80, conversion_rule="head",
          outside_bins=("0-0.1", "0.1-0.3", "0.3-inf"))
# Amendment 2 (12:03 UTC): the truth-anchor bars are v1's unrounded anchors from its pinned selection report
V1_OUTSIDE_FRAME = dict(key="acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v1/selection/"
                            "outside-frame-xy2-v1.json",
                        sha256="3e26180a5f6510f68adecb5bbb3d977aa03d469f29930dac38919d8533590555")
HEAD = "lifter[head]"


def _strip_proposal(rows: list[dict]) -> list[dict]:
    """Rows judged on another method than the gate summary's proposal_check: no proposal key or reading."""
    out = []
    for r in rows:
        r = dict(r, proposal_key=None)
        r.pop("proposal_value", None)
        out.append(r)
    return out


def _secondary(rows: list[dict], prefix: str) -> list[dict]:
    return [dict(r, id=f"{prefix}.{r['id']}", gating=False) for r in _strip_proposal(rows)]


def setup_rows_v2(report: dict, bundle_sha: str | None, split_sha: str | None) -> list[dict]:
    s = report.get("settings", {})
    split = report.get("split") or {}
    secondary = {r for r in (s.get("secondary_visibility") or "").split(",") if r}
    bundle = report.get("regressors_sha256")
    return [row("Setup", "visibility_is_inside", s.get("visibility"), SELECTED_RULE,
                PASS if s.get("visibility") == SELECTED_RULE else FAIL),
            row("Setup", "candidate_rules_as_gate_v1", s.get("candidate_visibility"), V2["candidates"],
                PASS if candidate_rules(s.get("candidate_visibility")) == candidate_rules(V2["candidates"]) else FAIL),
            row("Setup", "head_scored_as_secondary", sorted(secondary), V2["secondary"],
                PASS if V2["secondary"] in secondary else FAIL),
            row("Setup", "bundle_is_lifter_v2", bundle, bundle_sha or "the declared v2 bundle (--bundle-sha256)",
                NA if not bundle_sha else PASS if bundle == bundle_sha and bundle != BUNDLE_SHA256 else FAIL),
            row("Setup", "pairs_per_video", s.get("pairs_per_video"), V2["pairs_per_video"],
                PASS if s.get("pairs_per_video") == V2["pairs_per_video"] and not s.get("videos") else FAIL,
                detail=dict(videos_limit=s.get("videos"))),
            row("Setup", "bootstrap_resamples", s.get("bootstrap"), SAMPLE["bootstrap"],
                PASS if s.get("bootstrap") == SAMPLE["bootstrap"] else FAIL),
            row("Setup", "development_videos_excluded", len(report.get("excluded_videos", [])), SAMPLE["excluded_videos"],
                PASS if len(report.get("excluded_videos", [])) == SAMPLE["excluded_videos"] else FAIL),
            row("Setup", "lag_curves_stored", bool(s.get("lag_curves")), True, PASS if s.get("lag_curves") else FAIL),
            row("Setup", "split_is_the_declared_split", split.get("sha256"), split_sha or "the split file (--split-file)",
                NA if not split_sha else PASS if split.get("sha256") == split_sha else FAIL)]


def population_rows(report: dict, split: dict | None, split_sha: str | None, pairs_index: dict | None,
                    training: dict | None, bundle_sha: str | None) -> list[dict]:
    """Held-out channels only, verified disjoint from the training and lifter-val lists, and the training data and
    bundle provably built on this split."""
    if split is None:
        return [row("Population", rid, None, bar, NA, detail=dict(reason="no split file given"))
                for rid, bar in (("held_out_only", "every gated video held out"),
                                 ("split_disjoint", "no shared channel or video"),
                                 ("pairs_on_this_split", "pairs index of this split"),
                                 ("bundle_trained_on_this_split", "training report of this bundle and split"))]
    held = set(split["held_out"]["videos"])
    other = set(split["train"]["videos"]) | set(split["lifter_val"]["videos"])
    channels = set(split["train"]["channels"]) | set(split["lifter_val"]["channels"])
    videos = {p.get("video_id") for p in report.get("pairs", [])}
    outside = sorted(v for v in videos if v not in held or v in other)
    shared = dict(channels=len(set(split["held_out"]["channels"]) & channels), videos=len(held & other))
    out = [row("Population", "held_out_only", len(outside), 0, PASS if videos and not outside else FAIL if outside else NA,
               detail=dict(videos=len(videos), not_held_out=outside[:20])),
           row("Population", "split_disjoint", shared, "0 channels, 0 videos",
               PASS if not any(shared.values()) else FAIL)]
    if pairs_index is None:
        out.append(row("Population", "pairs_on_this_split", None, split_sha, NA,
                       detail=dict(reason="no pairs index given (build_uwb_lifter_pairs.py pairs-index.json)")))
    else:
        out.append(row("Population", "pairs_on_this_split", pairs_index.get("split_sha256"), split_sha,
                       PASS if pairs_index.get("split_sha256") == split_sha else FAIL))
    if training is None:
        out.append(row("Population", "bundle_trained_on_this_split", None, split_sha, NA,
                       detail=dict(reason="no training report given (train_uwb_lifter_v2.py validity report)")))
    else:
        ok = training.get("split_sha256") == split_sha and training.get("bundle_sha256") == bundle_sha
        out.append(row("Population", "bundle_trained_on_this_split",
                       dict(split=training.get("split_sha256"), bundle=training.get("bundle_sha256")),
                       dict(split=split_sha, bundle=bundle_sha), PASS if ok else FAIL))
    return out


def _sample(report: dict) -> set:
    return {(p.get("video_id"), p.get("clip_id")) for p in report.get("pairs", [])}


def _evaluated(report: dict) -> set:
    return {(p.get("video_id"), p.get("clip_id")) for p in report.get("pairs", []) if p.get("status") == "evaluated"}


# gate settings that may differ between the v2 run and its v1 reference (amendment 2: every other setting must match)
V1_REFERENCE_MAY_DIFFER = frozenset(("regressors", "regressors_sha256", "secondary_visibility", "out", "workers",
                                     "members_cache"))


def v1_reference_rows(report: dict, v1: dict | None) -> list[dict]:
    """The xy2 bundle on the identical held-out sample: a gating setup row (not-evaluable unless present and identical)
    and its accuracy medians beside v2's (reported)."""
    rid, bar = "v1_reference_on_identical_sample", "xy2 bundle, same split, settings and pairs"
    if v1 is None:
        return [row("v1 reference", rid, None, bar, NA, detail=dict(reason="no v1 reference report given"))]
    s, s1 = report.get("settings", {}), v1.get("settings", {})
    differs = sorted(k for k in set(s) | set(s1) if k not in V1_REFERENCE_MAY_DIFFER and s.get(k) != s1.get(k))
    problems = []
    if v1.get("regressors_sha256") != BUNDLE_SHA256:
        problems.append(f"bundle {v1.get('regressors_sha256')} is not xy2")
    if (v1.get("split") or {}).get("sha256") != (report.get("split") or {}).get("sha256"):
        problems.append("another split")
    if differs:
        problems.append(f"settings differ: {differs}")
    if sorted(v1.get("excluded_videos", [])) != sorted(report.get("excluded_videos", [])):
        problems.append("other excluded videos")
    if _sample(v1) != _sample(report):
        problems.append(f"another sample ({len(_sample(v1))} pairs against {len(_sample(report))})")
    elif _evaluated(v1) != _evaluated(report):
        problems.append(f"other evaluated pairs ({len(_evaluated(v1))} against {len(_evaluated(report))})")
    out = [row("v1 reference", rid, len(_sample(v1)), bar, NA if problems else PASS,
               detail=dict(reason="; ".join(problems) or None))]
    e2 = report["summary"]["methods"][report["summary"]["primary"]]
    e1 = (v1.get("summary") or {}).get("methods", {}).get((v1.get("summary") or {}).get("primary"), {})
    for g in GROUPS:
        a, b = e1.get("common", {}).get(g, {}), e2["common"].get(g, {})
        out.append(row("v1 reference", f"v1_reference.median.{g}", a.get("median"), f"v2 {b.get('median')}",
                       "reported", gating=False, rounded=ROUNDED_2DP,
                       detail=dict(v1=dict(median=a.get("median"), p90=a.get("p90"), ci95=a.get("median_ci95_by_video")),
                                   v2=dict(median=b.get("median"), p90=b.get("p90"),
                                           ci95=b.get("median_ci95_by_video")))))
    return out


def h2s_noninferiority_rows(h2s: dict | None, h2s_v1: dict | None, bundle: str | None, primary: str,
                            candidates) -> list[dict]:
    """Per group: v2's How2Sign-val median <= v1's + 0.5 deg (same primary method and candidate rules as the gate;
    each measured on >= 200 values from >= 5 videos)."""
    reason = None
    for name, x, sha in (("v2", h2s, bundle), ("v1", h2s_v1, BUNDLE_SHA256)):
        if x is None:
            reason = reason or f"no How2Sign-val medians of the {name} bundle given"
        elif x.get("regressors_sha256") != sha:
            reason = reason or f"the {name} How2Sign-val medians are of bundle {x.get('regressors_sha256')}, not {sha}"
        elif x["summary"].get("primary") != primary:
            reason = reason or f"the {name} How2Sign-val primary method is {x['summary'].get('primary')}, not {primary}"
        elif candidate_rules(x.get("candidate_visibility")) != candidate_rules(candidates):
            reason = reason or f"the {name} How2Sign-val candidate rules differ from the gate run's"
    out = []
    bar = V2["h2s_noninferiority"]
    for g in GROUPS:
        why, value = reason, None
        refs = {}
        if not why:
            for name, x in (("v2", h2s), ("v1", h2s_v1)):
                refs[name] = x["summary"]["methods"].get(primary, {}).get("common", {}).get(g, {})
                if refs[name].get("n", 0) < MIN_VALUES or refs[name].get("videos", 0) < MIN_VIDEOS:
                    why = f"{name} {g}: {refs[name].get('n', 0)} values from {refs[name].get('videos', 0)} videos"
        if not why:
            value = refs["v2"]["median"] - refs["v1"]["median"]
        out.append(row("How2Sign non-inferiority", f"how2sign_val_v2_minus_v1.{g}<={bar}", value, bar,
                       at_most(value, bar), rounded="difference of two 2-dp medians",
                       detail=dict(v2=refs.get("v2", {}).get("median"), v1=refs.get("v1", {}).get("median"),
                                   reason=why)))
    return out


def truth_anchor_rows(outside: dict | None, bundle: str | None, outside_v1: dict | None = None) -> list[dict]:
    """Full-frame two-view forearm error of the v2 bundle (the ungated lifter on full How2Sign frames with the whole
    arm in view: ``uwb_outside_frame_accuracy.anchor``) <= v1's, per side. The bars are read unrounded from v1's pinned
    selection report (amendment 2: 28.24 left, 29.07 right; the declaration wrote 28.2 and 29.1)."""
    reason = None if outside is not None else "no outside-frame report of the v2 bundle given"
    if not reason and outside.get("regressors_sha256") != bundle:
        reason = f"outside-frame report of bundle {outside.get('regressors_sha256')}, the gate ran {bundle}"
    if not reason and outside_v1 is None:
        reason = "no v1 outside-frame report given (the bars: lifter-v1/selection/outside-frame-xy2-v1.json)"
    if not reason and outside_v1.get("regressors_sha256") != BUNDLE_SHA256:
        reason = f"the v1 outside-frame report is of bundle {outside_v1.get('regressors_sha256')}, not xy2"
    out = []
    for side in ("Left", "Right"):
        a = {} if reason else (outside.get("anchor") or {}).get(side, {}).get("Forearm", {})
        b = {} if reason else (outside_v1.get("anchor") or {}).get(side, {}).get("Forearm", {})
        value = a.get("median") if a.get("n") else None
        bar = b.get("median") if b.get("n") else None
        why = reason or (None if bar is not None else f"v1's {side} forearm anchor is unmeasured")
        status = NA if why else at_most(value, bar)
        out.append(row("Truth anchor", f"full_frame_forearm_median.{side}<=v1", value, bar, status,
                       rounded="outside-frame reports, 2 dp", detail=dict(
                           n=a.get("n"), videos=a.get("videos"), ci95=a.get("median_ci95"), reason=why,
                           v1=dict(n=b.get("n"), median=bar), declared_rounded=V2["truth_anchor_declared"][side])))
    return out


def _outside_cell(tight: dict, decision: str, side: str) -> dict:
    return ((tight.get("by_decision") or {}).get(decision) or {}).get(side, {}).get("Forearm", {}).get("outside", {})


def chest_up_head_rows(outside: dict | None, bundle: str | None) -> list[dict]:
    """Per side, on the tight crops of How2Sign val signers 1 and 2 (``uwb_outside_frame_accuracy --tight-summary``),
    out-of-frame forearms (distance > 0): head survival >= 0.8x MediaPipe's where MediaPipe keeps >= 200 such frames
    from >= 5 videos; otherwise head survival >= ``inside``'s and ``head`` admissible (amendment 1, on those crops)."""
    reason = None
    tight = (outside or {}).get("tight")
    if outside is None:
        reason = "no outside-frame report of the v2 bundle given"
    elif outside.get("regressors_sha256") != bundle:
        reason = f"outside-frame report of bundle {outside.get('regressors_sha256')}, the gate ran {bundle}"
    elif not tight:
        reason = "the outside-frame report has no tight-crop summary (--tight-summary)"
    elif not all(d in tight.get("by_decision", {}) for d in ("head", "inside", "mediapipe")):
        reason = "head, inside or MediaPipe not measured on the tight crops"
    out = []
    for side in ("Left", "Right"):
        value, bar, status, detail = None, f">= {V2['chest_up_survival']}x MediaPipe", NA, dict(reason=reason)
        if not reason:
            head, mp, inside = (_outside_cell(tight, d, side) for d in ("head", "mediapipe", "inside"))
            mp_frames = round((mp.get("frames") or 0) * (mp.get("survival") or 0))
            mp_videos = (mp.get("kept") or {}).get("videos", 0)
            admissible = (tight.get("verdicts") or {}).get("head", {}).get("admissible")
            detail = dict(head_survival=head.get("survival"), mediapipe_survival=mp.get("survival"),
                          inside_survival=inside.get("survival"), outside_frames=head.get("frames"),
                          mediapipe_kept_frames=mp_frames, mediapipe_kept_videos=mp_videos, head_admissible=admissible,
                          head_failures=(tight.get("verdicts") or {}).get("head", {}).get("failures"))
            if head.get("survival") is None:
                detail["reason"] = "no out-of-frame forearm on the tight crops"
            elif mp_frames >= MIN_VALUES and mp_videos >= MIN_VIDEOS:
                value = head["survival"] / mp["survival"]
                status = at_least(value, V2["chest_up_survival"])
                detail["branch"] = "MediaPipe measured: ratio"
            else:
                # amendment 2: head must keep MORE than inside and admit >= 1 measured outside bin, else not-evaluable;
                # an inadmissible head fails (the declaration's admissibility)
                cells = ((tight.get("by_decision") or {}).get("head") or {}).get(side, {}).get("Forearm", {})
                admitted = [b for b in V2["outside_bins"] if (cells.get(b) or {}).get("admitted")
                            and not (cells.get(b) or {}).get("unmeasured")]
                value = head["survival"]
                bar = f"> inside's survival ({inside.get('survival')}), >= 1 admitted measured outside bin, admissible"
                beats = inside.get("survival") is not None and value > inside["survival"]
                status = FAIL if admissible is False else PASS if (admissible and beats and admitted) else NA
                detail.update(branch=f"MediaPipe keeps {mp_frames} frames from {mp_videos} videos: inside, admitted "
                                     "bins and admissibility", admitted_measured_outside_bins=admitted)
                if status == NA:
                    detail["reason"] = ("head keeps no more out-of-frame forearms than inside" if not beats
                                        else "head admits no measured outside bin")
        out.append(row("Chest-up survival (head)", f"chest_up_head_forearm_survival.{side}", value, bar, status,
                       rounded="ratio or value of 4-dp survivals (uwb_outside_frame_accuracy)", detail=detail))
    return out


def pooled_lag_rows(pairs: list | None, *, gating: bool = True, prefix: str = "") -> list[dict]:
    """Timing, redefined: per video with >= 3 arm-aligned stream-match pairs, the pooled lag (summed per-lag cost
    curves, ``uwb_lifter_gate.pooled_video_lags``) has |lag| <= 1 for >= 90% of videos; and no video's pooled lag beats
    lag 0 by more than 2 deg at |lag| > 1. A video without a common lag counts outside 1; one whose common lags lack 0
    counts as beating it."""
    import uwb_lifter_gate as gate
    lags = gate.pooled_video_lags(pairs or [], min_pairs=V2["pooled_lag_min_pairs"])
    share_id = f"{prefix}pooled_lag_within_1_share>={V2['pooled_lag_share']}"
    gain_id = f"{prefix}pooled_lag_beyond_1_gain_over_lag0<={V2['pooled_lag_gain']}"
    if not lags:
        reason = "no video with >= 3 arm-aligned stream-match pairs carrying lag curves"
        return [row("Timing", share_id, None, V2["pooled_lag_share"], NA, gating=gating, detail=dict(reason=reason)),
                row("Timing", gain_id, None, "0 videos", NA, gating=gating, detail=dict(reason=reason))]
    within = [v for v, x in lags.items() if x["lag"] is not None and abs(x["lag"]) <= V2["pooled_lag_bound"]]
    share = len(within) / len(lags)
    offenders = {v: x["gain_over_lag0"] for v, x in lags.items() if x["lag"] is not None
                 and abs(x["lag"]) > V2["pooled_lag_bound"]
                 and (x["gain_over_lag0"] is None or x["gain_over_lag0"] > V2["pooled_lag_gain"])}
    histogram = {}
    for x in lags.values():
        histogram[str(x["lag"])] = histogram.get(str(x["lag"]), 0) + 1
    gains = [x["gain_over_lag0"] for x in lags.values() if x["lag"] is not None
             and abs(x["lag"]) > V2["pooled_lag_bound"] and x["gain_over_lag0"] is not None]
    return [row("Timing", share_id, share, V2["pooled_lag_share"], at_least(share, V2["pooled_lag_share"]),
                gating=gating, detail=dict(videos=len(lags), within_1=len(within), pooled_lags=histogram)),
            row("Timing", gain_id, len(offenders), "0 videos", PASS if not offenders else FAIL, gating=gating,
                detail=dict(offenders=dict(sorted(offenders.items())[:20]), largest_gain_beyond_1=max(gains)
                            if gains else None))]


def lifter_val_gap_rows(report: dict, pose_report: dict | None) -> list[dict]:
    """Circularity control (reported): held-out minus lifter-val arm medians. Lifter-val is the trainer's direction
    error on every frame where our bone is valid; held-out is the gate's own-frame error of the inside and head
    methods (the closest gate quantities)."""
    if pose_report is None:
        return [row("Circularity", "lifter_val_gap", None, "reported", "reported", gating=False,
                    detail=dict(reason="no pose training report given"))]
    lv = (pose_report.get("lifter_val") or {}).get("pose") or {}
    out = []
    methods = report["summary"]["methods"]
    for g in ("upper_arms", "forearms"):
        held = {m: methods.get(m, {}).get("own", {}).get(g, {}).get("median") for m in (f"lifter[{SELECTED_RULE}]", HEAD)}
        value = {m: None if v is None or lv.get(g, {}).get("median") is None else round(v - lv[g]["median"], 2)
                 for m, v in held.items()}
        out.append(row("Circularity", f"held_out_minus_lifter_val.{g}", value, "reported", "reported", gating=False,
                       detail=dict(lifter_val=lv.get(g), held_out_own=held)))
    return out


def channel_ci_rows(e: dict) -> list[dict]:
    """Amendment 2 (reported): beside every gated by-video interval, the 95% interval of the median clustered by
    channel (2,000 resamples of channels; ``uwb_lifter_gate.summarise`` with channels)."""
    out = []
    for g, bar in BARS["median"].items():
        s = e["common"].get(g, {})
        ci = s.get("median_ci95_by_channel")
        out.append(row("Accuracy, medians", f"channel_ci.median.{g}", ci and ci[1], f"reported (gating: by-video "
                       f"{(s.get('median_ci95_by_video') or [None, None])[1]} <= {bar + BARS['ci_slack']})", "reported",
                       gating=False, rounded=ROUNDED_2DP,
                       detail=dict(ci95_by_channel=ci, ci95_by_video=s.get("median_ci95_by_video"),
                                   reason=None if ci else "no channel interval in the gate summary")))
    return out


def build_rows_v2(report: dict, *, v1: dict | None = None, h2s: dict | None = None, h2s_v1: dict | None = None,
                  downstream: dict | None = None, outside: dict | None = None, split: dict | None = None,
                  split_sha: str | None = None, pairs_index: dict | None = None, training: dict | None = None,
                  pose_report: dict | None = None, bundle_sha: str | None = None,
                  gate_sha: str | None = None, outside_v1: dict | None = None) -> list[dict]:
    summary = report["summary"]
    primary = summary["primary"]
    if primary not in summary["methods"]:
        raise SystemExit(f"the primary method {primary} is not in the gate summary")
    e = summary["methods"][primary]
    settings = report.get("settings", {})
    bundle = report.get("regressors_sha256")
    rows = setup_rows_v2(report, bundle_sha, split_sha)
    rows += population_rows(report, split, split_sha, pairs_index, training, bundle_sha)
    rows += v1_reference_rows(report, v1)
    rows += accuracy_rows(e) + channel_ci_rows(e)
    rows += relative_rows(e, h2s, bundle, primary, settings.get("candidate_visibility"))
    rows += h2s_noninferiority_rows(h2s, h2s_v1, bundle, primary, settings.get("candidate_visibility"))
    rows += truth_anchor_rows(outside, bundle, outside_v1) + offset_rows(e) + strata_rows(summary)
    if HEAD in summary["methods"]:
        rows += [dict(r, note=f"method {HEAD}") for r in _strip_proposal(validity_core_rows(summary["methods"][HEAD]))]
    else:
        rows += [row("Validity", rid, None, "", NA, detail=dict(reason=f"{HEAD} not in the gate summary"))
                 for rid in ("hand_agreement", "forearm_agreement_hips_visible", "uwb_only_rate")]
    rows += _secondary(validity_core_rows(e), "inside")
    rows += chest_up_head_rows(outside, bundle)
    mc = timing_rows(e, summary, report.get("pairs"))[0]
    rows += [mc] + pooled_lag_rows(report.get("pairs"))
    if v1 is not None:
        rows += pooled_lag_rows(v1.get("pairs"), gating=False, prefix="v1_reference.")
    rows += downstream_rows(downstream, gate_sha, report, expected_visibility=V2["conversion_rule"])
    if HEAD in summary["methods"]:
        rows += _secondary(accuracy_rows(dict(summary["methods"][HEAD], common=summary["methods"][HEAD]["own"])),
                           "head_kept")
    rows += lifter_val_gap_rows(report, pose_report) + diagnostic_rows(summary)
    return rows


def main_v2(args, store) -> int:
    report, gate_sha = load(store, args.gate_report)
    v1, v1_sha = load(store, args.v1_reference)
    h2s, h2s_sha = load(store, args.how2sign_val)
    h2s_v1, h2s_v1_sha = load(store, args.how2sign_val_v1)
    downstream, ds_sha = load(store, args.downstream)
    outside, of_sha = load(store, args.outside_frame)
    split, split_sha = load(store, args.split_file)
    pairs_index, pi_sha = load(store, args.pairs_index)
    training, tr_sha = load(store, args.training_report)
    pose_report, pr_sha = load(store, args.pose_report)
    outside_v1, of1_sha = load(store, args.outside_frame_v1)
    if outside_v1 is not None and of1_sha != V1_OUTSIDE_FRAME["sha256"]:
        raise SystemExit(f"--outside-frame-v1: SHA-256 {of1_sha} is not the pinned {V1_OUTSIDE_FRAME['sha256']}")
    rows = build_rows_v2(report, v1=v1, h2s=h2s, h2s_v1=h2s_v1, downstream=downstream, outside=outside, split=split,
                         split_sha=split_sha, pairs_index=pairs_index, training=training, pose_report=pose_report,
                         bundle_sha=args.bundle_sha256, gate_sha=gate_sha, outside_v1=outside_v1)
    check = cross_check(rows, report["summary"])
    verdict = overall(rows, check)
    out = dict(gate_version=2, verdict=verdict, rows=rows, cross_check=check, bars=BARS, bars_v2=V2,
               gating_strata=GATING_STRATA,
               inputs=dict(gate_report=[args.gate_report, gate_sha], v1_reference=[args.v1_reference, v1_sha],
                           how2sign_val=[args.how2sign_val, h2s_sha], how2sign_val_v1=[args.how2sign_val_v1, h2s_v1_sha],
                           downstream=[args.downstream, ds_sha], outside_frame=[args.outside_frame, of_sha],
                           split_file=[args.split_file, split_sha], pairs_index=[args.pairs_index, pi_sha],
                           training_report=[args.training_report, tr_sha], pose_report=[args.pose_report, pr_sha],
                           outside_frame_v1=[args.outside_frame_v1, of1_sha]),
               declaration=V2_DECLARATION)
    return finish(args, store, out, rows, verdict, check)


def finish(args, store, out: dict, rows: list[dict], verdict: str, check: dict) -> int:
    text = json.dumps(out, indent=1, sort_keys=True, default=str)
    Path(args.out).write_text(text)
    table = markdown(rows, verdict, check)
    if args.markdown:
        Path(args.markdown).write_text(table)
    if args.output_key:
        store.put(args.output_key, text.encode())
    if args.markdown_key:
        store.put(args.markdown_key, table.encode())
    print(table)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--gate-report", required=True)
    p.add_argument("--how2sign-val")
    p.add_argument("--downstream")
    p.add_argument("--outside-frame")
    p.add_argument("--out", required=True, help="local JSON")
    p.add_argument("--markdown", help="local Markdown table")
    p.add_argument("--output-key", help="also store the JSON here (refused if it exists)")
    p.add_argument("--markdown-key", help="also store the Markdown table here (refused if it exists)")
    p.add_argument("--gate-version", type=int, choices=(1, 2), default=1,
                   help="2: gate v2 (plan 2026-10-30 11:12 UTC); 1 (default): the gate v1 verdict, unchanged")
    v2 = p.add_argument_group("gate v2 inputs")
    v2.add_argument("--v1-reference", help="the gate run of the xy2 bundle on the identical held-out sample")
    v2.add_argument("--how2sign-val-v1", help="uwb_how2sign_val_medians.py of the xy2 bundle")
    v2.add_argument("--split-file", help="build_uwb_lifter_split.py output")
    v2.add_argument("--pairs-index", help="build_uwb_lifter_pairs.py pairs-index.json")
    v2.add_argument("--training-report", help="train_uwb_lifter_v2.py validity report (the gated bundle's)")
    v2.add_argument("--pose-report", help="train_uwb_lifter_v2.py pose report of the selected arm (lifter-val)")
    v2.add_argument("--bundle-sha256", help="the declared v2 bundle's SHA-256")
    v2.add_argument("--outside-frame-v1", help="v1's pinned selection report (truth-anchor bars), SHA-checked")
    args = p.parse_args(argv)
    v2_only = [a for a in ("v1_reference", "how2sign_val_v1", "split_file", "pairs_index", "training_report",
                           "pose_report", "bundle_sha256", "outside_frame_v1") if getattr(args, a)]
    if args.gate_version == 1 and v2_only:
        raise SystemExit(f"--{v2_only[0].replace('_', '-')} is a gate v2 input (--gate-version 2)")
    from train_text_to_motion import make_store
    store = make_store(args.store) if args.store != "none" else None
    for key in (args.output_key, args.markdown_key):
        if key and store.exists(key):
            raise SystemExit(f"{key} exists; a rerun writes elsewhere")
    if args.gate_version == 2:
        return main_v2(args, store)
    report, gate_sha = load(store, args.gate_report)
    h2s, h2s_sha = load(store, args.how2sign_val)
    downstream, ds_sha = load(store, args.downstream)
    outside, of_sha = load(store, args.outside_frame)
    rows = build_rows(report, h2s=h2s, downstream=downstream, outside=outside, gate_sha=gate_sha)
    check = cross_check(rows, report["summary"])
    verdict = overall(rows, check)
    out = dict(verdict=verdict, rows=rows, cross_check=check, bars=BARS, gating_strata=GATING_STRATA,
               inputs=dict(gate_report=[args.gate_report, gate_sha], how2sign_val=[args.how2sign_val, h2s_sha],
                           downstream=[args.downstream, ds_sha], outside_frame=[args.outside_frame, of_sha]),
               declaration="docs/NEXT_PHASE_PLAN.md: UWB lifter declared selection and gate (2026-10-30 03:34 UTC), "
                           "amendment 1 (03:59), owner (04:02), selection result inside (04:20)")
    text = json.dumps(out, indent=1, sort_keys=True, default=str)
    Path(args.out).write_text(text)
    table = markdown(rows, verdict, check)
    if args.markdown:
        Path(args.markdown).write_text(table)
    if args.output_key:
        store.put(args.output_key, text.encode())
    if args.markdown_key:
        store.put(args.markdown_key, table.encode())
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
