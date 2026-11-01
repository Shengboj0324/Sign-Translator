"""Path read, follow-up: which part of the monocular path loses v1's reading, and does an instrument never fine-tuned on two-view reproduce the loss? Report only: nothing trains, no generated form and no composition is read, and this code gates and selects nothing. ``scripts/path_read.py``'s lookups, conforming, cut and guards are reused unchanged.

The first read (``round20/path-read/report.json``, sha 67f858b8) was case (b) on both metrics: marks-free 2.653 against 4.621, likelihood minus blank +0.041 against +0.234. Swapping marks barely moved either path and trimming changed nothing, so the loss sits in the values. It could not say which values, or whether fine-tuning on two-view explains it.

**Question 1, which part.** v1, D20, cut to each two-view clip's frames, every form under two-view's marks, so only values differ. A channel-level factorial: ``real_two_view``; ``mono_body_two_view_hands``; ``two_view_body_mono_hands``; ``monocular_values_two_view_marks``. A part's share is what swapping it alone loses, over what swapping both loses, under one joint bootstrap (n 2,000, seed 0); a + b − 1, the two shares less one, is reported with its interval and shows what the swaps do together.

Each metric is read alone (BLEU, and likelihood minus blank):
- **body_channel_with_wrist:** that part's share at least 2/3 with a lower bound at least 1/3, and fingers' share under 1/3 with an upper bound under 1/2.
- **fingers:** the same with the parts exchanged.
- **both:** each share at least 1/3 with a lower bound above zero.
- Otherwise **unresolved**, and unresolved when what both lose has a lower bound at or under zero, or when the metrics fall in different cases.

The labels name what the channels hold: the body channel holds the wrist rotations, and the hands channels only the fingers, so hand orientation moves with the body. Hence a second split, reported beside the label by the same rule and setting no label: **``mono_arms_only``**, clean only because of what v1 reads. Rotations are parent-relative (hand = pelvis @ chest @ upper arm @ forearm @ wrist), so plainly swapping the arm rotations also turns the hand. But v1 reads a hand's points relative to its wrist, so holding each hand's world orientation (``wrist' = inv(forearm') @ forearm @ wrist``) leaves those points as they were: in v1's input, shoulders, elbows and wrist positions come from the monocular clip and every hand-local point from two-view. ``mono_hand_pose_only`` is the reverse. ``mono_hand_orientation_only`` holds two-view's arms and fingers and takes only each hand's world orientation from the monocular clip. These are forms v1 can read, not states either path produced. A read-time check compares each clip's arm and hand-local points with their sources in v1's view; a miss above 0.001 raises an ALERT.

**Question 2, does an instrument not fine-tuned on two-view reproduce the loss?** Marks free on the full clips, both forms all-valid: ``nearest_two_view`` and ``nearest_monocular``, read through v1 and through ``back-translator-kp2d-uwb-v2`` at step 76,000, sha ``f1ff9446...``, the checkpoint its best.json names, with ``uwb_real_two_view`` and ``uwb_blank`` beside them. ``instruments`` checks the run records.

**This instrument is v1's parent, not an independent reader:** ``ft-v1``'s run record names this key and hash as its ``init_key`` and ``init_sha256``, so it is also v1's pre-fine-tune weights, and there is one such instrument here, not two. ``back-translator-kp2d-h2s-v1`` trained on the same How2Sign FK keypoints as the fine-tune and is not used.
- **Trained on:** 323,978 UWB YouTube-ASL clips of MediaPipe Holistic 2D image landmarks from video (``keypoints2d.from_holistic``), with a pretrained T5 decoder (``google/flan-t5-base``), selected on the UWB monitor split at 6.09. How2Sign FK keypoints were its validation monitor only.
- **Which path it might favour is unknown, and may be two-view.** It trained on MediaPipe image xy. The monocular path uses MediaPipe world points, IK, nominal bones and orthographic FK, none of which it saw. Two-view is Sapiens triangulated in the frontal camera's frame, with the same IK, bones and FK. So neither path is its training input, and sharing a landmark source does not make monocular nearer. ``marginals`` reports each path's per-part distance to UWB's feature marginals on c02 train clips, which measures nearness and does not settle the bias.
- **Can its input see the difference?** Yes, as far as the input goes: it takes the same 144 features as v1, x, y and a mark per point, at 24 fps with the same normalisation, and ``pairing`` gives how far the paths sit apart there (first read: hands 0.587, wrists 0.231, elbows 0.128, shoulders 0.100). Whether it can read them is what the margin tests.
- **Its low How2Sign BLEU is a different measure.** The 1.1 in best.json (``val/bleu``, reported beside its read here) is zero-shot How2Sign validation at beam 1 on up to 512 clips truncated at 20 s; its UWB monitor reads 6.09 the same way. This job's reads use beam 4 on D20.

**Why a margin, not a gap over blank.** Every form here marks all its points; blank marks none, and UWB training filtered out all-zero input. So a form can read an offset O against blank from its flags alone, whatever it holds: s = (m + O) / (t + O) goes to 1 as O grows, and a content-blind instrument would read as keeping the gap. So ``uwb_*`` forms read each clip's reference and 4 others (``retrieval_candidates``, the same draw for every form), and the measure is each clip's **margin**: mean NLL of the mismatched references minus NLL of its own. Content-blind input gives a margin of zero, offset or not. v1 stays at one candidate for its control, so its share is taken over its blank.

**The positive control** (``monitor-control``): the trainer's fixed UWB monitor sample (``random.Random(1234)``, 512 clips, batch 64, beam 1, 20 s, the hash-pinned UWB index and its shards) must reproduce best.json's 6.09 within 0.05 BLEU and its monitor loss within 0.01. A miss voids Question 2 only: its label is withheld and it reads void, and Question 1 stands.

On likelihood, m_two and m_mono are the instrument's mean margins, s_u = m_mono / m_two, and s_v1 is v1's (monocular − blank) / (two-view − blank), each marks free, under one joint bootstrap:
- **Too small to read, unresolved:** m_two has a lower bound at or under zero, or s_u's interval is wider than 0.5, which could not tell a kept margin from a lost one.
- **loss_not_explained_by_fine_tune:** m_mono − m_two has an upper bound under zero, the paired s_u − s_v1 has a one-sided 95% upper bound at most 0.25 (s_u may sit below s_v1 by any amount), and s_u's upper bound is at most 0.75.
- **gap_not_reproduced:** s_u's lower bound is at least 0.75 and m_mono − m_two has an upper bound at or above zero, so a significant loss cannot read this way.
- Otherwise **unresolved**.
Neither label separates lost content from anything else: the parent shares v1's architecture, decoder and pretraining, and its own fit to UWB image landmarks is unmeasured. BLEU gets the same row over its blank and does not set the label. A label is a reading, not a rule: no step branches on it, it starts no training and no decision, and it makes no claim about composition.

**Labels are withheld** in a smoke and on an ALERT, and below 475 shared clips. On both questions: the path read's plumbing thresholds on validity, distance, the mirrored control, source hash and offset, and the floor. On Question 1 alone: ``monocular_values_two_view_marks`` off the first read (BLEU by over 0.05, likelihood by over 0.002; 2.546 and +0.097), or a read-time point check missing. On Question 2 alone: the nearest forms off the first read (4.621 and +0.234; 2.653 and +0.041), an unfitted form saying the same for every clip as its blank, or the positive control missing.

**Control:** ``real_two_view`` and ``blank`` under v1 against the pinned stored report, ``tok_probes.canary_read``'s limits: a miss, or another list, voids the read and writes a void record, and ``read`` refuses every other form without a passed control of this code and environment.

**Clips and rules:** as ``path_read``: exact D20 records, the 8 pinned train-split report shards, never C20, c00 or a test partition, and only a missing object loses a clip. ``--smoke-train 5 --smoke-longest 4 --smoke-never-tracked 2 --smoke-other-rate 1``: c02 train clips, among them one stored at a rate other than 24, the class the path read's first run met only on D20. ``merge`` accepts an existing report of this code, and ``upload`` skips a key holding the same bytes and refuses one holding others, so a retry on the same pod resumes.

    python scripts/path_parts.py instruments
    python scripts/path_parts.py preflight --out-dir W --final-key K --void-prefix V
    python scripts/path_parts.py read --forms real_two_view,blank --out-dir W/forms --instrument-cache W/i
    python scripts/path_parts.py control --out-dir W/forms --void-prefix V
    python scripts/path_parts.py read --forms <v1's parts and nearest forms> --resume --out-dir W/forms --instrument-cache W/i
    python scripts/path_parts.py monitor-control --out-dir W/forms --instrument-cache W/i
    python scripts/path_parts.py marginals --out-dir W/forms
    python scripts/path_parts.py read --forms <the uwb forms> --resume --out-dir W/forms --instrument-cache W/i
    python scripts/path_parts.py merge --out-dir W/forms --output-file W/report.json
    python scripts/path_parts.py check --report W/report.json

The job is ``runpod/fleet/path_parts.sh``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import functools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import path_read as R  # noqa: E402
import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v0 as V  # noqa: E402
import tok_v1 as T  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402
from path_read import DISTANCE_POINTS, FPS  # noqa: E402

PROBE = "path-parts"
JOB_FILE = "runpod/fleet/path_parts.sh"
PREFIX = f"{L.O}/round20/path-read-parts"
BOOT = 2000
ANCHORS = ("real_two_view", "blank")
PART_FORMS = ("monocular_values_two_view_marks", "mono_body_two_view_hands", "two_view_body_mono_hands", "mono_arms_only", "mono_hand_pose_only", "mono_hand_orientation_only")
NEAREST_FORMS = ("nearest_two_view", "nearest_monocular")
V1_FORMS = ANCHORS + PART_FORMS + NEAREST_FORMS
UNFITTED_FORMS = ("uwb_blank", "uwb_real_two_view", "uwb_nearest_two_view", "uwb_nearest_monocular")
FORMS = V1_FORMS + UNFITTED_FORMS
MONOCULAR_NEEDED = PART_FORMS + ("nearest_monocular", "uwb_nearest_monocular")
TOKENIZER_NEEDED = PART_FORMS + NEAREST_FORMS + ("uwb_nearest_two_view", "uwb_nearest_monocular")
UNFITTED_RUN = "back-translator-kp2d-uwb-v2"
UNFITTED_KEY = f"training-runs/{UNFITTED_RUN}/checkpoints/step-00076000.pt"
UNFITTED_SHA = "f1ff94466a9e12c4d281c2d3dbc281c1049bc081f9123957c5e2dacde9e1821b"
UWB_INDEX = "acquisition/2026-10-17-v1/derived/youtube-asl-uwb-keypoints-v1/shards-v1/index.json"
UWB_INDEX_SHA = "c71421b91629db0dae23584910bcad95ae104ff93a75e18e859399432fcb8700"
MONITOR = dict(seed=1234, clips=512, batch=64, seconds=20., beam=1)        # train_keypoints_to_text_t5's fixed monitor sample and evaluate
MONITOR_BLEU_TOLERANCE, MONITOR_LOSS_TOLERANCE = .05, .01
MONITOR_SMOKE_CLIPS = 64
MARGINAL_CLIPS, MARGINAL_UWB_CLIPS, MARGINAL_SEED = 100, 2000, "path-parts-marginals"
CANDIDATES = 5                                           # own reference and 4 others, uwb forms only
MINORITY_UPPER = .5
ISOLATION_TOLERANCE = 1e-3
ARM_POINTS = (0, 1, 2, 3, 4, 5, 6, 27)                   # keypoints2d: arms and each hand's wrist point, placed by the arm
HAND_LOCAL_POINTS = tuple(range(7, 27)) + tuple(range(28, 48))
HOW2SIGN_FK_INDEX = "acquisition/2026-10-17-v1/derived/full-v1/how2sign-fk-keypoints-v1/index.json"
FIT_ONE_THIRD, FIT_TWO_THIRDS = 1 / 3, 2 / 3
COMPARABLE = .25                                         # |s_u − s_v1|
KEPT = .75                                               # a share's bound
READABLE_WIDTH = .5                                      # s_u's interval
FIGURES = dict(nearest_two_view=(4.621, .234), nearest_monocular=(2.653, .041), monocular_values_two_view_marks=(2.546, .097))        # docs/NEXT_PHASE_PLAN.md, Path read result
FIGURES_OF_PARTS = ("monocular_values_two_view_marks",)
BLEU_TOLERANCE, LIKELIHOOD_TOLERANCE = .05, .002
LABEL_BODY, LABEL_FINGERS = "body_channel_with_wrist", "fingers"
LABEL_LOSS, LABEL_GAP = "loss_not_explained_by_fine_tune", "gap_not_reproduced"
PRESTATED = dict(
    body="**body_channel_with_wrist:** that part's share at least 2/3 with a lower bound at least 1/3, and fingers' share under 1/3 with an upper bound under 1/2.",
    both="**both:** each share at least 1/3 with a lower bound above zero.",
    readable="**Too small to read, unresolved:** m_two has a lower bound at or under zero, or s_u's interval is wider than 0.5, which could not tell a kept margin from a lost one.",
    loss="**loss_not_explained_by_fine_tune:** m_mono − m_two has an upper bound under zero, the paired s_u − s_v1 has a one-sided 95% upper bound at most 0.25 (s_u may sit below s_v1 by any amount), and s_u's upper bound is at most 0.75.",
    gap="**gap_not_reproduced:** s_u's lower bound is at least 0.75 and m_mono − m_two has an upper bound at or above zero, so a significant loss cannot read this way.",
    parent="**This instrument is v1's parent, not an independent reader:**",
    bias="**Which path it might favour is unknown, and may be two-view.**")
NOT_DECIDED = "report only: the header's readings are stated before the read; no rule, bar or verdict here gates anything"
DISCLOSURES = dict(
    body="The body channel holds the wrist rotations and the hands channels only the fingers, so hand orientation moves with the body in mono_body_two_view_hands and two_view_body_mono_hands. mono_arms_only, mono_hand_pose_only and mono_hand_orientation_only split what v1 sees instead.",
    arms="mono_arms_only holds each hand's world orientation by recomputing the wrist rotation, so it is a form v1 can read, not a state either path produced. Depth (z) of the hand's frame is held with it, which v1 sees only through the projection.",
    unfitted="back-translator-kp2d-uwb-v2 trained on MediaPipe image xy and never saw FK projections, IK, nominal bones or How2Sign. It is v1's parent, not an independent reader, and the direction of any bias between the paths is unknown.",
    margin="A margin needs the sentences to differ: a clip's mismatched references come from the same list, so a content-blind instrument's mean margin is zero within noise. It reads whether the clip tells its sentence from others, a narrower thing than reading it.",
    shares="s_u is a margin share and s_v1 a share over blank: v1 keeps one candidate for its control. They measure related things on different scales, so the comparison is one-sided and coarse.",
    fine_tune="v1 was fine-tuned from this same checkpoint on How2Sign FK keypoints built from the two-view manifest, selected on How2Sign val BLEU, the pool D20 comes from.",
    never_tracked="A joint its own path never tracked holds d2's neutral pose in a transplanted form, not zeros. Marks are two-view's in every part form, so a filled value can be read there: pairing.filled_under_two_view_marks counts them per part.",
    blank="Blank is V1's: the two-view clips' durations. Its length is therefore not exactly a full monocular clip's, and no separate monocular-length blank is read; the nearest forms' seconds_over_reference gives the ratio.",
    extraction="How2Sign's monocular states come from extract_video_states.py without upright_chest, so the chest is hip-gated; YouTube and Sem-Lex used the upright chest. Filled values come through the same fill in every path.")


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    try:
        return provenance.code_digest(extra_files=(JOB_FILE,))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


# ---- values: which rotations come from which path -----------------------------------------------------------------------------------

def filled_counts(trimmed, ref) -> dict:
    """{part: [frames-and-points two-view marks that the monocular clip did not track, all two-view marks]}: how much of a part form's monocular values under two-view's marks are nearest-filled."""
    from signtranslator.data.keypoints2d import from_state
    own, marks = from_state(trimmed)[1], from_state(ref)[1]
    return {part: [int((marks & ~own)[:, list(points)].sum()), int(marks[:, list(points)].sum())] for part, points in R.PARTS.items()}


def own_values(state, tokenizer):
    """``state``'s values on every frame: each joint's nearest tracked value, d2's neutral where it was never tracked (as the path read's transplants)."""
    return V.filled(state, neutral=V.neutral_pose(state, tokenizer))


def body_and_hands(body_from, hands_from, marks):
    """The body channel's values from one state, both hands channels' from another, under ``marks``. All three share their frames."""
    values = replace(hands_from, channels={**hands_from.channels, "body": body_from.channels["body"]})
    return V.with_marks(values, marks)


def forearm_world(body, side: str) -> np.ndarray:
    """(T, 3, 3) a forearm's world rotation (``canonical_skeleton.skeleton_positions``: pelvis @ chest @ upper arm @ forearm)."""
    from signtranslator.generation.canonical_skeleton import _matrices
    index = {label: k for k, label in enumerate(body.labels)}
    out = _matrices(body.values[:, index["Pelvis"]])
    for joint in ("Chest", f"{side}UpperArm", f"{side}Forearm"):
        out = out @ _matrices(body.values[:, index[joint]])
    return out


def arms_with_hand_pose(arms_from, pose_from):
    """A body channel with ``arms_from``'s rotations, except each wrist's, recomputed so the hand keeps ``pose_from``'s world orientation: ``wrist' = inv(forearm') @ forearm @ wrist``. Rotations are parent-relative, so without this an arm swap also turns the hand."""
    import torch
    from signtranslator.generation.canonical_skeleton import _matrices
    from signtranslator.pose.rotations import matrix_to_rotation_6d
    arms, pose = arms_from.channels["body"], pose_from.channels["body"]
    if arms.labels != pose.labels or arms.values.shape != pose.values.shape:
        raise SystemExit("the two body channels differ in joints or frames")
    values = arms.values.copy()
    for side in ("Left", "Right"):
        k = list(arms.labels).index(f"{side}Hand")
        hand = forearm_world(pose, side) @ _matrices(pose.values[:, k])
        wrist = np.swapaxes(forearm_world(arms, side), -1, -2) @ hand
        values[:, k] = matrix_to_rotation_6d(torch.from_numpy(np.ascontiguousarray(wrist))).numpy()
    return replace(arms, values=values)


def arms_and_hand_pose(arms_from, pose_from, marks):
    """In what v1 reads: shoulders, elbows and wrist positions from ``arms_from``, every hand-local point from ``pose_from``, under ``marks``."""
    hands = replace(pose_from, channels={**pose_from.channels, "body": arms_with_hand_pose(arms_from, pose_from)})
    return V.with_marks(hands, marks)


def orientation_only(two, mono, marks):
    """Two-view's arms and fingers, with each hand's world orientation taken from the monocular clip."""
    values = replace(two, channels={**two.channels, "body": arms_with_hand_pose(two, mono)})
    return V.with_marks(values, marks)


def view_gap(state, source, marks, points) -> float:
    """The largest difference on ``points``, in v1's normalised view, between ``state`` and ``source`` under the same marks."""
    from signtranslator.data.keypoints2d import from_state, normalise
    a, valid = from_state(state)
    b, other = from_state(V.with_marks(source, marks))
    gap = np.abs(normalise(a, valid, y_down=False) - normalise(b, other, y_down=False))[:, list(points)]
    return round(float(gap.max()), 6) if gap.size else 0.


def isolation_row(name: str, state, mono, two, marks) -> dict | None:
    """What this form's points should equal, checked in v1's view. None for a form with one source."""
    sources = dict(mono_body_two_view_hands=(mono, None), two_view_body_mono_hands=(two, None), mono_arms_only=(mono, two), mono_hand_pose_only=(two, mono), mono_hand_orientation_only=(two, None))
    if name not in sources:
        return None
    arm, hand = sources[name]
    return dict(arm=view_gap(state, arm, marks, ARM_POINTS), hand=None if hand is None else view_gap(state, hand, marks, HAND_LOCAL_POINTS))


def build_form(name: str, refs, clips, tokenizer) -> tuple[list[int], list, list, list | None]:
    """(record indices, states, the states motion is masked by, each clip's point check). Every part form is under two-view's marks on the cut frames."""
    from evaluate_text_to_motion import blanked
    every = list(range(len(refs)))
    kept = [c.index for c in clips]
    if name in ("real_two_view", "uwb_real_two_view"):
        return every, list(refs), list(refs), None
    if name in ("blank", "uwb_blank"):
        return every, blanked(refs), list(refs), None
    if name in ("nearest_two_view", "uwb_nearest_two_view"):
        return every, [T.cap_form("v0_nearest_fill", s, tokenizer) for s in refs], list(refs), None
    if name in ("nearest_monocular", "uwb_nearest_monocular"):
        return kept, [T.cap_form("v0_nearest_fill", c.conformed, tokenizer) for c in clips], [c.conformed for c in clips], None
    if name in PART_FORMS:
        states, masks, checks = [], [], []
        for c in clips:
            ref = refs[c.index]
            mono, two = own_values(c.trimmed, tokenizer), own_values(ref, tokenizer)
            if name == "monocular_values_two_view_marks":
                states.append(V.with_marks(mono, ref))
            elif name == "mono_body_two_view_hands":
                states.append(body_and_hands(mono, two, ref))
            elif name == "two_view_body_mono_hands":
                states.append(body_and_hands(two, mono, ref))
            elif name == "mono_arms_only":
                states.append(arms_and_hand_pose(mono, two, ref))
            elif name == "mono_hand_pose_only":
                states.append(arms_and_hand_pose(two, mono, ref))
            else:
                states.append(orientation_only(two, mono, ref))
            masks.append(ref)
            checks.append(isolation_row(name, states[-1], mono, two, ref))
        return kept, states, masks, checks
    raise SystemExit(f"unknown form {name}")


# ---- instruments ----------------------------------------------------------------------------------------------------------------------

def instrument_block(store) -> dict:
    """What each instrument trained on, from its own run record, and the checks this read rests on: the unfitted checkpoint is its run's best, trained from nothing on UWB alone, and is what v1 was fine-tuned from."""
    unfitted = json.loads(store.get(f"training-runs/{UNFITTED_RUN}/run.json"))
    best = json.loads(store.get(f"training-runs/{UNFITTED_RUN}/best.json"))
    v1 = json.loads(store.get(f"training-runs/{L.TRANSLATOR_RUN}/run.json"))
    a, b = unfitted["args"], v1["args"]
    problems = [what for what, ok in (("the unfitted checkpoint is not its run's best", best.get("checkpoint_key") == UNFITTED_KEY), ("the unfitted run was started from other weights", a.get("init_key") is None and a.get("init_sha256") is None),
                                      ("the unfitted run trained on more than UWB", a.get("train_index") == [UWB_INDEX]), ("the unfitted run selected on How2Sign", a.get("select_on") == "monitor" and a.get("monitor_index") == UWB_INDEX),
                                      ("v1 was not fine-tuned from this checkpoint", b.get("init_key") == UNFITTED_KEY and b.get("init_sha256") == UNFITTED_SHA), ("v1 did not fine-tune on How2Sign FK", b.get("train_index") == [HOW2SIGN_FK_INDEX]),
                                      ("the two instruments read different features", unfitted["config"].get("features") == v1["config"].get("features") and unfitted["config"].get("fps") == v1["config"].get("fps"))) if not ok]
    if problems:
        raise SystemExit(f"the instruments are not as the header states: {problems}")
    return dict(unfitted=dict(run=UNFITTED_RUN, checkpoint_key=UNFITTED_KEY, sha256=UNFITTED_SHA, step=best.get("step"), trained_on=a["train_index"], selected_on=a["select_on"], validation_monitor_only=a.get("val_index"), started_from=dict(weights=None, pretrained_decoder=a.get("t5_name")), train_clips=unfitted.get("train_clips"), validation={k: best.get("validation", {}).get(k) for k in ("val/bleu", "monitor/bleu", "monitor/loss")},
                              validation_note="val/bleu is zero-shot How2Sign validation at beam 1 on up to 512 clips truncated at 20 s; monitor/bleu is the UWB monitor at the same settings",
                              landmarks="MediaPipe Holistic 2D image landmarks from video (keypoints2d.from_holistic); never FK projections, IK, nominal bones or How2Sign", relation="v1's parent, not an independent reader", plan="docs/NEXT_PHASE_PLAN.md: best step 76,000, monitor BLEU 6.09; zero-shot How2Sign BLEU about 1.1"),
                v1=dict(run=L.TRANSLATOR_RUN, sha256=L.TRANSLATOR_SHA, trained_on=b["train_index"], selected_on=b["select_on"], started_from=b["init_key"], started_from_sha256=b["init_sha256"], train_clips=v1.get("train_clips"), landmarks="How2Sign FK keypoints from the two-view manifest (full-v1), after UWB pretraining"),
                pre_fine_tune_is_the_unfitted_checkpoint=True, same_input=dict(features=v1["config"]["features"], fps=v1["config"]["fps"], configs_identical=unfitted["config"] == v1["config"]))


def load_unfitted(store, args):
    """The unfitted instrument, SHA-pinned; no option names another. Its checkpoint is v1's size, so as ``s2_lib.instrument_payload`` it keeps a slim copy named by its hash."""
    import torch
    from score_back_translation import load_translator
    payload, digest = L.instrument_payload(store, UNFITTED_KEY, UNFITTED_SHA, args.instrument_cache)
    model, kind = load_translator(L._OneFile(payload), UNFITTED_KEY, L.TRANSLATOR_MANIFEST_SHA)
    del payload
    if kind != "keypoints":
        raise SystemExit(f"{UNFITTED_KEY}: not a 2D keypoint reader but {kind!r}")
    device = torch.device(args.device)
    return L.Reader(model.to(device).eval(), kind, dict(run=UNFITTED_RUN, checkpoint_key=UNFITTED_KEY, sha256=digest, kind=kind), device)


def reader_of(name: str, store, args, cache: dict):
    """The form's own instrument: its name says which, and no option can."""
    which = "unfitted" if name in UNFITTED_FORMS else "v1"
    if which not in cache:
        cache[which] = load_unfitted(store, args) if which == "unfitted" else P.load_reader(store, args)
    return cache[which]


# ---- the list -------------------------------------------------------------------------------------------------------------------------

def other_rate_records(store, k: int) -> list[dict]:
    """``k`` c02 train clips the pinned report rows say are stored at a rate other than 24, in sign order: a smoke must meet them."""
    pool = {r["sign"]: r for r in L.train_records(L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA), ("c02",))}
    found = []
    for key, sha in sorted(R.MONOCULAR_REPORTS.items()):
        R.check_report_key(key)
        raw = store.get(key)
        P.check_sha(raw, sha, key)
        for row in json.loads(raw)["results"]:
            if row.get("sample_id") in pool and row.get("frames") and row.get("seconds") and abs((row["frames"] - 1) / row["seconds"] - 24.) > .5:
                found.append(row["sample_id"])
    if len(found) < k:
        raise SystemExit(f"{len(found)} c02 train clips at another rate, of {k} wanted")
    records = [pool[s] for s in sorted(found)[:k]]
    L.check_train_records(records, ("c02",))
    return records


def open_list(store, args) -> tuple[list[dict], bool, dict]:
    if args.smoke_other_rate and not args.smoke_train:
        raise SystemExit("--smoke-other-rate belongs to a smoke")
    records, declared, described = R.open_list(store, args)
    if args.smoke_other_rate:
        have = {r["sign"] for r in records}
        records = list(records) + [r for r in other_rate_records(store, args.smoke_other_rate) if r["sign"] not in have]
        described = dict(described, clips=len(records), other_rate=args.smoke_other_rate)
    R.check_lookup_records(records, declared)
    return records, declared, described


# ---- commands -------------------------------------------------------------------------------------------------------------------------

def run_instruments(args) -> int:
    """The job's first step: the instruments are what the header says, from their run records. No checkpoint is fetched."""
    from train_text_to_motion import make_store
    P.check_pins()
    print(json.dumps(dict(instruments=instrument_block(make_store(args.store)))), flush=True)
    return 0


def run_preflight(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    if store.exists(args.final_key):
        raise SystemExit(f"{args.final_key} exists: this read is finished (write-once)")
    P.check_void(store, args, code_digest(), Path(args.out_dir))
    print(json.dumps(dict(preflight="clear", code_digest=code_digest())), flush=True)
    return 0


def run_read(args) -> int:
    import torch
    from train_text_to_motion import make_store
    P.check_pins()
    R.check_smoke(args.smoke_train)
    torch.set_num_threads(args.threads)
    names = [f for f in args.forms.split(",") if f]
    unknown = [f for f in names if f not in FORMS]
    if unknown:
        raise SystemExit(f"unknown forms {unknown}")
    out_dir = Path(args.out_dir)
    stored = [n for n in names if (out_dir / f"form-{n}.json").exists()]
    if stored and not args.resume:
        raise SystemExit(f"{out_dir / f'form-{stored[0]}.json'} exists (write-once)")
    names = [n for n in names if n not in stored]
    if not names:
        print(json.dumps(dict(read=0, stored=len(stored))), flush=True)
        return 0
    store = make_store(args.store)
    records, declared, described = open_list(store, args)
    digest, place = code_digest(), P.environment()
    if declared and any(n not in ANCHORS for n in names):
        R.passed_control(out_dir, digest, place)
    texts, signs, groups = [r["text"] for r in records], [r["sign"] for r in records], [r["signer_component"] for r in records]
    refs = L.load_states(store, records)
    clips, lost = [], []
    if any(n in MONOCULAR_NEEDED for n in names):
        clips, lost, template = R.open_monocular(store, records, refs, declared)
        path = out_dir / "monocular.json"
        if not path.exists():
            P.write_report_once(path, dict(kind=f"{PROBE}-monocular", report_only=True, declared_read=declared, code_digest=digest, clips_loaded=len(clips), lost=lost, conform=R.conform_tally(clips, template), pairing=[dict(R.pairing_row(c, refs[c.index]), validity=R.part_validity(c.conformed), filled_under_two_view_marks=filled_counts(c.trimmed, refs[c.index])) for c in clips]))     # no form here holds the monocular marks
        print(json.dumps(dict(monocular_clips=len(clips), lost=len(lost), stored_fps=dict(Counter(str(c.slice_of["stored_fps"]) for c in clips)))), flush=True)
    tokenizer = L.load_tokenizer(store) if any(n in TOKENIZER_NEEDED for n in names) else None
    readers: dict = {}
    for name in names:
        started = time.time()
        kept, states, masks, checks = build_form(name, refs, clips, tokenizer)
        built = time.time()
        reader = reader_of(name, store, args, readers)
        here_texts = [texts[i] for i in kept]
        raw = reader.read(states, texts, M.retrieval_candidates(len(texts), CANDIDATES if name in UNFITTED_FORMS else 1, seed=0)[kept], group=args.nll_group)     # one draw over the whole list, so a lost clip moves no other clip's candidates
        validity = [R.part_validity(s) for s in states]
        report = dict(kind=f"{PROBE}-form", report_only=True, form=name, declared_read=declared, list=described, code_digest=digest, environment=place, instrument=reader.info, tokenizer=dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA, gains=None) if name in TOKENIZER_NEEDED else None,
                      signs=[signs[i] for i in kept], groups=[groups[i] for i in kept], references=here_texts, lost=lost if name in MONOCULAR_NEEDED else [], hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(), isolation=checks,
                      seconds_over_reference=[round(len(s.timestamps) / max(len(refs[i].timestamps), 1), 5) for s, i in zip(states, kept)], valid_frame_share=[R.valid_frame_share(s) for s in states], part_validity={part: [v[part] for v in validity] for part in R.PARTS},
                      motion=L.motion_statistics(states, masks), hands=P.hand_statistics(states, masks), tracked=P.tracked_statistics(states),
                      build_seconds=round(built - started, 2), read_seconds=round(time.time() - built, 2), peak_rss_bytes=P.peak_rss_bytes(), gpu_peak_reserved_bytes=P.gpu_peak_reserved_bytes())
        P.write_report_once(out_dir / f"form-{name}.json", report)
        print(json.dumps({name: dict(clips=len(kept), bleu=M.bleu_ci_fast(raw["hypotheses"], here_texts)[0], build_seconds=report["build_seconds"], read_seconds=report["read_seconds"], peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2), gpu_reserved_bytes=report["gpu_peak_reserved_bytes"])}), flush=True)
        del states, raw
    return 0


def run_control(args) -> int:
    """Exit 3, voiding the read, unless real_two_view and blank under v1 reproduce the pinned stored report, ``tok_probes``' canary limits. A smoke reads train clips, so it checks the pin and the stored report's shape only, and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    store = make_store(args.store)
    raw = store.get(P.CANARY_REPORT)
    P.check_sha(raw, P.CANARY_REPORT_SHA, P.CANARY_REPORT)
    stored = json.loads(raw)
    if any(len(stored["forms"][n]["per_clip"]["nll"]) != len(stored["signs"]) for n in ("real", "blank")):
        raise SystemExit(f"{P.CANARY_REPORT}: no per-clip NLL for real and blank")
    read = {n: json.loads((out / f"form-{n}.json").read_bytes()) for n in ANCHORS}
    if not read["real_two_view"]["declared_read"]:
        print(json.dumps(dict(control="not applicable: a smoke reads train clips; the pin and the stored report's shape are checked")), flush=True)
        return 0
    digest, place = P.check_same_run(read, "the control's forms")
    try:
        canary = P.canary_read(dict(real=read["real_two_view"], blank=read["blank"]), stored)
    except SystemExit as error:             # another list or order: as void as a miss
        canary = dict(reads=None, misses=[str(error)], reproduced=False, stored_report=P.CANARY_REPORT, stored_sha256=P.CANARY_REPORT_SHA)
    result = dict(canary, code_digest=digest, environment=place)
    path = out / "control.json"
    if not path.exists():
        P.write_report_once(path, result)
    print(json.dumps(result), flush=True)
    if result["reproduced"]:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{digest}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-void", code_digest=digest, environment=place, control=result), indent=1, sort_keys=True).encode())
    print(f"ALERT real_two_view and blank do not reproduce the stored report: {result['misses']}. The read is void: no other form follows, no retry; a rerun needs new code and a recorded cause", flush=True)
    return 3


def run_monitor_control(args) -> int:
    """The positive control: the trainer's fixed monitor sample, its evaluate, the unfitted instrument. A miss is recorded and voids Question 2 at merge; the job goes on, since Question 1 does not rest on it. ``--smoke-clips`` takes the first clips and compares nothing, and says so."""
    import random
    from torch.utils.data import DataLoader, Subset
    from signtranslator.data.keypoint_shards import KeypointShardDataset, collate_keypoints
    from train_keypoints_to_text_t5 import evaluate
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    P.check_sha(store.get(UWB_INDEX), UWB_INDEX_SHA, UWB_INDEX)
    best = json.loads(store.get(f"training-runs/{UNFITTED_RUN}/best.json"))
    reader = load_unfitted(store, args)
    dataset = KeypointShardDataset(store, [UWB_INDEX], partition="monitor", target_fps=FPS, max_seconds=MONITOR["seconds"], limit=args.smoke_clips)
    clips = sorted(random.Random(MONITOR["seed"]).sample(range(len(dataset)), min(MONITOR["clips"], len(dataset))))
    metrics = evaluate(reader.model, list(DataLoader(Subset(dataset, clips), batch_size=MONITOR["batch"], collate_fn=collate_keypoints)), reader.device, beam=MONITOR["beam"], prefix="monitor")
    stored = dict(bleu=best["validation"]["monitor/bleu"], loss=best["validation"]["monitor/loss"])
    result = dict(kind=f"{PROBE}-monitor-control", report_only=True, code_digest=code_digest(), monitor_clips=len(dataset), sample=len(clips), settings=MONITOR, bleu=round(metrics["monitor/bleu"], 4), loss=round(metrics["monitor/loss"], 4), stored=stored, limits=dict(bleu=MONITOR_BLEU_TOLERANCE, loss=MONITOR_LOSS_TOLERANCE))
    if args.smoke_clips:
        result.update(smoke=True, reproduced=None, misses=[])
    else:
        result["misses"] = [why for why, bad in (("BLEU", abs(metrics["monitor/bleu"] - stored["bleu"]) > MONITOR_BLEU_TOLERANCE), ("loss", abs(metrics["monitor/loss"] - stored["loss"]) > MONITOR_LOSS_TOLERANCE)) if bad]
        result.update(smoke=False, reproduced=not result["misses"])
    P.write_report_once(Path(args.out_dir) / "monitor_control.json", result)
    print(json.dumps({k: result[k] for k in ("bleu", "loss", "stored", "sample", "monitor_clips", "reproduced", "misses")}), flush=True)
    if result["misses"]:
        print(f"ALERT the unfitted instrument does not reproduce its monitor read: {result['misses']}. Question 2 is void; Question 1 stands", flush=True)
    return 0


def part_points(points, valid) -> dict:
    """{part: (mean, std) over marked coordinates}, or None for a part with none."""
    out = {}
    for part, picks in DISTANCE_POINTS.items():
        mask = valid[:, list(picks)]
        values = points[:, list(picks)][mask]
        out[part] = (values.mean(0), values.std(0)) if len(values) else None
    return out


def marginal_row(path_points: dict, uwb: dict) -> dict:
    """{part: how far a path's mean sits from UWB's in UWB standard deviations, and the ratio of their spreads}, mean over x and y."""
    out = {}
    for part in DISTANCE_POINTS:
        if path_points[part] is None or uwb[part] is None:
            out[part] = None
            continue
        mean, std = path_points[part]
        base_mean, base_std = uwb[part]
        floor = np.maximum(base_std, 1e-6)
        out[part] = dict(mean_in_uwb_std=round(float(np.mean(np.abs(mean - base_mean) / floor)), 4), std_ratio=round(float(np.mean(std / floor)), 4))
    return out


def run_marginals(args) -> int:
    """Report only, no reader: each path's per-part distance to UWB's feature marginals, on a fixed draw of c02 train clips, UWB's from its first train clips. How far each path sits from what the unfitted instrument trained on; it does not settle the direction of any bias."""
    import random
    from signtranslator.data.keypoint_shards import KeypointShardDataset
    from signtranslator.data.keypoints2d import from_state, normalise
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    P.check_sha(store.get(UWB_INDEX), UWB_INDEX_SHA, UWB_INDEX)
    pool = L.train_records(L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA), ("c02",))
    records = random.Random(MARGINAL_SEED).sample(sorted(pool, key=lambda r: r["sign"]), min(args.clips, len(pool)))
    R.check_lookup_records(records, False)
    refs = L.load_states(store, records)
    clips, lost, _ = R.open_monocular(store, records, refs, False)
    dataset = KeypointShardDataset(store, [UWB_INDEX], partition="train", target_fps=FPS, max_seconds=MONITOR["seconds"], limit=args.uwb_clips)
    base_points = np.concatenate([c[0] for c in dataset.clips]).astype(np.float64)
    base_valid = np.concatenate([c[1] for c in dataset.clips])
    uwb = part_points(base_points, base_valid)

    def pooled(states):
        rows = [from_state(s) for s in states]
        return part_points(np.concatenate([normalise(p, v, y_down=False) for p, v in rows]).astype(np.float64), np.concatenate([v for _, v in rows]))
    result = dict(kind=f"{PROBE}-marginals", report_only=True, code_digest=code_digest(), clips=len(clips), lost=lost, uwb_clips=len(dataset), note="distance to what the unfitted instrument trained on, not its bias", two_view=marginal_row(pooled([refs[c.index] for c in clips]), uwb), monocular=marginal_row(pooled([c.trimmed for c in clips]), uwb))
    P.write_report_once(Path(args.out_dir) / "marginals.json", result)
    print(json.dumps(dict(marginals=dict(clips=len(clips), uwb_clips=len(dataset), hands_two_view=result["two_view"]["left_hand"], hands_monocular=result["monocular"]["left_hand"]))), flush=True)
    return 0


# ---- reading ---------------------------------------------------------------------------------------------------------------------------

def draw_indices(n: int) -> np.ndarray:
    """One bootstrap over the shared clips (n 2,000, seed 0): every instrument's draws pair clip by clip."""
    return np.random.default_rng(0).integers(0, n, (BOOT, n))


def share_rows(reads: dict, blank: str) -> tuple:
    """(point by metric and form, draws by metric and form): every form of ``reads`` against ``blank``'s NLL, one joint bootstrap over the shared clips (n 2,000, seed 0)."""
    refs = reads[blank]["references"]
    idx = draw_indices(len(refs))
    base = np.asarray(reads[blank]["nll"])[:, 0]
    point, draws = dict(bleu={}, likelihood={}), dict(bleu={}, likelihood={})
    for form, read in reads.items():
        stats = M.bleu_statistics(read["hypotheses"], refs)
        dnll = M.delta_nll(base, np.asarray(read["nll"])[:, 0])
        point["bleu"][form], draws["bleu"][form] = float(M.bleu_from_statistics(stats.sum(0))), M.bleu_from_statistics(stats[idx].sum(1))
        point["likelihood"][form], draws["likelihood"][form] = float(dnll.mean()), dnll[idx].mean(1)
    return point, draws


def difference(point: dict, draws: dict, a: str, b: str) -> dict:
    return dict(point=round(point[a] - point[b], 5), ci95=P._quantiles(draws[a] - draws[b]))


def ratio(point: dict, draws: dict, top: tuple, bottom: tuple) -> dict | None:
    """(a − b) / (c − d), None when the denominator is zero."""
    down = point[bottom[0]] - point[bottom[1]]
    if down == 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        resampled = (draws[top[0]] - draws[top[1]]) / (draws[bottom[0]] - draws[bottom[1]])
    return dict(point=round((point[top[0]] - point[top[1]]) / down, 4), ci95=P._quantiles(resampled))


def part_case(total: dict, first: dict | None, second: dict | None, names: tuple) -> str:
    """One metric, the header's rule: ``names[0]``, ``names[1]``, both or unresolved. ``first`` and ``second``: each part's share of ``total``, what swapping both loses."""
    if first is None or second is None or total["ci95"][0] <= 0:
        return "unresolved"
    a, b = first["point"], second["point"]
    if a >= FIT_TWO_THIRDS and first["ci95"][0] >= FIT_ONE_THIRD and b < FIT_ONE_THIRD and second["ci95"][1] < MINORITY_UPPER:
        return names[0]
    if b >= FIT_TWO_THIRDS and second["ci95"][0] >= FIT_ONE_THIRD and a < FIT_ONE_THIRD and first["ci95"][1] < MINORITY_UPPER:
        return names[1]
    return "both" if a >= FIT_ONE_THIRD and b >= FIT_ONE_THIRD and first["ci95"][0] > 0 and second["ci95"][0] > 0 else "unresolved"


def together(point: dict, draws: dict, first: str, second: str, both: str) -> dict | None:
    """a + b − 1: the two shares less one, with its interval. Zero when the swaps' losses add up."""
    base = "real_two_view"
    down = point[base] - point[both]
    if down == 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        resampled = ((draws[base] - draws[first]) + (draws[base] - draws[second])) / (draws[base] - draws[both]) - 1
    return dict(point=round((point[base] - point[first] + point[base] - point[second]) / down - 1, 4), ci95=P._quantiles(resampled))


def joint(cases: list[str]) -> str:
    """The metrics' shared case; unresolved when they disagree."""
    return cases[0] if len(set(cases)) == 1 else "unresolved"


def parts_block(reads: dict) -> dict:
    """Question 1: on each metric, what each swap loses from real_two_view and its share of what swapping both loses; the channel split sets the label, the view split is reported the same way."""
    point, draws = share_rows({n: reads[n] for n in ANCHORS + PART_FORMS}, "blank")
    out = dict(bootstrap=dict(n=BOOT, seed=0), clips=len(reads["blank"]["signs"]))
    both = "monocular_values_two_view_marks"
    cases, view = [], []
    for metric in ("bleu", "likelihood"):
        p, d = point[metric], draws[metric]
        total = difference(p, d, "real_two_view", both)
        share = {form: ratio(p, d, ("real_two_view", form), ("real_two_view", both)) for form in ("mono_body_two_view_hands", "two_view_body_mono_hands", "mono_arms_only", "mono_hand_pose_only", "mono_hand_orientation_only")}
        out[metric] = dict(cells={f: round(p[f], 5) for f in ANCHORS + PART_FORMS}, lost_by_swapping_both=total, lost=dict(body=difference(p, d, "real_two_view", "mono_body_two_view_hands"), hands=difference(p, d, "real_two_view", "two_view_body_mono_hands"), arms=difference(p, d, "real_two_view", "mono_arms_only"), hand_pose=difference(p, d, "real_two_view", "mono_hand_pose_only"), hand_orientation=difference(p, d, "real_two_view", "mono_hand_orientation_only")),
                           share=dict(body=share["mono_body_two_view_hands"], hands=share["two_view_body_mono_hands"], arms=share["mono_arms_only"], hand_pose=share["mono_hand_pose_only"], hand_orientation=share["mono_hand_orientation_only"]),
                           together=dict(channels=together(p, d, "mono_body_two_view_hands", "two_view_body_mono_hands", both), view=together(p, d, "mono_arms_only", "mono_hand_pose_only", both)))
        out[metric]["case"] = part_case(total, share["mono_body_two_view_hands"], share["two_view_body_mono_hands"], (LABEL_BODY, LABEL_FINGERS))
        out[metric]["view_case_report_only"] = part_case(total, share["mono_arms_only"], share["mono_hand_pose_only"], ("arms", "hand_pose"))
        cases.append(out[metric]["case"])
        view.append(out[metric]["view_case_report_only"])
    out["case"], out["view_case_report_only"] = joint(cases), joint(view)
    return out


def fit_case(own: dict, share_u: dict | None, drop: dict, difference_upper: float | None) -> str:
    """One metric, the header's rule. ``own``: the unfitted instrument's own measure on two-view (its margin on likelihood); ``share_u``: monocular's share of it; ``drop``: monocular minus two-view; ``difference_upper``: the paired s_u − s_v1's one-sided 95% upper bound."""
    if share_u is None or difference_upper is None or own["ci95"][0] <= 0 or share_u["ci95"][1] - share_u["ci95"][0] > READABLE_WIDTH:
        return "unresolved"
    if drop["ci95"][1] < 0 and difference_upper <= COMPARABLE and share_u["ci95"][1] <= KEPT:
        return LABEL_LOSS
    return LABEL_GAP if share_u["ci95"][0] >= KEPT and drop["ci95"][1] >= 0 else "unresolved"


def margins(read: dict) -> np.ndarray:
    """(n,) each clip's mean NLL of the mismatched references minus NLL of its own: above zero when the clip tells its sentence from others, zero for content-blind input whatever its offset."""
    nll = np.asarray(read["nll"])
    if nll.shape[1] < 2:
        raise SystemExit(f"{read.get('form')}: no mismatched reference to take a margin over")
    return nll[:, 1:].mean(1) - nll[:, 0]


def paired_upper(top: np.ndarray, bottom: np.ndarray, other_top: np.ndarray, other_bottom: np.ndarray) -> float | None:
    """The one-sided 95% upper bound of top / bottom − other_top / other_bottom, over paired draws."""
    with np.errstate(divide="ignore", invalid="ignore"):
        values = top / bottom - other_top / other_bottom
    values = values[np.isfinite(values)]
    return round(float(np.percentile(values, 95)), 4) if len(values) else None


def fit_block(reads: dict) -> dict:
    """Question 2. Likelihood reads the unfitted instrument's margin and sets the label; BLEU gets a row over its blank; v1's share is over its blank, its one candidate."""
    idx = draw_indices(len(reads["blank"]["signs"]))
    u_point, u_draws = share_rows({n: reads[n] for n in UNFITTED_FORMS}, "uwb_blank")
    v_point, v_draws = share_rows({n: reads[n] for n in ("blank",) + NEAREST_FORMS}, "blank")
    margin = {f: margins(reads[f]) for f in UNFITTED_FORMS}
    point = {f: float(v.mean()) for f, v in margin.items()}
    draws = {f: v[idx].mean(1) for f, v in margin.items()}
    point["zero"], draws["zero"] = 0., np.zeros(BOOT)
    two, mono = "uwb_nearest_two_view", "uwb_nearest_monocular"
    out = dict(bootstrap=dict(n=BOOT, seed=0), clips=len(idx[0]), candidates=CANDIDATES, thresholds=dict(comparable=COMPARABLE, kept=KEPT, readable_width=READABLE_WIDTH))
    own, drop = difference(point, draws, two, "zero"), difference(point, draws, mono, two)
    share_u = ratio(point, draws, (mono, "zero"), (two, "zero"))
    vp, vd = v_point["likelihood"], v_draws["likelihood"]
    share_v1 = ratio(vp, vd, ("nearest_monocular", "blank"), ("nearest_two_view", "blank"))
    upper = paired_upper(draws[mono], draws[two], vd["nearest_monocular"] - vd["blank"], vd["nearest_two_view"] - vd["blank"])
    up, ud = u_point["likelihood"], u_draws["likelihood"]
    out["likelihood"] = dict(unfitted=dict(margins={f: round(point[f], 5) for f in UNFITTED_FORMS}, own_margin=own, monocular_minus_two_view=drop, monocular_share_of_own_margin=share_u, real_two_view_margin=difference(point, draws, "uwb_real_two_view", "zero"), blank_margin=difference(point, draws, "uwb_blank", "zero"),
                                           over_blank=dict(cells={f: round(up[f], 5) for f in UNFITTED_FORMS}, note="flags on against all-zero input: an offset here says nothing about content", share=ratio(up, ud, (mono, "uwb_blank"), (two, "uwb_blank")))),
                             v1=dict(cells={f: round(vp[f], 5) for f in ("blank",) + NEAREST_FORMS}, own_gap=difference(vp, vd, "nearest_two_view", "blank"), monocular_minus_two_view=difference(vp, vd, "nearest_monocular", "nearest_two_view"), monocular_share_of_own_gap=share_v1),
                             share_difference=dict(point=None if share_u is None or share_v1 is None else round(share_u["point"] - share_v1["point"], 4), one_sided_upper_95=upper), case=fit_case(own, share_u, drop, upper))
    up, ud, vp, vd = u_point["bleu"], u_draws["bleu"], v_point["bleu"], v_draws["bleu"]
    own, drop = difference(up, ud, two, "uwb_blank"), difference(up, ud, mono, two)
    share_u, share_v1 = ratio(up, ud, (mono, "uwb_blank"), (two, "uwb_blank")), ratio(vp, vd, ("nearest_monocular", "blank"), ("nearest_two_view", "blank"))
    upper = paired_upper(ud[mono] - ud["uwb_blank"], ud[two] - ud["uwb_blank"], vd["nearest_monocular"] - vd["blank"], vd["nearest_two_view"] - vd["blank"])
    out["bleu"] = dict(unfitted=dict(cells={f: round(up[f], 5) for f in UNFITTED_FORMS}, own_gap=own, monocular_minus_two_view=drop, monocular_share_of_own_gap=share_u), v1=dict(cells={f: round(vp[f], 5) for f in ("blank",) + NEAREST_FORMS}, own_gap=difference(vp, vd, "nearest_two_view", "blank"), monocular_minus_two_view=difference(vp, vd, "nearest_monocular", "nearest_two_view"), monocular_share_of_own_gap=share_v1),
                       share_difference=dict(point=None if share_u is None or share_v1 is None else round(share_u["point"] - share_v1["point"], 4), one_sided_upper_95=upper), case=fit_case(own, share_u, drop, upper))
    out["case"] = out["likelihood"]["case"]
    return out


def figure_misses(reads: dict, forms) -> list[str]:
    """v1's reads against the first read's figures, on the whole form: this run's path holds or it does not."""
    base = np.asarray(reads["blank"]["nll"])[:, 0]
    keep = reads["blank"]["signs"]
    out = []
    for form in forms:
        bleu, likelihood = FIGURES[form]
        read = reads[form]
        here = M.bleu_ci_fast(read["hypotheses"], read["references"])[0]
        where = {s: i for i, s in enumerate(keep)}
        dnll = float(np.mean([base[where[s]] - row[0] for s, row in zip(read["signs"], read["nll"])]))
        if abs(here - bleu) > BLEU_TOLERANCE:
            out.append(f"{form} reads BLEU {here}, not the first read's {bleu}")
        if abs(dnll - likelihood) > LIKELIHOOD_TOLERANCE:
            out.append(f"{form} reads likelihood {dnll:.4f}, not the first read's {likelihood}")
    return out


def constant_reads(reads: dict) -> list[str]:
    """Forms the unfitted instrument reads exactly as its blank: it sees nothing there."""
    blank = dict(zip(reads["uwb_blank"]["signs"], reads["uwb_blank"]["hypotheses"]))
    return [f"{form} reads exactly as uwb_blank" for form in ("uwb_real_two_view", "uwb_nearest_two_view", "uwb_nearest_monocular") if all(blank[s] == h for s, h in zip(reads[form]["signs"], reads[form]["hypotheses"]))]


def isolation_misses(reads: dict) -> list[str]:
    """Each form's clips whose arm or hand-local points are off their sources in v1's view."""
    out = []
    for form, read in reads.items():
        for sign, row in zip(read["signs"], read.get("isolation") or ()):
            bad = {k: v for k, v in (row or {}).items() if v is not None and v > ISOLATION_TOLERANCE}
            if bad:
                out.append(f"{form}: {sign} is off its sources in v1's view: {bad}")
    return out


def filled_shares(rows: list[dict]) -> dict:
    """Per part, the share of two-view's marks on the shared clips where the monocular clip's value is nearest-filled."""
    out = {}
    for part in R.PARTS:
        filled, marked = sum(r["filled_under_two_view_marks"][part][0] for r in rows), sum(r["filled_under_two_view_marks"][part][1] for r in rows)
        out[part] = dict(filled=filled, marked=marked, share=round(filled / marked, 5) if marked else None)
    return out


def load_reads(out_dir: Path) -> tuple[dict, dict]:
    """(every form, the monocular record): one list, kind of read, code and environment. Instruments differ by form name, and each must be that name's."""
    path = out_dir / "monocular.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing")
    plan = json.loads(path.read_bytes())
    reads, first = {}, None
    for name in FORMS:
        form = out_dir / f"form-{name}.json"
        if not form.exists():
            raise SystemExit(f"{form} is missing")
        read = json.loads(form.read_bytes())
        if read.get("form") != name:
            raise SystemExit(f"{form} holds form {read.get('form')!r}")
        signature = (read["declared_read"], read["list"]["list"])
        if first is None:
            first = signature
        elif signature != first:
            raise SystemExit(f"{form}: another list, or a smoke mixed with a declared read")
        reads[name] = read
    instruments = {n: reads[n]["instrument"]["sha256"] for n in FORMS}
    if len({instruments[n] for n in V1_FORMS}) != 1 or len({instruments[n] for n in UNFITTED_FORMS}) != 1 or instruments["blank"] == instruments["uwb_blank"]:
        raise SystemExit(f"each form must be read by its own instrument: {instruments}")
    digest, _ = P.check_same_run(reads, "the forms")
    if plan.get("code_digest") != digest or plan.get("declared_read") != reads["real_two_view"]["declared_read"]:
        raise SystemExit(f"{path}: another run's record")
    return reads, plan


def already_merged(store, args) -> bool:
    """A retry's merge: an existing report of this code stands and is left alone; any other refuses."""
    if bool(args.output_key) == bool(args.output_file):
        raise SystemExit("give exactly one of --output-key and --output-file")
    where = args.output_key or args.output_file
    if not (store.exists(where) if args.output_key else Path(where).exists()):
        return False
    try:
        kept = json.loads(store.get(where) if args.output_key else Path(where).read_bytes())
    except ValueError:
        kept = {}
    if kept.get("code_digest") != code_digest():
        raise SystemExit(f"{where} exists (write-once) and is not this code's report")
    print(json.dumps(dict(merged=where, already=True, code_digest=kept["code_digest"])), flush=True)
    return True


def run_merge(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    if already_merged(store, args):
        return 0
    out_dir = Path(args.out_dir)
    all_reads, plan = load_reads(out_dir)
    first = all_reads["real_two_view"]
    declared = first["declared_read"]
    digest, place = first["code_digest"], first["environment"]
    control = R.passed_control(out_dir, digest, place) if declared else None
    shared = R.shared_signs(all_reads)
    if not shared:
        raise SystemExit("the forms share no clip")
    monitor, marginals = record_of(out_dir / "monitor_control.json", digest, declared), record_of(out_dir / "marginals.json", digest, None)
    reads = {n: R.restrict(r, shared) for n, r in all_reads.items()}
    keep = set(shared)
    rows = [r for r in plan["pairing"] if r["sign"] in keep]
    block = dict(clips=len(rows), source_hash_matches=sum(r["source_hash_matches"] for r in rows), max_offset_seconds=max(r["max_offset_seconds"] for r in rows), offsets=R.offset_distribution(rows), filled_under_two_view_marks=filled_shares(rows),
                 group_distance={g: R.mean_or_none([R.group_distance(r["distance"], g) for r in rows]) for g in R.GROUPS}, group_mirrored={g: R.mean_or_none([R.group_distance(r["mirrored"], g) for r in rows]) for g in R.GROUPS})
    common = R.plumbing_alerts(reads["real_two_view"]["part_validity"], {part: [r["validity"][part] for r in rows] for part in R.PARTS}, block, len(shared), declared)
    first_question = isolation_misses(all_reads) + (figure_misses(all_reads, FIGURES_OF_PARTS) if declared else [])
    monitor_misses = [f"the unfitted instrument missed its monitor read: {m}" for m in monitor["misses"]]
    second_question = constant_reads(reads) + monitor_misses + (figure_misses(all_reads, NEAREST_FORMS) if declared else [])
    parts = parts_block(reads)
    fit = fit_block(reads)
    smoke = [] if declared else ["a smoke carries no reading"]
    held_first, held_second = smoke + common + first_question, smoke + common + second_question
    alerts = common + first_question + second_question
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, declared_read=declared, smoke=None if declared else "SMOKE on c02 train clips: these numbers are plumbing, not a read", list=first["list"], code_digest=digest, environment=place, disclosures=DISCLOSURES, prestated=PRESTATED, instruments=instrument_block(store),
                  alerts=alerts, which_part=dict(parts, label=None if held_first else parts["case"], view_beside_label=parts["view_case_report_only"], withheld=held_first), fit_or_content=dict(fit, label=None if held_second else fit["case"], void=bool(monitor_misses), withheld=held_second), monitor_control=monitor, marginals=marginals,
                  clips=dict(list=len(first["signs"]), shared=len(shared), floor=R.SHARED_FLOOR, outside_shared=R.lost_rows(all_reads, shared)), forms={n: form_row(reads[n], reads["uwb_blank" if n in UNFITTED_FORMS else "blank"]) for n in FORMS}, pairing=block, conform=plan["conform"], per_signer=signer_rows(reads),
                  checks=dict(blank_identical_to_real_two_view=round(float(np.mean([a == b for a, b in zip(reads["blank"]["hypotheses"], reads["real_two_view"]["hypotheses"])])), 4), control=control), rerun=json.loads((out_dir / "rerun.json").read_bytes()) if (out_dir / "rerun.json").exists() else None,
                  resources={n: {k: r[k] for k in ("peak_rss_bytes", "gpu_peak_reserved_bytes", "build_seconds", "read_seconds")} for n, r in all_reads.items()})
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, declared_read=declared, shared=len(shared), of=len(first["signs"]), lost=report["clips"]["outside_shared"]["by_reason"], forms={n: {k: v for k, v in row.items() if k in ("bleu", "likelihood_minus_blank")} for n, row in report["forms"].items()},
                          which_part=dict(bleu=parts["bleu"]["share"], likelihood=parts["likelihood"]["share"], together=dict(bleu=parts["bleu"]["together"], likelihood=parts["likelihood"]["together"]), case=parts["case"], view_beside_label=parts["view_case_report_only"], label=report["which_part"]["label"], withheld=held_first),
                          fit_or_content=dict(margin=fit["likelihood"]["unfitted"]["margins"], share=fit["likelihood"]["unfitted"]["monocular_share_of_own_margin"], share_difference=fit["likelihood"]["share_difference"], v1=fit["likelihood"]["v1"]["monocular_share_of_own_gap"], bleu=fit["bleu"]["unfitted"], val_bleu_beside=report["instruments"]["unfitted"]["validation"]["val/bleu"], case=fit["case"], label=report["fit_or_content"]["label"], void=report["fit_or_content"]["void"], withheld=held_second),
                          monitor={k: monitor.get(k) for k in ("bleu", "stored", "reproduced")}, pairing={k: block[k] for k in ("clips", "source_hash_matches", "offsets", "filled_under_two_view_marks", "group_distance", "group_mirrored")}, alerts=alerts)), flush=True)
    return 0


def record_of(path: Path, digest: str, declared) -> dict:
    """A step's record of this code; ``declared`` also holds a monitor record to its kind of read."""
    if not path.exists():
        raise SystemExit(f"{path} is missing")
    record = json.loads(path.read_bytes())
    if record.get("code_digest") != digest or (declared is not None and record.get("smoke") is bool(declared)):
        raise SystemExit(f"{path}: another run's record")
    return record


def form_row(read: dict, blank: dict) -> dict:
    bleu, ci = M.bleu_ci_fast(read["hypotheses"], read["references"])
    dnll = M.delta_nll(np.asarray(blank["nll"])[:, 0], np.asarray(read["nll"])[:, 0])
    return dict(clips=len(read["signs"]), instrument=read["instrument"]["run"], bleu=bleu, ci95=ci, likelihood_minus_blank=M.bootstrap_mean(dnll, n=BOOT), valid_frame_share=M.bootstrap_mean(read["valid_frame_share"], n=BOOT), part_validity={p: M.bootstrap_mean(v, n=BOOT)["mean"] for p, v in read["part_validity"].items()}, seconds_over_reference=round(float(np.mean(read["seconds_over_reference"])), 4))


def signer_rows(reads: dict) -> dict:
    """Per signer, the two blocks' rows on that signer's shared clips. Their shares are noisy on few clips and set no label."""
    groups = dict(zip(reads["real_two_view"]["signs"], reads["real_two_view"]["groups"]))
    out = {}
    for signer in sorted(set(groups.values())):
        signs = [s for s in reads["real_two_view"]["signs"] if groups[s] == signer]
        here = {n: R.restrict(r, signs) for n, r in reads.items()}
        parts, fit = parts_block(here), fit_block(here)
        out[signer] = dict(clips=len(signs), which_part=dict(bleu=parts["bleu"]["share"], likelihood=parts["likelihood"]["share"]), fit_or_content=dict(likelihood=fit["likelihood"]["unfitted"]["monocular_share_of_own_margin"], margin=fit["likelihood"]["unfitted"]["own_margin"], v1_share=fit["likelihood"]["v1"]["monocular_share_of_own_gap"]))
    return out


def run_upload(args) -> int:
    """Every report under ``--out-dir``. A key holding the same bytes is left, one holding others refuses the whole upload before anything is written, so a retry resumes and nothing is overwritten. ``--partial`` is ``tok_probes``': keys stored are left alone."""
    from train_text_to_motion import make_store
    if args.partial:
        return P.run_upload(args)
    store = make_store(args.store)
    out = Path(args.out_dir)
    paths = sorted(q for q in out.rglob("*") if q.is_file() and q.suffix == ".json")
    if not paths:
        raise SystemExit(f"{out}: nothing to upload")
    keys = [(q, f"{args.prefix.rstrip('/')}/{q.relative_to(out)}") for q in paths]
    stored = [(q, k) for q, k in keys if store.exists(k)]
    for q, k in stored:
        if store.get(k) != q.read_bytes():
            raise SystemExit(f"{k} exists and holds other bytes (write-once); nothing uploaded")
    done = {k for _, k in stored}
    for q, k in keys:
        if k not in done:
            store.put(k, q.read_bytes())
    print(json.dumps(dict(uploaded=len(keys) - len(done), kept=len(done), prefix=args.prefix)), flush=True)
    return 0


def run_check(args) -> int:
    """Exit 3 on broken plumbing: not report_only, blank reading as real, a declared read without its passed control, no shared clip, any alert, or a label beside one. It judges no form."""
    report = json.loads(Path(args.report).read_text())
    checks, problems = report["checks"], list(report["alerts"])
    if report.get("report_only") is not True:
        problems.append("the report does not say report_only")
    if checks["blank_identical_to_real_two_view"] >= 1:
        problems.append("blank reads exactly as real_two_view")
    if report.get("declared_read") and not (checks.get("control") or {}).get("reproduced"):
        problems.append("a declared read without its passed control")
    if not report["clips"]["shared"]:
        problems.append("no shared clip")
    if any(report[q].get("label") is not None and report[q].get("withheld") for q in ("which_part", "fit_or_content")):
        problems.append("a report holding a label beside an alert")
    print(json.dumps(dict(problems=problems)), flush=True)
    for line in problems:
        print(f"ALERT {line}", flush=True)
    return 3 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    instruments = sub.add_parser("instruments", help="the job's first step: the instruments are what the header says, from their run records")
    instruments.add_argument("--store", default="r2")
    pre = sub.add_parser("preflight", help="the read is not finished or void")
    pre.add_argument("--store", default="r2")
    pre.add_argument("--out-dir", required=True)
    pre.add_argument("--final-key", required=True)
    pre.add_argument("--void-prefix", required=True)
    pre.add_argument("--rerun-cause")
    read = sub.add_parser("read")
    read.add_argument("--store", default="r2")
    read.add_argument("--out-dir", required=True)
    read.add_argument("--forms", default=",".join(FORMS))
    read.add_argument("--resume", action="store_true", help="leave out forms already read in --out-dir")
    read.add_argument("--nll-group", type=int, default=2)
    read.add_argument("--instrument-cache")
    read.add_argument("--smoke-train", type=int, default=0, help=f"read this many c02 train clips, at most {R.SMOKE_CLIPS}, not D20")
    read.add_argument("--smoke-longest", type=int, default=0)
    read.add_argument("--smoke-never-tracked", type=int, default=0)
    read.add_argument("--smoke-other-rate", type=int, default=0, help="a smoke's list also holds this many c02 train clips stored at a rate other than 24")
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    control = sub.add_parser("control", help="real_two_view and blank against the stored report; exit 3, voiding the read, otherwise")
    control.add_argument("--store", default="r2")
    control.add_argument("--out-dir", required=True)
    control.add_argument("--void-prefix", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--store", default="r2")
    merge.add_argument("--out-dir", required=True)
    merge.add_argument("--output-key")
    merge.add_argument("--output-file")
    monitor = sub.add_parser("monitor-control", help="the unfitted instrument's fixed UWB monitor sample against best.json")
    monitor.add_argument("--store", default="r2")
    monitor.add_argument("--out-dir", required=True)
    monitor.add_argument("--instrument-cache")
    monitor.add_argument("--device", default="cpu")
    monitor.add_argument("--smoke-clips", type=int, default=0, help="take the first clips and compare nothing")
    marginals = sub.add_parser("marginals", help="report only: each path's distance to UWB's feature marginals on c02 train clips")
    marginals.add_argument("--store", default="r2")
    marginals.add_argument("--out-dir", required=True)
    marginals.add_argument("--clips", type=int, default=MARGINAL_CLIPS)
    marginals.add_argument("--uwb-clips", type=int, default=MARGINAL_UWB_CLIPS)
    check = sub.add_parser("check")
    check.add_argument("--report", required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--store", default="r2")
    upload.add_argument("--out-dir", required=True)
    upload.add_argument("--prefix", required=True)
    upload.add_argument("--partial", action="store_true")
    log = sub.add_parser("upload-log")
    log.add_argument("--store", default="r2")
    log.add_argument("--file", required=True)
    log.add_argument("--key", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return dict(instruments=run_instruments, preflight=run_preflight, read=run_read, control=run_control, merge=run_merge, check=run_check, upload=run_upload, marginals=run_marginals, **{"upload-log": R.run_upload_log, "monitor-control": run_monitor_control})[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
