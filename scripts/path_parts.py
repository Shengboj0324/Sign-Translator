"""Path read, follow-up: which part of the monocular path loses v1's reading, and is the loss v1's fit or lost content? Report only: nothing trains, no generated form and no composition is read, and this code gates and selects nothing. ``scripts/path_read.py``'s lookups, conforming, cut and guards are reused unchanged.

The first read (``round20/path-read/report.json``, sha 67f858b8) was case (b) on both metrics: marks-free 2.653 against 4.621, likelihood minus blank +0.041 against +0.234. Swapping marks barely moved either path and trimming changed nothing, so the loss sits in the values. It could not say which values, or whether v1 is only fitted to two-view input.

**Question 1, which part.** v1, D20, cut to each two-view clip's frames, every form under two-view's marks, so only values differ. A channel-level factorial: ``real_two_view``; ``mono_body_two_view_hands``; ``two_view_body_mono_hands``; ``monocular_values_two_view_marks``. A part's share is what swapping it alone loses, over what swapping both loses, under one joint bootstrap (n 2,000, seed 0).

Each metric is read alone (BLEU, and likelihood minus blank):
- **body:** body's share at least 2/3 with a lower bound at least 1/3, and hands' share under 1/3.
- **hands:** the same with the parts exchanged.
- **both:** each share at least 1/3.
- Otherwise **unresolved**, and unresolved when what both lose has a lower bound at or under zero, or when the metrics fall in different cases.

The body channel holds the wrist rotations, and the hands channels only the fingers, so hand orientation moves with the body. Hence **``mono_arms_only``**, which is clean only because of what v1 reads. Rotations are parent-relative (hand = pelvis @ chest @ upper arm @ forearm @ wrist), so plainly swapping the arm rotations also turns the hand. But v1 reads a hand's points relative to its wrist, so holding each hand's world orientation (``wrist' = inv(forearm') @ forearm @ wrist``) leaves those points exactly as they were: in v1's input, shoulders, elbows and wrist positions come from the monocular clip and every hand-local point from two-view. ``mono_hand_pose_only`` is the reverse. These two are read by the same rule and labelled report-only. The split is in what v1 sees, not in the extraction: a solved rotation is not moved between parts in the source.

**Question 2, fit or content.** Marks free on the full clips, both forms all-valid: ``nearest_two_view`` and ``nearest_monocular``, read through v1 and through an instrument never fitted to two-view keypoints, with its own blank.

The unfitted instrument is ``back-translator-kp2d-uwb-v2`` at step 76,000, sha ``f1ff9446...``, the checkpoint its best.json names. It is also v1's own pre-fine-tune weights: ``ft-v1``'s run record names this key and hash as its ``init_key`` and ``init_sha256``, and ``instruments`` checks that. So there is one unfitted instrument here, not two. ``back-translator-kp2d-h2s-v1`` is not one: it trained on the same How2Sign FK keypoints as the fine-tune.
- **Trained on:** 323,978 UWB YouTube-ASL clips, MediaPipe Holistic 2D landmarks from video (``keypoints2d.from_holistic``), selected on the UWB monitor split. How2Sign FK keypoints were its validation monitor only and never selected or trained it. Zero-shot on How2Sign it reads about 1.1 BLEU, so **likelihood is read first** and BLEU is reported beside it.
- **Can its input see the difference?** Yes, as far as the input goes: it takes the same 144 features as v1 (x, y and a mark per point of ``keypoints2d.features``), at the same 24 fps with the same normalisation, through the same FK projection, and ``pairing`` gives how far the two paths sit apart there (first read: hands 0.587, wrists 0.231, elbows 0.128, shoulders 0.100). Whether it can read them is another matter: it never saw FK's nominal bones, and its near-floor BLEU says it reads this domain poorly. The readable condition below is that test.
- **It is not neutral.** It trained on MediaPipe, the landmark source of the monocular path, so it may favour monocular. A loss that survives it is therefore the stronger result. A gap that vanishes can be v1's fit to two-view or this instrument's fit to MediaPipe, and this read cannot separate those two.

Shares: s = (monocular − blank) / (two-view − blank), an instrument's own marks-free gap, under one joint bootstrap. On likelihood, with s_u for the unfitted instrument and s_v1 for v1:
- **Too small to read, unresolved:** the unfitted instrument's own two-view minus blank has a lower bound at or under zero, or s_u's interval is wider than 0.5 (which could not tell a kept gap from a lost one).
- **More likely content:** the same direction (monocular minus two-view has an upper bound under zero), comparable (|s_u − s_v1| at most 0.25), and s_u's upper bound at most 0.75.
- **More likely v1's fit:** the gap vanishes or reverses: s_u's lower bound at least 0.75.
- Otherwise **unresolved**.
BLEU gets the same row and does not set the label, being near this instrument's floor. A label is a reading, not a rule: no step branches on it, it starts no training and no decision, and it makes no claim about composition.

**Labels are withheld** on any ALERT, in a smoke, and below 475 shared clips. ALERT: v1's reads off the first read's figures (BLEU by over 0.05, likelihood by over 0.002: nearest two-view 4.621 and +0.234, nearest monocular 2.653 and +0.041, monocular values under two-view marks 2.546 and +0.097), the path read's plumbing thresholds on validity, distance, the mirrored control, source hash and offset, or the unfitted instrument saying the same for every clip of a form as for its blank.

**Control:** ``real_two_view`` and ``blank`` under v1 against the pinned stored report, ``tok_probes.canary_read``'s limits: a miss voids the read, and ``read`` refuses every other form without a passed control of this code and environment.

**Clips and rules:** as ``path_read``: exact D20 records, the 8 pinned train-split report shards, never C20, c00 or a test partition, and only a missing object loses a clip. ``--smoke-train 5 --smoke-longest 4 --smoke-never-tracked 2 --smoke-other-rate 1``: c02 train clips, among them one stored at a rate other than 24, the class the path read's first run met only on D20.

    python scripts/path_parts.py instruments
    python scripts/path_parts.py preflight --out-dir W --final-key K --void-prefix V
    python scripts/path_parts.py read --forms real_two_view,blank --out-dir W/forms --instrument-cache W/i
    python scripts/path_parts.py control --out-dir W/forms --void-prefix V
    python scripts/path_parts.py read --forms <v1's parts and nearest forms> --resume --out-dir W/forms --instrument-cache W/i
    python scripts/path_parts.py read --forms uwb_blank,uwb_nearest_two_view,uwb_nearest_monocular --resume --out-dir W/forms --instrument-cache W/i
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

PROBE = "path-parts"
JOB_FILE = "runpod/fleet/path_parts.sh"
PREFIX = f"{L.O}/round20/path-read-parts"
BOOT = 2000
ANCHORS = ("real_two_view", "blank")
PART_FORMS = ("monocular_values_two_view_marks", "mono_body_two_view_hands", "two_view_body_mono_hands", "mono_arms_only", "mono_hand_pose_only")
NEAREST_FORMS = ("nearest_two_view", "nearest_monocular")
V1_FORMS = ANCHORS + PART_FORMS + NEAREST_FORMS
UNFITTED_FORMS = ("uwb_blank", "uwb_nearest_two_view", "uwb_nearest_monocular")
FORMS = V1_FORMS + UNFITTED_FORMS
MONOCULAR_NEEDED = PART_FORMS + ("nearest_monocular", "uwb_nearest_monocular")
TOKENIZER_NEEDED = PART_FORMS + NEAREST_FORMS + ("uwb_nearest_two_view", "uwb_nearest_monocular")
UNFITTED_RUN = "back-translator-kp2d-uwb-v2"
UNFITTED_KEY = f"training-runs/{UNFITTED_RUN}/checkpoints/step-00076000.pt"
UNFITTED_SHA = "f1ff94466a9e12c4d281c2d3dbc281c1049bc081f9123957c5e2dacde9e1821b"
UWB_INDEX = "acquisition/2026-10-17-v1/derived/youtube-asl-uwb-keypoints-v1/shards-v1/index.json"
HOW2SIGN_FK_INDEX = "acquisition/2026-10-17-v1/derived/full-v1/how2sign-fk-keypoints-v1/index.json"
FIT_ONE_THIRD, FIT_TWO_THIRDS = 1 / 3, 2 / 3
COMPARABLE = .25                                         # |s_u − s_v1|
KEPT = .75                                               # a share's bound
READABLE_WIDTH = .5                                      # s_u's interval
FIGURES = dict(nearest_two_view=(4.621, .234), nearest_monocular=(2.653, .041), monocular_values_two_view_marks=(2.546, .097))        # docs/NEXT_PHASE_PLAN.md, Path read result
BLEU_TOLERANCE, LIKELIHOOD_TOLERANCE = .05, .002
PRESTATED = dict(
    body="**body:** body's share at least 2/3 with a lower bound at least 1/3, and hands' share under 1/3.",
    both="**both:** each share at least 1/3.",
    readable="**Too small to read, unresolved:** the unfitted instrument's own two-view minus blank has a lower bound at or under zero, or s_u's interval is wider than 0.5 (which could not tell a kept gap from a lost one).",
    content="**More likely content:** the same direction (monocular minus two-view has an upper bound under zero), comparable (|s_u − s_v1| at most 0.25), and s_u's upper bound at most 0.75.",
    fit="**More likely v1's fit:** the gap vanishes or reverses: s_u's lower bound at least 0.75.",
    neutral="**It is not neutral.** It trained on MediaPipe, the landmark source of the monocular path, so it may favour monocular.")
NOT_DECIDED = "report only: the header's readings are stated before the read; no rule, bar or verdict here gates anything"
DISCLOSURES = dict(
    body="The body channel holds the wrist rotations and the hands channels only the fingers, so hand orientation moves with the body in mono_body_two_view_hands and two_view_body_mono_hands. mono_arms_only and mono_hand_pose_only split what v1 sees instead.",
    arms="mono_arms_only holds each hand's world orientation by recomputing the wrist rotation, so it is a form v1 can read, not a state either path produced. Depth (z) of the hand's frame is held with it, which v1 sees only through the projection.",
    unfitted="back-translator-kp2d-uwb-v2 trained on MediaPipe Holistic 2D landmarks from video and never on FK projections or How2Sign. It is not neutral: it may favour the monocular path's landmark source, and it reads this domain near its floor.",
    fine_tune="v1 was fine-tuned from this same checkpoint on How2Sign FK keypoints built from the two-view manifest, selected on How2Sign val BLEU, the pool D20 comes from.",
    never_tracked="A joint its own path never tracked holds d2's neutral pose in a transplanted form, not zeros. Marks are two-view's in every part form, so a filled value can be read there.",
    extraction="How2Sign's monocular states come from extract_video_states.py without upright_chest, so the chest is hip-gated; YouTube and Sem-Lex used the upright chest. Filled values come through the same fill in every path.")


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    try:
        return provenance.code_digest(extra_files=(JOB_FILE,))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


# ---- values: which rotations come from which path -----------------------------------------------------------------------------------

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


def build_form(name: str, refs, clips, tokenizer) -> tuple[list[int], list, list]:
    """(record indices, states, the states motion is masked by). Every part form is under two-view's marks on the cut frames."""
    from evaluate_text_to_motion import blanked
    every = list(range(len(refs)))
    kept = [c.index for c in clips]
    if name == "real_two_view":
        return every, list(refs), list(refs)
    if name in ("blank", "uwb_blank"):
        return every, blanked(refs), list(refs)
    if name in ("nearest_two_view", "uwb_nearest_two_view"):
        return every, [T.cap_form("v0_nearest_fill", s, tokenizer) for s in refs], list(refs)
    if name in ("nearest_monocular", "uwb_nearest_monocular"):
        return kept, [T.cap_form("v0_nearest_fill", c.conformed, tokenizer) for c in clips], [c.conformed for c in clips]
    if name in PART_FORMS:
        states, masks = [], []
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
            else:
                states.append(arms_and_hand_pose(two, mono, ref))
            masks.append(ref)
        return kept, states, masks
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
    return dict(unfitted=dict(run=UNFITTED_RUN, checkpoint_key=UNFITTED_KEY, sha256=UNFITTED_SHA, step=best.get("step"), trained_on=a["train_index"], selected_on=a["select_on"], validation_monitor_only=a.get("val_index"), started_from=None, train_clips=unfitted.get("train_clips"),
                              landmarks="MediaPipe Holistic 2D from video (keypoints2d.from_holistic); never FK projections, never How2Sign", plan="docs/NEXT_PHASE_PLAN.md: best step 76,000, monitor BLEU 6.09; zero-shot How2Sign BLEU about 1.1"),
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
            P.write_report_once(path, dict(kind=f"{PROBE}-monocular", report_only=True, declared_read=declared, code_digest=digest, clips_loaded=len(clips), lost=lost, conform=R.conform_tally(clips, template), pairing=[dict(R.pairing_row(c, refs[c.index]), validity=R.part_validity(c.conformed)) for c in clips]))     # no form here holds the monocular marks
        print(json.dumps(dict(monocular_clips=len(clips), lost=len(lost), stored_fps=dict(Counter(str(c.slice_of["stored_fps"]) for c in clips)))), flush=True)
    tokenizer = L.load_tokenizer(store) if any(n in TOKENIZER_NEEDED for n in names) else None
    readers: dict = {}
    for name in names:
        started = time.time()
        kept, states, masks = build_form(name, refs, clips, tokenizer)
        built = time.time()
        reader = reader_of(name, store, args, readers)
        here_texts = [texts[i] for i in kept]
        raw = reader.read(states, here_texts, M.retrieval_candidates(len(here_texts), 1, seed=0), group=args.nll_group)
        validity = [R.part_validity(s) for s in states]
        report = dict(kind=f"{PROBE}-form", report_only=True, form=name, declared_read=declared, list=described, code_digest=digest, environment=place, instrument=reader.info, tokenizer=dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA, gains=None) if name in TOKENIZER_NEEDED else None,
                      signs=[signs[i] for i in kept], groups=[groups[i] for i in kept], references=here_texts, lost=lost if name in MONOCULAR_NEEDED else [], hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(),
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
    result = dict(P.canary_read(dict(real=read["real_two_view"], blank=read["blank"]), stored), code_digest=digest, environment=place)
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


# ---- reading ---------------------------------------------------------------------------------------------------------------------------

def share_rows(reads: dict, blank: str) -> tuple:
    """(point by metric and form, draws by metric and form): every form of ``reads`` against ``blank``'s NLL, one joint bootstrap over the shared clips (n 2,000, seed 0)."""
    refs = reads[blank]["references"]
    idx = np.random.default_rng(0).integers(0, len(refs), (BOOT, len(refs)))
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
    if a >= FIT_TWO_THIRDS and first["ci95"][0] >= FIT_ONE_THIRD and b < FIT_ONE_THIRD:
        return names[0]
    if b >= FIT_TWO_THIRDS and second["ci95"][0] >= FIT_ONE_THIRD and a < FIT_ONE_THIRD:
        return names[1]
    return "both" if a >= FIT_ONE_THIRD and b >= FIT_ONE_THIRD else "unresolved"


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
        share = {form: ratio(p, d, ("real_two_view", form), ("real_two_view", both)) for form in ("mono_body_two_view_hands", "two_view_body_mono_hands", "mono_arms_only", "mono_hand_pose_only")}
        out[metric] = dict(cells={f: round(p[f], 5) for f in ANCHORS + PART_FORMS}, lost_by_swapping_both=total, lost=dict(body=difference(p, d, "real_two_view", "mono_body_two_view_hands"), hands=difference(p, d, "real_two_view", "two_view_body_mono_hands"), arms=difference(p, d, "real_two_view", "mono_arms_only"), hand_pose=difference(p, d, "real_two_view", "mono_hand_pose_only")),
                           share=dict(body=share["mono_body_two_view_hands"], hands=share["two_view_body_mono_hands"], arms=share["mono_arms_only"], hand_pose=share["mono_hand_pose_only"]))
        out[metric]["case"] = part_case(total, share["mono_body_two_view_hands"], share["two_view_body_mono_hands"], ("body", "hands"))
        out[metric]["view_case_report_only"] = part_case(total, share["mono_arms_only"], share["mono_hand_pose_only"], ("arms", "hand_pose"))
        cases.append(out[metric]["case"])
        view.append(out[metric]["view_case_report_only"])
    out["case"], out["view_case_report_only"] = joint(cases), joint(view)
    return out


def fit_case(gap: dict, share_u: dict | None, drop: dict, share_v1: dict | None) -> str:
    """One metric, the header's rule: content, fit or unresolved. ``gap``: the unfitted instrument's own two-view minus blank; ``share_u``, ``share_v1``: each instrument's monocular share of its gap; ``drop``: its monocular minus two-view."""
    if share_u is None or share_v1 is None or gap["ci95"][0] <= 0 or share_u["ci95"][1] - share_u["ci95"][0] > READABLE_WIDTH:
        return "unresolved"
    if drop["ci95"][1] < 0 and abs(share_u["point"] - share_v1["point"]) <= COMPARABLE and share_u["ci95"][1] <= KEPT:
        return "content"
    return "fit" if share_u["ci95"][0] >= KEPT else "unresolved"


def fit_block(reads: dict) -> dict:
    """Question 2: each instrument's marks-free pair against its own blank. Likelihood sets the label; BLEU gets the same row."""
    u_point, u_draws = share_rows({n: reads[n] for n in UNFITTED_FORMS}, "uwb_blank")
    v_point, v_draws = share_rows({n: reads[n] for n in ("blank",) + NEAREST_FORMS}, "blank")
    out = dict(bootstrap=dict(n=BOOT, seed=0), clips=len(reads["blank"]["signs"]), thresholds=dict(comparable=COMPARABLE, kept=KEPT, readable_width=READABLE_WIDTH))
    for metric in ("likelihood", "bleu"):
        up, ud, vp, vd = u_point[metric], u_draws[metric], v_point[metric], v_draws[metric]
        gap = difference(up, ud, "uwb_nearest_two_view", "uwb_blank")
        drop = difference(up, ud, "uwb_nearest_monocular", "uwb_nearest_two_view")
        share_u = ratio(up, ud, ("uwb_nearest_monocular", "uwb_blank"), ("uwb_nearest_two_view", "uwb_blank"))
        share_v1 = ratio(vp, vd, ("nearest_monocular", "blank"), ("nearest_two_view", "blank"))
        out[metric] = dict(unfitted=dict(cells={f: round(up[f], 5) for f in UNFITTED_FORMS}, own_gap=gap, monocular_minus_two_view=drop, monocular_share_of_own_gap=share_u),
                           v1=dict(cells={f: round(vp[f], 5) for f in ("blank",) + NEAREST_FORMS}, own_gap=difference(vp, vd, "nearest_two_view", "blank"), monocular_minus_two_view=difference(vp, vd, "nearest_monocular", "nearest_two_view"), monocular_share_of_own_gap=share_v1),
                           share_difference=None if share_u is None or share_v1 is None else round(share_u["point"] - share_v1["point"], 4), case=fit_case(gap, share_u, drop, share_v1))
    out["case"] = out["likelihood"]["case"]
    return out


def figure_misses(reads: dict) -> list[str]:
    """v1's reads against the first read's figures, on the whole form: this run's path holds or it does not."""
    base = np.asarray(reads["blank"]["nll"])[:, 0]
    keep = reads["blank"]["signs"]
    out = []
    for form, (bleu, likelihood) in FIGURES.items():
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
    return [f"{form} reads exactly as uwb_blank" for form in ("uwb_nearest_two_view", "uwb_nearest_monocular") if all(blank[s] == h for s, h in zip(reads[form]["signs"], reads[form]["hypotheses"]))]


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


def run_merge(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    L.refuse_existing(store, args.output_key, args.output_file)
    out_dir = Path(args.out_dir)
    all_reads, plan = load_reads(out_dir)
    first = all_reads["real_two_view"]
    declared = first["declared_read"]
    digest, place = first["code_digest"], first["environment"]
    control = R.passed_control(out_dir, digest, place) if declared else None
    shared = R.shared_signs(all_reads)
    if not shared:
        raise SystemExit("the forms share no clip")
    reads = {n: R.restrict(r, shared) for n, r in all_reads.items()}
    keep = set(shared)
    rows = [r for r in plan["pairing"] if r["sign"] in keep]
    block = dict(clips=len(rows), source_hash_matches=sum(r["source_hash_matches"] for r in rows), max_offset_seconds=max(r["max_offset_seconds"] for r in rows), offsets=R.offset_distribution(rows),
                 group_distance={g: R.mean_or_none([R.group_distance(r["distance"], g) for r in rows]) for g in R.GROUPS}, group_mirrored={g: R.mean_or_none([R.group_distance(r["mirrored"], g) for r in rows]) for g in R.GROUPS})
    alerts = R.plumbing_alerts(reads["real_two_view"]["part_validity"], {part: [r["validity"][part] for r in rows] for part in R.PARTS}, block, len(shared), declared) + constant_reads(reads)
    if declared:
        alerts += figure_misses(all_reads)
    parts = parts_block(reads)
    fit = fit_block(reads)
    withheld = ([] if declared else ["a smoke carries no reading"]) + list(alerts)
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, declared_read=declared, smoke=None if declared else "SMOKE on c02 train clips: these numbers are plumbing, not a read", list=first["list"], code_digest=digest, environment=place, disclosures=DISCLOSURES, prestated=PRESTATED, instruments=instrument_block(store),
                  alerts=alerts, withheld=withheld, which_part=dict(parts, label=None if withheld else parts["case"]), fit_or_content=dict(fit, label=None if withheld else fit["case"]), clips=dict(list=len(first["signs"]), shared=len(shared), floor=R.SHARED_FLOOR, outside_shared=R.lost_rows(all_reads, shared)),
                  forms={n: form_row(reads[n], reads["uwb_blank" if n in UNFITTED_FORMS else "blank"]) for n in FORMS}, pairing=block, conform=plan["conform"], per_signer=signer_rows(reads),
                  checks=dict(blank_identical_to_real_two_view=round(float(np.mean([a == b for a, b in zip(reads["blank"]["hypotheses"], reads["real_two_view"]["hypotheses"])])), 4), control=control), rerun=json.loads((out_dir / "rerun.json").read_bytes()) if (out_dir / "rerun.json").exists() else None,
                  resources={n: {k: r[k] for k in ("peak_rss_bytes", "gpu_peak_reserved_bytes", "build_seconds", "read_seconds")} for n, r in all_reads.items()})
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, declared_read=declared, shared=len(shared), of=len(first["signs"]), lost=report["clips"]["outside_shared"]["by_reason"], forms={n: {k: v for k, v in row.items() if k in ("bleu", "likelihood_minus_blank")} for n, row in report["forms"].items()},
                          which_part=dict(bleu=parts["bleu"]["share"], likelihood=parts["likelihood"]["share"], case=parts["case"], view_case=parts["view_case_report_only"], label=report["which_part"]["label"]), fit_or_content=dict(likelihood=fit["likelihood"]["unfitted"], v1=fit["likelihood"]["v1"], bleu=fit["bleu"]["unfitted"], case=fit["case"], label=report["fit_or_content"]["label"]), pairing=block, withheld=withheld, alerts=alerts)), flush=True)
    return 0


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
        out[signer] = dict(clips=len(signs), which_part=dict(bleu=parts["bleu"]["share"], likelihood=parts["likelihood"]["share"]), fit_or_content=dict(likelihood=fit["likelihood"]["unfitted"], v1_share=fit["likelihood"]["v1"]["monocular_share_of_own_gap"]))
    return out


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
    if report["alerts"] and (report["which_part"].get("label") is not None or report["fit_or_content"].get("label") is not None):
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
    return dict(instruments=run_instruments, preflight=run_preflight, read=run_read, control=run_control, merge=run_merge, check=run_check, upload=P.run_upload, **{"upload-log": R.run_upload_log})[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
