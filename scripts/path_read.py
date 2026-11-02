"""Path read: does instrument v1 read real D20 through the monocular lifting path materially lower than through two-view? Report only: nothing trains, no generated form and no composition is read, and this code gates and selects nothing.

The word clips composition uses come almost entirely from monocular-path training data (0.27% two-view among single-word clips: ``word_clip_path`` in ``reports/wrist_by_signer.json``, branch ``comp-a6``), and they reach v1 all-valid through d2, while "real D20" has been read through two-view with real's marks.

**How this will be read (stated before any read; the commit holding this text precedes the job).** The label rests on the marks-free pair, on the shared clips: d = nearest_monocular − the stored two-view v0_nearest_fill. V0 discredited the marked read: nearest fill moved two-view's BLEU up, 4.493 to 4.621, and its likelihood down, +0.2920 to +0.2337, so the two metrics disagree on what marks do. real_monocular − real_two_view is kept as the secondary pair. Each metric is read alone, BLEU against 0.5 and likelihood minus blank against 0.052 nats (0.5 BLEU is 17.8% of the 2.813 gap, and 17.8% of +0.2920 is 0.052).

(a) monocular within the threshold of two-view with overlapping CI and a paired lower bound at or above minus the threshold: the path is not a large cap.
(b) lower by more than the threshold: v1 reads this path lower; lost content, marks, length and v1's fit are not separated.
(c) higher: the instrument may prefer monocular, investigate before believing it.
Otherwise unresolved, and unresolved when BLEU and likelihood fall in different cases.

"Higher" means d above the threshold, or d > 0 with a paired interval wholly above zero; a higher read inside noise is (a), and its sign is shown. Exactly the threshold is within it. (b) has no interval condition; whether the intervals overlap is shown beside it. The case is a reading, not a rule: no step of the job branches on it, and it starts no training and no decision. No reading here makes a claim about composition.

**Fine-tuning.** v1 was pretrained on UWB YouTube-ASL MediaPipe 2D keypoints, then fine-tuned on How2Sign FK keypoints built from the two-view manifest (full-v1) and selected on How2Sign val BLEU, the pool D20 comes from. A lower monocular read can be v1's fit to two-view input as well as lost content, and this read cannot separate them.

**If round_trip_monocular holds while real_monocular drops,** suspect marks or noise that d2 removes, not lost signing.

**The label is withheld** on any ALERT, in a smoke too, and below 475 of D20's 500 shared. ``check`` prints each ALERT and exits 3. Fixed here, before any read:
- Validity, mean per-clip share of frames, either path: shoulders, each elbow and each wrist at least 0.5; each hand at least 0.3. Monocular over two-view at least 0.5 per part. (D20's report rows find a pose on every frame and the hands on 0.96 and 0.94 of frames, so these catch a broken path, not ordinary tracking. They do not see the chest: see below.)
- Keypoint distance, monocular against two-view on shared timestamps, v1's normalised view, pooled over clips: shoulders at most 0.5 and elbows and wrists at most 1.0 shoulder widths; hands at most 3.0 palm widths.
- The negative control: a mirrored monocular clip (``pose.mirror.mirror_state``) must sit farther from two-view than the clip itself, on elbows, wrists and hands.
- Pairing: every two-view clip's ``source_sha256`` is its monocular row's hash, and the largest offset on the stored clocks is at most 0.001 s.

**Forms** (D20, instrument v1, the selection and scoring of ``tok_v1`` and ``tok_v0``; no option names another list or instrument):

- ``real_two_view``, ``blank``: V1's. ``rt_nearest_two_view``: V1's ``rt_nearest``, read again in this run. The control: on all 500 these must reproduce their stored reads (``tok_probes.canary_read`` for the first two, the same limits for the third: BLEU within 0.05, at least 0.98 of hypotheses identical, mean per-clip NLL within 0.001). A miss voids the read, and ``read`` refuses a monocular form without a passed control of this code and environment.
- ``real_monocular``: the same clips' monocular states (``derived/how2sign-video-estimate-v1``), at 24 fps, conformed by ``layout_adapter.conform`` to the template word clips are conformed to (``round15_compose_eval``: the first How2Sign train state of r10, ``TEMPLATE_SIGN``). Their extraction is ``scripts/extract_video_states.py`` as ``docs/RUNPOD.md`` documents it: without ``upright_chest``, so the chest is hip-gated and a frame without hips marks no shoulder, elbow, wrist or hand for v1. YouTube and Sem-Lex were extracted with the upright chest (``extract_video_corpus.py``, ``extract_semlex_videos.py``), so this path is not exactly theirs.
- ``real_monocular_unconformed``: conforming only renames the frame and convention of valid channels and empties channels with no valid frame; v1 reads body and hand rotations and their marks (``keypoints2d.from_state``). So each clip's v1 input is compared before any read. If it is byte-identical on every clip, a read can only repeat ``real_monocular``: the form is skipped, the comparison stands as what conforming costs (nothing), and ``monocular.json`` says why. If any clip differs, the form is read.
- ``nearest_monocular``: ``tok_v1.cap_form("v0_nearest_fill", ...)`` on the conformed clip, every frame valid, no encoder or decoder: the marks-free read, paired with the stored two-view form (``tok_v1.STORED_FORMS``).
- ``round_trip_monocular``: V1's ``rt_nearest`` on the conformed clip, since word clips pass through the tokenizer, beside this run's and the stored two-view read (4.124 on all 500), and beside 3.8.
- ``real_monocular_trimmed``: cut to the frames of its two-view clip, so length is taken out, then resampled on two-view's own timestamps.

**Clocks.** A two-view clip's stored timestamps are a slice of its monocular clip's (``two_view_states.triangulated_state``: ``times = monocular.timestamps[frames]``), so the two are compared on the stored clocks, each clip's two-view timestamps read from its own stored state, before any resampling. After resampling they cannot be: ``at_rate`` builds its 24 fps grid from each clip's own first frame, so a 30 or 60 fps clip's two grids differ in phase by up to a frame (a first run stopped on one at 0.0333 s while its stored clocks agreed exactly). The cut takes each two-view timestamp's nearest stored monocular frame. The job stops when any offset exceeds a quarter of that clip's own stored frame interval (10.4 ms at 24 fps, 8.3 ms at 30, 4.2 ms at 60), or the frames found are not consecutive. A nearest-frame offset cannot pass half a frame inside a clip, so half would only ever catch a clip's ends: a quarter stops a clock off the frame grid, and a clip reaching past its monocular clip's ends. Timestamps are copied, not derived, so none is expected: measured before this threshold was fixed, on the stored clocks of 46 D20 clips including all 18 report rows at a rate other than 24, the largest offset is 0.0 s and every source hash is equal. Above 0.001 s an offset raises an ALERT even inside the tolerance, and every clip's offset goes in the report with its distribution. A shift of whole frames shows on no clock: it rests on the source hash, the start frame reported, and keypoint distance.
- Report only, the 2x2 of values and marks on those frames, beside ``real_two_view`` and ``real_monocular_trimmed``: ``monocular_values_two_view_marks`` (``V.with_marks(V.filled(monocular), two_view)``) and ``two_view_values_monocular_marks``. A joint its own path never tracked holds d2's neutral (``V.neutral_pose``), not zeros.

**Clips.** Monocular states are found by exact D20 record: the record's ``sign`` as a dictionary key among the rows of the 8 report shards pinned below, each by key and hash; the state key must then be exactly ``<prefix>/states/<sign>.npz``. Nothing is listed and no substring is matched. All 500 of D20 sit in those shards (checked 2026-11-01 from index objects; 0 in the official-val report), so the official val and test reports are never opened.
A record that is not a How2Sign val clip of c01, c03, c05 (or, in a smoke, a c02 train clip) is refused; so is c00, whatever else it says. C20 cannot be named: ``s2_lib.load_read_list`` refuses any other key unread.
Only a missing object loses a clip: no row, no state in its row, or no object at its key, each listed. A hash mismatch, a network error, a refused conform, a row naming another object, or a memory error stops the job. Every comparison runs on the clips every form read.

**Per form** (``merge``): BLEU with ``bleu_ci_fast``'s interval, likelihood minus blank, share of the real-minus-blank gap, valid-frame share and per-part validity (shoulders, each elbow, wrist and hand), paired differences against ``real_two_view``, per-signer rows, clips lost with reasons. ``same_path``: each difference and share under one joint bootstrap (n 2,000, seed 0): the marks-free gap, its ratio to the two-view gap, the round trip's share of it, and the same for the marked pair.
A stored figure S against two-view reads S / ratio against monocular, within the same blank; that division is left to the reader, since no generated report is opened here. ``pairing``: the distances and the mirrored control per part, and, report only, the per-clip likelihood difference regressed on keypoint distance.

**Stored reads.** The job's first step (``stored-check``) reads both stored forms and checks them against the plan's figures: each by its pinned hash: ``v0_nearest_fill`` BLEU 4.621, likelihood +0.2337; ``rt_nearest`` BLEU 4.124 [3.485, 4.764], likelihood +0.1883.

**Smoke** (``--smoke-train 5 --smoke-longest 4 --smoke-never-tracked 2``): c02 train clips, as ``tok_v1``'s, through every form, and D20's lookup from index objects alone. No D20 state is read and no instrument sees D20 in the smoke.

    python scripts/path_read.py stored-check
    python scripts/path_read.py preflight --out-dir W --final-key K --void-prefix V
    python scripts/path_read.py lookups --out-dir W/forms
    python scripts/path_read.py read --forms real_two_view,blank,rt_nearest_two_view --out-dir W/forms --instrument-cache W/i
    python scripts/path_read.py control --out-dir W/forms --void-prefix V
    python scripts/path_read.py read --forms <the monocular forms> --resume --out-dir W/forms --instrument-cache W/i
    python scripts/path_read.py merge --out-dir W/forms --output-file W/report.json
    python scripts/path_read.py check --report W/report.json

The job is ``runpod/fleet/path_read.sh``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import functools
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v0 as V  # noqa: E402
import tok_v1 as T  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402

PROBE = "path-read"
JOB_FILE = "runpod/fleet/path_read.sh"
FPS = L.FPS
PREFIX = f"{L.O}/round20/path-read"
SMOKE_CLIPS = 5
BOOT = 2000
ANCHORS = ("real_two_view", "blank", "rt_nearest_two_view")
UNCONFORMED = "real_monocular_unconformed"
MONOCULAR_FORMS = ("real_monocular", UNCONFORMED, "nearest_monocular", "round_trip_monocular", "real_monocular_trimmed", "monocular_values_two_view_marks", "two_view_values_monocular_marks")
FORMS = ANCHORS + MONOCULAR_FORMS
TOKENIZER_FORMS = ("rt_nearest_two_view", "nearest_monocular", "round_trip_monocular", "monocular_values_two_view_marks", "two_view_values_monocular_marks")
MONOCULAR = "acquisition/2026-10-17-v1/derived/how2sign-video-estimate-v1"
REPORTS = f"{MONOCULAR}/reports/"
STATES = f"{MONOCULAR}/states/"
# the only monocular objects this code opens before a state: How2Sign's official train split, every clip of D20 and of the smoke. Never val or test.
REPORT_NAME = re.compile(r"train_rgb_front_clips-shard-\d{3}-of-008\.json")
MONOCULAR_REPORTS = {f"{REPORTS}train_rgb_front_clips-shard-{i:03d}-of-008.json": sha for i, sha in enumerate((
    "c4b5c02d8244ad61e61da231385318f07a73f738452f806d2ce3c2150d02fda7", "8caf0176fcf6318f4aa623658016676a89112bdc7d8c0c083a366e9885f135a8",
    "710e68880f745848a60efda1e2d3b217e9c64f64626ed03fbf62a75883b8a824", "2222b896371a3d1f0949671c241ba0d9d667dbda34ea80acd71bb13df14018de",
    "b4f8e34a4e98b4a4676df8ad693defe9a6e991fde20231f5308bcf28c20d784c", "6f2cd81d87be0386709ffe2ec73d566f932891a39bce01396a79e6c47baa697f",
    "a675fd716b2bc1fb3b398e25863c5851afaa25eca1f73025e256d4f3ee7cf407", "26361e9311dfce0ec5c6d2197c3f0c3e02a04e43da34c6f7e97d8a5fe8036804"))}
TEMPLATE_SIGN = "--7E2sU6zP4_10-5-rgb_front"            # r10's first How2Sign train record (c02): train_motion_tokens.layout_template's choice
STORED_RT = f"{L.O}/tok-probes-v1/forms/form-rt_nearest.json"
STORED_RT_SHA256 = "4577be5a56205038e97c6ff2b091015e9cf3a072f5f1ae40a7708c26819d2817"     # printed by stored-check on the first run
STORED_RT_BLEU, STORED_RT_CI, STORED_RT_LIKELIHOOD = 4.124, [3.485, 4.764], .1883     # docs/NEXT_PHASE_PLAN.md, V1 result
STORED_RT_DIGEST = "abcb46bd"                           # V1's code digest, the ledger's prefix
STORED_NEAREST, STORED_NEAREST_SHA256 = T.STORED_FORMS["v0_nearest_fill"]
STORED_NEAREST_BLEU, STORED_NEAREST_LIKELIHOOD = 4.621, .2337
THRESHOLD = dict(bleu=.5, likelihood=.052)
BAR = 3.8
SHARED_FLOOR = 475
SLICE_TOLERANCE = .25                                   # of a clip's own stored frame interval: stops above it; the offset itself is held to OFFSET_CEILING
OFFSET_CEILING = .001
PARTS = dict(shoulders=(0, 1), left_elbow=(2,), right_elbow=(3,), left_wrist=(4,), right_wrist=(5,), left_hand=tuple(range(6, 27)), right_hand=tuple(range(27, 48)))      # keypoints2d.POINTS
DISTANCE_POINTS = dict(PARTS, left_hand=tuple(range(7, 27)), right_hand=tuple(range(28, 48)))      # a hand's 20 local points: its wrist point is placed by the arm
GROUPS = dict(shoulders=("shoulders",), elbows=("left_elbow", "right_elbow"), wrists=("left_wrist", "right_wrist"), hands=("left_hand", "right_hand"))
VALID_FLOOR = dict(shoulders=.5, left_elbow=.5, right_elbow=.5, left_wrist=.5, right_wrist=.5, left_hand=.3, right_hand=.3)
VALID_RATIO_FLOOR = .5
DISTANCE_CEILING = dict(shoulders=.5, elbows=1., wrists=1., hands=3.)
MIRRORED_GROUPS = ("elbows", "wrists", "hands")
PRESTATED = dict(a="(a) monocular within the threshold of two-view with overlapping CI and a paired lower bound at or above minus the threshold: the path is not a large cap.",
                 b="(b) lower by more than the threshold: v1 reads this path lower; lost content, marks, length and v1's fit are not separated.",
                 c="(c) higher: the instrument may prefer monocular, investigate before believing it.",
                 unresolved="Otherwise unresolved, and unresolved when BLEU and likelihood fall in different cases.",
                 settled='"Higher" means d above the threshold, or d > 0 with a paired interval wholly above zero; a higher read inside noise is (a), and its sign is shown.',
                 zero_read="**If round_trip_monocular holds while real_monocular drops,** suspect marks or noise that d2 removes, not lost signing.")
NOT_DECIDED = "report only: the header's reading is stated before the read; no rule, bar or verdict here gates anything"
DISCLOSURES = dict(
    instrument="v1 was pretrained on UWB YouTube-ASL MediaPipe 2D keypoints, then fine-tuned on How2Sign FK keypoints built from the two-view manifest (full-v1) and selected on How2Sign val BLEU, the pool D20 comes from. A lower monocular read can be v1's fit to two-view input as well as lost content; this read cannot separate them.",
    extraction="How2Sign's monocular states come from extract_video_states.py without upright_chest (docs/RUNPOD.md), so the chest is hip-gated; YouTube and Sem-Lex used the upright chest. part_validity shows what that leaves v1 here.",
    frames="A two-view clip is a slice of the same front clip's frames (triangulate_how2sign_sessions), so monocular clips run about 1.00-1.05x longer with extra frames at the ends. real_monocular_trimmed takes that out; seconds_over_reference gives the ratio per form. A clip stored at another rate resamples from its own first frame, so an untrimmed real_monocular sits on another phase of the 24 fps grid than its two-view clip; the trimmed form does not.",
    blank="Blank is V1's: the two-view clips' durations. Its length is therefore not exactly the monocular clip's, and no separate monocular-length blank is read.",
    marks="Validity differs by path, and v1 reads per-point marks, so part_validity belongs beside each read. nearest_monocular and round_trip_monocular are all-valid by construction, like their stored two-view forms.",
    two_by_two="monocular_values_two_view_marks and two_view_values_monocular_marks are report only. Marks also move the shoulder origin and unit, which normalise takes from the marked frames, and removing a mark hides that frame's values.",
    scope="This reads How2Sign's signers through the monocular path. The word clips come from other corpora through that path (ASLLVD, RIT, WLASL, Sem-Lex, YouTube, ASLLRP), with other framing and signers, so this measures the path, not those corpora.",
    stored="The stored rt_nearest and v0_nearest_fill were read by V1's and V0's jobs; their environments are shown beside this read's, and their likelihoods are taken against this read's blank.",
    panels="Inside a form's panel, wrist_speed, wrist_spread, hand_speed and hand_variation name 'over_real'; real there is real_two_view. Each form's motion is measured on its own path's tracked frames.")
PER_CLIP = ("signs", "groups", "references", "hypotheses", "nll", "seconds_over_reference", "valid_frame_share")


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    try:
        return provenance.code_digest(extra_files=(JOB_FILE,))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


# ---- what may be looked up -------------------------------------------------------------------------------------------------------

def check_lookup_records(records, declared: bool) -> None:
    """The only records a monocular lookup takes: D20's validation clips, or a smoke's c02 train clips. c00 and any other partition are refused, whatever else a record says."""
    for r in records:
        if r.get("signer_component") == "c00":
            raise SystemExit(f"{r.get('sign')}: How2Sign signer 9 (c00) is never read")
        if r.get("corpus") != L.H2S:
            raise SystemExit(f"{r.get('sign')}: not a How2Sign record")
    P.check_read_list(records, declared)


def check_report_key(key: str) -> None:
    """A report is opened only if it is pinned and named as an official train shard, tested before it is fetched."""
    if key not in MONOCULAR_REPORTS or not key.startswith(REPORTS) or not REPORT_NAME.fullmatch(key[len(REPORTS):]):
        raise SystemExit(f"{key}: not a pinned train-split report; refused unread")


def monocular_rows(store, wanted) -> dict:
    """{sign: its report row}, only for the exact signs in ``wanted``: other clips' rows are not kept. Every pinned shard is read and hash-checked; a sign with two different states refuses."""
    wanted, rows = set(wanted), {}
    for key, sha in sorted(MONOCULAR_REPORTS.items()):
        check_report_key(key)
        raw = store.get(key)
        P.check_sha(raw, sha, key)
        for row in json.loads(raw)["results"]:
            sign = row.get("sample_id")
            if sign not in wanted:
                continue
            here = dict(state_key=row.get("state_key"), state_sha256=row.get("state_sha256"), status=row.get("status"), shard=key[len(REPORTS):])
            if sign in rows and (rows[sign]["state_key"], rows[sign]["state_sha256"]) != (here["state_key"], here["state_sha256"]):
                raise SystemExit(f"{sign}: two different monocular states in the reports")
            rows.setdefault(sign, here)
    return rows


def state_key_of(sign: str) -> str:
    return f"{STATES}{sign}.npz"


def lost_clip(sign: str, reason: str, detail: str | None = None) -> dict:
    return dict(sign=sign, reason=reason, detail=detail)


def match_monocular(records, rows: dict, declared: bool) -> tuple[list[dict], list[dict]]:
    """(the records with a monocular state, in list order, as {index, sign, state_key, state_sha256}; the others, listed). A record matches its own row by its exact sign, and that row must name exactly this clip's state."""
    check_lookup_records(records, declared)
    found, lost = [], []
    for i, r in enumerate(records):
        sign, row = r["sign"], rows.get(r["sign"])
        if row is None:
            lost.append(lost_clip(sign, "no_monocular_row"))
        elif not row.get("state_key") or not row.get("state_sha256"):
            lost.append(lost_clip(sign, "no_state_in_row", row.get("status")))
        elif row["state_key"] != state_key_of(sign):
            raise SystemExit(f"{sign}: its row names {row['state_key']}, not this clip's monocular state: refused unread")
        else:
            found.append(dict(index=i, sign=sign, state_key=row["state_key"], state_sha256=row["state_sha256"]))
    return found, lost


def load_template(store):
    """The layout word clips are conformed to (``train_motion_tokens.layout_template`` on r10): the first How2Sign train record, checked by name, partition and signer."""
    from evaluate_text_to_motion import load_state
    manifest = L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA)
    record = next((r for r in manifest["records"] if r.get("corpus") == L.H2S and r.get("partition") == "train"), None)
    if record is None or record["sign"] != TEMPLATE_SIGN or record.get("signer_component") == "c00":
        raise SystemExit(f"the template record is {record and record.get('sign')}, not {TEMPLATE_SIGN}")
    return load_state(store, record["continuous"]["state_key"], record["continuous"]["state_sha256"])


# ---- monocular clips -------------------------------------------------------------------------------------------------------------

@dataclass
class Clip:
    index: int
    sign: str
    raw: object
    conformed: object
    trimmed: object
    slice_of: dict


def layout_of(state) -> dict:
    """The layout of the channels v1 reads. A clip's other channels may be unavailable under placeholder names in either path, which ``conform`` handles and v1 never sees."""
    return {name: (state.channels[name].labels, state.channels[name].coordinate_frame, state.channels[name].units, state.channels[name].convention) for name in V.MASKED}


def reader_input(state) -> np.ndarray:
    """What v1 receives from a state at its rate (``score_back_translation.translate``)."""
    from signtranslator.data.keypoints2d import features, from_state, normalise
    points, valid = from_state(state)
    return features(normalise(points, valid, y_down=False), valid)


def valid_frame_share(state) -> float:
    """The share of v1's points the state marks valid, over all frames."""
    from signtranslator.data.keypoints2d import from_state
    return round(float(from_state(state)[1].mean()), 5)


def part_validity(state) -> dict:
    """{part: the share of frames and points of that part marked valid, as v1 reads them}."""
    from signtranslator.data.keypoints2d import from_state
    valid = from_state(state)[1]
    return {part: round(float(valid[:, list(points)].mean()), 5) for part, points in PARTS.items()}


def key_distance(a, b) -> dict:
    """{part: mean L2 between two states of equal length in v1's normalised view, over points marked in both; None where none is}: shoulder widths, and palm widths for a hand's local points."""
    from signtranslator.data.keypoints2d import from_state, normalise
    pa, va = from_state(a)
    pb, vb = from_state(b)
    gap = np.linalg.norm(normalise(pa, va, y_down=False) - normalise(pb, vb, y_down=False), axis=-1)
    both = va & vb
    out = {}
    for part, points in DISTANCE_POINTS.items():
        mask = both[:, list(points)]
        out[part] = round(float(gap[:, list(points)][mask].mean()), 5) if mask.any() else None
    return out


def trim_index(times, two_view_times) -> tuple[np.ndarray | None, float | None]:
    """(the frames of the stored clock ``times`` nearest each stored two-view timestamp, the largest offset): None when an offset exceeds a quarter of this clip's own frame interval, or the frames found are not consecutive."""
    times, wanted = np.asarray(times, np.float64), np.asarray(two_view_times, np.float64)
    tolerance = SLICE_TOLERANCE * float(np.median(np.diff(times)))
    near = np.clip(np.searchsorted(times, wanted), 1, len(times) - 1)
    near = np.where(np.abs(times[near - 1] - wanted) <= np.abs(times[near] - wanted), near - 1, near)
    offset = float(np.abs(times[near] - wanted).max())
    if offset > tolerance or (np.diff(near) != 1).any():
        return None, offset
    return near, offset


def cut_frames(state, index):
    """The frames ``index`` of ``state``, each at its own timestamp."""
    channels = {n: replace(c, values=c.values[index], valid=c.valid[index], observed=c.observed[index], confidence=c.confidence[index]) for n, c in state.channels.items()}
    return replace(state, timestamps=state.timestamps[index], channels=channels)


def load_clip(store, item: dict, template, ref, two_view_times) -> tuple[Clip | None, dict | None]:
    """(the clip, None), or (None, why) for a missing object alone: anything else wrong with a clip stops the job. Conformed as ``round20_lib.utterance_clip`` conforms a word's material: to the template, then to 24 fps. ``two_view_times``: the two-view clip's stored clock, which the cut is taken on."""
    from evaluate_text_to_motion import at_rate, load_state
    from signtranslator.data.layout_adapter import conform
    sign = item["sign"]
    if item["state_key"] != state_key_of(sign):
        raise SystemExit(f"{sign}: refused {item['state_key']}: not this clip's monocular state")
    if not store.exists(item["state_key"]):
        return None, lost_clip(sign, "state_missing", item["state_key"])
    try:
        raw = load_state(store, item["state_key"], item["state_sha256"])
    except ValueError as error:
        raise SystemExit(f"{sign}: its state does not load: {error}") from error
    if len(raw.timestamps) < 2:
        raise SystemExit(f"{sign}: {len(raw.timestamps)} frames in its monocular state")
    try:
        done = conform(raw, template)
    except (ValueError, KeyError) as error:
        raise SystemExit(f"{sign}: conform refused it: {type(error).__name__}: {error}") from error
    conformed = at_rate(done, FPS)
    if layout_of(conformed) != layout_of(ref):
        raise SystemExit(f"{sign}: conformed, v1's channels are not in its two-view clip's layout")
    index, offset = trim_index(raw.timestamps, two_view_times)
    if index is None:
        raise SystemExit(f"{sign}: its two-view clip is not a slice of its monocular clip on the stored clocks: largest offset {offset:.4f} s")
    trimmed = at_rate(replace(cut_frames(done, index), timestamps=np.asarray(two_view_times, np.float64)), FPS)      # on two-view's clock, so at_rate lands on two-view's own grid
    if len(trimmed.timestamps) != len(ref.timestamps):
        raise SystemExit(f"{sign}: cut and resampled it holds {len(trimmed.timestamps)} frames, its two-view clip {len(ref.timestamps)}")
    slice_of = dict(start_frame=int(index[0]), frames_two_view=len(ref.timestamps), frames_monocular=len(conformed.timestamps), stored_fps=round(1 / float(np.median(np.diff(raw.timestamps))), 2), max_offset_seconds=round(offset, 6), source_hash_matches=ref.channels["body"].source_sha256 == item["state_sha256"])
    return Clip(item["index"], sign, at_rate(raw, FPS), conformed, trimmed, slice_of), None


def open_monocular(store, records, refs, declared: bool) -> tuple[list[Clip], list[dict], object]:
    """Every record's monocular clip, the missing ones with their reasons, and the template. Each two-view clip's stored clock is read again from its own stored state: ``refs`` are at 24 fps."""
    from evaluate_text_to_motion import load_state
    found, lost = match_monocular(records, monocular_rows(store, [r["sign"] for r in records]), declared)
    template = load_template(store)

    def one(item):
        record = records[item["index"]]["continuous"]
        return load_clip(store, item, template, refs[item["index"]], load_state(store, record["state_key"], record["state_sha256"]).timestamps)
    with ThreadPoolExecutor(4) as pool:
        loaded = list(pool.map(one, found))
    clips = [c for c, _ in loaded if c is not None]
    lost += [why for _, why in loaded if why is not None]
    if not clips:
        raise SystemExit(f"no clip has a usable monocular state: {Counter(x['reason'] for x in lost)}")
    return clips, lost, template


def pairing_row(clip: Clip, ref) -> dict:
    """What pairs this clip with its two-view clip, and how far apart v1 sees them: on shared timestamps, itself and mirrored."""
    from signtranslator.pose.mirror import mirror_state
    return dict(clip.slice_of, sign=clip.sign, distance=key_distance(clip.trimmed, ref), mirrored=key_distance(mirror_state(clip.trimmed), ref))


def conform_tally(clips: list[Clip], template) -> dict:
    """Per channel, how ``conform`` treated it across clips (``tokenizer_domain_read.conform_kind``), and channels the template does not carry."""
    from tokenizer_domain_read import conform_audit
    kinds, extra = Counter(), Counter()
    for clip in clips:
        audit = conform_audit(clip.raw, template)
        kinds.update(f"{name}:{kind}" for name, kind in audit["kinds"].items())
        extra.update(audit["extra"])
    return dict(kinds=dict(sorted(kinds.items())), dropped=dict(sorted(extra.items())))


def unconformed_plan(clips: list[Clip]) -> dict:
    """Whether conforming changes anything v1 receives; the form is read only if it does on some clip."""
    differ = [c.sign for c in clips if not np.array_equal(reader_input(c.raw), reader_input(c.conformed))]
    if differ:
        reason = f"conforming changed v1's input on {len(differ)} of {len(clips)} clips, so the form is read"
    else:
        reason = (f"not read: v1's input is byte-identical with and without conforming on all {len(clips)} clips (conform renames the frame and convention of valid channels and empties channels with no valid frame; "
                  "v1 reads body and hand rotations and their marks). A read could only repeat real_monocular: conforming costs v1 nothing")
    return dict(read=bool(differ), clips_compared=len(clips), clips_differing=len(differ), differing=differ, reason=reason)


# ---- forms -----------------------------------------------------------------------------------------------------------------------

def build_form(name: str, refs, clips: list[Clip], tokenizer) -> tuple[list[int], list, list]:
    """(record indices, states, the states motion is masked by). A form's motion is measured on its own path's tracked frames."""
    from evaluate_text_to_motion import blanked
    every = list(range(len(refs)))
    kept = [c.index for c in clips]
    if name == "real_two_view":
        return every, list(refs), list(refs)
    if name == "blank":
        return every, blanked(refs), list(refs)
    if name == "rt_nearest_two_view":
        return every, [T.cap_form("rt_nearest", s, tokenizer) for s in refs], list(refs)
    if name == "real_monocular":
        return kept, [c.conformed for c in clips], [c.conformed for c in clips]
    if name == UNCONFORMED:
        return kept, [c.raw for c in clips], [c.raw for c in clips]
    if name == "nearest_monocular":
        return kept, [T.cap_form("v0_nearest_fill", c.conformed, tokenizer) for c in clips], [c.conformed for c in clips]
    if name == "round_trip_monocular":
        return kept, [T.cap_form("rt_nearest", c.conformed, tokenizer) for c in clips], [c.conformed for c in clips]
    if name == "real_monocular_trimmed":
        return kept, [c.trimmed for c in clips], [c.trimmed for c in clips]
    if name == "monocular_values_two_view_marks":
        return kept, [V.with_marks(V.filled(c.trimmed, neutral=V.neutral_pose(c.trimmed, tokenizer)), refs[c.index]) for c in clips], [refs[c.index] for c in clips]
    if name == "two_view_values_monocular_marks":
        return kept, [V.with_marks(V.filled(refs[c.index], neutral=V.neutral_pose(refs[c.index], tokenizer)), c.trimmed) for c in clips], [c.trimmed for c in clips]
    raise SystemExit(f"unknown form {name}")


def passed_control(out_dir: Path, digest: str, place: dict) -> dict:
    """The control of this code and environment, passed; anything else refuses."""
    path = out_dir / "control.json"
    control = json.loads(path.read_bytes()) if path.exists() else None
    if control is None or control.get("reproduced") is not True or control.get("code_digest") != digest or control.get("environment") != place:
        raise SystemExit(f"{path}: no passed control of this code and environment; the read is void")
    return control


def run_preflight(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    if store.exists(args.final_key):
        raise SystemExit(f"{args.final_key} exists: this read is finished (write-once)")
    P.check_void(store, args, code_digest(), Path(args.out_dir))
    print(json.dumps(dict(preflight="clear", code_digest=code_digest())), flush=True)
    return 0


def check_smoke(n: int) -> None:
    if n > SMOKE_CLIPS:
        raise SystemExit(f"a smoke reads at most {SMOKE_CLIPS} clips, not {n}")


def run_stored_check(args) -> int:
    """The job's first step: both stored forms against the plan's figures, each likelihood against the pinned report's blank. Prints each file's hash."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    raw = store.get(P.CANARY_REPORT)
    P.check_sha(raw, P.CANARY_REPORT_SHA, P.CANARY_REPORT)
    blank = np.asarray(json.loads(raw)["forms"]["blank"]["per_clip"]["nll"])
    rt, rt_sha = load_stored_rt_nearest(store)
    nearest, nearest_sha = load_stored_nearest(store)
    rows = []
    for name, form, sha, want in (("rt_nearest", rt, rt_sha, STORED_RT_LIKELIHOOD), ("v0_nearest_fill", nearest, nearest_sha, STORED_NEAREST_LIKELIHOOD)):
        likelihood = float(M.delta_nll(blank, np.asarray(form["nll"])[:, 0]).mean())
        if abs(likelihood - want) > P.CANARY["nll"]:
            raise SystemExit(f"{name}: likelihood minus the pinned blank is {likelihood:.4f}, not the plan's {want}")
        bleu, ci = M.bleu_ci_fast(form["hypotheses"], form["references"])
        rows.append(dict(form=name, sha256=sha, bleu=bleu, ci95=ci, likelihood_minus_blank=round(likelihood, 4), clips=len(form["signs"])))
    print(json.dumps(dict(stored_check=rows)), flush=True)
    return 0


def run_lookups(args) -> int:
    """The lookup alone, from index objects: no state, no reader. Written once, so a job shows its unmatched clips before it spends anything."""
    from train_text_to_motion import make_store
    P.check_pins()
    check_smoke(args.smoke_train)
    out = Path(args.out_dir)
    store = make_store(args.store)
    records, declared, described = L.open_read_list(store, args.smoke_train)
    check_lookup_records(records, declared)
    rows = monocular_rows(store, [r["sign"] for r in records])
    found, lost = match_monocular(records, rows, declared)
    report = dict(kind=f"{PROBE}-lookups", report_only=True, declared_read=declared, list=described, clips=len(records), matched=len(found), lost=lost, reports_opened=sorted(MONOCULAR_REPORTS), code_digest=code_digest(),
                  signers=dict(sorted(Counter(records[f["index"]].get("signer_component") for f in found).items())), shards=dict(sorted(Counter(rows[f["sign"]]["shard"] for f in found).items())), states=[f["state_key"] for f in found])
    path = out / "lookups.json"
    if path.exists():                       # a resume: the same answer leaves the file, another refuses
        kept = json.loads(path.read_bytes())
        if any(kept.get(k) != report[k] for k in ("declared_read", "clips", "matched", "lost", "states", "code_digest")):
            raise SystemExit(f"{path} holds another lookup (write-once)")
    else:
        P.write_report_once(path, report)
    print(json.dumps(dict(lookups=str(out / "lookups.json"), declared_read=declared, clips=len(records), matched=len(found), lost=len(lost), signers=report["signers"], reports_opened=len(MONOCULAR_REPORTS))), flush=True)
    return 0


def open_list(store, args) -> tuple[list[dict], bool, dict]:
    """The list a read takes: D20, or a smoke's c02 train clips with, as ``tok_v1``'s, the longest and some with a hand never tracked."""
    if (args.smoke_longest or args.smoke_never_tracked) and not args.smoke_train:
        raise SystemExit("--smoke-longest and --smoke-never-tracked belong to a smoke")
    records, declared, described = L.open_read_list(store, args.smoke_train)
    if args.smoke_longest:
        have = {r["sign"] for r in records}
        records = list(records) + [r for r in P.longest_train_records(store, args.smoke_longest) if r["sign"] not in have]
        described = dict(described, clips=len(records), longest=args.smoke_longest)
    if args.smoke_never_tracked:
        have = {r["sign"] for r in records}
        records = list(records) + [r for r in V.never_tracked_records(store, args.smoke_never_tracked) if r["sign"] not in have]
        described = dict(described, clips=len(records), never_tracked=args.smoke_never_tracked)
    check_lookup_records(records, declared)
    return records, declared, described


def run_read(args) -> int:
    import torch
    from train_text_to_motion import make_store
    P.check_pins()
    check_smoke(args.smoke_train)
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
    plan_path = out_dir / "monocular.json"
    if UNCONFORMED in names and plan_path.exists() and not json.loads(plan_path.read_bytes())["unconformed"]["read"]:      # a resume: already settled, not read
        names.remove(UNCONFORMED)
    if not names:
        print(json.dumps(dict(read=0, stored=len(stored))), flush=True)
        return 0
    store = make_store(args.store)
    records, declared, described = open_list(store, args)
    digest, place = code_digest(), P.environment()
    if declared and any(n in MONOCULAR_FORMS for n in names):
        passed_control(out_dir, digest, place)
    texts, signs, groups = [r["text"] for r in records], [r["sign"] for r in records], [r["signer_component"] for r in records]
    refs = L.load_states(store, records)
    clips, lost = [], []
    if any(n in MONOCULAR_FORMS for n in names):
        clips, lost, template = open_monocular(store, records, refs, declared)
        plan = unconformed_plan(clips)
        if not plan_path.exists():
            P.write_report_once(plan_path, dict(kind=f"{PROBE}-monocular", report_only=True, declared_read=declared, code_digest=digest, clips_loaded=len(clips), lost=lost, conform=conform_tally(clips, template), unconformed=plan,
                                                pairing=[pairing_row(c, refs[c.index]) for c in clips]))
        print(json.dumps(dict(monocular_clips=len(clips), lost=len(lost), unconformed=plan["reason"])), flush=True)
        if UNCONFORMED in names and not plan["read"]:
            names.remove(UNCONFORMED)
    tokenizer = L.load_tokenizer(store) if any(n in TOKENIZER_FORMS for n in names) else None
    reader = P.load_reader(store, args)
    for name in names:
        started = time.time()
        kept, states, masks = build_form(name, refs, clips, tokenizer)
        built = time.time()
        here_texts = [texts[i] for i in kept]
        raw = reader.read(states, here_texts, M.retrieval_candidates(len(here_texts), 1, seed=0), group=args.nll_group)
        validity = [part_validity(s) for s in states]
        report = dict(kind=f"{PROBE}-form", report_only=True, form=name, declared_read=declared, list=described, code_digest=digest, environment=place, instrument=reader.info, tokenizer=dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA, gains=None) if name in TOKENIZER_FORMS else None,
                      signs=[signs[i] for i in kept], groups=[groups[i] for i in kept], references=here_texts, lost=lost if name in MONOCULAR_FORMS else [], hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(),
                      seconds_over_reference=[round(len(s.timestamps) / max(len(refs[i].timestamps), 1), 5) for s, i in zip(states, kept)], valid_frame_share=[valid_frame_share(s) for s in states], part_validity={part: [v[part] for v in validity] for part in PARTS},
                      motion=L.motion_statistics(states, masks), hands=P.hand_statistics(states, masks), tracked=P.tracked_statistics(states),
                      build_seconds=round(built - started, 2), read_seconds=round(time.time() - built, 2), peak_rss_bytes=P.peak_rss_bytes(), gpu_peak_reserved_bytes=P.gpu_peak_reserved_bytes())
        P.write_report_once(out_dir / f"form-{name}.json", report)
        print(json.dumps({name: dict(clips=len(kept), bleu=M.bleu_ci_fast(raw["hypotheses"], here_texts)[0], build_seconds=report["build_seconds"], read_seconds=report["read_seconds"], peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2), gpu_reserved_bytes=report["gpu_peak_reserved_bytes"])}), flush=True)
        del states, raw
    return 0


def round_trip_control(read: dict, stored: dict) -> dict:
    """This run's rt_nearest_two_view against V1's stored read, the canary's limits. Clips in another order refuse."""
    if read["signs"] != stored["signs"] or read["references"] != stored["references"]:
        raise SystemExit("the stored rt_nearest read other clips or another order")
    bleu = M.bleu_ci_fast(read["hypotheses"], read["references"])[0]
    identical = float(np.mean([a == b for a, b in zip(read["hypotheses"], stored["hypotheses"])]))
    gap = float(np.mean(np.abs(np.asarray(read["nll"])[:, 0] - np.asarray(stored["nll"])[:, 0])))
    row = dict(bleu=bleu, stored_bleu=STORED_RT_BLEU, bleu_gap=round(abs(bleu - STORED_RT_BLEU), 4), identical=round(identical, 4), nll_mean_absolute_difference=round(gap, 6))
    row["misses"] = [f"rt_nearest {why}" for why, bad in (("BLEU", row["bleu_gap"] > P.CANARY["bleu_tolerance"]), ("hypotheses", identical < P.CANARY["identical"]), ("NLL", gap > P.CANARY["nll"])) if bad]
    return row


def run_control(args) -> int:
    """Exit 3, voiding the read, unless real_two_view and blank reproduce the pinned stored report and rt_nearest_two_view V1's stored read, through ``tok_probes``' canary limits. A smoke reads train clips, so it checks the pins and the stored reports' shape only, and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    store = make_store(args.store)
    raw = store.get(P.CANARY_REPORT)
    P.check_sha(raw, P.CANARY_REPORT_SHA, P.CANARY_REPORT)
    stored = json.loads(raw)
    if any(len(stored["forms"][n]["per_clip"]["nll"]) != len(stored["signs"]) for n in ("real", "blank")):
        raise SystemExit(f"{P.CANARY_REPORT}: no per-clip NLL for real and blank")
    stored_rt, _ = load_stored_rt_nearest(store)
    read = {n: json.loads((out / f"form-{n}.json").read_bytes()) for n in ANCHORS}
    if not read["real_two_view"]["declared_read"]:
        print(json.dumps(dict(control="not applicable: a smoke reads train clips; the pins and the stored reports' shape are checked")), flush=True)
        return 0
    digest, place = P.check_same_run(read, "the control's forms")
    canary = P.canary_read(dict(real=read["real_two_view"], blank=read["blank"]), stored)
    trip = round_trip_control(read["rt_nearest_two_view"], stored_rt)
    misses = canary["misses"] + trip["misses"]
    result = dict(canary, round_trip=trip, misses=misses, reproduced=not misses, code_digest=digest, environment=place)
    path = out / "control.json"
    if not path.exists():
        P.write_report_once(path, result)
    print(json.dumps(result), flush=True)
    if result["reproduced"]:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{digest}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-void", code_digest=digest, environment=place, control=result), indent=1, sort_keys=True).encode())
    print(f"ALERT the control's forms do not reproduce their stored reads: {misses}. The read is void: no monocular form follows, no retry; a rerun needs new code and a recorded cause", flush=True)
    return 3


# ---- shared clips and the panels -------------------------------------------------------------------------------------------------

def restrict(read: dict, signs) -> dict:
    """``read`` on ``signs``, in that order, every per-clip field with them."""
    where = {s: i for i, s in enumerate(read["signs"])}
    pick = [where[s] for s in signs]
    out = dict(read)
    for key in PER_CLIP:
        if key in read:
            out[key] = [read[key][i] for i in pick]
    if "part_validity" in read:
        out["part_validity"] = {k: [v[i] for i in pick] for k, v in read["part_validity"].items()}
    out["motion"] = {k: [v[i] for i in pick] for k, v in read["motion"].items()}
    out["hands"] = {h: {k: [v[i] for i in pick] for k, v in row.items()} for h, row in read["hands"].items()}
    out["tracked"] = {k: [v[i] for i in pick] for k, v in read["tracked"].items()}
    return out


def shared_signs(reads: dict) -> list[str]:
    """The clips every form read, in real_two_view's order. Refuses a form holding a clip outside the list, or another reference for one."""
    base = reads["real_two_view"]
    text = dict(zip(base["signs"], base["references"]))
    for name, read in reads.items():
        for sign, reference in zip(read["signs"], read["references"]):
            if text.get(sign) != reference:
                raise SystemExit(f"{name}: {sign} is not this list's clip with this reference")
    common = set.intersection(*(set(r["signs"]) for r in reads.values()))
    return [s for s in base["signs"] if s in common]


def lost_rows(reads: dict, shared) -> dict:
    """Every clip outside the shared set, with the reason each form lost it. A clip gone without a reason refuses."""
    keep, by_clip = set(shared), {}
    for name, read in reads.items():
        for row in read.get("lost") or []:
            by_clip.setdefault(row["sign"], {})[name] = dict(reason=row["reason"], detail=row.get("detail"))
    unexplained = [s for s in reads["real_two_view"]["signs"] if s not in keep and s not in by_clip]
    if unexplained:
        raise SystemExit(f"{len(unexplained)} clips fall outside the shared set with no reason, for example {unexplained[0]}")
    reasons = Counter(reason for forms in by_clip.values() for reason in {r["reason"] for r in forms.values()})
    return dict(count=len(by_clip), by_reason=dict(sorted(reasons.items())), clips=[dict(sign=s, forms=by_clip[s]) for s in sorted(by_clip)])


def aliased(reads: dict) -> dict:
    """``tok_probes.form_panel`` reads real under the name real."""
    return {("real" if k == "real_two_view" else k): v for k, v in reads.items()}


def panel(reads: dict, name: str) -> dict:
    out = P.form_panel(aliased(reads), "real" if name == "real_two_view" else name)
    out["paired"]["real_two_view"] = out["paired"].pop("real")
    out["identical_to_real_two_view"] = out.pop("identical_to_real")
    out["valid_frame_share"] = M.bootstrap_mean(reads[name]["valid_frame_share"])
    out["part_validity"] = {part: M.bootstrap_mean(v) for part, v in reads[name]["part_validity"].items()}
    out["clips"] = len(reads[name]["signs"])
    return out


SIGNER_ROW = ("clips", "bleu", "share_of_gap", "likelihood_minus_blank", "valid_frame_share", "part_validity", "tracked_share", "seconds_over_reference", "paired")


def signer_rows(reads: dict) -> dict:
    """Per signer, the same panels on that signer's shared clips."""
    groups = dict(zip(reads["real_two_view"]["signs"], reads["real_two_view"]["groups"]))
    out = {}
    for signer in sorted(set(groups.values())):
        signs = [s for s in reads["real_two_view"]["signs"] if groups[s] == signer]
        here = {n: restrict(r, signs) for n, r in reads.items()}
        panels = {n: panel(here, n) for n in reads}
        out[signer] = {n: {k: p[k] for k in SIGNER_ROW} for n, p in panels.items()}
    return out


def same_path(reads: dict, stored: dict) -> dict:
    """Differences and shares under one joint bootstrap over the shared clips (n 2,000, seed 0), on BLEU and likelihood minus blank. ``stored``: the stored forms on these clips, {} in a smoke. A row whose form was not read is None."""
    forms = dict(reads, **stored)
    refs = reads["real_two_view"]["references"]
    idx = np.random.default_rng(0).integers(0, len(refs), (BOOT, len(refs)))
    blank = np.asarray(reads["blank"]["nll"])[:, 0]
    dnll = {f: M.delta_nll(blank, np.asarray(r["nll"])[:, 0]) for f, r in forms.items()}
    stats = {f: M.bleu_statistics(r["hypotheses"], refs) for f, r in forms.items()}

    def block(metric: str) -> dict:
        if metric == "likelihood_minus_blank":
            point = {f: float(dnll[f].mean()) for f in forms}
            draws = {f: dnll[f][idx].mean(1) for f in forms}
        else:
            point = {f: float(M.bleu_from_statistics(stats[f].sum(0))) for f in forms}
            draws = {f: M.bleu_from_statistics(stats[f][idx].sum(1)) for f in forms}

        def difference(a, b):
            return dict(point=round(point[a] - point[b], 5), ci95=P._quantiles(draws[a] - draws[b])) if a in point and b in point else None

        def share(a, of):
            if a not in point or of not in point or point[of] == point["blank"]:
                return None
            with np.errstate(divide="ignore", invalid="ignore"):
                resampled = (draws[a] - draws["blank"]) / (draws[of] - draws["blank"])
            return dict(point=round((point[a] - point["blank"]) / (point[of] - point["blank"]), 4), ci95=P._quantiles(resampled))
        return dict(marks_free=dict(monocular_gap=difference("nearest_monocular", "blank"), two_view_gap=difference("stored_v0_nearest_fill", "blank"), monocular_gap_over_two_view_gap=share("nearest_monocular", "stored_v0_nearest_fill"),
                                    monocular_minus_two_view=difference("nearest_monocular", "stored_v0_nearest_fill"), round_trip_monocular_share_of_monocular_gap=share("round_trip_monocular", "nearest_monocular"),
                                    round_trip_monocular_share_of_two_view_gap=share("round_trip_monocular", "stored_v0_nearest_fill"), stored_rt_nearest_share_of_two_view_gap=share("stored_rt_nearest", "stored_v0_nearest_fill")),
                    marked=dict(monocular_gap=difference("real_monocular", "blank"), two_view_gap=difference("real_two_view", "blank"), monocular_gap_over_two_view_gap=share("real_monocular", "real_two_view"), monocular_minus_two_view=difference("real_monocular", "real_two_view"),
                                trimmed_minus_two_view=difference("real_monocular_trimmed", "real_two_view"), round_trip_monocular_share_of_monocular_gap=share("round_trip_monocular", "real_monocular"), round_trip_monocular_share_of_two_view_gap=share("round_trip_monocular", "real_two_view")),
                    round_trip=dict(monocular_minus_real_monocular=difference("round_trip_monocular", "real_monocular"), monocular_minus_two_view_this_run=difference("round_trip_monocular", "rt_nearest_two_view"), monocular_minus_stored=difference("round_trip_monocular", "stored_rt_nearest"), this_run_minus_stored=difference("rt_nearest_two_view", "stored_rt_nearest")),
                    values_and_marks=dict(report_only=True, two_view_values_two_view_marks=point.get("real_two_view"), monocular_values_monocular_marks=point.get("real_monocular_trimmed"), monocular_values_two_view_marks=point.get("monocular_values_two_view_marks"), two_view_values_monocular_marks=point.get("two_view_values_monocular_marks"),
                                          values_under_two_view_marks=difference("monocular_values_two_view_marks", "real_two_view"), values_under_monocular_marks=difference("real_monocular_trimmed", "two_view_values_monocular_marks"),
                                          marks_on_two_view_values=difference("two_view_values_monocular_marks", "real_two_view"), marks_on_monocular_values=difference("real_monocular_trimmed", "monocular_values_two_view_marks")))
    return dict(bleu=block("bleu"), likelihood_minus_blank=block("likelihood_minus_blank"), bootstrap=dict(n=BOOT, seed=0), clips=len(refs))


def beside_bar(read: dict) -> dict:
    """The round trip beside 3.8: where it reads, and whether its interval clears the bar."""
    bleu, ci = M.bleu_ci_fast(read["hypotheses"], read["references"])
    return dict(bar=BAR, bleu=bleu, ci95=ci, minus_bar=round(bleu - BAR, 3), lower_bound_above_bar=ci[0] > BAR, upper_bound_below_bar=ci[1] < BAR)


def check_stored_shape(form: dict, kind: str, name: str, key: str) -> None:
    differs = [what for what, ok in (("kind", form.get("kind") == kind), ("form", form.get("form") == name), ("declared_read", form.get("declared_read") is True), ("report_only", form.get("report_only") is True),
                                     ("list", (form.get("list") or {}).get("sha256") == L.D20_SHA), ("clips", len(form.get("signs") or ()) == L.D20_CLIPS), ("instrument", (form.get("instrument") or {}).get("sha256") == L.TRANSLATOR_SHA),
                                     ("tokenizer", (form.get("tokenizer") or {}).get("sha256") == L.TOKENIZER_SHA)) if not ok]
    if differs:
        raise SystemExit(f"{key}: not the stored read: {differs}")


def load_stored_rt_nearest(store) -> tuple[dict, str]:
    """(V1's stored rt_nearest form on all of D20, its pinned hash), with V1's kind, form, code, list, instrument and tokenizer, and its BLEU and interval recomputed to the plan's figures."""
    raw = store.get(STORED_RT)
    P.check_sha(raw, STORED_RT_SHA256, STORED_RT)
    form = json.loads(raw)
    check_stored_shape(form, "tok-v1-form", "rt_nearest", STORED_RT)
    if not str(form.get("code_digest")).startswith(STORED_RT_DIGEST):
        raise SystemExit(f"{STORED_RT}: not the stored read: another code digest, {form.get('code_digest')}")
    bleu, ci = M.bleu_ci_fast(form["hypotheses"], form["references"])
    if bleu != STORED_RT_BLEU or ci != STORED_RT_CI:
        raise SystemExit(f"{STORED_RT}: reads {bleu} {ci}, not the plan's {STORED_RT_BLEU} {STORED_RT_CI}")
    return form, STORED_RT_SHA256


def load_stored_nearest(store) -> tuple[dict, str]:
    """(V0's stored v0_nearest_fill on all of D20, its pinned hash), its BLEU recomputed to the plan's figure."""
    raw = store.get(STORED_NEAREST)
    P.check_sha(raw, STORED_NEAREST_SHA256, STORED_NEAREST)
    form = json.loads(raw)
    check_stored_shape(form, "tok-v0-form", "v0_nearest_fill", STORED_NEAREST)
    bleu = M.bleu_ci_fast(form["hypotheses"], form["references"])[0]
    if bleu != STORED_NEAREST_BLEU:
        raise SystemExit(f"{STORED_NEAREST}: reads {bleu}, not the plan's {STORED_NEAREST_BLEU}")
    return form, STORED_NEAREST_SHA256


def stored_row(form: dict, sha: str, key: str, reads: dict, place: dict, declared: bool) -> tuple[dict, dict | None]:
    """(a stored form's row, the form on the shared clips). A smoke reads other clips, so the form is checked and not compared."""
    bleu, ci = M.bleu_ci_fast(form["hypotheses"], form["references"])
    row = dict(key=key, sha256=sha, code_digest=form["code_digest"], environment=form.get("environment"), same_environment=form.get("environment") == place, all_500=dict(bleu=bleu, ci95=ci, clips=len(form["signs"])))
    if not declared:
        return dict(row, shared=None, note="a smoke reads train clips: the stored form is checked, not compared"), None
    signs = reads["real_two_view"]["signs"]
    text = dict(zip(form["signs"], form["references"]))
    if any(text.get(s) != r for s, r in zip(signs, reads["real_two_view"]["references"])):
        raise SystemExit(f"{key}: another reference for a shared clip, or a clip it did not read")
    kept = restrict(form, signs)
    bleu, ci = M.bleu_ci_fast(kept["hypotheses"], kept["references"])
    row["shared"] = dict(clips=len(signs), bleu=bleu, ci95=ci, likelihood_minus_blank=M.bootstrap_mean(M.delta_nll(np.asarray(reads["blank"]["nll"])[:, 0], np.asarray(kept["nll"])[:, 0])))
    return row, kept


# ---- the reading -----------------------------------------------------------------------------------------------------------------

def overlaps(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def prestated_case(difference: float, paired_ci, a_ci, b_ci, threshold: float) -> str:
    """Which of the header's cases one metric falls in: a, b, c or unresolved. A label on a reading stated before the read; nothing branches on it."""
    if difference < -threshold:
        return "b"
    if difference > threshold or (difference > 0 and paired_ci[0] > 0):
        return "c"
    return "a" if overlaps(a_ci, b_ci) and paired_ci[0] >= -threshold else "unresolved"


def joint_case(cases: dict) -> str:
    """The metrics' shared case; unresolved when they disagree."""
    values = set(cases.values())
    return values.pop() if len(values) == 1 else "unresolved"


def pair_facts(monocular: dict, two_view: dict, blank: dict) -> dict:
    """One pair, each metric: both reads with their intervals, the paired difference with its interval, the sign, whether the intervals overlap, and that metric's case."""
    refs = monocular["references"]
    base = np.asarray(blank["nll"])[:, 0]
    a, b = M.delta_nll(base, np.asarray(monocular["nll"])[:, 0]), M.delta_nll(base, np.asarray(two_view["nll"])[:, 0])
    bleu_a, ci_a = M.bleu_ci_fast(monocular["hypotheses"], refs)
    bleu_b, ci_b = M.bleu_ci_fast(two_view["hypotheses"], refs)
    bleu = dict(monocular=bleu_a, monocular_ci95=ci_a, two_view=bleu_b, two_view_ci95=ci_b, **M.paired_bleu_fast(M.bleu_statistics(monocular["hypotheses"], refs), M.bleu_statistics(two_view["hypotheses"], refs), n=BOOT))
    mean_a, mean_b = M.bootstrap_mean(a, n=BOOT), M.bootstrap_mean(b, n=BOOT)
    likelihood = dict(monocular=mean_a["mean"], monocular_ci95=mean_a["ci95"], two_view=mean_b["mean"], two_view_ci95=mean_b["ci95"], **M.paired_mean_difference(a, b, n=BOOT))
    out = dict(bleu=bleu, likelihood_minus_blank=likelihood)
    for metric, row, threshold in (("bleu", bleu, THRESHOLD["bleu"]), ("likelihood_minus_blank", likelihood, THRESHOLD["likelihood"])):
        row.update(threshold=threshold, monocular_higher=row["difference"] > 0, intervals_overlap=overlaps(row["monocular_ci95"], row["two_view_ci95"]), case=prestated_case(row["difference"], row["ci95"], row["monocular_ci95"], row["two_view_ci95"], threshold))
    out["case"] = joint_case(dict(bleu=bleu["case"], likelihood=likelihood["case"]))
    return out


def reading_block(reads: dict, stored: dict, alerts: list[str], declared: bool) -> dict:
    """The label rests on the marks-free pair. Withheld in a smoke and on any alert, which includes a shared count under the floor."""
    marks_free = pair_facts(reads["nearest_monocular"], stored["stored_v0_nearest_fill"], reads["blank"]) if "stored_v0_nearest_fill" in stored else None
    marked = pair_facts(reads["real_monocular"], reads["real_two_view"], reads["blank"])
    withheld = ([] if declared else ["a smoke carries no reading"]) + list(alerts)
    return dict(prestated=PRESTATED, marks_free=marks_free, marked_secondary=marked, case=None if withheld or marks_free is None else marks_free["case"], withheld=withheld, case_note="a label on the reading stated in the script's header; nothing branches on it")


def mean_or_none(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 5) if values else None


def group_distance(row: dict, group: str) -> float | None:
    return mean_or_none([row[part] for part in GROUPS[group]])


def regression(x, y) -> dict | None:
    """OLS of y on x with a bootstrap over clips (n 2,000, seed 0). Report only. None with under 3 clips or no spread."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    a, b = np.array(pairs, np.float64).T
    if a.std() == 0:
        return None
    slope, intercept = np.polyfit(a, b, 1)
    idx = np.random.default_rng(0).integers(0, len(a), (BOOT, len(a)))
    with np.errstate(divide="ignore", invalid="ignore"):
        centred = a[idx] - a[idx].mean(1, keepdims=True)
        draws = (centred * (b[idx] - b[idx].mean(1, keepdims=True))).sum(1) / (centred ** 2).sum(1)
    return dict(report_only=True, clips=len(a), slope=round(float(slope), 5), slope_ci95=P._quantiles(draws), intercept=round(float(intercept), 5), r=round(float(np.corrcoef(a, b)[0, 1]), 4) if b.std() else None)


def offset_distribution(rows: list[dict]) -> dict:
    """Every clip's largest offset on the stored clocks, summarised; the per-clip rows stay in ``monocular.json``. Rates as stored, counted."""
    values = np.array([r["max_offset_seconds"] for r in rows], np.float64)
    return dict(clips=len(values), nonzero=int((values > 0).sum()), median=round(float(np.median(values)), 6), p90=round(float(np.percentile(values, 90)), 6), p99=round(float(np.percentile(values, 99)), 6), max=round(float(values.max()), 6),
                stored_fps={str(k): v for k, v in sorted(Counter(r["stored_fps"] for r in rows).items())})


def pairing_block(rows: list[dict], reads: dict, stored: dict) -> dict:
    """Pairing on the shared clips: the hash, the offset, the distances and mirrored control per part and group, and, report only, per-clip likelihood differences on distance."""
    base = np.asarray(reads["blank"]["nll"])[:, 0]

    def dnll(read):
        return M.delta_nll(base, np.asarray(read["nll"])[:, 0])
    distances = {part: M.bootstrap_mean([r["distance"][part] for r in rows if r["distance"][part] is not None]) for part in DISTANCE_POINTS}
    mirrored = {part: M.bootstrap_mean([r["mirrored"][part] for r in rows if r["mirrored"][part] is not None]) for part in DISTANCE_POINTS}
    x = {group: [group_distance(r["distance"], group) for r in rows] for group in GROUPS}
    y = dict(marked=dnll(reads["real_monocular"]) - dnll(reads["real_two_view"]), trimmed=dnll(reads["real_monocular_trimmed"]) - dnll(reads["real_two_view"]))
    if "stored_v0_nearest_fill" in stored:
        y["marks_free"] = dnll(reads["nearest_monocular"]) - dnll(stored["stored_v0_nearest_fill"])
    return dict(clips=len(rows), source_hash_matches=sum(r["source_hash_matches"] for r in rows), max_offset_seconds=max(r["max_offset_seconds"] for r in rows), offsets=offset_distribution(rows), frames_ratio_median=round(float(np.median([r["frames_monocular"] / r["frames_two_view"] for r in rows])), 4),
                distance=distances, mirrored=mirrored, group_distance={g: mean_or_none(v) for g, v in x.items()}, group_mirrored={g: mean_or_none([group_distance(r["mirrored"], g) for r in rows]) for g in GROUPS},
                regression={name: {group: regression(x[group], values.tolist()) for group in GROUPS} for name, values in y.items()})


def plumbing_alerts(two_view: dict, monocular: dict, pairing: dict, shared: int, declared: bool) -> list[str]:
    """Every threshold fixed in the header that this read breaks. It judges plumbing, no form."""
    alerts = []
    for part, floor in VALID_FLOOR.items():
        a, b = float(np.mean(two_view[part])), float(np.mean(monocular[part]))
        if a < floor:
            alerts.append(f"two-view {part} valid on {a:.3f} of frames, under {floor}")
        if b < floor:
            alerts.append(f"monocular {part} valid on {b:.3f} of frames, under {floor}")
        if a > 0 and b / a < VALID_RATIO_FLOOR:
            alerts.append(f"monocular {part} validity is {b / a:.3f} of two-view's, under {VALID_RATIO_FLOOR}")
    for group, ceiling in DISTANCE_CEILING.items():
        here = pairing["group_distance"][group]
        if here is None:
            alerts.append(f"{group}: no frame both paths mark")
        elif here > ceiling:
            alerts.append(f"{group}: keypoint distance {here:.3f}, over {ceiling}")
    for group in MIRRORED_GROUPS:
        here, control = pairing["group_distance"][group], pairing["group_mirrored"][group]
        if here is not None and control is not None and control <= here:
            alerts.append(f"{group}: a mirrored monocular clip sits as near two-view as the clip itself ({control:.3f} against {here:.3f})")
    if pairing["source_hash_matches"] != pairing["clips"]:
        alerts.append(f"{pairing['clips'] - pairing['source_hash_matches']} of {pairing['clips']} two-view clips do not name their monocular row's hash")
    if pairing["max_offset_seconds"] > OFFSET_CEILING:
        alerts.append(f"stored clocks differ by up to {pairing['max_offset_seconds']:.4f} s, over {OFFSET_CEILING}")
    if declared and shared < SHARED_FLOOR:
        alerts.append(f"{shared} shared clips, under {SHARED_FLOOR}")
    return alerts


def load_reads(out_dir: Path) -> tuple[dict, dict]:
    """(every form read, what the monocular step recorded). All one list, instrument, kind of read, code and environment; the unconformed form is present exactly when the record says it was read."""
    plan_path = out_dir / "monocular.json"
    if not plan_path.exists():
        raise SystemExit(f"{plan_path} is missing")
    plan = json.loads(plan_path.read_bytes())
    names = [n for n in FORMS if n != UNCONFORMED or plan["unconformed"]["read"]]
    reads, first = {}, None
    for name in names:
        path = out_dir / f"form-{name}.json"
        if not path.exists():
            raise SystemExit(f"{path} is missing")
        read = json.loads(path.read_bytes())
        if read.get("form") != name:
            raise SystemExit(f"{path} holds form {read.get('form')!r}")
        signature = (read["instrument"]["sha256"], read["declared_read"], read["list"]["list"])
        if first is None:
            first = signature
        elif signature != first:
            raise SystemExit(f"{path}: another instrument, another list, or a smoke mixed with a declared read")
        reads[name] = read
    if not plan["unconformed"]["read"] and (out_dir / f"form-{UNCONFORMED}.json").exists():
        raise SystemExit(f"{UNCONFORMED} was read although the record says it was not")
    digest, _ = P.check_same_run(reads, "the forms")
    if plan.get("code_digest") != digest or plan.get("declared_read") != reads["real_two_view"]["declared_read"]:
        raise SystemExit(f"{plan_path}: another run's record")
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
    control = passed_control(out_dir, digest, place) if declared else None
    shared = shared_signs(all_reads)
    if not shared:
        raise SystemExit("the forms share no clip")
    reads = {n: restrict(r, shared) for n, r in all_reads.items()}
    rt_form, rt_sha = load_stored_rt_nearest(store)
    nearest_form, nearest_sha = load_stored_nearest(store)
    rt_row, rt_kept = stored_row(rt_form, rt_sha, STORED_RT, reads, place, declared)
    nearest_row, nearest_kept = stored_row(nearest_form, nearest_sha, STORED_NEAREST, reads, place, declared)
    stored = dict(stored_rt_nearest=rt_kept, stored_v0_nearest_fill=nearest_kept) if declared else {}
    panels = {n: panel(reads, n) for n in reads}
    keep = set(shared)
    pairing = pairing_block([r for r in plan["pairing"] if r["sign"] in keep], reads, stored)
    alerts = plumbing_alerts(reads["real_two_view"]["part_validity"], reads["real_monocular"]["part_validity"], pairing, len(shared), declared)
    lookups_path = out_dir / "lookups.json"
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, declared_read=declared, smoke=None if declared else "SMOKE on c02 train clips: these numbers are plumbing, not a read", list=first["list"], instrument=first["instrument"], tokenizer=reads["round_trip_monocular"]["tokenizer"],
                  code_digest=digest, environment=place, disclosures=DISCLOSURES, alerts=alerts, reading=reading_block(reads, stored, alerts, declared),
                  clips=dict(list=len(first["signs"]), shared=len(shared), floor=SHARED_FLOOR, outside_shared=lost_rows(all_reads, shared)), forms=panels, per_signer=signer_rows(reads), same_path=same_path(reads, stored), round_trip_beside_bar=beside_bar(reads["round_trip_monocular"]),
                  stored=dict(rt_nearest=rt_row, v0_nearest_fill=nearest_row), pairing=pairing, monocular=dict(conform=plan["conform"], unconformed=plan["unconformed"], clips_loaded=plan["clips_loaded"]), lookups=json.loads(lookups_path.read_bytes()) if lookups_path.exists() else None,
                  rerun=json.loads((out_dir / "rerun.json").read_bytes()) if (out_dir / "rerun.json").exists() else None,
                  checks=dict(blank_identical_to_real_two_view=panels["blank"]["identical_to_real_two_view"], control=control),
                  resources={n: {k: r[k] for k in ("peak_rss_bytes", "gpu_peak_reserved_bytes", "build_seconds", "read_seconds")} for n, r in all_reads.items()})
    L.write_once(store, args.output_key, args.output_file, report)
    summary = {n: dict(bleu=p["bleu"]["bleu"], ci95=p["bleu"]["ci95"], gap_share=p["share_of_gap"]["bleu"], likelihood_minus_blank=p["likelihood_minus_blank"]["mean"], valid_frame_share=p["valid_frame_share"]["mean"]) for n, p in panels.items()}
    reading = report["reading"]
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, declared_read=declared, shared=len(shared), of=len(first["signs"]), lost=report["clips"]["outside_shared"]["by_reason"], forms=summary, validity={n: {part: float(np.mean(v)) for part, v in reads[n]["part_validity"].items()} for n in ("real_two_view", "real_monocular")},
                          pairing={k: pairing[k] for k in ("clips", "source_hash_matches", "max_offset_seconds", "offsets", "frames_ratio_median", "group_distance", "group_mirrored")}, round_trip_beside_bar=report["round_trip_beside_bar"], marked_case=reading["marked_secondary"]["case"], case=reading["case"], withheld=reading["withheld"], alerts=alerts,
                          stored=dict(rt_nearest=rt_row["all_500"], v0_nearest_fill=nearest_row["all_500"]))), flush=True)
    return 0


def run_check(args) -> int:
    """Exit 3 on broken plumbing: not report_only, blank reading as real, a declared read without its passed control, no shared clip, or any threshold fixed in the header, in a smoke too. It judges no form."""
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
    if problems and report.get("reading", {}).get("case") is not None:
        problems.append("a report holding a case label beside an alert")
    print(json.dumps(dict(problems=problems)), flush=True)
    for line in problems:
        print(f"ALERT {line}", flush=True)
    return 3 if problems else 0


def run_upload_log(args) -> int:
    """The job's log, write-once."""
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if store.exists(args.key):
        raise SystemExit(f"{args.key} exists (write-once)")
    store.put(args.key, Path(args.file).read_bytes())
    print(json.dumps(dict(log=args.key)), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    stored = sub.add_parser("stored-check", help="the job's first step: both stored forms against the plan's figures")
    stored.add_argument("--store", default="r2")
    pre = sub.add_parser("preflight", help="the read is not finished or void")
    pre.add_argument("--store", default="r2")
    pre.add_argument("--out-dir", required=True)
    pre.add_argument("--final-key", required=True)
    pre.add_argument("--void-prefix", required=True)
    pre.add_argument("--rerun-cause")
    lookups = sub.add_parser("lookups", help="index objects only: which clips have a monocular state")
    lookups.add_argument("--store", default="r2")
    lookups.add_argument("--out-dir", required=True)
    lookups.add_argument("--smoke-train", type=int, default=0, help=f"this many c02 train clips, at most {SMOKE_CLIPS}, not D20")
    read = sub.add_parser("read")
    read.add_argument("--store", default="r2")
    read.add_argument("--out-dir", required=True)
    read.add_argument("--forms", default=",".join(FORMS))
    read.add_argument("--resume", action="store_true", help="leave out forms already read in --out-dir")
    read.add_argument("--nll-group", type=int, default=2)
    read.add_argument("--instrument-cache")
    read.add_argument("--smoke-train", type=int, default=0, help=f"read this many c02 train clips, at most {SMOKE_CLIPS}, not D20")
    read.add_argument("--smoke-longest", type=int, default=0, help="a smoke's list also holds this many of c02 train's longest clips")
    read.add_argument("--smoke-never-tracked", type=int, default=0, help="a smoke's list also holds this many c02 train clips with a hand never tracked")
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    control = sub.add_parser("control", help="the control's forms against their stored reads; exit 3, voiding the read, otherwise")
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
    return dict(preflight=run_preflight, lookups=run_lookups, read=run_read, control=run_control, merge=run_merge, check=run_check, upload=P.run_upload, **{"stored-check": run_stored_check, "upload-log": run_upload_log})[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
