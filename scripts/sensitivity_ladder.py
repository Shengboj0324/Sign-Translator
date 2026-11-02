"""Sensitivity ladder: how much does each instrument's reading fall when real two-view motion is damaged in known parts by known amounts? Report only: nothing trains, no generated form and no composition is read, and this code gates and selects nothing. It reads D20 through v1 and v1's parent (``--set base``), and later the mono-tuned instrument (``--set mono``).

**Why.** The parent reads both paths equally on its clip-specific margin (99.8% [92.6, 107.6]) while the paths sit 0.587 palm widths apart on the hands (mirror control 1.286). That share means something about hands only if the parent's margin moves when hands move.

**How this will be read (stated before any read; the commit holding this text precedes the job).**
- **If the parent's margin holds with hands still, its 99.8% says nothing about hands.** Holds: ``hands_still``'s share of ``nearest_two_view``'s margin has a lower bound at least 0.75. Only that rung may say it: ``fingers_frozen`` freezes finger articulation alone, and hand-local points keep moving with wrist and forearm rotation.
- Falls: that share's upper bound is at most 0.75, and the paired drop's lower bound is above zero: the parent reads the hands, and its 99.8% does bear on them.
- Otherwise, or if the instrument's own margin has a lower bound at or under zero, unresolved.
- **Finger articulation, on the margin.** ``fingers_frozen`` holds the clip's own middle frame, which leaves one real handshape in the input; ``fingers_neutral`` holds a constant neutral hand. If the own pose falls, the instrument reads finger motion. If the own pose holds and neutral falls, it reads a static handshape, not motion. If both hold, it reads no finger content. Otherwise unresolved. The still baseline keeps the clip's handshape, so a neutral share is biased low: use it to confirm a hold, or read its raw paired drop against ``fingers_frozen`` (``neutral_against_frozen``).
- **v1's case is computed on the margin over its still too.** Its likelihood over blank inherits a presence offset, so that label cannot carry the bold sentence, and up to about 0.18 of a share may be offset.
- **The size of finger change its interval can detect:** the mean 95% half-width of the three noise levels' paired drops, over the slope of the drop on distance (through the origin, the levels' measured distances), in palm widths. ``sees_path_distance`` true, at or under 0.587, is necessary, not sufficient: the 0.587 includes hand orientation, and finger noise has none. "Reads fingers" also needs the noise rungs to agree: drops that rise with distance, the largest with a lower bound above zero. Above 0.587 it could not see a change the size of the path's, and its equal reading of the paths is no evidence on fingers. No slope above zero reads "not detectable in this range".
The ladder is a reading, not a rule: no step branches on it, it starts no training and no decision, and it makes no claim about composition.

**What it cannot show.** Noise is not the path's difference, whose structure is the tracker's; the 0.587 also holds hand orientation, which ``finger_noise_*`` leaves alone (``wrist_turned`` covers that). An instrument can read a damage poorly and still read another. Sensitivity on real signing says nothing about generated motion.

**Forms.** Every damaged form starts from ``nearest_two_view``: real two-view D20, nearest-filled with V0's fill, all points marked, so marks do not vary between rungs (``hand_dropout_marked`` alone changes them). Lengths never change. Noise is fixed by clip, not by rung: the same pattern scaled.
- ``real_two_view``, ``blank``, ``nearest_two_view``, ``still_two_view``: the parts job's, read again here (``still_two_view``: the nearest fill with its middle frame held, flags on).
- ``fingers_frozen``: finger articulation frozen: both hands channels hold the clip's own middle frame, one real handshape; the body channel, so arms and each wrist's rotation, moves, and hand-local points with it.
- ``fingers_neutral``: both hands channels hold a constant neutral pose (``tok_v0.neutral_pose``), the body channel moving.
- ``hands_still``: fingers and each hand's world orientation held at the middle frame, arms moving: hand-local points stand in v1's view.
- ``finger_noise_015``, ``_030``, ``_060``: a smooth rotation noise on every finger joint (per joint and axis, unit-variance Gaussian through a 7-frame box, about 0.3 s), scaled per clip so the mean distance in v1's view of both hands' 40 local points is 0.15, 0.30 or 0.60 palm widths (``path_read.key_distance``, the unit of 0.587).
- ``hand_dropout_marked``: each hand, independently, unmarked on about 25% of frames, MediaPipe's published per-hand rate as the research report gives it (not checked against the source here); a two-state chain with mean run 6 frames (0.25 s), since tracking loss comes in runs: the run length is this read's assumption. ``hand_dropout_filled``: the same frames nearest-filled and marked, how word clips reach an instrument.
- ``wrist_turned``: the same noise on both wrist rotations, fingers kept, scaled to 0.587 palm widths. Wrist positions do not move. ``wrist_moved``: the noise on both forearms with each hand's world orientation held (``path_parts.arms_with_hand_pose``), scaled to the path's 0.231 shoulder widths at the wrists. Hand-local points do not move.
- ``arms_frozen``: the whole body channel holds its middle frame except each wrist rotation, recomputed so the hand's world orientation is kept: arm points stand, hand-local points as they were.

**Each instrument** (one process; the form's name says which, no option can) reads a clip's reference and 4 others, the parts job's draw. **Margin:** mean NLL of the mismatched minus own, per clip, over ``still_two_view``'s. **Likelihood:** own NLL over that instrument's blank. Both are given for every instrument under one joint bootstrap (n 2,000, seed 0), with BLEU. The rungs' shares, paired drops, measured distances and the same rows split by whether v1 trained on the clip (``parent_lib``: 234 of D20 are in its training), report only.

**Controls.** Before the smoke, ``stored-check`` fetches and hashes every stored form and asserts 500 signs, and 5 NLL columns on the parent's, printing each form's environment beside this pod's: a pin failure stops the job with status 1, not a void. v1's ``real_two_view`` and ``blank`` against the pinned stored report (``tok_probes.canary_read``), and in ``--set base`` the parent's four anchors and v1's ``nearest_two_view`` against the parts job's stored forms (at least 0.98 of hypotheses identical, BLEU within 0.05, mean per-clip NLL within 0.001). A miss voids the read, writes a void record and stops it; ``read`` refuses every other form without a passed control of this code and environment. The parent's monitor control is not repeated: reproducing the parts job's forms carries it.
**Alerts** (the label is withheld on any, and in a smoke): any clip or a rung's mean over 5% off its target distance, a rung moving a part it must hold (over 0.001 in v1's view), ``hands_still``'s hand-local points moving (over 0.001), dropout's realised rate outside 0.2-0.3 on D20, an instrument's hash not its pin, fewer than 500 clips on D20, D20's count in v1's training list not 234, or the parent's margins off by over 0.005 from the parts read: 0.2372 on ``nearest_two_view`` over its still, 0.0610 raw on ``still_two_view``.

**Clips and rules.** D20 by ``s2_lib.load_read_list``, which refuses C20 and any other list unread; c00 and other partitions are refused. Monocular states are never opened. ``--smoke-train 5 --smoke-longest 2 --smoke-never-tracked 1``: c02 train clips through every rung; no D20 state in a smoke.

    python scripts/sensitivity_ladder.py instruments --set base
    python scripts/sensitivity_ladder.py preflight --out-dir W --final-key K --void-prefix V
    python scripts/sensitivity_ladder.py read --set base --forms control --out-dir W/forms --instrument-cache W/i
    python scripts/sensitivity_ladder.py control --set base --out-dir W/forms --void-prefix V
    python scripts/sensitivity_ladder.py read --set base --forms rest --resume --out-dir W/forms --instrument-cache W/i
    python scripts/sensitivity_ladder.py merge --set base --out-dir W/forms --output-file W/report.json
    python scripts/sensitivity_ladder.py check --report W/report.json

The job is ``runpod/fleet/sensitivity_ladder.sh``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import functools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import parent_lib as Q  # noqa: E402
import path_parts as PP  # noqa: E402
import path_read as R  # noqa: E402
import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v0 as V  # noqa: E402
import tok_v1 as T  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402

PROBE = "sensitivity-ladder"
JOB_FILE = "runpod/fleet/sensitivity_ladder.sh"
PREFIX = f"{L.O}/round20/sensitivity-ladder"
SETS = dict(base=("v1", "parent"), mono=("mono",))
INSTRUMENT_SHA = dict(v1=L.TRANSLATOR_SHA, parent=PP.UNFITTED_SHA)
BASE = "nearest_two_view"
STILL = "still_two_view"
ANCHORS = ("real_two_view", "blank", BASE, STILL)
NOISE = dict(finger_noise_015=.15, finger_noise_030=.30, finger_noise_060=.60)
RUNGS = ("fingers_frozen", "fingers_neutral", "hands_still", *NOISE, "hand_dropout_marked", "hand_dropout_filled", "wrist_turned", "wrist_moved", "arms_frozen")
FORMS = ANCHORS + RUNGS
PATH_HANDS, MIRROR_HANDS, PATH_WRISTS = .587, 1.286, .231
TARGETS = dict(NOISE, wrist_turned=PATH_HANDS, wrist_moved=PATH_WRISTS)
TARGET_GROUP = dict(dict.fromkeys(NOISE, "hands"), wrist_turned="hands", wrist_moved="wrists")
HOLDS = dict(fingers_frozen=("shoulders", "elbows", "wrists"), fingers_neutral=("shoulders", "elbows", "wrists"), hands_still=("shoulders", "elbows", "wrists"), finger_noise_015=("shoulders", "elbows", "wrists"), finger_noise_030=("shoulders", "elbows", "wrists"), finger_noise_060=("shoulders", "elbows", "wrists"), wrist_turned=("shoulders", "elbows", "wrists"), wrist_moved=("hands",), arms_frozen=("hands",))
BOX, HOLD_TOLERANCE, TARGET_TOLERANCE = 7, 1e-3, .05
DROP_RATE, DROP_RUN, DROP_FLOOR, DROP_CEILING = .25, 6., .2, .3
SEED = 20261105
STREAM = dict(fingers=0, wrist_turned=1, wrist_moved=2, dropout=3)
KEPT = .75
CASE_RUNGS = ("fingers_frozen", "fingers_neutral", "hands_still")
V1_TRAINING_IN_D20 = 234
PARENT_FIGURES = dict(nearest_over_still=.2372, still_raw=.0610)
FIGURE_TOLERANCE = .005
FIT_STEPS, FIT_TOLERANCE, MAX_SCALE = 8, .01, np.pi
BOOT = Q.BOOT
PRESTATED = dict(
    holds="**If the parent's margin holds with hands still, its 99.8% says nothing about hands.** Holds: ``hands_still``'s share of ``nearest_two_view``'s margin has a lower bound at least 0.75.",
    falls="Falls: that share's upper bound is at most 0.75, and the paired drop's lower bound is above zero: the parent reads the hands, and its 99.8% does bear on them.",
    otherwise="Otherwise, or if the instrument's own margin has a lower bound at or under zero, unresolved.",
    fingers="If the own pose falls, the instrument reads finger motion. If the own pose holds and neutral falls, it reads a static handshape, not motion. If both hold, it reads no finger content.",
    neutral="The still baseline keeps the clip's handshape, so a neutral share is biased low: use it to confirm a hold, or read its raw paired drop against ``fingers_frozen`` (``neutral_against_frozen``).",
    v1="Its likelihood over blank inherits a presence offset, so that label cannot carry the bold sentence, and up to about 0.18 of a share may be offset.",
    detect="``sees_path_distance`` true, at or under 0.587, is necessary, not sufficient: the 0.587 includes hand orientation, and finger noise has none.")
NOT_DECIDED = "report only: the header's readings are stated before the read; no rule, bar or verdict here gates anything"
DISCLOSURES = dict(
    training="v1 trained on 234 of D20's clips and was selected on a pool holding 265 more, through two-view. by_v1_training splits every row; the parent never trained on How2Sign.",
    noise="Noise is not the path's difference, whose structure is the tracker's. 0.587 also holds hand orientation, which finger noise leaves alone, so sees_path_distance true is necessary, not sufficient.",
    frozen="fingers_frozen holds the clip's own middle frame, which leaves one real handshape in the input, and hand-local points keep moving with wrist and forearm rotation: it freezes finger articulation, not the hand. Only hands_still may say nothing about hands.",
    neutral="The still baseline keeps the clip's handshape, so fingers_neutral's share is biased low: it can confirm a hold, or read neutral_against_frozen, its raw paired drop against fingers_frozen.",
    presence="v1's likelihood over blank inherits a presence offset: up to about 0.18 of a share may be offset, so the margin reading is what can carry the bold sentence.",
    dropout="The 25% per-hand rate is MediaPipe's published figure as the research report gives it, not checked against the source here; the 6-frame mean run is this read's assumption. The distance row of a marked dropout covers only the frames both mark.",
    candidates="Every instrument reads 5 candidates here, so v1 has a margin too. Its first column is its one-candidate read's own NLL.")


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    try:
        return provenance.code_digest(extra_files=(JOB_FILE,))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


def form_name(instrument: str, form: str) -> str:
    return f"{instrument}__{form}"


def split_name(name: str) -> tuple[str, str]:
    """(instrument, form) of a form's name; anything not a known pair refuses."""
    instrument, _, form = name.partition("__")
    if instrument not in ("v1", "parent", "mono") or form not in FORMS:
        raise SystemExit(f"unknown form {name}")
    return instrument, form


def control_forms(which: str) -> tuple[str, ...]:
    """The forms the control compares: v1's canary always; in the base set, v1's stored nearest fill and the parent's four anchors."""
    canary = (form_name("v1", "real_two_view"), form_name("v1", "blank"))
    if which == "mono":
        return canary
    return canary + (form_name("v1", BASE),) + tuple(form_name("parent", f) for f in ANCHORS)


def all_forms(which: str) -> tuple[str, ...]:
    held = control_forms(which)
    return held + tuple(n for n in (form_name(i, f) for i in SETS[which] for f in FORMS) if n not in held)


def resolve_forms(spec: str, which: str) -> list[str]:
    """``control``, ``rest``, or a comma list of this set's forms."""
    if spec == "control":
        return list(control_forms(which))
    if spec == "rest":
        return [n for n in all_forms(which) if n not in control_forms(which)]
    names = [f for f in spec.split(",") if f]
    outside = [n for n in names if n not in all_forms(which)]
    if outside:
        raise SystemExit(f"not forms of set {which}: {outside}")
    return names


# ---- damage --------------------------------------------------------------------------------------------------------------------------

def rng_of(clip: int, stream: str) -> np.random.Generator:
    return np.random.default_rng([SEED, clip, STREAM[stream]])


def smooth_noise(rng, frames: int, joints: int) -> np.ndarray:
    """(T, J, 3) unit-variance Gaussian through a box of ``BOX`` frames, per joint and axis."""
    z = rng.normal(size=(frames + BOX - 1, joints, 3))
    total = np.concatenate([np.zeros((1, joints, 3)), np.cumsum(z, 0)])
    return (total[BOX:] - total[:-BOX]) / np.sqrt(BOX)


def rotation_matrices(axis_angle: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle to (..., 3, 3), Rodrigues; zero is the identity."""
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    unit = np.divide(axis_angle, theta, out=np.zeros_like(axis_angle), where=theta > 0)
    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
    zero = np.zeros_like(x)
    K = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*x.shape, 3, 3)
    sin, cos = np.sin(theta)[..., None], np.cos(theta)[..., None]
    return np.eye(3) + sin * K + (1 - cos) * K @ K


def turned(channel, joints, noise: np.ndarray, scale: float):
    """``channel`` with these joints' rotations turned by ``noise`` x ``scale``, in each joint's own frame; values stay 6D. ``noise`` is (T, len(joints), 3)."""
    import torch
    from signtranslator.generation.canonical_skeleton import _matrices
    from signtranslator.pose.rotations import matrix_to_rotation_6d
    values = channel.values.copy()
    for k, joint in enumerate(joints):
        out = _matrices(values[:, joint]) @ rotation_matrices(noise[:, k] * scale)
        values[:, joint] = matrix_to_rotation_6d(torch.from_numpy(np.ascontiguousarray(out))).numpy()
    return replace(channel, values=values)


def held_fingers(base):
    """Both hands channels at their middle frame; the body channel moves."""
    middle = len(base.timestamps) // 2
    channels = dict(base.channels)
    for name in T.HANDS:
        c = base.channels[name]
        channels[name] = replace(c, values=np.repeat(c.values[[middle]], len(base.timestamps), 0))
    return replace(base, channels=channels)


def neutral_fingers(base, tokenizer):
    """Both hands channels at a constant neutral pose; the body channel moves."""
    neutral = V.neutral_pose(base, tokenizer)
    channels = dict(base.channels)
    for name in T.HANDS:
        c, values = base.channels[name], neutral.channels[name].values
        if values.shape != c.values.shape:
            raise SystemExit(f"{name}: the neutral pose is {values.shape}, the clip {c.values.shape}")
        channels[name] = replace(c, values=np.repeat(values[[0]], len(base.timestamps), 0).astype(c.values.dtype))
    return replace(base, channels=channels)


def hands_still_state(base):
    """Fingers and each hand's world orientation at the middle frame, arms moving: hand-local points stand in v1's view."""
    held = held_fingers(base)
    return replace(held, channels={**held.channels, "body": PP.arms_with_hand_pose(base, PP.held_still(base))})


def local_motion(state) -> float:
    """The largest move of any hand-local point from its first frame, in v1's view."""
    from signtranslator.data.keypoints2d import from_state, normalise
    points, valid = from_state(state)
    view = normalise(points, valid, y_down=False)[:, list(PP.HAND_LOCAL_POINTS)]
    return float(np.abs(view - view[[0]]).max())


def finger_noised(base, noise: dict, scale: float):
    channels = dict(base.channels)
    for name in T.HANDS:
        channels[name] = turned(base.channels[name], range(base.channels[name].values.shape[1]), noise[name], scale)
    return replace(base, channels=channels)


def finger_noise_of(base, clip: int) -> dict:
    rng = rng_of(clip, "fingers")
    return {name: smooth_noise(rng, len(base.timestamps), base.channels[name].values.shape[1]) for name in T.HANDS}


def body_joints(base, labels) -> list[int]:
    names = list(base.channels["body"].labels)
    return [names.index(x) for x in labels]


def wrist_turned_state(base, noise: np.ndarray, scale: float):
    return replace(base, channels={**base.channels, "body": turned(base.channels["body"], body_joints(base, ("LeftHand", "RightHand")), noise, scale)})


def wrist_moved_state(base, noise: np.ndarray, scale: float):
    moved = replace(base, channels={**base.channels, "body": turned(base.channels["body"], body_joints(base, ("LeftForearm", "RightForearm")), noise, scale)})
    return replace(base, channels={**base.channels, "body": PP.arms_with_hand_pose(moved, base)})


def arms_frozen_state(base):
    return replace(base, channels={**base.channels, "body": PP.arms_with_hand_pose(PP.held_still(base), base)})


def dropout_frames(rng, frames: int) -> np.ndarray:
    """(T,) True on dropped frames: a two-state chain at rate ``DROP_RATE``, mean run ``DROP_RUN`` frames."""
    leave = 1 / DROP_RUN
    enter = leave * DROP_RATE / (1 - DROP_RATE)
    draws = rng.random(frames + 1)
    out = np.zeros(frames, bool)
    state = bool(draws[0] < DROP_RATE)
    for t in range(frames):
        out[t] = state
        state = bool(draws[t + 1] >= leave) if state else bool(draws[t + 1] < enter)
    return out


def dropped_state(base, clip: int) -> tuple:
    """(``base`` with each hand unmarked on its dropped frames, values zero as an untracked frame stores them; each hand's realised rate)."""
    rng = rng_of(clip, "dropout")
    channels, rates = dict(base.channels), {}
    for name in T.HANDS:
        lost = dropout_frames(rng, len(base.timestamps))
        c = base.channels[name]
        channels[name] = replace(c, values=np.where(lost[:, None, None], 0., c.values).astype(c.values.dtype), valid=c.valid & ~lost[:, None], observed=c.observed & ~lost[:, None], confidence=np.where(lost[:, None], 0., c.confidence).astype(c.confidence.dtype))
        rates[name] = float(lost.mean())
    return replace(base, channels=channels), rates


def group_distances(state, base) -> dict:
    """{group: distance from ``base`` in v1's view}, the units of the path read's pairing."""
    row = R.key_distance(state, base)
    return {group: R.group_distance(row, group) for group in R.GROUPS}


def fit_scale(build, base, group: str, target: float) -> tuple:
    """(the state whose distance on ``group`` is ``target``, its scale, its distance): the scale is corrected in proportion, a few rounds, from a tenth of a radian. A clip that cannot reach the target keeps its nearest, and ``merge`` raises it."""
    scale, best = .1, None
    for _ in range(FIT_STEPS):
        state = build(scale)
        distance = group_distances(state, base)[group]
        if distance is None:
            raise SystemExit(f"{group}: no point both forms mark")
        if best is None or abs(distance - target) < abs(best[2] - target):
            best = (state, scale, distance)
        if abs(distance - target) <= FIT_TOLERANCE * target:
            break
        scale = min(scale * target / max(distance, 1e-9), MAX_SCALE)
    return best


def rung(name: str, base, clip: int, tokenizer=None) -> tuple:
    """(the damaged state, what was done to this clip)."""
    if name == "fingers_frozen":
        state, extra = held_fingers(base), {}
    elif name == "fingers_neutral":
        state, extra = neutral_fingers(base, tokenizer), {}
    elif name == "hands_still":
        state = hands_still_state(base)
        extra = dict(local_motion=round(local_motion(state), 6))
    elif name in NOISE:
        noise = finger_noise_of(base, clip)
        state, scale, _ = fit_scale(lambda s: finger_noised(base, noise, s), base, "hands", NOISE[name])
        extra = dict(scale=round(scale, 5))
    elif name == "wrist_turned":
        noise = smooth_noise(rng_of(clip, "wrist_turned"), len(base.timestamps), 2)
        state, scale, _ = fit_scale(lambda s: wrist_turned_state(base, noise, s), base, "hands", PATH_HANDS)
        extra = dict(scale=round(scale, 5))
    elif name == "wrist_moved":
        noise = smooth_noise(rng_of(clip, "wrist_moved"), len(base.timestamps), 2)
        state, scale, _ = fit_scale(lambda s: wrist_moved_state(base, noise, s), base, "wrists", PATH_WRISTS)
        extra = dict(scale=round(scale, 5))
    elif name in ("hand_dropout_marked", "hand_dropout_filled"):
        state, rates = dropped_state(base, clip)
        if name == "hand_dropout_filled":
            state = V.filled(state, neutral=base)
        extra = dict(dropped=rates)
    elif name == "arms_frozen":
        state, extra = arms_frozen_state(base), {}
    else:
        raise SystemExit(f"unknown rung {name}")
    return state, dict(extra, distance=group_distances(state, base))


def build_states(form: str, refs, tokenizer) -> tuple:
    """(states, the states motion is masked by, per-clip rows or None)."""
    from evaluate_text_to_motion import blanked
    if form == "real_two_view":
        return list(refs), list(refs), None
    if form == "blank":
        return blanked(refs), list(refs), None
    bases = [T.cap_form("v0_nearest_fill", s, tokenizer) for s in refs]
    if form == BASE:
        return bases, list(refs), None
    if form == STILL:
        return [PP.held_still(b) for b in bases], list(refs), None
    rows = [rung(form, b, i, tokenizer) for i, b in enumerate(bases)]
    return [r[0] for r in rows], list(bases), [r[1] for r in rows]


# ---- commands -------------------------------------------------------------------------------------------------------------------------

def load_instrument(instrument: str, store, args):
    if instrument == "v1":
        return P.load_reader(store, args)
    if instrument == "parent":
        return PP.load_unfitted(store, args)
    return Q.load_mono(store, args)


def run_instruments(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    if args.set == "base":
        block = PP.instrument_block(store)
    else:
        if not store.exists(Q.TRAINED_RECORD):
            raise SystemExit(f"{Q.TRAINED_RECORD}: no trained record; the mono-tuned instrument is not ready")
        block = json.loads(store.get(Q.TRAINED_RECORD))
    print(json.dumps(dict(instruments=block)), flush=True)
    return 0


def run_stored_check(args) -> int:
    """Before the smoke: every stored form fetched and hashed, 500 signs, 5 NLL columns on the parent's, each form's environment beside this pod's. A pin failure stops with status 1: it is no void, nothing was read."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    P.check_sha(store.get(P.CANARY_REPORT), P.CANARY_REPORT_SHA, P.CANARY_REPORT)
    rows = {}
    for instrument, table, columns in (("parent", Q.STORED_PARENT, Q.CANDIDATES), ("v1", Q.STORED_V1, None)):
        for name in table:
            form = Q.stored_form(store, table, name)
            if len(form.get("signs") or ()) != L.D20_CLIPS:
                raise SystemExit(f"{instrument} {name}: {len(form.get('signs') or ())} signs, not {L.D20_CLIPS}")
            shape = np.asarray(form["nll"]).shape
            if columns is not None and shape[1:] != (columns,):
                raise SystemExit(f"{instrument} {name}: NLL of shape {shape}, not {columns} columns")
            rows[f"{instrument}.{name}"] = dict(sha256=table[name][1], signs=len(form["signs"]), nll_columns=shape[1], environment=form.get("environment"))
    print(json.dumps(dict(stored_check=rows, environment_here=P.environment())), flush=True)
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
    names = resolve_forms(args.forms, args.set)
    out_dir = Path(args.out_dir)
    stored = [n for n in names if (out_dir / f"form-{n}.json").exists()]
    if stored and not args.resume:
        raise SystemExit(f"{out_dir / f'form-{stored[0]}.json'} exists (write-once)")
    names = [n for n in names if n not in stored]
    if not names:
        print(json.dumps(dict(read=0, stored=len(stored))), flush=True)
        return 0
    store = make_store(args.store)
    records, declared, described = R.open_list(store, args)
    digest, place = code_digest(), P.environment()
    if declared and any(n not in control_forms(args.set) for n in names):
        R.passed_control(out_dir, digest, place)
    texts, signs, groups = [r["text"] for r in records], [r["sign"] for r in records], [r["signer_component"] for r in records]
    refs = L.load_states(store, records)
    tokenizer = L.load_tokenizer(store) if any(split_name(n)[1] != "real_two_view" and split_name(n)[1] != "blank" for n in names) else None
    candidates = M.retrieval_candidates(len(texts), Q.CANDIDATES, seed=0)
    readers: dict = {}
    by_form: dict = {}
    for name in names:
        by_form.setdefault(split_name(name)[1], []).append(name)
    for form, group in by_form.items():
        started = time.time()
        states, masks, rows = build_states(form, refs, tokenizer)
        built = time.time()
        validity = [R.part_validity(s) for s in states]
        shared = dict(motion=L.motion_statistics(states, masks), hands=P.hand_statistics(states, masks), tracked=P.tracked_statistics(states))
        for name in group:
            instrument = split_name(name)[0]
            if instrument not in readers:
                readers[instrument] = load_instrument(instrument, store, args)
            reader = readers[instrument]
            begun = time.time()
            raw = reader.read(states, texts, candidates, group=args.nll_group)
            report = dict(kind=f"{PROBE}-form", report_only=True, form=form, instrument_name=instrument, declared_read=declared, list=described, code_digest=digest, environment=place, instrument=reader.info, tokenizer=dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA) if form not in ("real_two_view", "blank") else None,
                          signs=signs, groups=groups, references=texts, hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(), rows=rows, valid_frame_share=[R.valid_frame_share(s) for s in states], part_validity={part: [v[part] for v in validity] for part in R.PARTS}, **shared,
                          build_seconds=round(built - started, 2), read_seconds=round(time.time() - begun, 2), peak_rss_bytes=P.peak_rss_bytes(), gpu_peak_reserved_bytes=P.gpu_peak_reserved_bytes())
            P.write_report_once(out_dir / f"form-{name}.json", report)
            print(json.dumps({name: dict(clips=len(signs), build_seconds=report["build_seconds"], read_seconds=report["read_seconds"], peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2))}), flush=True)
            del raw
        del states, masks, rows
    return 0


def run_control(args) -> int:
    """Exit 3, voiding the read, unless v1's real and blank reproduce the pinned report and, in the base set, the parent's anchors and v1's nearest fill their stored forms. A smoke reads train clips, so it checks the pins and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    store = make_store(args.store)
    raw = store.get(P.CANARY_REPORT)
    P.check_sha(raw, P.CANARY_REPORT_SHA, P.CANARY_REPORT)
    canary_report = json.loads(raw)
    read = {n: json.loads((out / f"form-{n}.json").read_bytes()) for n in control_forms(args.set)}
    if not read[form_name("v1", "real_two_view")]["declared_read"]:
        print(json.dumps(dict(control="not applicable: a smoke reads train clips; the pin is checked and nothing compared")), flush=True)
        return 0
    digest, place = P.check_same_run(read, "the control's forms")
    misses, rows = [], {}
    try:
        canary = P.canary_read(dict(real=read[form_name("v1", "real_two_view")], blank=read[form_name("v1", "blank")]), canary_report)
    except SystemExit as error:
        canary = dict(reads=None, misses=[str(error)], reproduced=False)
    misses += canary["misses"]
    if args.set == "base":
        pairs = [(form_name("v1", BASE), Q.STORED_V1, BASE)] + [(form_name("parent", f), Q.STORED_PARENT, f) for f in ANCHORS]
        for name, table, stored_name in pairs:
            stored = Q.stored_form(store, table, stored_name)      # a pin failure stops here with status 1: nothing is voided
            try:
                rows[name] = Q.same_read(read[name], stored)
            except SystemExit as error:
                rows[name] = dict(misses=[str(error)])
            misses += [f"{name} {m}" for m in rows[name]["misses"]]
    result = dict(canary=canary, stored=rows, misses=misses, reproduced=not misses, code_digest=digest, environment=place)
    path = out / "control.json"
    if not path.exists():
        P.write_report_once(path, result)
    print(json.dumps(result), flush=True)
    if result["reproduced"]:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{digest}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-void", code_digest=digest, environment=place, control=result), indent=1, sort_keys=True).encode())
    print(f"ALERT the control's forms do not reproduce their stored reads: {misses}. The read is void: no other form follows, no retry; a rerun needs new code and a recorded cause", flush=True)
    return 3


# ---- reading ---------------------------------------------------------------------------------------------------------------------------

def per_clip(reads: dict) -> tuple:
    """(margin over the still by form, own NLL gain over blank by form, BLEU statistics by form), one instrument."""
    still = Q.margins(reads[STILL])
    base = np.asarray(reads["blank"]["nll"])[:, 0]
    refs = reads["blank"]["references"]
    return {f: Q.margins(r) - still for f, r in reads.items()}, {f: M.delta_nll(base, np.asarray(r["nll"])[:, 0]) for f, r in reads.items()}, {f: M.bleu_statistics(r["hypotheses"], refs) for f, r in reads.items()}


def mean_row(values: np.ndarray, idx: np.ndarray) -> dict:
    return dict(point=round(float(values.mean()), 5), ci95=P._quantiles(values[idx].mean(1)))


def paired_share(values: np.ndarray, base: np.ndarray, idx: np.ndarray) -> dict | None:
    """mean(values) / mean(base) with its joint interval; None with no denominator."""
    if base.mean() == 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(point=round(float(values.mean() / base.mean()), 4), ci95=P._quantiles(values[idx].mean(1) / base[idx].mean(1)))


def frozen_case(own: dict, share: dict | None, drop: dict) -> str:
    """The header's rule on one instrument's measure. ``own``: its own measure on the base; ``share``: fingers_frozen's share of it; ``drop``: base minus fingers_frozen."""
    if share is None or own["ci95"][0] <= 0:
        return "unresolved"
    if share["ci95"][0] >= KEPT:
        return "holds"
    return "falls" if share["ci95"][1] <= KEPT and drop["ci95"][0] > 0 else "unresolved"


def finger_reading(own: str, neutral: str) -> str:
    """The header's rule on finger articulation: the cases of the own-pose and the neutral rungs."""
    if own == "falls":
        return "reads_finger_motion"
    if own == "holds" and neutral == "falls":
        return "static_handshape_not_motion"
    if own == "holds" and neutral == "holds":
        return "no_finger_content"
    return "unresolved"


def detectable(values: dict, idx: np.ndarray, distances: dict) -> dict:
    """The size of finger change the interval can detect (header), in palm widths: mean half-width of the noise rungs' paired drops, over the slope of the drop on measured distance through the origin."""
    forms = list(NOISE)
    d = np.array([distances[f] for f in forms], np.float64)
    drops = np.array([float((values[BASE] - values[f]).mean()) for f in forms])
    draws = np.stack([(values[BASE] - values[f])[idx].mean(1) for f in forms])
    width = np.mean([(np.percentile(row, 97.5) - np.percentile(row, 2.5)) / 2 for row in draws])
    slope = float(d @ drops / (d @ d))
    slopes = d @ draws / (d @ d)
    seen = [f for f in forms if np.percentile((values[BASE] - values[f])[idx].mean(1), 2.5) > 0]
    size = round(float(width / slope), 4) if slope > 0 else None
    sees = None if size is None else bool(size <= PATH_HANDS)
    agree = bool((np.diff(drops) >= 0).all() and np.percentile(draws[-1], 2.5) > 0)
    return dict(noise_rungs_agree=agree, reads_fingers=bool(sees and agree), levels={f: round(float(x), 4) for f, x in zip(forms, d)}, drops={f: round(float(x), 5) for f, x in zip(forms, drops)}, half_width=round(float(width), 5), slope=round(slope, 5), slope_ci95=P._quantiles(slopes), detectable_palm_widths=size, first_rung_detected=seen[0] if seen else None, sees_path_distance=sees, sees_path_distance_note="necessary, not sufficient: the 0.587 includes hand orientation, and finger noise has none; reads_fingers also needs the noise rungs to agree", path_distance=PATH_HANDS, mirror_control=MIRROR_HANDS, note=None if size is not None else "not detectable in this range")


def instrument_rows(reads: dict, keep: np.ndarray | None = None) -> dict:
    """One instrument's rows on the clips ``keep`` (all by default), under one bootstrap."""
    if keep is not None:
        picks = np.flatnonzero(keep)
        reads = {f: dict(r, nll=[r["nll"][i] for i in picks], hypotheses=[r["hypotheses"][i] for i in picks], references=[r["references"][i] for i in picks]) for f, r in reads.items()}
    margin, likelihood, stats = per_clip(reads)
    n = len(reads["blank"]["references"])
    idx = np.random.default_rng(0).integers(0, n, (BOOT, n))
    out = {}
    for form in reads:
        bleu = float(M.bleu_from_statistics(stats[form].sum(0)))
        boot = M.bleu_from_statistics(stats[form][idx].sum(1))
        base_bleu = M.bleu_from_statistics(stats[BASE][idx].sum(1))
        out[form] = dict(bleu=round(bleu, 4), bleu_ci95=P._quantiles(boot), bleu_drop=dict(point=round(float(M.bleu_from_statistics(stats[BASE].sum(0))) - bleu, 4), ci95=P._quantiles(base_bleu - boot)), likelihood_minus_blank=mean_row(likelihood[form], idx), likelihood_drop=mean_row(likelihood[BASE] - likelihood[form], idx),
                         likelihood_share=paired_share(likelihood[form], likelihood[BASE], idx), margin_over_still=mean_row(margin[form], idx), margin_drop=mean_row(margin[BASE] - margin[form], idx), margin_share=paired_share(margin[form], margin[BASE], idx))
    return out


def distance_means(reads: dict) -> dict:
    """{rung: {group: mean measured distance}}, one instrument's forms (all see the same states)."""
    out = {}
    for form in RUNGS:
        rows = reads[form]["rows"]
        out[form] = {group: R.mean_or_none([r["distance"][group] for r in rows]) for group in R.GROUPS}
    return out


def dropout_rates(reads: dict) -> dict:
    return {hand: round(float(np.mean([r["dropped"][hand] for r in reads["hand_dropout_marked"]["rows"]])), 4) for hand in T.HANDS}


def plumbing_alerts(reads: dict, distances: dict, rates: dict, instrument: str, expected_sha: str | None, declared: bool) -> list[str]:
    alerts = []
    for form, target in TARGETS.items():
        here = distances[form][TARGET_GROUP[form]]
        if here is None or abs(here - target) > TARGET_TOLERANCE * target:
            alerts.append(f"{instrument} {form}: mean {TARGET_GROUP[form]} distance {here}, target {target}")
    for form, target in TARGETS.items():
        rows = reads[form]["rows"]
        bad = [s for s, r in zip(reads[form]["signs"], rows) if r["distance"][TARGET_GROUP[form]] is None or abs(r["distance"][TARGET_GROUP[form]] - target) > TARGET_TOLERANCE * target]
        if bad:
            alerts.append(f"{instrument} {form}: {len(bad)} clips miss the target {target} by over {TARGET_TOLERANCE:.0%}, for example {bad[0]}")
    moved = [s for s, r in zip(reads["hands_still"]["signs"], reads["hands_still"]["rows"]) if r["local_motion"] > HOLD_TOLERANCE]
    if moved:
        alerts.append(f"{instrument} hands_still: hand-local points move on {len(moved)} clips, for example {moved[0]}")
    for form, groups in HOLDS.items():
        for group in groups:
            here = distances[form][group]
            if here is not None and here > HOLD_TOLERANCE:
                alerts.append(f"{instrument} {form}: {group} moved {here}, over {HOLD_TOLERANCE}")
    for hand, rate in rates.items():
        if declared and not DROP_FLOOR <= rate <= DROP_CEILING:
            alerts.append(f"{instrument} dropout's realised rate on {hand} is {rate}, outside {DROP_FLOOR}-{DROP_CEILING}")
    shas = {r["instrument"]["sha256"] for r in reads.values()}
    if len(shas) != 1 or (expected_sha is not None and shas != {expected_sha}):
        alerts.append(f"{instrument}: read by {sorted(shas)}, not its pin {expected_sha}")
    if declared and len(reads["blank"]["signs"]) != L.D20_CLIPS:
        alerts.append(f"{instrument}: {len(reads['blank']['signs'])} clips, not D20's {L.D20_CLIPS}")
    return alerts


def parent_figures(reads: dict) -> dict:
    """The parent's margins the parts read recorded: logged, and compared on D20."""
    return dict(nearest_over_still=round(float(per_clip(reads)[0][BASE].mean()), 5), still_raw=round(float(Q.margins(reads[STILL]).mean()), 5))


def figure_alerts(figures: dict) -> list[str]:
    return [f"the parent's {name} margin is {figures[name]}, over {FIGURE_TOLERANCE} from the parts read's {want}" for name, want in PARENT_FIGURES.items() if abs(figures[name] - want) > FIGURE_TOLERANCE]


def expected_sha_of(instrument: str, store) -> str:
    if instrument in INSTRUMENT_SHA:
        return INSTRUMENT_SHA[instrument]
    if not store.exists(Q.TRAINED_RECORD):
        raise SystemExit(f"{Q.TRAINED_RECORD}: no trained record")
    return json.loads(store.get(Q.TRAINED_RECORD))["sha256"]


def load_reads(out_dir: Path, instrument: str) -> dict:
    reads = {}
    for form in FORMS:
        path = out_dir / f"form-{form_name(instrument, form)}.json"
        if not path.exists():
            raise SystemExit(f"{path} is missing")
        read = json.loads(path.read_bytes())
        if read.get("form") != form or read.get("instrument_name") != instrument:
            raise SystemExit(f"{path} holds {read.get('instrument_name')} {read.get('form')}")
        reads[form] = read
    return reads


def run_merge(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    L.refuse_existing(store, args.output_key, args.output_file)
    out_dir = Path(args.out_dir)
    by_instrument = {i: load_reads(out_dir, i) for i in SETS[args.set]}
    everything = {f"{i}__{f}": r for i, reads in by_instrument.items() for f, r in reads.items()}
    digest, place = P.check_same_run(everything, "the forms")
    if digest != code_digest():
        raise SystemExit(f"the forms are of code {digest}, not this code {code_digest()}: a stale read")
    signatures = {(r["declared_read"], r["list"]["list"], tuple(r["signs"])) for r in everything.values()}
    if len(signatures) != 1:
        raise SystemExit("the forms read other clips, another list, or a smoke mixed with a declared read")
    first = next(iter(everything.values()))
    declared = first["declared_read"]
    control = R.passed_control(out_dir, digest, place) if declared else None
    training = Q.v1_training_signs(store)
    masks = Q.subset_masks(first["signs"], training)
    blocks, alerts = {}, []
    for instrument, reads in by_instrument.items():
        distances, rates = distance_means(reads), dropout_rates(reads)
        alerts += plumbing_alerts(reads, distances, rates, instrument, expected_sha_of(instrument, store), declared)
        rows = instrument_rows(reads)
        margin, likelihood, _ = per_clip(reads)
        n = len(first["signs"])
        idx = np.random.default_rng(0).integers(0, n, (BOOT, n))
        cases = {metric: {name: frozen_case(rows[BASE][own], rows[name][share], rows[name][drop]) for name in CASE_RUNGS} for metric, (share, drop, own) in (("margin", ("margin_share", "margin_drop", "margin_over_still")), ("likelihood", ("likelihood_share", "likelihood_drop", "likelihood_minus_blank")))}
        reading = dict(hands_still=cases["margin"]["hands_still"], says_nothing_about_hands=cases["margin"]["hands_still"] == "holds", finger_articulation=finger_reading(cases["margin"]["fingers_frozen"], cases["margin"]["fingers_neutral"]))
        neutral_against_frozen = dict(margin=mean_row(margin["fingers_frozen"] - margin["fingers_neutral"], idx), likelihood=mean_row(likelihood["fingers_frozen"] - likelihood["fingers_neutral"], idx))
        hand_distance = {f: distances[f]["hands"] for f in NOISE}
        blocks[instrument] = dict(cases=cases, reading_on_margin=reading, neutral_against_frozen=neutral_against_frozen, forms=rows, distances=distances, dropout_rates=rates, detectable=dict(margin=detectable(margin, idx, hand_distance), likelihood=detectable(likelihood, idx, hand_distance)), by_v1_training={name: dict(clips=int(mask.sum()), forms=instrument_rows(reads, mask)) if mask.sum() >= 2 else None for name, mask in masks.items()})
    figures = parent_figures(by_instrument["parent"]) if "parent" in by_instrument else None
    if declared:
        if figures is not None:
            alerts += figure_alerts(figures)
        if int(masks["in_v1_training"].sum()) != V1_TRAINING_IN_D20:
            alerts.append(f"{int(masks['in_v1_training'].sum())} of D20 are in v1's training list, not the disclosed {V1_TRAINING_IN_D20}")
    held = ([] if declared else ["a smoke carries no reading"]) + alerts
    for block in blocks.values():
        block["label"] = None if held else block["reading_on_margin"]
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, set=args.set, declared_read=declared, smoke=None if declared else "SMOKE on c02 train clips: these numbers are plumbing, not a read", list=first["list"], code_digest=digest, environment=place, prestated=PRESTATED, disclosures=DISCLOSURES, alerts=alerts, withheld=held, instruments={i: r["blank"]["instrument"] for i, r in by_instrument.items()}, ladder=blocks, parent_figures=figures,
                  clips=dict(list=len(first["signs"]), in_v1_training=int(masks["in_v1_training"].sum()), outside_v1_training=int(masks["outside_v1_training"].sum())), checks=dict(control=control, blank_identical_to_base={i: round(float(np.mean([a == b for a, b in zip(r["blank"]["hypotheses"], r[BASE]["hypotheses"])])), 4) for i, r in by_instrument.items()}),
                  rerun=json.loads((out_dir / "rerun.json").read_bytes()) if (out_dir / "rerun.json").exists() else None, resources={n: {k: r[k] for k in ("peak_rss_bytes", "gpu_peak_reserved_bytes", "build_seconds", "read_seconds")} for n, r in everything.items()})
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, declared_read=declared, withheld=held, labels={i: b["label"] for i, b in blocks.items()}, parent_figures=figures, readings={i: b["reading_on_margin"] for i, b in blocks.items()}, cases={i: b["cases"] for i, b in blocks.items()}, neutral_against_frozen={i: b["neutral_against_frozen"] for i, b in blocks.items()}, detectable={i: {m: b["detectable"][m]["detectable_palm_widths"] for m in ("margin", "likelihood")} for i, b in blocks.items()}, ladder={i: {f: {k: b["forms"][f][k] for k in ("bleu", "likelihood_minus_blank", "margin_over_still", "margin_share", "likelihood_share")} for f in FORMS} for i, b in blocks.items()}, distances={i: b["distances"] for i, b in blocks.items()}, alerts=alerts)), flush=True)
    return 0


def run_check(args) -> int:
    """Exit 3 on broken plumbing: not report_only, blank reading as the base, a declared read without its passed control, any alert, or a label beside one. It judges no form."""
    report = json.loads(Path(args.report).read_text())
    problems = list(report["alerts"])
    if report.get("report_only") is not True:
        problems.append("the report does not say report_only")
    if any(v >= 1 for v in report["checks"]["blank_identical_to_base"].values()):
        problems.append("blank reads exactly as the base")
    if report.get("declared_read") and not (report["checks"].get("control") or {}).get("reproduced"):
        problems.append("a declared read without its passed control")
    if report["withheld"] and any(b.get("label") is not None for b in report["ladder"].values()):
        problems.append("a report holding a label beside an alert")
    print(json.dumps(dict(problems=problems)), flush=True)
    for line in problems:
        print(f"ALERT {line}", flush=True)
    return 3 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def with_set(q):
        q.add_argument("--store", default="r2")
        q.add_argument("--set", default="base", choices=tuple(SETS))
    with_set(sub.add_parser("instruments", help="the job's first step: what each instrument trained on, from its record; no checkpoint is fetched"))
    stored = sub.add_parser("stored-check", help="before the smoke: every stored form by its hash; a pin failure exits 1")
    stored.add_argument("--store", default="r2")
    pre = sub.add_parser("preflight", help="the read is not finished or void")
    pre.add_argument("--store", default="r2")
    pre.add_argument("--out-dir", required=True)
    pre.add_argument("--final-key", required=True)
    pre.add_argument("--void-prefix", required=True)
    pre.add_argument("--rerun-cause")
    read = sub.add_parser("read")
    with_set(read)
    read.add_argument("--out-dir", required=True)
    read.add_argument("--forms", default="control", help="control, rest, or a comma list of this set's forms")
    read.add_argument("--resume", action="store_true")
    read.add_argument("--nll-group", type=int, default=4)
    read.add_argument("--instrument-cache")
    read.add_argument("--smoke-train", type=int, default=0, help=f"read this many c02 train clips, at most {R.SMOKE_CLIPS}, not D20")
    read.add_argument("--smoke-longest", type=int, default=0)
    read.add_argument("--smoke-never-tracked", type=int, default=0)
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    control = sub.add_parser("control", help="the control's forms against their stored reads; exit 3, voiding the read, otherwise")
    with_set(control)
    control.add_argument("--out-dir", required=True)
    control.add_argument("--void-prefix", required=True)
    merge = sub.add_parser("merge")
    with_set(merge)
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
    return dict(instruments=run_instruments, preflight=run_preflight, read=run_read, control=run_control, merge=run_merge, check=run_check, upload=PP.run_upload, **{"upload-log": R.run_upload_log, "stored-check": run_stored_check})[args.command](args)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
