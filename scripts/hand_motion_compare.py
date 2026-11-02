"""Do generated hands move like real ones? A direct measure of finger motion, stated before computing. REPORT ONLY: it decides nothing, adopts nothing, and no rule reads it.

No training, no pods, no back-translator and no recogniser run. Nothing reads a recogniser's score: from a cache's index only the entry, its seconds, its scorer's name (recogniser or
prior PMI), the chosen sample's number, key and hash are taken. Local, CPU, one BLAS thread.

**Why.** Both back-translators lose about 80% of their clip-specific signal when real finger motion is frozen (docs/NEXT_PHASE_PLAN.md, "Sensitivity ladder result"). So: do generated hands move like real ones?

**What was seen before this header was committed.** The field names of r10's train word records and one c02 train record; the structure of one 3D-LEX state (`accomplish`) and one generated state (`accept`, draw 1), with
their per-joint mean angles from identity; the rest offsets of two 3D-LEX GLBs (bones along local +y; hand x runs pinky to index on the right hand and index to pinky on the left). No speed, range, variety or match number on any source.

**Sources.**
- `generated`: A5 draws 1, 2 and 3, each entry's best-of-16 sample (`round20/comp-a5/draw<k>-cache/attempt1/index.json`). A draw's seven added words are left out: they have no real control and no 3D-LEX entry.
- `phase_a`: `round20/phase-a/cache-w14b-nogains-r10/index.json`, the same recipe on other seeds. It stored best of 4, 8 and 16, so some entries have stored losers. The caches store chosen samples only: **the other samples of the 16 do not exist in R2**.
- `fresh` (if W14b samples on this Mac within limits): new samples of a fixed draw of entries with no selection at all, on CPU, so sample k here is not the GPU's sample k and only the distribution compares.
- `capture`: `derived/3d-lex-canonical-v4`, 24 fps. Admission by `corpus_manifest.usable_channels(record, "skeleton")`; a hand not admitted is masked, even when its state marks it valid. Good, not ground truth: gloves confuse some handshapes, thumbs were never glove-validated, and its values are solved. Wrist orientation flips under occlusion, so only finger articulation is read, never a wrist rotation.
- `real_words`: train records of `asllvd-text-v1`, `rit-text-r6`, `wlasl-text-r6` in `round10-mixed-v1/manifest.json`, by exact `corpus` and `partition == "train"`; served as the dataset serves them (`corpus_manifest.load_sample`: usable channels, trim window, mirror flag); at most 6 clips per entry, a fixed draw going round the performers. Monocular path. The generator trained on these clips, so they are not independent of it.
- `sentences_two_view`, `sentences_monocular`: a fixed draw of 150 How2Sign c02 train clips by exact `corpus`, `partition == "train"` and `signer_component == "c02"`, through both paths. A reference on speed, never compared by word.
- **Refused, raising, before any fetch:** any record whose `partition` is not exactly `train`, any `c00`, any How2Sign signer but c02, any other corpus. The test split's monocular report is never opened.

**Validity.** Real states: only frames `valid & observed` count. Capture: `observed` is false everywhere by construction (solved values), so admitted and `valid`, and the report says so. If a real source turns out to have no observed frames at all, the report names it and reads it on `valid`, marked. Generated: every frame; the report states how generated output marks validity. A speed needs both frames of a step valid on that joint at 24 fps. A handshape needs all 15 joints valid. A hand with fewer than 12 such frames (0.5 s) gives no row, and is counted as excluded, never as zero.

**Common form.** All at 24 fps (`evaluate_text_to_motion.at_rate`). Low-pass: 4th-order zero-phase Butterworth at 6 and 3 Hz over runs of valid frames (`composition_v2.lowpass_values`), then Gram-Schmidt. Joints by label, never by position. Sources that resample (ASLLVD 60 fps, RIT 30, WLASL various) lose some jitter to interpolation; this is stated, not corrected.

**Dominant hand.** Of the hands with a row, the one with more arm travel per frame (upper arm and forearm geodesic step, 6 Hz). Arm only, so it cannot select on finger motion. Equal travel, or a single row, takes the row with more frames, then right.

**1. Motion.** Per hand, per clip. Geodesic joint speed (rad/s) mean and p90, raw, 6 Hz and 3 Hz. Bend speed: change in the angle between a bone and its parent bone (rad/s), which drops twist, so 3D-LEX's solved twist cannot flatter it against swing-IK states, which have none. Active share: frames whose mean joint speed at 6 Hz is at least 0.5 rad/s (and 1.0). Range per joint: p95 minus p5 of bend at 6 Hz. Jitter: lag-1 autocorrelation of the rotation-vector step (white jitter near -0.5, smooth motion near +1), and 1 minus speed at 6 Hz over raw speed. Per word the mean over its clips or samples. Intervals: 95% percentile bootstrap over words (2,000), ratios paired on the common words. Sentences bootstrap over clips.

**2. Variety.** Handshape features of the dominant hand at the sign's core (central 50% of whole-hand-valid frames in time, per-feature median; sensitivity 30%): per finger MCP bend, PIP plus DIP bend (radians), adjacent-finger spread angle (radians), thumb tip to index tip, thumb tip to index knuckle and thumb tip to little knuckle (palm widths on nominal bones, no source's own lengths), and thumb IP bend. Every one is unchanged by mirroring. Mean pairwise distance between different words, unscaled, on common words. Effective clusters: k-means k = 32, seed 0, fitted once on real and capture words only, so a generator cannot move the clusters; `exp(entropy)` and inverse Simpson of the words' occupancy, with every source cut to the same number of words. Word separation: chance a same-word pair (other draw, or other performer) sits closer than a different-word pair.

**3. Word match to capture, no text instrument.** Entry equal to the 3D-LEX name by exact string after `compositional.normalise`: no stemming, no inflection, no variant; multi-word and digit entries cannot match. Exclusions counted. English labels are not sign identity: `requires_review` entries stay in, and a homograph is a miss. For each word its capture entry's rank among all capture entries with a dominant row, by Euclidean distance, features z-scored within each source (primary; absorbs each path's constant offset) and unscaled (secondary). Top-1, top-5, median rank, per sample or clip then per word; centroids secondary. Against analytic chance and a shuffled control (word labels permuted, 200 times). Positive control: real clips of the same glosses, same features, same candidates. Beside it, secondary: each generated sample's rank against real-word centroids, and real clips' leave-one-performer-out rank.

**4. Fidelity to the tokenizer.** d2 (`training-runs/tokens-rvq-d0b-r3d-d2`, step 5000, EMA) round-trips a fixed draw of 300 real word clips (`build_recon_states.decode_state`, no decoder gains, as A5), only if it runs on this Mac within limits; otherwise the report says so and specifies the pod job. Measured on the original's valid frames only: speed and range kept, geodesic error, word match kept.

**Best of 16.** Stored best of 16 against stored losers of the same entry (paired), recogniser-scored entries beside PMI-scored ones as a control; `fresh` against chosen by distribution if it ran.

**What each result would mean, stated before computing. None of it decides anything.**
- **"Far" means a ratio whose 95% interval lies wholly below 0.7, or wholly above 1/0.7.** Compared at 6 and 3 Hz first, because monocular jitter inflates real's raw speed.
- Generated finger speed or range far below real word clips: generated hands move too little, and are a concrete target for work on the generator.
- Generated close at raw speed but far below after low-pass: generated hand motion is mostly jitter, which is not articulation.
- Generated far above real: hands move too much or too noisily, a target of another kind. Near real on speed: the finger signal the instruments miss is not explained by amount of motion.
- Real's own speed is largely jitter (autocorrelation near or below 0, jitter share above 0.5): real speed is not a fair target on raw numbers, and only low-passed comparisons carry weight.
- Generated variety far below real (pairwise distance or effective clusters): the generator is collapsing to few handshapes, also a concrete target. Variety near real but word separation far below real: hands vary but not by word, so the handshape is not tied to the word.
- **If real controls cannot match 3D-LEX, the word-match test is uninformative.** Uninformative means: real's top-5 interval overlaps the shuffled control's 95% range, or its top-5 is under twice chance. Then generated's match is reported and given no reading.
- If real controls do match: generated's lift over chance as a share of real's says how much achievable word-specific handshape generated hands carry. A share with an interval below 0.5 means generated handshapes are much less word-specific than real ones. A generated rank better than real's is suspicious first, and read as an artefact of selection or of few words.
- Round trip loses much of real's motion (speed or range kept far below 1): the tokenizer caps what any prior can produce, and a prior change alone cannot fix it. Round trip keeps it while generated is far below: the loss is in the prior or in sampling.
- Best of 16 above stored losers on recogniser-scored entries, and not on PMI-scored ones: the recogniser's choice favours hand motion. No difference: selection is not the source of generated hands' motion. Stored losers are winners of smaller subsets, so this under-reads any effect.
- Capture and real disagree on speed while generated sits with one of them: which is right is not settled here; gloves and monocular estimation both err.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

SEED = "hand-motion-compare"
FPS = 24.
MIN_FRAMES = 12
CLIPS_PER_ENTRY, SENTENCE_CLIPS, ROUND_TRIP_CLIPS, FRESH_ENTRIES, FRESH_SAMPLES = 6, 150, 300, 150, 4
K_CLUSTERS, SHUFFLES, BOOT = 32, 200, 2000
ACTIVE = (.5, 1.)
FAR = .7
O = "acquisition/2026-10-17-v1/youtube-asl-v1"
A = "acquisition/2026-10-17-v1"
EVAL_MIX = f"{O}/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
GENERATOR_MIX_SHA = "710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d"
W14B = "training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt"
D2 = "training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt"
CAPTURE_MANIFEST = f"{A}/derived/3d-lex-corpus-manifest-v2/manifest.json"
DRAW_INDEX = f"{O}/round20/comp-a5/draw{{k}}-cache/attempt1/index.json"
PHASE_A_INDEX = f"{O}/round20/phase-a/cache-w14b-nogains-r10/index.json"
MONOCULAR = f"{A}/derived/how2sign-video-estimate-v1"
H2S = "how2sign-two-view-text-v1"
WORD_CORPORA = ("asllvd-text-v1", "rit-text-r6", "wlasl-text-r6")
H2S_SIGNER = "c02"
DRAWS = (1, 2, 3)
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
JOINTS = tuple(f"{f}{k}" for f in FINGERS for k in (1, 2, 3))
FEATURES = tuple(f"{f}_{p}" for f in FINGERS[1:] for p in ("mcp", "pip_dip")) + ("spread_index_middle", "spread_middle_ring", "spread_ring_pinky", "thumb_tip_index_tip", "thumb_tip_index_knuckle", "thumb_tip_pinky_knuckle", "thumb_ip")
ARM_LABELS = {"Left": (("LeftUpperArm", "LeftForearm"), ("LeftArm", "LeftForeArm")), "Right": (("RightUpperArm", "RightForearm"), ("RightArm", "RightForeArm"))}
_Y = np.array([0., 1., 0.])
_THUMB_REST_SWING = np.array([2 ** -.5, 2 ** -.5, 0.])
METRICS = ("speed_raw", "speed_6hz", "speed_3hz", "p90_raw", "p90_6hz", "p90_3hz", "bend_raw", "bend_6hz", "bend_3hz", "active_0_5", "active_1_0", "range_mean", "autocorr_raw", "autocorr_6hz", "jitter_share")


# ---- what may be read ------------------------------------------------------------------------------------------------

def refuse(record: dict) -> dict:
    """The record, or SystemExit before anything is fetched: only train records of the named corpora, and of How2Sign only c02. Exact fields; a missing field is a refusal."""
    if record.get("partition") != "train":
        raise SystemExit("refused: a record whose partition is not exactly train")
    if record.get("signer_component") == "c00":
        raise SystemExit("refused: How2Sign signer 9 (c00) is never read")
    corpus = record.get("corpus")
    if corpus == H2S:
        if record.get("signer_component") != H2S_SIGNER:
            raise SystemExit(f"refused: How2Sign here is {H2S_SIGNER} only")
    elif corpus not in WORD_CORPORA:
        raise SystemExit("refused: a record of a corpus this read does not name")
    return record


def word_records(records) -> dict:
    """{entry: [train records]} of the named word corpora, by exact fields, with a state and usable channels."""
    from signtranslator.data_engineering.corpus_manifest import usable_channels
    from signtranslator.generation.compositional import normalise
    out = {}
    for r in records:
        if r.get("corpus") in WORD_CORPORA and r.get("partition") == "train" and r.get("text") and usable_channels(r, "continuous"):
            out.setdefault(" ".join(normalise(r["text"])), []).append(refuse(r))
    return out


def sentence_records(records) -> list:
    """A fixed draw of How2Sign c02 train records, by exact fields."""
    from signtranslator.data_engineering.corpus_manifest import usable_channels
    pool = sorted((r for r in records if r.get("corpus") == H2S and r.get("partition") == "train" and r.get("signer_component") == H2S_SIGNER and usable_channels(r, "continuous")), key=lambda r: r["sign"])
    return [refuse(r) for r in random.Random(f"{SEED}|sentences").sample(pool, min(SENTENCE_CLIPS, len(pool)))]


def pick_clips(records, entry: str, n: int = CLIPS_PER_ENTRY) -> list:
    """At most n records of an entry: a fixed shuffle, going round the performers."""
    order = sorted(records, key=lambda r: r["sign"])
    random.Random(f"{SEED}|clips|{entry}").shuffle(order)
    by = {}
    for r in order:
        by.setdefault(r.get("performer_sha256"), []).append(r)
    out = []
    while len(out) < n and any(by.values()):
        for p in sorted(by, key=str):
            if by[p] and len(out) < n:
                out.append(by[p].pop(0))
    return out


def index_rows(payload: dict, ns=(16,)) -> list:
    """From a word cache's index only what names a state: never a score, probability, rank or summary."""
    rows = []
    for e in payload["entries"]:
        for n in ns:
            c = e["chosen"].get(str(n))
            if c:
                rows.append(dict(entry=e["entry"], seconds=e["seconds"], scorer=e["scorer"], n=n, index=c["index"], state_key=c["state_key"], state_sha256=c["state_sha256"]))
    return rows


def phase_a_rows(payload: dict) -> list:
    """One row per distinct stored sample of each entry, role best16 or loser."""
    out = []
    by = {}
    for row in index_rows(payload, (4, 8, 16)):
        by.setdefault(row["entry"], []).append(row)
    for entry, rows in sorted(by.items()):
        best = next(r["index"] for r in rows if r["n"] == 16)
        seen = set()
        for r in sorted(rows, key=lambda x: -x["n"]):
            if r["index"] in seen:
                continue
            seen.add(r["index"])
            out.append(dict(r, role="best16" if r["index"] == best else "loser"))
    return out


def capture_match(entry: str) -> str | None:
    """The 3D-LEX name an entry can match: itself when it is one run of letters, or nothing. No stemming, no inflection, no variant."""
    from signtranslator.generation.compositional import normalise
    words = normalise(entry)
    return words[0] if len(words) == 1 and words[0].isalpha() and words[0] == entry else None


# ---- rotations and hands ---------------------------------------------------------------------------------------------

def matrices(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(T, J, 3, 3) of 6D columns; identity where the mask is false, so a zero payload never makes a NaN."""
    from signtranslator.generation.motion_metrics import _matrices
    v = np.array(values, dtype=np.float64)
    v[~mask] = (1., 0., 0., 0., 1., 0.)
    return _matrices(v)


def log_steps(R: np.ndarray) -> np.ndarray:
    """(T-1, J, 3) rotation vectors of R_t^T R_{t+1}."""
    Q = np.swapaxes(R[:-1], -1, -2) @ R[1:]
    w = .5 * np.stack((Q[..., 2, 1] - Q[..., 1, 2], Q[..., 0, 2] - Q[..., 2, 0], Q[..., 1, 0] - Q[..., 0, 1]), -1)
    s = np.linalg.norm(w, axis=-1)
    theta = np.arctan2(s, (np.trace(Q, axis1=-2, axis2=-1) - 1) / 2)
    scale = np.where(s > 1e-12, theta / np.clip(s, 1e-12, None), 1.)
    return w * scale[..., None]


def lowpassed(values: np.ndarray, mask: np.ndarray, hz: float) -> np.ndarray:
    """6D values filtered over each joint's valid runs and put back on SO(3); nothing crosses a gap."""
    from signtranslator.generation.composition_v2 import lowpass_values
    from signtranslator.generation.compositional import _gram_schmidt
    v, bad = _gram_schmidt(lowpass_values(values, mask, hz, fps=FPS))
    return np.where(bad[..., None], values, v)


def hand_channel(state, side: str, use_observed: bool):
    """(T, 15, 6) values, (T, 15) mask in JOINTS order, by label. An unknown layout raises."""
    c = state.channels[f"{side.lower()}_hand"]
    labels = {label: k for k, label in enumerate(c.labels)}
    index = [labels[f"{side}Hand{j}"] for j in JOINTS]
    mask = c.valid & c.observed if use_observed else c.valid
    return np.asarray(c.values, dtype=np.float64)[:, index], mask[:, index]


def is_capture(state, side: str) -> bool:
    """3D-LEX's local rotations hold the glTF rest (bones along +y, the hand's x mirrored on the left); others are swing-IK with the thumb rest of ``video_landmarks``."""
    return state.channels[f"{side.lower()}_hand"].convention.startswith("gltf-node-local")


def rests(capture: bool) -> np.ndarray:
    """(15, 3) each joint's rest bone direction in its own frame."""
    out = np.tile(_Y, (15, 1))
    if not capture:
        out[:3] = _THUMB_REST_SWING
    return out


def bends(R: np.ndarray, capture: bool) -> np.ndarray:
    """(T, 15) the angle between a bone and its parent bone (for a first joint, the rest direction in the hand frame), radians. Twist does not enter."""
    r = rests(capture)
    return np.arccos(np.clip(((R @ r[:, :, None])[..., 0] * r).sum(-1), -1., 1.))


def hand_points(R: np.ndarray, side: str, capture: bool) -> dict:
    """{name: (T, 3)} thumb tip, index tip, index knuckle, little knuckle in the hand's frame, on nominal bones (``canonical_skeleton``'s) for every source."""
    from signtranslator.generation.canonical_skeleton import _FINGER_LENGTHS, _KNUCKLES
    r = rests(capture)
    sign = -1. if capture and side == "Left" else 1.      # 3D-LEX's left hand x runs index to pinky
    T = len(R)
    out = {}
    for f, name in enumerate(FINGERS):
        base = np.array([sign * _KNUCKLES[name][0], _KNUCKLES[name][1], 0.])
        p = np.tile(base, (T, 1))
        frame = np.tile(np.eye(3), (T, 1, 1))
        out[f"{name}1"] = p
        for k in range(3):
            frame = frame @ R[:, 3 * f + k]
            p = p + _FINGER_LENGTHS[k] * (frame @ r[3 * f + k])
        out[f"{name}4"] = p
    return out


def frame_features(R: np.ndarray, side: str, capture: bool) -> np.ndarray:
    """(T, len(FEATURES)): radians and palm widths on nominal bones; unchanged by mirroring (module docstring)."""
    from signtranslator.generation.canonical_skeleton import _KNUCKLES
    b = bends(R, capture)
    columns = []
    for f in range(1, 5):
        columns += [b[:, 3 * f], b[:, 3 * f + 1] + b[:, 3 * f + 2]]
    azimuth = []
    for f in range(1, 5):
        d = R[:, 3 * f, :, 1]                 # the proximal bone in the hand frame: R ŷ
        azimuth.append(np.arctan2(d[:, 0], d[:, 1]))
    for a, c in zip(azimuth[:-1], azimuth[1:]):
        columns.append(np.abs(a - c))
    p = hand_points(R, side, capture)
    palm = float(np.hypot(_KNUCKLES["Index"][0] - _KNUCKLES["Pinky"][0], _KNUCKLES["Index"][1] - _KNUCKLES["Pinky"][1]))
    for x, y in (("Thumb4", "Index4"), ("Thumb4", "Index1"), ("Thumb4", "Pinky1")):
        columns.append(np.linalg.norm(p[x] - p[y], axis=-1) / palm)
    columns.append(b[:, 2])
    return np.stack(columns, -1)


def core_features(features: np.ndarray, whole: np.ndarray, frac: float) -> np.ndarray | None:
    """The per-feature median over the central ``frac`` of whole-hand-valid frames in time; None under the frame minimum."""
    k = np.flatnonzero(whole)
    if len(k) < MIN_FRAMES:
        return None
    keep = max(int(round(frac * len(k))), 1)
    start = (len(k) - keep) // 2
    return np.median(features[k[start:start + keep]], axis=0)


def _speeds(R: np.ndarray, mask: np.ndarray):
    """(per-step per-joint speeds with NaN where a step is not valid, rotation vectors)."""
    v = log_steps(R)
    ok = mask[:-1] & mask[1:]
    speed = np.linalg.norm(v, axis=-1) * FPS
    return np.where(ok, speed, np.nan), v, ok


def _autocorr(v: np.ndarray, ok: np.ndarray) -> float | None:
    both = ok[:-1] & ok[1:]
    if not both.any():
        return None
    top = (v[:-1] * v[1:]).sum(-1)[both].sum()
    bottom = (.5 * ((v[:-1] ** 2).sum(-1) + (v[1:] ** 2).sum(-1)))[both].sum()
    return float(top / bottom) if bottom > 1e-18 else None


def _round(x, n=5):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def hand_row(values: np.ndarray, mask: np.ndarray, side: str, capture: bool) -> dict | None:
    """One hand of one clip; None when it holds under ``MIN_FRAMES`` whole-hand-valid frames. ``values``, ``mask``: (T, 15, 6), (T, 15) in JOINTS order at 24 fps."""
    whole = mask.all(axis=1)
    if whole.sum() < MIN_FRAMES:
        return None
    out = dict(frames=int(whole.sum()))
    R = {"raw": matrices(values, mask)}
    for name, hz in (("6hz", 6.), ("3hz", 3.)):
        R[name] = matrices(lowpassed(values, mask, hz), mask)
    speed_mean = {}
    for name, rotation in R.items():
        speed, v, ok = _speeds(rotation, mask)
        if not ok.any():
            return None
        out[f"speed_{name}"] = _round(np.nanmean(speed))
        out[f"p90_{name}"] = _round(np.nanpercentile(speed, 90))
        speed_mean[name] = np.nanmean(speed)
        b = bends(rotation, capture)
        db = np.abs(np.diff(b, axis=0)) * FPS
        out[f"bend_{name}"] = _round(db[ok].mean())
        if name in ("raw", "6hz"):
            out[f"autocorr_{name}"] = _round(_autocorr(v, ok))
        if name == "6hz":
            whole_steps = ok.all(axis=1)
            if whole_steps.any():
                per_step = speed[whole_steps].mean(axis=1)
                out["active_0_5"] = _round((per_step >= ACTIVE[0]).mean())
                out["active_1_0"] = _round((per_step >= ACTIVE[1]).mean())
            out["speed_per_joint_6hz"] = [_round(x, 4) for x in np.nanmean(speed, axis=0)]
            spans = []
            for j in range(15):
                seen = b[mask[:, j], j]
                spans.append(float(np.percentile(seen, 95) - np.percentile(seen, 5)) if len(seen) >= 2 else np.nan)
            out["range_per_joint"] = [_round(x, 4) for x in spans]
            out["range_mean"] = _round(np.nanmean(spans))
    out["jitter_share"] = _round(1 - speed_mean["6hz"] / speed_mean["raw"]) if speed_mean["raw"] > 1e-9 else None
    features = frame_features(R["raw"], side, capture)
    out["core"] = [_round(x, 4) for x in core_features(features, whole, .5)]
    out["core_30"] = [_round(x, 4) for x in core_features(features, whole, .3)]
    return out


def arm_travel(state, side: str, use_observed: bool) -> float | None:
    """Upper arm and forearm geodesic step per frame at 6 Hz: arm only, so it cannot select on fingers."""
    c = state.channels["body"]
    labels = {label: k for k, label in enumerate(c.labels)}
    names = next((n for n in ARM_LABELS[side] if all(x in labels for x in n)), None)
    if names is None:
        return None
    index = [labels[n] for n in names]
    mask = (c.valid & c.observed if use_observed else c.valid)[:, index]
    if mask.sum() < 2:
        return None
    values = np.asarray(c.values, dtype=np.float64)[:, index]
    speed, _, ok = _speeds(matrices(lowpassed(values, mask, 6.), mask), mask)
    return float(np.nanmean(speed)) if ok.any() else None


def dominant_of(rows: dict, travel: dict) -> str | None:
    """Of the hands with a row, the one with more arm travel per frame; equal or unknown takes more frames, then right."""
    have = [s for s in ("Right", "Left") if rows.get(s)]
    if not have:
        return None
    if len(have) == 1:
        return have[0]
    a, b = travel.get("Right"), travel.get("Left")
    if a is not None and b is not None and a != b:
        return "Right" if a > b else "Left"
    return "Right" if rows["Right"]["frames"] >= rows["Left"]["frames"] else "Left"


def clip_row(state, use_observed: bool) -> dict:
    """Both hands of a state at 24 fps, its dominant hand, and how its hands mark validity."""
    from evaluate_text_to_motion import at_rate
    s = at_rate(state, FPS)
    hands, travel, validity = {}, {}, {}
    for side in ("Right", "Left"):
        c = s.channels[f"{side.lower()}_hand"]
        validity[side] = dict(valid=_round(c.valid.all(axis=1).mean(), 4), observed=_round((c.valid & c.observed).all(axis=1).mean(), 4), confidence=_round(float(c.confidence.mean()), 4))
        values, mask = hand_channel(s, side, use_observed)
        hands[side] = hand_row(values, mask, side, is_capture(s, side))
        travel[side] = arm_travel(s, side, use_observed)
    return dict(seconds=_round(float(s.timestamps[-1] - s.timestamps[0]) if len(s.timestamps) > 1 else 0., 3), hands=hands, dominant=dominant_of(hands, travel), travel=dict((k, _round(v)) for k, v in travel.items()), validity=validity)


# ---- statistics ------------------------------------------------------------------------------------------------------

def boot_mean(values, *, boot: int = BOOT, seed: int = 0) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return dict(n=0, mean=None, ci95=None)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (boot, len(v)))].mean(axis=1)
    return dict(n=len(v), mean=_round(v.mean(), 4), ci95=[_round(np.percentile(means, 2.5), 4), _round(np.percentile(means, 97.5), 4)])


def boot_ratio(a, b, *, boot: int = BOOT, seed: int = 0) -> dict:
    """mean(a) / mean(b) on the units holding both, the same units resampled in both."""
    x = np.array([(p, q) for p, q in zip(a, b) if p is not None and q is not None and np.isfinite(p) and np.isfinite(q)], dtype=float)
    if not len(x) or x[:, 1].mean() <= 0:
        return dict(n=len(x), ratio=None, ci95=None, far=None)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(x), (boot, len(x)))
    ratios = x[pick, 0].mean(axis=1) / np.clip(x[pick, 1].mean(axis=1), 1e-12, None)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    far = "below" if hi < FAR else "above" if lo > 1 / FAR else None
    return dict(n=len(x), ratio=_round(x[:, 0].mean() / x[:, 1].mean(), 4), ci95=[_round(lo, 4), _round(hi, 4)], far=far)


def word_means(rows, get) -> dict:
    """{word: mean of ``get(row)`` over its rows}, rows whose value is None left out."""
    by = {}
    for r in rows:
        v = get(r)
        if v is not None:
            by.setdefault(r["entry"], []).append(v)
    return {k: float(np.mean(v)) for k, v in by.items()}


def pairwise_distance(X: np.ndarray, *, boot: int = 500, seed: int = 0) -> dict:
    """Mean distance between different items, with the items resampled."""
    if len(X) < 2:
        return dict(n=len(X), mean=None, ci95=None)
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    full = D.sum() / (len(X) * (len(X) - 1))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(boot):
        k = rng.integers(0, len(X), len(X))
        sub = D[k][:, k]
        different = k[:, None] != k[None]
        out.append(sub[different].mean())
    return dict(n=len(X), mean=_round(full, 4), ci95=[_round(np.percentile(out, 2.5), 4), _round(np.percentile(out, 97.5), 4)])


def kmeans(X: np.ndarray, k: int = K_CLUSTERS, *, seed: int = 0, iterations: int = 100) -> np.ndarray:
    """Centres by k-means++ and Lloyd, fixed by the seed."""
    rng = np.random.default_rng(seed)
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min([((X - c) ** 2).sum(-1) for c in centres], axis=0)
        centres.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    C = np.array(centres)
    for _ in range(iterations):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        new = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    return C


def effective_clusters(X: np.ndarray, centres: np.ndarray, *, boot: int = 1000, seed: int = 0) -> dict:
    """exp(entropy) and inverse Simpson of the items' occupancy of fixed centres, with the items resampled."""
    a = ((X[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
    k = len(centres)

    def read(labels):
        p = np.bincount(labels, minlength=k) / len(labels)
        q = p[p > 0]
        return float(np.exp(-(q * np.log(q)).sum())), float(1 / (q ** 2).sum())
    rng = np.random.default_rng(seed)
    draws = np.array([read(a[rng.integers(0, len(a), len(a))]) for _ in range(boot)])
    e, s = read(a)
    return dict(n=len(X), exp_entropy=_round(e, 3), exp_entropy_ci95=[_round(x, 3) for x in np.percentile(draws[:, 0], [2.5, 97.5])], inverse_simpson=_round(s, 3), inverse_simpson_ci95=[_round(x, 3) for x in np.percentile(draws[:, 1], [2.5, 97.5])], occupied=int(len(set(a.tolist()))), largest_share=_round(np.bincount(a, minlength=k).max() / len(a), 4))


def separation(samples: dict, *, seed: int = 0, others: int = 40) -> dict:
    """Per word with at least two samples: the chance a different word's sample sits further from a sample than the word's own other sample does; ties count half. Bootstrap over words."""
    rng = np.random.default_rng(seed)
    words = sorted(samples)
    pool = [(w, x) for w in words for x in samples[w]]
    if len({w for w, _ in pool}) < 2:
        return dict(words=0, mean=None, ci95=None)
    values = []
    for w in words:
        own = samples[w]
        if len(own) < 2:
            continue
        hits = []
        for i in range(len(own)):
            for j in range(len(own)):
                if i == j:
                    continue
                same = np.linalg.norm(own[i] - own[j])
                for t in rng.integers(0, len(pool), others):
                    if pool[t][0] == w:
                        continue
                    d = np.linalg.norm(own[i] - pool[t][1])
                    hits.append(1. if d > same else .5 if d == same else 0.)
        if hits:
            values.append(float(np.mean(hits)))
    row = boot_mean(values, seed=seed)
    return dict(words=row["n"], mean=row["mean"], ci95=row["ci95"])


def zscored(X: np.ndarray) -> np.ndarray:
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 1e-9, sd, 1.)


def ranks_of(queries: np.ndarray, truth: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1-based rank of target ``truth[i]`` among all targets by distance from query i; ties take their middle rank, so identical candidates cannot flatter or punish a word."""
    D = np.linalg.norm(queries[:, None] - targets[None], axis=-1)
    own = D[np.arange(len(queries)), truth][:, None]
    return (D < own).sum(1) + .5 * ((D == own).sum(1) - 1) + 1


def rank_summary(words, ranks, candidates: int, *, seed: int = 0) -> dict:
    """Top-1, top-5 and median rank, per sample then per word, bootstrapped over words."""
    by = {}
    for w, r in zip(words, ranks):
        by.setdefault(w, []).append(float(r))
    top1, top5, med = [np.array([f(np.array(v)) for v in by.values()]) for f in (lambda v: (v <= 1).mean(), lambda v: (v <= 5).mean(), lambda v: v.mean())]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(by), (BOOT, len(by)))
    interval = lambda x, f: [_round(y, 4) for y in np.percentile(f(x[pick]), [2.5, 97.5])]   # noqa: E731
    return dict(words=len(by), samples=len(words), candidates=candidates, top1=_round(top1.mean(), 4), top1_ci95=interval(top1, lambda a: a.mean(1)), top5=_round(top5.mean(), 4), top5_ci95=interval(top5, lambda a: a.mean(1)), median_rank=_round(np.median(med), 2), median_rank_ci95=interval(med, lambda a: np.median(a, axis=1)))


def shuffled_control(queries: np.ndarray, words: list, targets: np.ndarray, target_of: dict, *, shuffles: int = SHUFFLES, seed: int = 0) -> dict:
    """Word labels permuted among the query words: each word is scored against another word's entry."""
    rng = np.random.default_rng(seed)
    names = sorted(set(words))
    index = {w: k for k, w in enumerate(names)}
    q = np.array([index[w] for w in words])
    truth = np.array([target_of[w] for w in names])
    D = np.linalg.norm(queries[:, None] - targets[None], axis=-1)
    rows = []
    for _ in range(shuffles):
        p = rng.permutation(len(names))
        t = truth[p][q]
        own = D[np.arange(len(q)), t][:, None]
        r = (D < own).sum(1) + .5 * ((D == own).sum(1) - 1) + 1
        per = np.zeros((len(names), 2))
        count = np.bincount(q, minlength=len(names))
        np.add.at(per[:, 0], q, r <= 1)
        np.add.at(per[:, 1], q, r <= 5)
        rows.append((per[:, 0] / count).mean())
        rows[-1] = (rows[-1], (per[:, 1] / count).mean(), np.median(np.bincount(q, weights=r, minlength=len(names)) / count))
    rows = np.array(rows)
    return dict(shuffles=shuffles, top1=_round(rows[:, 0].mean(), 4), top1_range95=[_round(x, 4) for x in np.percentile(rows[:, 0], [2.5, 97.5])], top5=_round(rows[:, 1].mean(), 4), top5_range95=[_round(x, 4) for x in np.percentile(rows[:, 1], [2.5, 97.5])], median_rank=_round(rows[:, 2].mean(), 2), median_rank_range95=[_round(x, 2) for x in np.percentile(rows[:, 2], [2.5, 97.5])])


def match(words: list, X: np.ndarray, names: list, T: np.ndarray, *, scale: bool = True, shuffles: int = SHUFFLES) -> dict:
    """Each sample's rank of its word's target among all targets, with chance and a shuffled control. Words without a target must be out already."""
    target_of = {n: k for k, n in enumerate(names)}
    q, t = (zscored(X), zscored(T)) if scale else (X, T)
    truth = np.array([target_of[w] for w in words])
    row = rank_summary(words, ranks_of(q, truth, t), len(names))
    row["chance"] = dict(top1=_round(1 / len(names), 4), top5=_round(min(5 / len(names), 1.), 4), median_rank=(len(names) + 1) / 2)
    row["shuffled"] = shuffled_control(q, words, t, target_of, shuffles=shuffles)
    return row


def informative(real: dict) -> dict:
    """The stated rule: real's top-5 interval clear of the shuffled control's range, and at least twice chance. Otherwise generated's match gets no reading."""
    lo = real["top5_ci95"][0]
    clear = lo > real["shuffled"]["top5_range95"][1]
    twice = real["top5"] >= 2 * real["chance"]["top5"]
    return dict(top5_clear_of_shuffled=bool(clear), top5_at_least_twice_chance=bool(twice), informative=bool(clear and twice))


def lift_share(generated: dict, real: dict) -> float | None:
    """Generated's top-5 lift over chance as a share of real's."""
    gain = real["top5"] - real["chance"]["top5"]
    return _round((generated["top5"] - generated["chance"]["top5"]) / gain, 4) if gain > 0 else None


# ---- reading the store -----------------------------------------------------------------------------------------------

def _state(store, key: str, sha: str):
    from evaluate_text_to_motion import load_state
    return load_state(store, key, sha)


def _jobs_generated(store) -> list:
    jobs = []
    for k in DRAWS:
        raw = store.get(DRAW_INDEX.format(k=k))
        payload = json.loads(raw)
        for row in index_rows(payload):
            jobs.append(dict(row, source=f"draw{k}", id=f"draw{k}|{row['entry']}"))
    for row in phase_a_rows(json.loads(store.get(PHASE_A_INDEX))):
        jobs.append(dict(row, source="phase_a", id=f"phase_a|{row['entry']}|{row['index']}"))
    return jobs


def _manifest_records(store) -> list:
    raw = store.get(EVAL_MIX)
    if hashlib.sha256(raw).hexdigest() != EVAL_MIX_SHA:
        raise SystemExit("the mix differs from its pin")
    return json.loads(raw)["records"]


def _train_only(store):
    """(word records by entry, the sentence draw); nothing else of the mix stays in memory, and none of it is printed."""
    records = _manifest_records(store)
    return word_records(records), sentence_records(records)


def _monocular_index(store, wanted: set) -> dict:
    """{sample id: (key, sha)} of the wanted c02 train clips, from the train report only: the test split's report stays unopened."""
    keys = [k for k in store.list(f"{MONOCULAR}/reports/") if Path(k).name.startswith("train")]
    if not keys:
        raise SystemExit("no train report of How2Sign's monocular states")
    out = {}
    for key in keys:
        for row in json.loads(store.get(key))["results"]:
            if row.get("sample_id") in wanted and row.get("state_key") and row.get("state_sha256"):
                out[row["sample_id"]] = (row["state_key"], row["state_sha256"])
    return out


def _append(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _seen(path: Path) -> set:
    return {json.loads(line)["id"] for line in path.open()} if path.exists() else set()


def collect_generated(store, work: Path) -> None:
    out = work / "generated.jsonl"
    seen = _seen(out)
    jobs = [j for j in _jobs_generated(store) if j["id"] not in seen]
    start = time.time()
    for n, j in enumerate(jobs):
        row = clip_row(_state(store, j["state_key"], j["state_sha256"]), False)
        _append(out, dict(id=j["id"], source=j["source"], entry=j["entry"], scorer=j["scorer"], role=j.get("role", "best16"), index=j["index"], **row))
        if (n + 1) % 200 == 0:
            print(f"generated {n + 1}/{len(jobs)} {time.time() - start:.0f}s", flush=True)


def collect_capture(store, work: Path) -> None:
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_sample, usable_channels
    out = work / "capture.jsonl"
    seen = _seen(out)
    records = json.loads(store.get(CAPTURE_MANIFEST))["records"]
    cache = ContentCache(work / "cache-capture", 1_500_000_000)
    for r in records:
        if not usable_channels(r, "skeleton") or f"capture|{r['sign']}" in seen:
            continue
        state = load_sample(r, "skeleton", cache=cache, fetch=store.get)
        _append(out, dict(id=f"capture|{r['sign']}", source="capture", entry=r["sign"], performer=r["performer_sha256"][:8], requires_review=(r.get("gloss") or {}).get("requires_review"), admitted=[s for s in ("Left", "Right") if ((r.get("hands") or {}).get(s) or {}).get("status") == "validated"], **clip_row(state, False)))


def collect_real_words(store, work: Path) -> None:
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_sample
    entries = sorted({row["entry"] for row in index_rows(json.loads(store.get(DRAW_INDEX.format(k=1))))})
    pools, _ = _train_only(store)
    out = work / "real_words.jsonl"
    seen = _seen(out)
    cache = ContentCache(work / "cache-real", 1_500_000_000)
    absent = [e for e in entries if e not in pools]
    print(f"real words: {len(entries)} entries, {len(absent)} with no train word clip", flush=True)
    jobs = [(e, r) for e in entries for r in pick_clips(pools.get(e, []), e)]
    start = time.time()
    for n, (e, r) in enumerate(jobs):
        if f"real|{r['sign']}" in seen:
            continue
        refuse(r)
        try:
            row = clip_row(load_sample(r, "continuous", cache=cache, fetch=store.get), True)
        except (ValueError, RuntimeError) as bad:            # a state whose window holds nothing: counted, not hidden
            row = dict(hands=dict(Right=None, Left=None), dominant=None, failed=type(bad).__name__)
        _append(out, dict(id=f"real|{r['sign']}", source="real_words", entry=e, corpus=r["corpus"], performer=r.get("performer_sha256", "")[:8], **row))
        if (n + 1) % 200 == 0:
            print(f"real {n + 1}/{len(jobs)} {time.time() - start:.0f}s", flush=True)


def collect_sentences(store, work: Path) -> None:
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_sample
    _, chosen = _train_only(store)
    index = _monocular_index(store, {r["sign"] for r in chosen})
    out = work / "sentences.jsonl"
    seen = _seen(out)
    cache = ContentCache(work / "cache-sent", 1_500_000_000)
    for r in chosen:
        refuse(r)
        if f"two|{r['sign']}" not in seen:
            _append(out, dict(id=f"two|{r['sign']}", source="sentences_two_view", entry=r["sign"], clip=hashlib.sha256(r["sign"].encode()).hexdigest()[:10], **clip_row(load_sample(r, "continuous", cache=cache, fetch=store.get), True)))
        if r["sign"] in index and f"mono|{r['sign']}" not in seen:
            _append(out, dict(id=f"mono|{r['sign']}", source="sentences_monocular", entry=r["sign"], clip=hashlib.sha256(r["sign"].encode()).hexdigest()[:10], **clip_row(_state(store, *index[r["sign"]]), True)))
    print(f"sentences: {len(chosen)} drawn, {len(index)} with a monocular state", flush=True)


def _temperature():
    return dict(body=1.1, left_hand=1.1, right_hand=1.1, face=1.0)


def collect_fresh(store, work: Path, *, entries: int = FRESH_ENTRIES, samples: int = FRESH_SAMPLES, seconds_limit: float = 1500.) -> dict:
    """New samples with no selection at all (CPU; not the GPU's bits), the composition's recipe without gains. Nothing is stored in R2."""
    import torch
    from signtranslator.generation.motion_rvq import generate_rvq_motion
    from signtranslator.generation.rerank import rerank_seeds
    from signtranslator.pose.resample import resample_motion
    from evaluate_text_to_motion import load_generator
    from exposure_diagnostic import make_post
    from round20_lib import draw_seed
    torch.set_num_threads(1)
    rows = index_rows(json.loads(store.get(DRAW_INDEX.format(k=1))))
    picked = random.Random(f"{SEED}|fresh").sample(sorted(rows, key=lambda r: r["entry"]), min(entries, len(rows)))
    start = time.time()
    _, _, sample = load_generator(store, W14B, GENERATOR_MIX_SHA, "cpu", temperature=_temperature(), decoder_gains=False)
    post = make_post(sample.decoder_gains)
    out = work / "fresh.jsonl"
    seen = _seen(out)
    print(f"fresh: generator loaded in {time.time() - start:.0f}s", flush=True)
    done = 0
    for row in picked:
        for k, seed in enumerate(rerank_seeds(draw_seed(row["entry"], 100), samples)):
            sid = f"fresh|{row['entry']}|{k}"
            if sid in seen:
                continue
            if time.time() - start > seconds_limit:
                print("fresh: time limit reached", flush=True)
                return dict(samples=done, stopped="time limit")
            raw, _ = generate_rvq_motion(sample.prior, sample.tokenizer, [row["entry"]], fps=FPS, durations=[float(row["seconds"])], guidance=1.5, temperature=_temperature(), seed=seed, return_codes=True)
            _append(out, dict(id=sid, source="fresh", entry=row["entry"], scorer=row["scorer"], index=k, **clip_row(post(resample_motion(raw[0], FPS)), False)))
            done += 1
            if done % 25 == 0:
                print(f"fresh {done} {time.time() - start:.0f}s", flush=True)
    return dict(samples=done)


def round_trip_pair(state, decoded) -> dict:
    """Original and decoded on the original's observed-valid frames only; hands the decoded copy marks everywhere valid are masked as the original's."""
    out = {}
    for side in ("Right", "Left"):
        values, mask = hand_channel(state, side, True)
        if mask.all(axis=1).sum() < MIN_FRAMES:
            out[side] = None
            continue
        capture = is_capture(state, side)
        c = decoded.channels[f"{side.lower()}_hand"]
        if len(c.values) != len(values):
            raise RuntimeError("the decoded copy has other frames")
        index = {label: k for k, label in enumerate(c.labels)}
        decoded_values = np.asarray(c.values, dtype=np.float64)[:, [index[f"{side}Hand{j}"] for j in JOINTS]]
        decoded_values = np.where(mask[..., None], decoded_values, 0.)
        a, b = matrices(values, mask), matrices(decoded_values, mask)
        Q = np.swapaxes(a, -1, -2) @ b
        w = .5 * np.stack((Q[..., 2, 1] - Q[..., 1, 2], Q[..., 0, 2] - Q[..., 2, 0], Q[..., 1, 0] - Q[..., 0, 1]), -1)
        error = np.degrees(np.arctan2(np.linalg.norm(w, axis=-1), (np.trace(Q, axis1=-2, axis2=-1) - 1) / 2))
        out[side] = dict(original=hand_row(values, mask, side, capture), decoded=hand_row(decoded_values, mask, side, capture), error_degrees=_round(error[mask].mean(), 3), error_per_joint=[_round(error[mask[:, j], j].mean(), 2) if mask[:, j].any() else None for j in range(15)])
    return out


def collect_round_trip(store, work: Path, *, clips: int = ROUND_TRIP_CLIPS, seconds_limit: float = 900.) -> dict:
    """d2 on a fixed draw of the real word clips, as ``build_recon_states.decode_state``, no gains (A5's recipe)."""
    import torch
    from signtranslator.data.layout_adapter import conform
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_sample
    from build_recon_states import decode_state, load_tokenizer
    from evaluate_text_to_motion import at_rate
    from exposure_diagnostic import make_post
    torch.set_num_threads(1)
    pools, sentences = _train_only(store)
    entries = sorted({row["entry"] for row in index_rows(json.loads(store.get(DRAW_INDEX.format(k=1))))})
    jobs = [(e, r) for e in entries for r in pick_clips(pools.get(e, []), e)]
    jobs = random.Random(f"{SEED}|round-trip").sample(jobs, min(clips, len(jobs)))
    cache = ContentCache(work / "cache-rt", 1_000_000_000)
    template = load_sample(refuse(sorted(sentences, key=lambda r: r["sign"])[0]), "continuous", cache=cache, fetch=store.get)
    start = time.time()
    tokenizer, saved = load_tokenizer(store.get(D2), "cpu")
    post = make_post(None)
    print(f"round trip: tokenizer loaded in {time.time() - start:.0f}s", flush=True)
    out = work / "round_trip.jsonl"
    seen = _seen(out)
    done = 0
    for e, r in jobs:
        if f"rt|{r['sign']}" in seen:
            continue
        if time.time() - start > seconds_limit:
            print("round trip: time limit reached", flush=True)
            return dict(clips=done, stopped="time limit")
        refuse(r)
        try:
            state = at_rate(conform(load_sample(r, "continuous", cache=cache, fetch=store.get), template), FPS)
            decoded = decode_state(state, tokenizer, tokenizer.config.motion, post, text=e)
            row = round_trip_pair(state, decoded)
            row["dominant"] = dominant_of({k: (v or {}).get("original") for k, v in row.items()}, {k: arm_travel(state, k, True) for k in ("Right", "Left")})
        except (ValueError, RuntimeError) as bad:
            row, _ = dict(Right=None, Left=None, failed=type(bad).__name__), None
        _append(out, dict(id=f"rt|{r['sign']}", source="round_trip", entry=e, corpus=r["corpus"], **row))
        done += 1
        if done % 25 == 0:
            print(f"round trip {done} {time.time() - start:.0f}s", flush=True)
    return dict(clips=done)


# ---- the report ------------------------------------------------------------------------------------------------------

def hand_of(row: dict, which: str) -> dict | None:
    side = row.get("dominant") if which == "dominant" else which
    return (row.get("hands") or {}).get(side) if side else None


def _load(work: Path, name: str) -> list:
    path = work / f"{name}.jsonl"
    return [json.loads(line) for line in path.open()] if path.exists() else []


def metric_means(rows: list, which: str) -> dict:
    """{metric: {word: mean}} of one hand."""
    out = {}
    for m in METRICS:
        out[m] = word_means(rows, lambda r, m=m: (hand_of(r, which) or {}).get(m))
    return out


def motion_table(rows: list, which: str) -> dict:
    means = metric_means(rows, which)
    return {m: boot_mean(list(means[m].values())) for m in METRICS}


def ratio_table(a: list, b: list, which: str) -> dict:
    """a over b on the words holding both, resampling words."""
    x, y = metric_means(a, which), metric_means(b, which)
    out = {}
    for m in METRICS:
        common = sorted(set(x[m]) & set(y[m]))
        out[m] = boot_ratio([x[m][w] for w in common], [y[m][w] for w in common])
    return out


def per_joint(rows: list, which: str, key: str) -> dict:
    by = {}
    for r in rows:
        h = hand_of(r, which)
        if h and h.get(key):
            by.setdefault(r["entry"], []).append([np.nan if v is None else v for v in h[key]])
    return {w: np.nanmean(np.array(v), axis=0) for w, v in by.items()}


def joint_table(a: list, b: list, c: list) -> dict:
    """The dominant hand per joint: generated, real, capture and generated over each, on each pair's common words."""
    out = {}
    for key in ("speed_per_joint_6hz", "range_per_joint"):
        x, y, z = (per_joint(rows, "dominant", key) for rows in (a, b, c))
        rows = {}
        for j, name in enumerate(JOINTS):
            ry = sorted(set(x) & set(y))
            rz = sorted(set(x) & set(z))
            rows[name] = dict(generated=_round(np.nanmean([x[w][j] for w in x]), 3), real=_round(np.nanmean([y[w][j] for w in y]), 3), capture=_round(np.nanmean([z[w][j] for w in z]), 3),
                              over_real=_round(np.nanmean([x[w][j] for w in ry]) / np.nanmean([y[w][j] for w in ry]), 3) if ry else None, over_capture=_round(np.nanmean([x[w][j] for w in rz]) / np.nanmean([z[w][j] for w in rz]), 3) if rz else None)
        out[key] = rows
    return out


def features_by_word(rows: list, *, key: str = "core") -> dict:
    """{word: [dominant-hand feature vectors]}."""
    out = {}
    for r in rows:
        h = hand_of(r, "dominant")
        if h and h.get(key) and all(v is not None for v in h[key]):
            out.setdefault(r["entry"], []).append(np.array(h[key], dtype=float))
    return out


def one_per_word(by: dict, words: list, seed: int) -> np.ndarray:
    rng = random.Random(f"{SEED}|single|{seed}")
    return np.array([rng.choice(by[w]) for w in words])


def distance_ratio(X: np.ndarray, Y: np.ndarray, *, boot: int = 500, seed: int = 0) -> dict:
    """Mean pairwise distance of X over Y, on the same words, with the same resampled words in both."""
    Dx, Dy = np.linalg.norm(X[:, None] - X[None], axis=-1), np.linalg.norm(Y[:, None] - Y[None], axis=-1)
    rng = np.random.default_rng(seed)
    n = len(X)
    out = []
    for _ in range(boot):
        k = rng.integers(0, n, n)
        d = k[:, None] != k[None]
        out.append(Dx[k][:, k][d].mean() / Dy[k][:, k][d].mean())
    lo, hi = np.percentile(out, [2.5, 97.5])
    return dict(n=n, ratio=_round(Dx.sum() / Dy.sum(), 4), ci95=[_round(lo, 4), _round(hi, 4)], far="below" if hi < FAR else "above" if lo > 1 / FAR else None)


def cluster_ratio(X: np.ndarray, Y: np.ndarray, centres: np.ndarray, *, boot: int = 1000, seed: int = 0) -> dict:
    """exp(entropy) of X over Y, on the same words, with the same resampled words in both."""
    a = ((X[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
    b = ((Y[:, None] - centres[None]) ** 2).sum(-1).argmin(1)

    def e(labels):
        p = np.bincount(labels, minlength=len(centres)) / len(labels)
        q = p[p > 0]
        return float(np.exp(-(q * np.log(q)).sum()))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(boot):
        k = rng.integers(0, len(a), len(a))
        out.append(e(a[k]) / e(b[k]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return dict(n=len(a), ratio=_round(e(a) / e(b), 4), ci95=[_round(lo, 4), _round(hi, 4)], far="below" if hi < FAR else "above" if lo > 1 / FAR else None)


def variety_report(generated: dict, real: dict, capture: dict, *, draws) -> dict:
    """``generated``: {draw: {word: [features]}}."""
    common = sorted(set.intersection(*[set(generated[d]) for d in draws], set(real)))
    three = sorted(set(common) & set(capture))
    out = dict(words_generated_and_real=len(common), words_all_three=len(three), per_draw={})
    centroids = np.array([np.mean(v, axis=0) for _, v in sorted(real.items())])
    entries = np.array([v[0] for _, v in sorted(capture.items())])
    centres = kmeans(np.concatenate([centroids, entries]))
    out["kmeans"] = dict(k=K_CLUSTERS, seed=0, fitted_on=f"{len(centroids)} real word centroids and {len(entries)} capture entries, unscaled; never generated")
    real_single = [one_per_word(real, common, s) for s in range(3)]
    out["real_words"] = dict(pairwise=[pairwise_distance(x) for x in real_single])
    out["real_words"]["clusters"] = [effective_clusters(x, centres) for x in real_single]
    for d in draws:
        x = one_per_word(generated[d], common, 0)
        out["per_draw"][str(d)] = dict(pairwise=pairwise_distance(x), clusters=effective_clusters(x, centres), pairwise_over_real=[distance_ratio(x, y) for y in real_single], clusters_over_real=[cluster_ratio(x, y, centres) for y in real_single])
    if three:
        cap = np.array([capture[w][0] for w in three])
        out["capture_on_common_words"] = dict(pairwise=pairwise_distance(cap), clusters=effective_clusters(cap, centres))
        out["generated_over_capture"] = {str(d): dict(pairwise=distance_ratio(one_per_word(generated[d], three, 0), cap), clusters=cluster_ratio(one_per_word(generated[d], three, 0), cap, centres)) for d in draws}
        out["real_over_capture"] = dict(pairwise=distance_ratio(one_per_word(real, three, 0), cap), clusters=cluster_ratio(one_per_word(real, three, 0), cap, centres))
    pooled = {w: [x for d in draws for x in generated[d].get(w, [])] for w in common}
    per_performer = {w: real[w] for w in common if len(real[w]) >= 2}
    both = sorted(w for w in common if len(pooled[w]) >= 2 and w in per_performer)
    out["separation"] = dict(generated=separation({w: pooled[w] for w in both}), real_words=separation({w: per_performer[w] for w in both}), note="chance that another word's sample sits further than the word's own other sample; 0.5 is no word signal. Generated: other draws. Real: other clips, so other performers where it can, on the same words.")
    return out


def _flat(by: dict, words) -> tuple[list, np.ndarray]:
    w = [x for x in words for _ in by[x]]
    return w, np.array([v for x in words for v in by[x]])


def word_match_report(sources: dict, capture: dict, *, centroid_form: bool = True) -> dict:
    """Each source's rank of its word's capture entry; every source's own words and the words all sources hold."""
    names = sorted(capture)
    T = np.array([capture[n][0] for n in names])
    out = dict(candidates=len(names))
    matched = {k: sorted(set(v) & set(names)) for k, v in sources.items()}
    common = sorted(set.intersection(*[set(v) for v in matched.values()]))
    out["common_words"] = len(common)
    for name, by in sources.items():
        row = {}
        for label, words in (("own_words", matched[name]), ("common_words", common)):
            if not words:
                continue
            w, X = _flat(by, words)
            row[label] = dict(z_scored=match(w, X, names, T), unscaled=match(w, X, names, T, scale=False, shuffles=50))
            if centroid_form:
                C = np.array([np.mean(by[x], axis=0) for x in words])
                row[label]["centroid_z_scored"] = match(list(words), C, names, T, shuffles=50)
        out[name] = row
    return out


def top5_share(generated: dict, real: dict, capture: dict, words, *, boot: int = BOOT, seed: int = 0) -> dict:
    """Generated's top-5 lift over chance as a share of real's on the same words, words resampled; z-scored within each source."""
    names = sorted(capture)
    T = zscored(np.array([capture[n][0] for n in names]))
    index = {n: k for k, n in enumerate(names)}

    def per_word(by):
        w, X = _flat(by, words)
        r = ranks_of(zscored(X), np.array([index[x] for x in w]), T)
        out = {}
        for x, y in zip(w, r):
            out.setdefault(x, []).append(y <= 5)
        return np.array([np.mean(out[x]) for x in words])
    g, q = per_word(generated), per_word(real)
    chance = 5 / len(names)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(words), (boot, len(words)))
    gains = q[pick].mean(1) - chance
    shares = (g[pick].mean(1) - chance) / np.where(np.abs(gains) > 1e-9, gains, np.nan)
    return dict(words=len(words), share=_round((g.mean() - chance) / (q.mean() - chance), 4) if q.mean() != chance else None, ci95=[_round(x, 4) for x in np.nanpercentile(shares, [2.5, 97.5])])


def to_real_centroids(generated: dict, real_rows: list) -> dict:
    """Secondary: generated samples against real word centroids, and real clips against their word's centroid without their own performer. Z-scored within each source. The generator trained on these clips, so this is not independent."""
    by, perf = {}, {}
    for r in real_rows:
        h = hand_of(r, "dominant")
        if h and h.get("core") and all(v is not None for v in h["core"]):
            by.setdefault(r["entry"], []).append(np.array(h["core"], float))
            perf.setdefault(r["entry"], []).append(r.get("performer"))
    names = sorted(by)
    C = np.array([np.mean(by[n], axis=0) for n in names])
    mean, sd = C.mean(0), np.where(C.std(0) > 1e-9, C.std(0), 1.)
    T = (C - mean) / sd
    index = {n: k for k, n in enumerate(names)}
    words = sorted(set(generated) & set(names))
    w, X = _flat(generated, words)
    out = dict(candidates=len(names), generated=rank_summary(w, ranks_of(zscored(X), np.array([index[x] for x in w]), T), len(names)))
    all_clips = np.array([v for n in names for v in by[n]])
    cm, cs = all_clips.mean(0), np.where(all_clips.std(0) > 1e-9, all_clips.std(0), 1.)
    w2, r2 = [], []
    for n in names:
        for clip, p in zip(by[n], perf[n]):
            rest = [v for v, q in zip(by[n], perf[n]) if q != p]
            if not rest:
                continue
            d = np.linalg.norm((clip - cm) / cs - T, axis=-1)
            own = np.linalg.norm((clip - cm) / cs - (np.mean(rest, axis=0) - mean) / sd)
            d[index[n]] = own
            w2.append(n)
            r2.append((d < own).sum() + 1)
    out["real_leave_performer_out"] = rank_summary(w2, r2, len(names)) if w2 else None
    out["chance_top5"] = _round(min(5 / len(names), 1.), 4)
    return out


def best_of_16(phase_a: list, fresh: list, chosen: list) -> dict:
    """Stored best against stored losers of the same entry, by scorer; and chosen draws against fresh unselected samples."""
    out = {}
    for scorer in ("recogniser", "prior_pmi"):
        rows = [r for r in phase_a if r["scorer"] == scorer]
        best, loser = [r for r in rows if r["role"] == "best16"], [r for r in rows if r["role"] == "loser"]
        out[scorer] = dict(entries_with_a_stored_loser=len({r["entry"] for r in loser}))
        keep = {r["entry"] for r in loser}
        best = [r for r in best if r["entry"] in keep]
        out[scorer]["best_over_loser"] = {m: v for m, v in ratio_table(best, loser, "dominant").items() if m in ("speed_6hz", "speed_3hz", "bend_6hz", "active_0_5", "range_mean", "speed_raw")}
        out[scorer]["best_over_loser_right_hand"] = {m: v for m, v in ratio_table(best, loser, "Right").items() if m in ("speed_6hz", "bend_6hz", "range_mean")}
    if fresh:
        words = {r["entry"] for r in fresh}
        out["chosen_over_fresh"] = {}
        for scorer in ("recogniser", "prior_pmi", None):
            a = [r for r in chosen if r["entry"] in words and (scorer is None or r["scorer"] == scorer)]
            b = [r for r in fresh if scorer is None or r["scorer"] == scorer]
            out["chosen_over_fresh"][scorer or "all"] = {m: v for m, v in ratio_table(a, b, "dominant").items() if m in ("speed_raw", "speed_6hz", "speed_3hz", "bend_6hz", "active_0_5", "range_mean")}
    return out


def round_trip_report(rows: list, capture: dict, generated: list) -> dict:
    good = [r for r in rows if r.get("dominant")]
    out = dict(clips_read=len(rows), clips_with_a_hand=len(good), failed=sum(1 for r in rows if r.get("failed")), recipe="d2 step 5000 EMA, codes decoded as a generation, no gains (A5), on the original's observed-valid frames")
    for which in ("Right", "Left", "dominant"):
        o, d = [], []
        errors = []
        for r in good:
            side = r["dominant"] if which == "dominant" else which
            part = r.get(side)
            if part and part.get("original") and part.get("decoded"):
                o.append(dict(entry=r["entry"], dominant="x", hands=dict(x=part["original"])))
                d.append(dict(entry=r["entry"], dominant="x", hands=dict(x=part["decoded"])))
                errors.append((r["entry"], part["error_degrees"]))
        if not o:
            continue
        table = ratio_table(d, o, "dominant")
        by = {}
        for w, e in errors:
            by.setdefault(w, []).append(e)
        out[which] = dict(clips=len(o), kept=table, error_degrees=boot_mean([float(np.mean(v)) for v in by.values()]), original=motion_table(o, "dominant"), decoded=motion_table(d, "dominant"))
        if which == "dominant":
            out["generated_over_decoded"] = ratio_table(generated, d, "dominant")
            per = [r[r["dominant"]]["error_per_joint"] for r in good if r.get(r["dominant"])]
            out["error_per_joint_degrees"] = {j: _round(np.nanmean([np.nan if p[k] is None else p[k] for p in per]), 2) for k, j in enumerate(JOINTS)}
            originals, decodeds = features_by_word(o), features_by_word(d)
            out["word_match"] = word_match_report(dict(original=originals, decoded=decodeds), capture, centroid_form=False)
    return out


def validity_report(groups: dict) -> dict:
    out = {}
    for name, rows in groups.items():
        rows = list(rows)
        have = [r for r in rows if r.get("dominant")]
        row = dict(rows=len(rows), with_a_hand=len(have), excluded_no_hand=len(rows) - len(have), failed=sum(1 for r in rows if r.get("failed")), dominant_right_share=_round(np.mean([r["dominant"] == "Right" for r in have]), 3) if have else None)
        for side in ("Right", "Left"):
            v = [(r.get("validity") or {}).get(side) for r in rows if (r.get("validity") or {}).get(side)]
            if v:
                row[side] = dict(rows_with_hand=sum(1 for r in rows if hand_of(r, side)), mean_valid_frames=_round(np.mean([x["valid"] for x in v]), 4), mean_valid_and_observed_frames=_round(np.mean([x["observed"] for x in v]), 4), mean_confidence=_round(np.mean([x["confidence"] for x in v]), 4))
        out[name] = row
    return out


def exclusions(entries: list, capture_all: dict, capture_ok: dict) -> dict:
    """How the generator's entries match 3D-LEX, by the stated rule."""
    multi = [e for e in entries if len(e.split()) > 1]
    other = [e for e in entries if len(e.split()) == 1 and capture_match(e) is None]
    single = [e for e in entries if capture_match(e)]
    absent = [e for e in single if e not in capture_all]
    no_hand = [e for e in single if e in capture_all and e not in capture_ok]
    return dict(entries=len(entries), multi_word_cannot_match=len(multi), digits_or_apostrophes_cannot_match=len(other), single_word=len(single), not_in_converted_3d_lex=len(absent), in_3d_lex_without_an_admitted_hand_row=len(no_hand), matched=len(single) - len(absent) - len(no_hand), examples_multi_word=multi[:5], examples_other=other[:5], examples_no_admitted_hand=no_hand[:5])


def analyse(work: Path) -> dict:
    gen = _load(work, "generated")
    draws = {k: [r for r in gen if r["source"] == f"draw{k}"] for k in DRAWS}
    pooled = [r for k in DRAWS for r in draws[k]]
    phase = [r for r in gen if r["source"] == "phase_a"]
    phase_best = [r for r in phase if r["role"] == "best16"]
    capture_rows, real, sentences, fresh, trip = _load(work, "capture"), _load(work, "real_words"), _load(work, "sentences"), _load(work, "fresh"), _load(work, "round_trip")
    two, mono = [r for r in sentences if r["source"] == "sentences_two_view"], [r for r in sentences if r["source"] == "sentences_monocular"]
    by_corpus = {c: [r for r in real if r.get("corpus") == c] for c in WORD_CORPORA}
    sides = ("Right", "Left", "dominant")
    motion = {}
    for name, rows in dict(generated_three_draws=pooled, draw1=draws[1], draw2=draws[2], draw3=draws[3], phase_a_best16=phase_best, capture=capture_rows, real_words=real, sentences_two_view=two, sentences_monocular=mono, fresh_unselected=fresh, **{f"real_{c}": v for c, v in by_corpus.items()}).items():
        if rows:
            motion[name] = {h: motion_table(rows, h) for h in sides}
    ratios = {"generated_over_real_words": {h: ratio_table(pooled, real, h) for h in sides}, "generated_over_capture": {h: ratio_table(pooled, capture_rows, h) for h in sides}, "real_words_over_capture": {h: ratio_table(real, capture_rows, h) for h in sides},
              "per_draw_over_real_words_dominant": {f"draw{k}": ratio_table(draws[k], real, "dominant") for k in DRAWS}, "generated_over_real_by_corpus_dominant": {c: ratio_table(pooled, v, "dominant") for c, v in by_corpus.items()},
              "monocular_over_two_view_same_clips_dominant": ratio_table(mono, two, "dominant")}
    speed = {}
    for name, rows in dict(sentences_two_view=two, sentences_monocular=mono).items():
        if rows:
            speed[f"generated_over_{name}"] = {m: boot_ratio_independent(motion["generated_three_draws"]["dominant"][m], motion[name]["dominant"][m]) for m in ("speed_raw", "speed_6hz", "speed_3hz", "bend_6hz", "range_mean", "active_0_5")}
    ratios["generated_over_sentences_not_paired"] = speed
    f_draw = {k: features_by_word(draws[k]) for k in DRAWS}
    f_real, f_cap = features_by_word(real), features_by_word(capture_rows)
    variety = variety_report(f_draw, f_real, f_cap, draws=DRAWS)
    pooled_f = {}
    for k in DRAWS:
        for w, v in f_draw[k].items():
            pooled_f.setdefault(w, []).extend(v)
    sources = dict(generated_three_draws=pooled_f, phase_a_best16=features_by_word(phase_best), real_words=f_real, **{f"real_{c}": features_by_word(v) for c, v in by_corpus.items()})
    wm = word_match_report(dict((k, sources[k]) for k in ("generated_three_draws", "real_words")), f_cap)
    wm["other_sources"] = {k: word_match_report({k: sources[k]}, f_cap, centroid_form=False) for k in sources if k not in ("generated_three_draws", "real_words") and sources[k]}
    f30 = dict(generated_three_draws={}, real_words=features_by_word(real, key="core_30"))
    for k in DRAWS:
        for w, v in features_by_word(draws[k], key="core_30").items():
            f30["generated_three_draws"].setdefault(w, []).extend(v)
    wm["sensitivity_core_30"] = word_match_report(f30, features_by_word(capture_rows, key="core_30"), centroid_form=False)
    common = sorted(set(pooled_f) & set(f_real) & set(f_cap))
    rule = informative(wm["real_words"]["common_words"]["z_scored"])
    wm["uninformative_rule"] = rule
    wm["generated_share_of_real_lift"] = top5_share(pooled_f, f_real, f_cap, common) if common else None
    wm["to_real_centroids"] = to_real_centroids(pooled_f, real)
    entries = sorted({r["entry"] for r in draws[1]})
    capture_all = {r["entry"] for r in capture_rows}
    return dict(kind="hand-motion-compare", report_only=True, decides_nothing=True, sources=dict(generated_rows=len(pooled), phase_a_rows=len(phase), capture_rows=len(capture_rows), real_word_rows=len(real), sentence_rows=len(sentences), fresh_rows=len(fresh), round_trip_rows=len(trip)),
                validity=validity_report(dict(generated=pooled, phase_a=phase, capture=capture_rows, real_words=real, sentences_two_view=two, sentences_monocular=mono, fresh=fresh)), generated_validity_note="generated output marks every frame and joint valid, none observed, confidence 0: it never says a hand is absent, and both hands always have a row, while real hands often have no detection",
                motion=motion, ratios=ratios, per_joint_dominant=joint_table(pooled, real, capture_rows), variety=variety, word_match_exclusions=exclusions(entries, capture_all, f_cap), word_match=wm, best_of_16=best_of_16(phase, fresh, pooled), round_trip=round_trip_report(trip, f_cap, pooled) if trip else dict(ran=False))


def boot_ratio_independent(a: dict, b: dict) -> dict:
    """Two means from unrelated units (words and sentence clips): the ratio with each interval's half-width combined on the log scale. A reference only."""
    if not a.get("mean") or not b.get("mean") or not a.get("ci95") or not b.get("ci95") or min(a["ci95"][0], b["ci95"][0]) <= 0:
        return dict(ratio=None, ci95=None)
    half = lambda r: (np.log(r["ci95"][1]) - np.log(r["ci95"][0])) / 2       # noqa: E731
    h = float(np.hypot(half(a), half(b)))
    ratio = a["mean"] / b["mean"]
    return dict(ratio=_round(ratio, 4), ci95=[_round(ratio * np.exp(-h), 4), _round(ratio * np.exp(h), 4)], far="below" if ratio * np.exp(h) < FAR else "above" if ratio * np.exp(-h) > 1 / FAR else None, note="unpaired: different words and clips")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=("generated", "capture", "real_words", "sentences", "fresh", "round_trip", "analyse"))
    p.add_argument("--work", required=True, help="the local folder for row files and downloads")
    p.add_argument("--out", default=str(ROOT / "reports/hand_motion_compare.json"))
    args = p.parse_args(argv)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if args.command == "analyse":
        report = analyse(work)
        Path(args.out).write_text(json.dumps(report, indent=1, default=float) + "\n")
        print(f"{args.out} {hashlib.sha256(Path(args.out).read_bytes()).hexdigest()}", flush=True)
        return 0
    from signtranslator.data.text_motion_dataset import R2Store
    import os
    store = R2Store(os.environ)
    fn = dict(generated=collect_generated, capture=collect_capture, real_words=collect_real_words, sentences=collect_sentences, fresh=collect_fresh, round_trip=collect_round_trip)[args.command]
    fn(store, work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
