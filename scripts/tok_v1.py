"""V1: does neutral fill limit generated signing, and how much would a tokenizer keep without it? Report only; nothing trains; this code declares nothing and holds no rule.

V0 read neutral fill as the main cost of the all-valid path on v1 (``identity`` against ``v0_nearest_fill``). d2's reconstruction and skeleton losses are masked by validity, so its decoder had no target on filled frames: neutral reaches d2
through the encoder's input and the commitment and codebook terms, and what the decoder does there is free. D20 only, instrument v1, through ``tok_probes`` and ``tok_v0``'s reads, canary and digest, write-once under ``tok-probes-v1/``.
``--smoke-train N`` reads c02 train clips and says so.

**(a) The true cap.** ``rt_nearest``: real with nearest-valid fill, all valid, through d2's encoder, codes and decoder, no gains. ``no_quant_nearest``: the same with no codes. ``rt_post_fill``: the neutral-filled clip decoded as d2 was trained, then filled
from its own tracked frames, so the input is what d2 knows and the output has no neutral. Compared with ``v0_nearest_fill`` (the input, no network) and ``round_trip`` (neutral-filled, read as is). ``rt_post_fill`` above ``rt_nearest`` means the input mismatch matters.

**The flag** (``flag-fit``, c02 train only, fixed before any read: ``reports/tok_v1_flag.json``, SHA-pinned). The decoder's output on a filled frame is not the fill: decoded-neutral frames sit far from identity. So a part's frame is flagged when its mean geodesic angle
to the part's decoded-neutral mean pose is under the part's threshold. Pose and threshold (best catch minus false alarm on whole degrees) come from one set of clips; catch and false-alarm rates are reported on held-out clips. Parts: each hand's finger joints, and each wrist rotation (reported, not decided on).

**(b) Motion only, no reader** (``neutral``). ``generated``: the share of a composed form's hand frames under the flag, minus d2's decoded real on tracked frames, clip by clip, within wrist-activity terciles (edges from the same c02 fit), so a resting clip is not read as neutral. ``positive_control``: the
flag's catch rate on D20's own decoded-neutral frames. Flag sensitivity at half and one and a half times the threshold. ``word_alignment`` (report only): a word's flagged frames against where its training clips were untracked, within each time bin, with a shuffled-entry null. ``corpus_neutral``: untracked hand frames per corpus in W14b's
and d2's training mixes. W14b masks its cross entropy where a part is unobserved (``part_valid_loss``), so generated neutral may well be absent. The wrist-rotation row is reported: wrist rotation is often untracked and yet supervised in the body channel.

**(c) A sensitivity, never a score.** ``a9p``: a9@predicted, composed again by ``round15_compose_eval --export-only`` (composed states were never stored; settings are pinned and checked). ``a9p_replaced``: its flagged hand frames hold that hand's nearest unflagged frame. ``a9p_control``: the same hold, with the same run lengths per hand, on unflagged frames at random,
so the edit itself is controlled. The sensitivity is ``a9p_replaced`` − ``a9p_control``. It modifies generated output after the fact: never a score, never M1, never selection. Hand speed and variation are reported beside it, wrist only as a check. ``composed-check`` compares ``a9p`` with the stored a10 report on hypotheses and NLL.

``reads`` in the merged report gives each difference and share with a bootstrap interval, on likelihood minus blank first, then BLEU. ``scripts/tok_v1_rule.py`` reads it; the job never calls that.

The job is ``runpod/fleet/tok_v1.sh``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
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

import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v0 as V  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402

PROBE = "tok-v1"
JOB_FILE = "runpod/fleet/tok_v1.sh"
FPS = L.FPS
HANDS = ("left_hand", "right_hand")
PREFIX = f"{L.O}/tok-probes-v1"
CAP_FORMS = ("real", "blank", "identity", "round_trip", "v0_nearest_fill", "rt_nearest", "no_quant_nearest", "rt_post_fill")
SENSITIVITY_FORMS = ("a9p", "a9p_replaced", "a9p_control")
FORMS = CAP_FORMS + SENSITIVITY_FORMS
ANCHORS = ("real", "blank", "identity", "round_trip", "v0_nearest_fill")
COMPOSED_FORM = "core_crossfade@predicted"
A9_INDEX = f"{PREFIX}/composed/a9/index.json"
A3_INDEX = f"{PREFIX}/composed/a3-top10/index.json"
A9_CACHE, A9_CACHE_SHA = f"{L.O}/round20/phase-a/cache-w14b-nogains-r10/index.json", "507ee9cc8f436107863d203ec844e46114d55d7a053883f030d901b07c7f137b"
A3_CACHE, A3_CACHE_SHA = f"{L.O}/round20/comp-a/cache-a3-top30-nogains/index.json", "2d614f416abf2d7558c36b7ba117d6042aac81be31373f31878654dab2dfff5c"
A10_REPORT, A10_SHA = f"{L.O}/round20/phase-a/a10-predicted-nogains-r10.json", "1e912c9a3cac0f8c581e1542c53de41c8152196d371ee61e450e691836af537d"
TOKENS_FILE, TOKENS_SHA = "reports/comp_a3_tokens.json", "c1105410ef8d502cfcf145768329fe1f72886106afdf83cd0474bfdd654cb776"
STORED_FORMS = {"identity": (f"{L.O}/tok-probes/forms/form-identity.json", "24958febc663ab00f984f8f24bea791e3e5a0904a191311e49a213e4f58e0356"),
                "round_trip": (f"{L.O}/tok-probes/forms/form-round_trip.json", "18e6b17524db4df91b9770d197d7dd45d1a7314cd3685bd2008e5342243da4fb"),
                "v0_nearest_fill": (f"{L.O}/tok-probes-v0/forms/form-v0_nearest_fill.json", "e1bfd0ac62d72593b58370e48477a8db9bb7c85c6cc54b57790db64d7637939f")}
FLAG_FILE, FLAG_SHA = "reports/tok_v1_flag.json", "6cb4ffb75bb94fb384bfcdf6dec91b1defad778b0e307c564983a83b36349e04"
FLAG_FIT_CLIPS, FLAG_HOLD_CLIPS, FLAG_SEED = 300, 150, 20261104
FLAG_GRID = tuple(float(d) for d in range(2, 46))
FLAG_SCALES = (.5, 1.5)                  # the flag's sensitivity, reported
PARTS = dict(left_hand=("left_hand", None, 0), right_hand=("right_hand", None, 1), left_wrist=("body", "LeftHand", 0), right_wrist=("body", "RightHand", 1))
HAND_PARTS, WRIST_PARTS = ("left_hand", "right_hand"), ("left_wrist", "right_wrist")
TERCILES = ("low", "mid", "high")
BINS, NULL_SHUFFLES = 20, 200
WORD_CLIPS, CORPUS_CLIPS, SAMPLE_SEED, FETCH_THREADS = 8, 300, 20261103, 16
BOOT = 2000
_SHARED_EXPORT = dict(mix_sha256=L.EVAL_MIX_SHA, sentences_sha256=L.D20_SHA, selection="stage1", timing="sign-fitted", word_cache_key=A9_CACHE, word_cache_sha256=A9_CACHE_SHA, word_cache_n=16,
                      predicted_duration=True, expect_duration_model="-0.083,0.3691,18289", expect_model_sha256="42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21", recognizer_run="sign-recognizer-v2",
                      variants="core_crossfade", export_only=True, export_forms=COMPOSED_FORM)
EXPORT_PINS = dict(a9=dict(_SHARED_EXPORT, core_margin=.1, vocabulary_extra_top=None, word_cache_extra_sha256=None),
                   a3=dict(_SHARED_EXPORT, core_margin=0., vocabulary_extra_top=10, vocabulary_extra_sha256=TOKENS_SHA, word_cache_extra_sha256=A3_CACHE_SHA))
COMPARATORS = dict(identity=("real",), round_trip=("identity",), v0_nearest_fill=("identity", "real"), rt_nearest=("v0_nearest_fill", "round_trip"), no_quant_nearest=("v0_nearest_fill", "rt_nearest"),
                   rt_post_fill=("rt_nearest", "round_trip"), a9p_replaced=("a9p_control", "a9p"), a9p_control=("a9p",))
NOT_DECIDED = "report only: the declaration's reading is its own; no rule, bar or verdict is computed here"
DISCLOSURES = dict(
    sensitivity="a9p_replaced and a9p_control modify generated output after the fact. It is a sensitivity of the instrument to these frames: never a score, never M1, never selection.",
    decoder="d2's reconstruction and skeleton losses are masked by validity, so its decoder had no target on filled frames. Neutral reaches d2 through the encoder's input and the commitment and codebook terms.",
    flag="The flag reads distance to d2's decoded-neutral mean pose, fit on c02 train clips. Its held-out false-alarm rate is not zero: a generated hand near that pose reads as flagged whatever produced it, and the terciles only control for activity.",
    alignment="Report only. A word's untracked share comes from at most 8 of its training clips, by time-normalised position, with no alignment of the sign inside the clip.",
    a3_tokens="The a3 cache holds only some of the top 10 tokens as entries; tokens_requested and entries_found say how many.",
    cap="rt_nearest sends d2 input it was not trained on (no neutral stretches), so it reads what d2 keeps as it stands, not what a retrained d2 would.",
    a9p="Composed again, not stored: composed states were never saved. Its settings are checked against pins, and composed-check compares its hypotheses and NLL with the stored a10 read.")


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    try:
        return provenance.code_digest(extra_files=(JOB_FILE,))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


# ---- the flag and neutral frames: pure measures ---------------------------------------------------------------------------------

def part_arrays(state, part: str) -> tuple:
    """((T, J, 6) values, (T, J) validity) of a part's joints."""
    name, joint, _ = PARTS[part]
    c = state.channels[name]
    if joint is None:
        return c.values, c.valid
    k = list(c.labels).index(joint)
    return c.values[:, [k]], c.valid[:, [k]]


def part_distance(state, part: str, pose) -> np.ndarray:
    """(T,) mean geodesic degrees to ``pose`` (J, 6) over the frame's valid joints; NaN on a frame with none."""
    from signtranslator.generation.motion_metrics import geodesic_degrees
    values, valid = part_arrays(state, part)
    if not valid.any():
        return np.full(len(values), np.nan)
    target = np.broadcast_to(np.asarray(pose, np.float64), values.shape)
    angle = np.where(valid, geodesic_degrees(np.where(valid[..., None], values, target), target), 0.)
    count = valid.sum(1)
    return np.where(count > 0, angle.sum(1) / np.maximum(count, 1), np.nan)


def mean_pose(frames: np.ndarray) -> np.ndarray:
    """(J, 6) the mean of (N, J, 6) rotations, back on the rotation manifold."""
    from signtranslator.generation.motion_metrics import _matrices
    matrices = _matrices(np.asarray(frames, np.float64).mean(0))
    return np.concatenate((matrices[..., 0], matrices[..., 1]), axis=-1)


def activity(state) -> np.ndarray:
    """(T, 2) each wrist's speed in v1's view, smoothed over 5 frames, gaps interpolated; NaN where a wrist is never seen."""
    steps = P.wrist_steps(state)
    out = np.full(steps.shape, np.nan)
    for k in range(2):
        seen = np.isfinite(steps[:, k])
        if seen.any():
            frames = np.arange(len(steps))
            out[:, k] = np.convolve(np.pad(np.interp(frames, frames[seen], steps[seen, k]), 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return out


def tercile_of(speed: np.ndarray, edges) -> np.ndarray:
    """(T,) 0, 1 or 2, or -1 where the speed is unknown."""
    return np.where(np.isfinite(speed), np.searchsorted(edges, np.nan_to_num(speed), side="right"), -1)


def decoded_state(tokenizer, state):
    """d2's raw decoder output for this clip, as it was trained: neutral-filled input, every frame the decoder's, no gains. A generator's states pass through this decoder."""
    import torch
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.text_to_motion import assemble_state, diffusion_targets, native_values, unit_to_raw
    cfg = tokenizer.config.motion
    with torch.no_grad():
        unit = diffusion_targets(cfg, collate_multichannel([state]))
        rec = tokenizer.decode_latents(tokenizer.quantize(tokenizer.encode(unit))[0], unit.shape[1])
        raw = unit_to_raw(cfg, rec.double())
    return assemble_state(cfg, "x", native_values(cfg, {k: v[0] for k, v in raw.items()}), len(state.timestamps), FPS, "decoded", "decoded-clock")


def flag_rates(distance: np.ndarray, neutral: np.ndarray, tracked: np.ndarray, degrees: float) -> dict:
    under = np.isfinite(distance) & (distance < degrees)
    return dict(catch=round(float(under[neutral].mean()), 4) if neutral.any() else None, false_alarm=round(float(under[tracked].mean()), 4) if tracked.any() else None,
                neutral_frames=int(neutral.sum()), tracked_frames=int(tracked.sum()))


def part_frames(real, decoded, part: str) -> tuple:
    """(real's frames with the part wholly untracked, real's frames with it wholly tracked)."""
    _, valid = part_arrays(real, part)
    return ~valid.any(1), valid.all(1)


def fit_flag(pairs) -> dict:
    """Per part, from (real, decoded) clips: the decoded-neutral mean pose and the whole degree with the best catch minus false alarm; each wrist's activity tercile edges, on decoded-tracked frames."""
    parts, speeds = {}, {0: [], 1: []}
    for part, (_, _, side) in PARTS.items():
        chosen = []
        for real, decoded in pairs:
            neutral, _ = part_frames(real, decoded, part)
            if neutral.any():
                chosen.append(part_arrays(decoded, part)[0][neutral])
        if not chosen:
            raise SystemExit(f"{part}: no decoded-neutral frames in the fit clips")
        pose = mean_pose(np.concatenate(chosen))
        d, n, t = [], [], []
        for real, decoded in pairs:
            neutral, tracked = part_frames(real, decoded, part)
            d.append(part_distance(decoded, part, pose))
            n.append(neutral)
            t.append(tracked)
        d, n, t = np.concatenate(d), np.concatenate(n), np.concatenate(t)
        score = [(flag_rates(d, n, t, g)["catch"] or 0.) - (flag_rates(d, n, t, g)["false_alarm"] or 0.) for g in FLAG_GRID]
        degrees = FLAG_GRID[int(np.argmax(score))]
        parts[part] = dict(pose=np.round(pose, 6).tolist(), degrees=degrees, fit=flag_rates(d, n, t, degrees))
    for real, decoded in pairs:
        speed = activity(decoded)
        for side, hand in enumerate(HAND_PARTS):
            speeds[side].append(speed[part_frames(real, decoded, hand)[1], side])
    edges = {str(side): np.round(np.percentile(np.concatenate(v), [100 / 3, 200 / 3]), 6).tolist() for side, v in speeds.items()}
    return dict(parts=parts, tercile_edges=edges)


def flag_distance(state, part: str, flag: dict) -> np.ndarray:
    return part_distance(state, part, flag["parts"][part]["pose"])


def flagged(state, part: str, flag: dict, scale: float = 1.) -> np.ndarray:
    d = flag_distance(state, part, flag)
    return np.isfinite(d) & (d < flag["parts"][part]["degrees"] * scale)


def holdout_rates(flag: dict, pairs) -> dict:
    out = {}
    for part in PARTS:
        d, n, t = [], [], []
        for real, decoded in pairs:
            neutral, tracked = part_frames(real, decoded, part)
            d.append(flag_distance(decoded, part, flag))
            n.append(neutral)
            t.append(tracked)
        out[part] = flag_rates(np.concatenate(d), np.concatenate(n), np.concatenate(t), flag["parts"][part]["degrees"])
    return out


def tercile_shares(state, flag: dict, parts, *, tracked_by=None, scale: float = 1.) -> dict:
    """This clip's share of frames flagged within each tercile, pooled over ``parts``: only frames ``tracked_by`` tracked wholly, if given. None where a tercile holds no frame."""
    speed = activity(state)
    under, totals = {t: 0 for t in TERCILES}, {t: 0 for t in TERCILES}
    for part in parts:
        side = PARTS[part][2]
        labels = tercile_of(speed[:, side], flag["tercile_edges"][str(side)])
        mask = flagged(state, part, flag, scale)
        seen = part_arrays(state, part)[1].any(1) if tracked_by is None else part_arrays(tracked_by, part)[1].all(1)
        for k, name in enumerate(TERCILES):
            here = seen & (labels == k)
            totals[name] += int(here.sum())
            under[name] += int((mask & here).sum())
    return {t: (round(under[t] / totals[t], 5) if totals[t] else None) for t in TERCILES}


def paired_terciles(a: list[dict], b: list[dict]) -> dict:
    """Per tercile, a − b over clips both measure."""
    out = {}
    for t in TERCILES:
        pairs = [(x[t], y[t]) for x, y in zip(a, b) if x[t] is not None and y[t] is not None]
        out[t] = dict(M.paired_mean_difference(*np.array(pairs).T, n=BOOT), clips=len(pairs)) if pairs else None
    return out


def positive_control(pairs, flag: dict) -> dict:
    """The flag's catch rate on these clips' decoded-neutral hand frames, pooled, and per clip."""
    rows, hit, total = [], 0, 0
    for real, decoded in pairs:
        clip_hit, clip_total = 0, 0
        for part in HAND_PARTS:
            neutral, _ = part_frames(real, decoded, part)
            clip_hit += int((flagged(decoded, part, flag) & neutral).sum())
            clip_total += int(neutral.sum())
        hit, total = hit + clip_hit, total + clip_total
        if clip_total:
            rows.append(clip_hit / clip_total)
    return dict(catch_rate=round(hit / total, 4) if total else None, frames=total, clips=len(rows), per_clip=M.bootstrap_mean(rows, n=BOOT))


def flat_share(rows: list[dict]) -> dict:
    """Per tercile over clips: the mean share, with a bootstrap."""
    return {t: M.bootstrap_mean([r[t] for r in rows if r[t] is not None], n=BOOT) for t in TERCILES}


def replaced_neutral(state, flag: dict) -> tuple:
    """(``state`` with each hand's flagged frames holding the hand's nearest unflagged frame, counts). A hand wholly flagged stays and is counted."""
    from dataclasses import replace
    channels, marked, frames, untouched = dict(state.channels), 0, 0, 0
    for part in HAND_PARTS:
        c = state.channels[part]
        frames += len(c.values)
        mask = flagged(state, part, flag)
        if not mask.any():
            continue
        if mask.all():
            untouched += 1
            continue
        channels[part] = _held(c, mask)
        marked += int(mask.sum())
    return replace(state, channels=channels), dict(flagged_frames=marked, hand_frames=frames, hands_all_flagged=untouched)


def _held(channel, mask: np.ndarray):
    from dataclasses import replace
    index = V.nearest_index(~mask)
    return replace(channel, values=channel.values[index].astype(channel.values.dtype), valid=channel.valid[index], observed=channel.observed[index], confidence=channel.confidence[index])


def run_lengths(mask: np.ndarray) -> list[int]:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return (edges[1::2] - edges[::2]).tolist()


def control_mask(mask: np.ndarray, rng) -> tuple:
    """(a mask with ``mask``'s run lengths placed at random on its unflagged frames, the frames of runs that did not fit). Runs stay apart by a frame, as flagged runs are."""
    free, placed, missed = ~mask, np.zeros_like(mask), 0
    for size in sorted(run_lengths(mask), reverse=True):
        starts = [t for t in range(len(mask) - size + 1) if free[t:t + size].all()]
        if not starts:
            missed += size
            continue
        t = int(starts[int(rng.integers(len(starts)))])
        placed[t:t + size] = True
        free[max(t - 1, 0):t + size + 1] = False
    return placed, missed


def controlled_hold(state, flag: dict, clip: int) -> tuple:
    """(``state`` with the replaced form's hold applied to the same run lengths on unflagged frames, counts): the edit without the flag."""
    from dataclasses import replace
    channels, held, missed = dict(state.channels), 0, 0
    for k, part in enumerate(HAND_PARTS):
        mask = flagged(state, part, flag)
        if not mask.any() or mask.all():
            continue
        placed, lost = control_mask(mask, np.random.default_rng([SAMPLE_SEED, clip, k]))
        missed += lost
        if placed.any():
            channels[part] = _held(state.channels[part], placed)
            held += int(placed.sum())
    return replace(state, channels=channels), dict(control_frames=held, control_frames_unplaced=missed)


def time_bins(values: np.ndarray, bins: int = BINS) -> np.ndarray:
    """(bins,) means over time-normalised position; NaN in an empty bin."""
    T = len(values)
    index = np.minimum((np.arange(T) * bins / max(T, 1)).astype(int), bins - 1)
    out = np.full(bins, np.nan)
    for b in range(bins):
        seen = np.isfinite(values) & (index == b)
        if seen.any():
            out[b] = float(values[seen].mean())
    return out


def untracked_profile(clips, hand: str) -> np.ndarray:
    """(bins,) over training clips: the share of a hand's joints untracked at each time-normalised position."""
    rows = [time_bins((~c.channels[hand].valid).mean(1).astype(np.float64)) for c in clips]
    if not rows:
        return np.full(BINS, np.nan)
    stacked = np.stack(rows)
    counts = np.isfinite(stacked).sum(0)
    return np.where(counts > 0, np.nansum(stacked, 0) / np.maximum(counts, 1), np.nan)


def _bin_gap(generated: np.ndarray, untracked: np.ndarray) -> float | None:
    """Mean flag share where training was mostly untracked minus where mostly tracked, within each hand and time bin, averaged over the cells holding both. (E, hands, bins) arrays."""
    gaps = []
    for h in range(generated.shape[1]):
        for b in range(generated.shape[2]):
            g, u = generated[:, h, b], untracked[:, h, b]
            seen = np.isfinite(g) & np.isfinite(u)
            gone, kept = seen & (u >= .5), seen & (u < .5)
            if gone.any() and kept.any():
                gaps.append(float(g[gone].mean() - g[kept].mean()))
    return float(np.mean(gaps)) if gaps else None


def word_alignment(rows: list[dict]) -> dict:
    """Report only. Within each hand and time bin, the flag share where a word's training clips were mostly untracked minus where mostly tracked, averaged over bins, a bootstrap over entries and a null with entries' untracked profiles shuffled.
    ``rows``: {entry, generated: {hand: (bins,) flag share}, untracked: {hand: (bins,) share}}."""
    if not rows:
        return dict(entries=0)
    g = np.stack([np.stack([r["generated"][h] for h in HAND_PARTS]) for r in rows])       # (E, 2, bins)
    u = np.stack([np.stack([r["untracked"][h] for h in HAND_PARTS]) for r in rows])
    rng = np.random.default_rng(SAMPLE_SEED)
    E = len(rows)

    point = _bin_gap(g, u)
    boots = [_bin_gap(g[i], u[i]) for i in rng.integers(0, E, (BOOT // 10, E))]
    nulls = [_bin_gap(g, u[rng.permutation(E)]) for _ in range(NULL_SHUFFLES)]
    return dict(entries=E, bins=BINS, report_only=True, within_bin_difference=None if point is None else round(point, 5), ci95=P._quantiles([b for b in boots if b is not None]),
                shuffled_entry_null=dict(shuffles=NULL_SHUFFLES, mean=round(float(np.mean([n for n in nulls if n is not None])), 5) if any(n is not None for n in nulls) else None, ci95=P._quantiles([n for n in nulls if n is not None])))


# ---- stored material ---------------------------------------------------------------------------------------------------------------

def export_differences(index: dict, arm: str) -> dict:
    """Pinned settings an export's run did not use: {setting: [pinned, stored]}."""
    settings = index.get("settings") or {}
    return {k: [v, settings.get(k)] for k, v in EXPORT_PINS[arm].items() if settings.get(k) != v}


def load_composed(store, index_key: str, signs, arm: str = "a9") -> list:
    """Each sign's composed state from an export index, in list order. A missing sign or settings other than the pins refuse."""
    from evaluate_text_to_motion import load_state
    index = json.loads(store.get(index_key))
    if index.get("kind") != "round15-composed-states":
        raise SystemExit(f"{index_key}: not a composed-state export")
    differs = export_differences(index, arm)
    if differs:
        raise SystemExit(f"{index_key}: its settings differ from the pins: {differs}")
    rows = {r["sign"]: r for r in index["states"] if r["form"] == COMPOSED_FORM}
    missing = [s for s in signs if s not in rows]
    if missing:
        raise SystemExit(f"{index_key}: {len(missing)} signs are not exported, for example {missing[0]}")
    with ThreadPoolExecutor(FETCH_THREADS) as pool:
        return list(pool.map(lambda s: load_state(store, rows[s]["state_key"], rows[s]["state_sha256"]), signs))


def cache_entries(store, key: str, sha: str, only=None) -> list[dict]:
    """(entry, the best-of-16 state's key and hash) of a word cache, SHA-pinned; ``only`` keeps those entries."""
    raw = store.get(key)
    P.check_sha(raw, sha, key)
    rows = []
    for e in json.loads(raw)["entries"]:
        if only is None or e["entry"] in only:
            chosen = e["chosen"]["16"]
            rows.append(dict(entry=e["entry"], state_key=chosen["state_key"], state_sha256=chosen["state_sha256"]))
    return rows


def top_tokens(top: int) -> set:
    """The first ``top`` ranks of the committed list, SHA-pinned, as ``--vocabulary-extra-top`` takes them."""
    raw = (ROOT / TOKENS_FILE).read_bytes()
    P.check_sha(raw, TOKENS_SHA, TOKENS_FILE)
    return {t["token"] for t in json.loads(raw)["tokens"] if t["rank"] <= top}


def word_training_clips(manifest, cache, fetch, entries, *, clips: int = WORD_CLIPS, corpora=None) -> dict:
    """{entry: states} of each entry's training clips, a fixed draw of at most ``clips`` each, the windows training reads (``load_sample``). Never c00, never test."""
    from signtranslator.data_engineering.corpus_manifest import load_sample
    from signtranslator.generation.compositional import isolated_corpora, normalise
    corpora = set(corpora or isolated_corpora(manifest))
    wanted, pools = set(entries), {}
    for r in manifest["records"]:
        if r.get("partition") != "train" or r.get("corpus") not in corpora or not r.get("text") or r.get("signer_component") == "c00":
            continue
        entry = " ".join(normalise(r["text"]))
        if entry in wanted:
            pools.setdefault(entry, []).append(r)
    draws = {entry: random.Random(f"{SAMPLE_SEED}/{entry}").sample(sorted(pool, key=lambda r: r["sign"]), min(clips, len(pool))) for entry, pool in pools.items()}
    order = [(entry, r) for entry, draw in draws.items() for r in draw]
    with ThreadPoolExecutor(FETCH_THREADS) as threads:                 # fetches batched: a GPU pod does not idle clip by clip
        states = list(threads.map(lambda item: load_sample(item[1], "continuous", cache=cache, fetch=fetch), order))
    out = {entry: [] for entry in draws}
    for (entry, _), state in zip(order, states):
        out[entry].append(state)
    return out


def corpus_neutral(manifest, cache, fetch, *, clips: int = CORPUS_CLIPS, components=None) -> dict:
    """Per corpus in a training mix, a sample: the share of hand-frames with every joint untracked, and with any joint untracked (each neutral-filled by ``diffusion_targets``), as training reads the clips."""
    from signtranslator.data_engineering.corpus_manifest import load_sample
    corpora: dict = {}
    for r in manifest["records"]:
        if r.get("partition") == "train" and r.get("text") and (r.get("continuous") or {}).get("state_key") and r.get("signer_component") != "c00" \
                and (components is None or r.get("signer_component") in components):
            corpora.setdefault(r.get("corpus") or manifest.get("corpus_id"), []).append(r)
    out = {}
    for corpus, pool in sorted(corpora.items()):
        draw = random.Random(f"{SAMPLE_SEED}/{corpus}").sample(sorted(pool, key=lambda r: r["sign"]), min(clips, len(pool)))
        every, some = [], []
        with ThreadPoolExecutor(FETCH_THREADS) as threads:
            states = list(threads.map(lambda r: load_sample(r, "continuous", cache=cache, fetch=fetch), draw))
        for state in states:
            whole = [float((~state.channels[h].valid).all(1).mean()) for h in HANDS]
            part = [float((~state.channels[h].valid).any(1).mean()) for h in HANDS]
            every.append(np.mean(whole))
            some.append(np.mean(part))
        out[corpus] = dict(records=len(pool), sampled=len(draw), every_joint_untracked=M.bootstrap_mean(every, n=BOOT), any_joint_untracked=M.bootstrap_mean(some, n=BOOT))
    total = sum(v["records"] for v in out.values())
    if total:
        out["weighted_by_records"] = dict(every_joint_untracked=round(sum(v["records"] * v["every_joint_untracked"]["mean"] for v in out.values()) / total, 5),
                                          any_joint_untracked=round(sum(v["records"] * v["any_joint_untracked"]["mean"] for v in out.values()) / total, 5))
    return out


# ---- forms -------------------------------------------------------------------------------------------------------------------------

def nearest_input(state, tokenizer):
    return V.v0_form("v0_nearest_fill", state, None, 0, tokenizer)


def cap_form(name: str, state, tokenizer):
    from evaluate_text_to_motion import blanked
    if name == "real":
        return state
    if name == "blank":
        return blanked([state])[0]
    if name in ("identity", "round_trip"):
        return P.decoded_form(name, state, tokenizer, None)
    filled = nearest_input(state, tokenizer)
    if name == "v0_nearest_fill":
        return filled
    if name == "rt_nearest":
        return P.decoded_form("round_trip", filled, tokenizer, None)
    if name == "no_quant_nearest":
        return P.decoded_form("no_quant", filled, tokenizer, None)
    if name == "rt_post_fill":
        return V.filled(decoded_state(tokenizer, state), tracked=state)
    raise SystemExit(f"unknown form {name}")


def build_form(name: str, refs, tokenizer, composed, flag) -> tuple[list, list | None]:
    """(states, counts per clip, for the edited forms)."""
    if name in CAP_FORMS:
        return [cap_form(name, s, tokenizer) for s in refs], None
    if composed is None:
        raise SystemExit(f"{name} needs the composed export")
    if name == "a9p":
        return list(composed), None
    if name == "a9p_replaced":
        pairs = [replaced_neutral(s, flag) for s in composed]
    elif name == "a9p_control":
        pairs = [controlled_hold(s, flag, i) for i, s in enumerate(composed)]
    else:
        raise SystemExit(f"unknown form {name}")
    return [p[0] for p in pairs], [p[1] for p in pairs]


def load_flag() -> dict:
    """The flag, fixed before any read: SHA-pinned, fit on c02 train clips only."""
    raw = (ROOT / FLAG_FILE).read_bytes()
    P.check_sha(raw, FLAG_SHA, FLAG_FILE)
    flag = json.loads(raw)
    if flag.get("clips", {}).get("components") != ["c02"] or flag["clips"].get("partitions") != ["train"]:
        raise SystemExit(f"{FLAG_FILE}: the flag is fit on c02 train clips only")
    return flag


# ---- commands -------------------------------------------------------------------------------------------------------------------------

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
    torch.set_num_threads(args.threads)
    names = [f for f in args.forms.split(",") if f]
    unknown = [f for f in names if f not in FORMS]
    if unknown:
        raise SystemExit(f"unknown forms {unknown}")
    if (args.smoke_longest or args.smoke_never_tracked) and not args.smoke_train:
        raise SystemExit("--smoke-longest and --smoke-never-tracked belong to a smoke")
    out_dir = Path(args.out_dir)
    stored = [n for n in names if (out_dir / f"form-{n}.json").exists()]
    if stored and not args.resume:
        raise SystemExit(f"{out_dir / f'form-{stored[0]}.json'} exists (write-once)")
    names = [n for n in names if n not in stored]
    if not names:
        print(json.dumps(dict(read=0, stored=len(stored))), flush=True)
        return 0
    store = make_store(args.store)
    records, declared, described = L.open_read_list(store, args.smoke_train)
    if args.smoke_longest:
        have = {r["sign"] for r in records}
        records = list(records) + [r for r in P.longest_train_records(store, args.smoke_longest) if r["sign"] not in have]
        described = dict(described, clips=len(records), longest=args.smoke_longest)
    if args.smoke_never_tracked:
        have = {r["sign"] for r in records}
        records = list(records) + [r for r in V.never_tracked_records(store, args.smoke_never_tracked) if r["sign"] not in have]
        described = dict(described, clips=len(records), never_tracked=args.smoke_never_tracked)
    P.check_read_list(records, declared)
    texts, signs = [r["text"] for r in records], [r["sign"] for r in records]
    refs = L.load_states(store, records)
    tokenizer = L.load_tokenizer(store)
    composed, flag = None, load_flag() if any(n in SENSITIVITY_FORMS for n in names) else None
    if any(n in SENSITIVITY_FORMS for n in names):
        if declared:
            composed = load_composed(store, A9_INDEX, signs)
        else:                                                           # a smoke's stand-in on train clips, so nothing generated is read
            composed = [decoded_state(tokenizer, s) for s in refs]
    reader = P.load_reader(store, args)
    candidates = M.retrieval_candidates(len(texts), 1, seed=0)
    digest, place = code_digest(), P.environment()
    for name in names:
        started = time.time()
        states, counts = build_form(name, refs, tokenizer, composed, flag)
        built = time.time()
        raw = reader.read(states, texts, candidates, group=args.nll_group)
        report = dict(kind=f"{PROBE}-form", report_only=True, form=name, declared_read=declared, list=described, code_digest=digest, environment=place,
                      instrument=reader.info, tokenizer=dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA, gains=None), signs=signs, references=texts, counts=counts,
                      hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(),
                      seconds_over_reference=[round(len(s.timestamps) / max(len(r.timestamps), 1), 5) for s, r in zip(states, refs)],
                      motion=L.motion_statistics(states, refs), hands=P.hand_statistics(states, refs), tracked=P.tracked_statistics(states),
                      build_seconds=round(built - started, 2), read_seconds=round(time.time() - built, 2), peak_rss_bytes=P.peak_rss_bytes(), gpu_peak_reserved_bytes=P.gpu_peak_reserved_bytes())
        P.write_report_once(out_dir / f"form-{name}.json", report)
        print(json.dumps({name: dict(build_seconds=report["build_seconds"], read_seconds=report["read_seconds"], peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2),
                                     gpu_reserved_bytes=report["gpu_peak_reserved_bytes"])}), flush=True)
        del states, raw
    return 0


def run_flag_fit(args) -> int:
    """c02 train clips only, one fixed draw: fit on the first ``--fit-clips``, rates on the held-out rest. Written once, then pinned in this script."""
    from train_text_to_motion import make_store
    P.check_pins()
    path = Path(args.output_file)
    if path.exists():
        raise SystemExit(f"{path} exists (write-once)")
    store = make_store(args.store)
    pool = L.train_records(L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA), ("c02",))
    records = random.Random(FLAG_SEED).sample(pool, min(args.fit_clips + args.hold_clips, len(pool)))
    L.check_train_records(records, ("c02",))
    tokenizer = L.load_tokenizer(store)
    pairs = []
    for start in range(0, len(records), 32):
        chunk = records[start:start + 32]
        pairs += [(s, decoded_state(tokenizer, s)) for s in L.load_states(store, chunk)]
    flag = fit_flag(pairs[:args.fit_clips])
    report = dict(kind=f"{PROBE}-flag", report_only=True, clips=dict(components=sorted({r["signer_component"] for r in records}), partitions=sorted({r["partition"] for r in records}), corpora=sorted({r["corpus"] for r in records}),
                                                                      fit=args.fit_clips, held_out=len(pairs) - args.fit_clips, seed=FLAG_SEED, signs_sha256=hashlib.sha256(json.dumps([r["sign"] for r in records]).encode()).hexdigest()),
                  tokenizer=dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA), rule="per part: mean geodesic degrees to the decoded-neutral mean pose under the whole degree with the best catch minus false alarm on the fit clips",
                  held_out=holdout_rates(flag, pairs[args.fit_clips:]), peak_rss_bytes=P.peak_rss_bytes(), **flag)
    path.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(dict(flag=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), held_out=report["held_out"], degrees={p: report["parts"][p]["degrees"] for p in PARTS})), flush=True)
    return 0


def run_neutral(args) -> int:
    """(b), motion only, no reader. A smoke measures c02 train clips and reads no cache or composed form."""
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from evaluate_text_to_motion import load_state
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    if (out / "neutral.json").exists():
        raise SystemExit(f"{out / 'neutral.json'} exists (write-once)")
    store = make_store(args.store)
    records, declared, described = L.open_read_list(store, args.smoke_train)
    P.check_read_list(records, declared)
    flag = load_flag()
    refs = L.load_states(store, records)
    tokenizer = L.load_tokenizer(store)
    pairs = [(s, decoded_state(tokenizer, s)) for s in refs]
    decoded_hands = [tercile_shares(d, flag, HAND_PARTS, tracked_by=s) for s, d in pairs]
    decoded_wrists = [tercile_shares(d, flag, WRIST_PARTS, tracked_by=s) for s, d in pairs]
    report = dict(kind=f"{PROBE}-neutral", report_only=True, declared_read=declared, list=described, code_digest=code_digest(), environment=P.environment(), flag=dict(file=FLAG_FILE, sha256=FLAG_SHA, degrees={p: flag["parts"][p]["degrees"] for p in PARTS},
                  held_out=flag.get("held_out")), signs=[r["sign"] for r in records], positive_control=positive_control(pairs, flag),
                  populations=dict(decoded_real_tracked_frames=flat_share(decoded_hands), real_tracked_frames=flat_share([tercile_shares(s, flag, HAND_PARTS, tracked_by=s) for s in refs])),
                  wrist_rotation=dict(report_only=True, untracked_share_of_frames=round(float(np.mean([(~part_arrays(s, p)[1].any(1)).mean() for s in refs for p in WRIST_PARTS])), 5), decoded_real_tracked_frames=flat_share(decoded_wrists)))
    with tempfile.TemporaryDirectory() as t:
        cache = ContentCache(Path(t) / "c", 2 * 2 ** 30)
        if declared:
            train = L.load_manifest_at(store, L.TRAIN_MIX, L.TRAIN_MIX_SHA)     # W14b's own training mix
            top = top_tokens(10)
            words = dict(a9_words=cache_entries(store, A9_CACHE, A9_CACHE_SHA), a3_new_words=cache_entries(store, A3_CACHE, A3_CACHE_SHA, top))
            report["a3_tokens"] = dict(tokens_requested=len(top), entries_found=len(words["a3_new_words"]), missing=sorted(top - {r["entry"] for r in words["a3_new_words"]}))
            alignment = {}
            for name, rows in words.items():
                with ThreadPoolExecutor(FETCH_THREADS) as pool:
                    states = dict(zip((r["entry"] for r in rows), pool.map(lambda r: load_state(store, r["state_key"], r["state_sha256"]), rows)))
                report["populations"][name] = flat_share([tercile_shares(s, flag, HAND_PARTS) for s in states.values()])
                training = word_training_clips(train, cache, store.get, states)
                alignment[name] = word_alignment([dict(entry=e, generated={h: time_bins(flagged(s, h, flag).astype(np.float64)) for h in HAND_PARTS}, untracked={h: untracked_profile(training[e], h) for h in HAND_PARTS})
                                                  for e, s in states.items() if training.get(e)])
                alignment[name]["entries_without_training_clips"] = sum(not training.get(e) for e in states)
            report["word_alignment"] = alignment
            report["generated"] = {}
            for name, key, arm in (("a9_predicted", A9_INDEX, "a9"), ("a3_top10_predicted", A3_INDEX, "a3")):
                composed = load_composed(store, key, report["signs"], arm)
                shares = [tercile_shares(s, flag, HAND_PARTS) for s in composed]
                report["populations"][name] = flat_share(shares)
                report["generated"][name] = dict(minus_decoded_real=paired_terciles(shares, decoded_hands),
                                                 sensitivity={str(c): paired_terciles([tercile_shares(s, flag, HAND_PARTS, scale=c) for s in composed], [tercile_shares(d, flag, HAND_PARTS, tracked_by=r, scale=c) for r, d in pairs]) for c in FLAG_SCALES})
                report["wrist_rotation"][name] = paired_terciles([tercile_shares(s, flag, WRIST_PARTS) for s in composed], decoded_wrists)
            run = json.loads(store.get(f"{L.TOKENIZER.split('/checkpoints/')[0]}/run.json"))
            d2 = L.load_manifest_at(store, run["manifest_key"], run["manifest_sha256"])
            report["corpus_neutral"] = dict(w14b=corpus_neutral(train, cache, store.get), d2=corpus_neutral(d2, cache, store.get), d2_manifest=dict(key=run["manifest_key"], sha256=run["manifest_sha256"]))
        else:
            report["corpus_neutral"] = dict(smoke_c02=corpus_neutral(L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA), cache, store.get, clips=args.smoke_train, components=("c02",)))
    report["peak_rss_bytes"] = P.peak_rss_bytes()
    P.write_report_once(out / "neutral.json", report)
    print(json.dumps(dict(neutral=str(out / "neutral.json"), peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2))), flush=True)
    return 0


def run_path_check(args) -> int:
    """Anchors against their stored reads (each SHA-pinned), the canary's limits. Exit 3, voiding the read, on a miss. A smoke checks the pins only, and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    out = Path(args.out_dir)
    stored = {}
    for form, (key, sha) in STORED_FORMS.items():
        raw = store.get(key)
        P.check_sha(raw, sha, key)
        stored[form] = json.loads(raw)
    probe = json.loads((out / "form-identity.json").read_bytes())
    if not probe["declared_read"]:
        print(json.dumps(dict(path_check="not applicable: a smoke reads train clips; the pins are checked")), flush=True)
        return 0
    rows, misses = {}, []
    for form, kept in stored.items():
        read = json.loads((out / f"form-{form}.json").read_bytes())
        if read["signs"] != kept["signs"] or read["references"] != kept["references"]:
            raise SystemExit(f"the stored {form} read other clips or another order")
        identical = float(np.mean([a == b for a, b in zip(read["hypotheses"], kept["hypotheses"])]))
        gap = float(np.mean(np.abs(np.asarray(read["nll"])[:, 0] - np.asarray(kept["nll"])[:, 0])))
        rows[form] = dict(identical=round(identical, 4), nll_mean_absolute_difference=round(gap, 6))
        misses += [f"{form} {why}" for why, bad in (("hypotheses", identical < P.CANARY["identical"]), ("NLL", gap > P.CANARY["nll"])) if bad]
    result = dict(reads=rows, limits=dict(identical=P.CANARY["identical"], nll=P.CANARY["nll"]), misses=misses, reproduced=not misses, code_digest=probe["code_digest"], environment=probe["environment"])
    if not (out / "path_check.json").exists():
        P.write_report_once(out / "path_check.json", result)
    print(json.dumps(result), flush=True)
    if not misses:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{probe['code_digest']}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-void", code_digest=probe["code_digest"], path_check=result), indent=1, sort_keys=True).encode())
    print(f"ALERT the anchors do not reproduce their stored reads: {misses}. The read is void: no retry; a rerun needs new code and a recorded cause", flush=True)
    return 3


def run_composed_check(args) -> int:
    """a9p against the stored a10 report: hypotheses and per-clip NLL, the canary's limits. It reports, and Generated and Sensitivity depend on it. A smoke checks the pin only, and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    raw = store.get(A10_REPORT)
    P.check_sha(raw, A10_SHA, A10_REPORT)
    out = Path(args.out_dir)
    read = json.loads((out / "form-a9p.json").read_bytes())
    if not read["declared_read"]:
        print(json.dumps(dict(composed_check="not applicable: a smoke reads train clips; the pin is checked")), flush=True)
        return 0
    stored = json.loads(raw)
    if read["signs"] != stored["signs"]:
        raise SystemExit("the stored a10 report read other clips or another order")
    identical = float(np.mean([a == b for a, b in zip(read["hypotheses"], stored["forms"][COMPOSED_FORM]["hypotheses"])]))
    gap = float(np.mean(np.abs(np.asarray(read["nll"])[:, 0] + np.asarray(stored["likelihood"]["per_form"][COMPOSED_FORM]))))    # stored: log-probability per token
    settings = {arm: export_differences(json.loads(store.get(key)), arm) for arm, key in (("a9", A9_INDEX), ("a3", A3_INDEX))}
    result = dict(identical=round(identical, 4), nll_mean_absolute_difference=round(gap, 6), limits=dict(identical=P.CANARY["identical"], nll=P.CANARY["nll"]), export_settings_differ=settings, settings_match=not any(settings.values()),
                  reproduced=identical >= P.CANARY["identical"] and gap <= P.CANARY["nll"], stored=A10_REPORT, stored_sha256=A10_SHA, code_digest=read["code_digest"])
    P.write_report_once(out / "composed_check.json", result)
    print(json.dumps(result), flush=True)
    if not result["reproduced"] or not result["settings_match"]:
        print(f"ALERT a9p did not reproduce the stored a10 read, or its settings differ: identical {result['identical']}, NLL {result['nll_mean_absolute_difference']}, settings {settings}. Generated and Sensitivity are void", flush=True)
    return 0


def declared_reads(reads: dict) -> dict:
    """Differences and shares, as facts, on likelihood minus blank and on BLEU, jointly resampled over clips (n 2,000, seed 0). A block is None when a form is missing."""
    refs = reads["real"]["references"]
    idx = np.random.default_rng(0).integers(0, len(refs), (BOOT, len(refs)))
    blank = np.asarray(reads["blank"]["nll"])[:, 0]
    forms = [f for f in FORMS if f in reads]
    dnll = {f: M.delta_nll(blank, np.asarray(reads[f]["nll"])[:, 0]) for f in forms}
    stats = {f: M.bleu_statistics(reads[f]["hypotheses"], refs) for f in forms}

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
        return dict(rt_nearest_minus_round_trip=difference("rt_nearest", "round_trip"), rt_nearest_minus_v0_nearest_fill=difference("rt_nearest", "v0_nearest_fill"),
                    no_quant_nearest_minus_rt_nearest=difference("no_quant_nearest", "rt_nearest"), no_quant_nearest_minus_v0_nearest_fill=difference("no_quant_nearest", "v0_nearest_fill"),
                    v0_nearest_fill_minus_identity=difference("v0_nearest_fill", "identity"), cap_share_of_nearest_fill=share("rt_nearest", "v0_nearest_fill"), rt_post_fill_minus_rt_nearest=difference("rt_post_fill", "rt_nearest"),
                    round_trip_share_of_real=share("round_trip", "real"), rt_nearest_share_of_real=share("rt_nearest", "real"), a9p_replaced_minus_a9p_control=difference("a9p_replaced", "a9p_control"),
                    a9p_replaced_minus_a9p=difference("a9p_replaced", "a9p"), a9p_control_minus_a9p=difference("a9p_control", "a9p"))
    return dict(primary="likelihood_minus_blank", likelihood_minus_blank=block("likelihood_minus_blank"), bleu=block("bleu"), bootstrap=dict(n=BOOT, seed=0))


def run_merge(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store) if args.output_key else None
    L.refuse_existing(store, args.output_key, args.output_file)
    out_dir = Path(args.out_dir)
    names = list(FORMS) if args.forms is None else [f for f in args.forms.split(",") if f]
    for anchor in ("real", "blank"):
        if anchor not in names:
            raise SystemExit(f"--forms must include {anchor}: every pair reads it")
    reads = L.load_form_reads(out_dir, names)
    first = next(iter(reads.values()))
    digest, place = P.check_same_run(reads, "the forms")
    canary, path_check = None, None
    if first["declared_read"]:
        path = out_dir / "canary.json"
        canary = json.loads(path.read_bytes()) if path.exists() else None
        if canary is None or canary.get("reproduced") is not True or canary.get("code_digest") != digest or canary.get("environment") != place:
            raise SystemExit(f"{path}: no passed canary of this code and environment; the read is void")
        path = out_dir / "path_check.json"
        path_check = json.loads(path.read_bytes()) if path.exists() else None
        if path_check is None or path_check.get("reproduced") is not True or path_check.get("code_digest") != digest:
            raise SystemExit(f"{path}: no passed path check of this code; the read is void")
    neutral_path = out_dir / "neutral.json"
    neutral = json.loads(neutral_path.read_bytes()) if neutral_path.exists() else None
    if neutral is not None and (neutral.get("code_digest") != digest or neutral.get("environment") != place or neutral.get("declared_read") != first["declared_read"]):
        raise SystemExit(f"{neutral_path}: another run's measure")
    composed_path = out_dir / "composed_check.json"
    panels = {name: P.form_panel(reads, name, tuple(c for c in COMPARATORS.get(name, ()) if c in reads)) for name in names}
    counts = reads["a9p_replaced"]["counts"] if "a9p_replaced" in reads else None
    control = reads["a9p_control"]["counts"] if "a9p_control" in reads else None
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, declared_read=first["declared_read"], list=first["list"], instrument=first["instrument"],
                  tokenizer=first["tokenizer"], clips=len(first["signs"]), code_digest=digest, environment=place, disclosures=DISCLOSURES,
                  rerun=json.loads((out_dir / "rerun.json").read_bytes()) if (out_dir / "rerun.json").exists() else None,
                  smoke=None if first["declared_read"] else "SMOKE on c02 train clips: these numbers are plumbing, not a read", forms=panels, reads=declared_reads(reads), neutral=neutral,
                  sensitivity_counts=None if counts is None else dict(flagged_frames=sum(c["flagged_frames"] for c in counts), hand_frames=sum(c["hand_frames"] for c in counts), hands_all_flagged=sum(c["hands_all_flagged"] for c in counts),
                                                                      flagged_share=round(sum(c["flagged_frames"] for c in counts) / max(sum(c["hand_frames"] for c in counts), 1), 5),
                                                                      control_frames=None if control is None else sum(c["control_frames"] for c in control), control_frames_unplaced=None if control is None else sum(c["control_frames_unplaced"] for c in control)),
                  sensitivity_motion={name: {k: panels[name][k] for k in ("hand_speed", "hand_variation", "wrist_speed")} for name in ("a9p", "a9p_replaced", "a9p_control") if name in panels},
                  hypotheses={name: reads[name]["hypotheses"] for name in names},
                  checks=dict(blank_identical_to_real=panels["blank"]["identical_to_real"], canary=canary, path_check=path_check,
                              composed_check=json.loads(composed_path.read_bytes()) if composed_path.exists() else None),
                  resources={name: dict(peak_rss_bytes=reads[name]["peak_rss_bytes"], gpu_peak_reserved_bytes=reads[name]["gpu_peak_reserved_bytes"], build_seconds=reads[name]["build_seconds"],
                                        read_seconds=reads[name]["read_seconds"]) for name in names})
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, forms=len(panels))), flush=True)
    return 0


def run_check(args) -> int:
    """Exit 3 on broken plumbing: not report_only, blank reading as real, or a declared read without its canary, path check or motion measure. It judges no form."""
    report = json.loads(Path(args.report).read_text())
    checks, problems = report["checks"], []
    if report.get("report_only") is not True:
        problems.append("the report does not say report_only")
    if checks["blank_identical_to_real"] >= 1:
        problems.append("blank reads exactly as real")
    if report.get("declared_read"):
        for name in ("canary", "path_check"):
            if not (checks.get(name) or {}).get("reproduced"):
                problems.append(f"a declared read without its {name}")
        if not report.get("neutral"):
            problems.append("a declared read without its motion measure")
    print(json.dumps(dict(problems=problems)), flush=True)
    for line in problems:
        print(f"ALERT {line}", flush=True)
    return 3 if problems else 0


def run_export_list(args) -> int:
    """The list ``--export-only`` composes: D20's signs in order, or a smoke's first 5. Prints the file's hash."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    records, declared, _ = L.open_read_list(store, 0)
    P.check_read_list(records, declared)
    signs = [r["sign"] for r in records][:args.first or None]
    payload = json.dumps(dict(name=f"tok-v1-d20-first-{args.first}" if args.first else "tok-v1-d20", signs=signs), indent=1).encode()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(payload)
    print(hashlib.sha256(payload).hexdigest(), flush=True)
    return 0


def run_check_export(args) -> int:
    """A composed export's stored settings against the pins, before anything hangs on them. Refuses a mismatch."""
    from train_text_to_motion import make_store
    differs = export_differences(json.loads(make_store(args.store).get(args.key)), args.arm)
    if differs:
        raise SystemExit(f"{args.key}: its settings differ from the pins: {differs}")
    print(json.dumps(dict(export=args.key, arm=args.arm, settings="as pinned")), flush=True)
    return 0


def run_stored(args) -> int:
    from train_text_to_motion import make_store
    return 0 if make_store(args.store).exists(args.key) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--store", default="r2")
    pre.add_argument("--out-dir", required=True)
    pre.add_argument("--final-key", required=True)
    pre.add_argument("--void-prefix", required=True)
    pre.add_argument("--rerun-cause")
    read = sub.add_parser("read")
    read.add_argument("--store", default="r2")
    read.add_argument("--out-dir", required=True)
    read.add_argument("--forms", default=",".join(FORMS))
    read.add_argument("--resume", action="store_true")
    read.add_argument("--nll-group", type=int, default=2)
    read.add_argument("--instrument-cache")
    read.add_argument("--smoke-train", type=int, default=0, help="read this many c02 train clips, not D20")
    read.add_argument("--smoke-longest", type=int, default=0)
    read.add_argument("--smoke-never-tracked", type=int, default=0)
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    neutral = sub.add_parser("neutral", help="(b), motion only")
    neutral.add_argument("--store", default="r2")
    neutral.add_argument("--out-dir", required=True)
    neutral.add_argument("--smoke-train", type=int, default=0)
    canary = sub.add_parser("canary")
    canary.add_argument("--store", default="r2")
    canary.add_argument("--out-dir", required=True)
    canary.add_argument("--void-prefix", required=True)
    path = sub.add_parser("path-check")
    path.add_argument("--store", default="r2")
    path.add_argument("--out-dir", required=True)
    path.add_argument("--void-prefix", required=True)
    composed = sub.add_parser("composed-check")
    composed.add_argument("--store", default="r2")
    composed.add_argument("--out-dir", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--store", default="r2")
    merge.add_argument("--out-dir", required=True)
    merge.add_argument("--forms")
    merge.add_argument("--output-key")
    merge.add_argument("--output-file")
    check = sub.add_parser("check")
    check.add_argument("--report", required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--store", default="r2")
    upload.add_argument("--out-dir", required=True)
    upload.add_argument("--prefix", required=True)
    upload.add_argument("--partial", action="store_true")
    export = sub.add_parser("export-list")
    export.add_argument("--store", default="r2")
    export.add_argument("--out", required=True)
    export.add_argument("--first", type=int, default=0)
    fit = sub.add_parser("flag-fit", help="c02 train only: the flag's pose and threshold, with rates on held-out clips")
    fit.add_argument("--store", default="r2")
    fit.add_argument("--output-file", required=True)
    fit.add_argument("--fit-clips", type=int, default=FLAG_FIT_CLIPS)
    fit.add_argument("--hold-clips", type=int, default=FLAG_HOLD_CLIPS)
    export_check = sub.add_parser("check-export", help="a composed export's settings against the pins")
    export_check.add_argument("--store", default="r2")
    export_check.add_argument("--key", required=True)
    export_check.add_argument("--arm", required=True, choices=("a9", "a3"))
    stored = sub.add_parser("stored")
    stored.add_argument("--store", default="r2")
    stored.add_argument("--key", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return dict(preflight=run_preflight, read=run_read, neutral=run_neutral, canary=P.run_canary, merge=run_merge, check=run_check, upload=P.run_upload, stored=run_stored,
                **{"path-check": run_path_check, "composed-check": run_composed_check, "export-list": run_export_list, "flag-fit": run_flag_fit, "check-export": run_check_export})[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
