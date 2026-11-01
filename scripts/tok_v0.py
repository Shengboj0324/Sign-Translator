"""V0: do real's validity marks carry content, or does v1 only mind what marking every frame valid does? (Report only; nothing trains, nothing is declared by this code.)

The tokenizer probes read ``round_trip_masked`` and ``round_trip`` far apart, and ``identity`` (no network at all) between them. Before that reads as "the tokenizer is fine",
this asks whether the marks themselves tell v1 about the sentence. D20 only, instrument v1, through ``tok_probes.py``'s reads, canary, digest and panels, write-once under
``tok-probes-v0/``. No option names another list or instrument. ``--smoke-train N`` reads c02 train clips and says so.

**Forms.** ``real``, ``blank``, ``identity`` (as ``tok_probes``), and:

- ``v0_blank_real_marks``: a constant neutral pose, with this clip's validity. All it can tell v1 is the marks. **Control:** ``v0_blank_other_marks``, the same pose with the partner's
  marks, stretched to this clip's length by nearest frame. (This control is not on the task's list: without it, blank against this form also moves zeros to a pose, so it could
  not show that the pattern is this sentence's.)
- ``v0_other_motion``: the partner's motion, time-scaled to this clip's length, with this clip's validity. **Comparator:** ``v0_other_motion_other_marks``, the same motion under the
  partner's own marks.
- ``v0_other_marks``: this clip's values with the partner's marks. A frame the partner marks and this clip did not track holds this clip's nearest tracked value.
- ``v0_random_marks``: this clip's values with its own marks reordered: its runs of identical frame patterns in a random order (seeded per clip and form). Each joint's tracked rate
  is exact and run lengths stay, so random flicker does not stand in for a wrong pattern. Filled as ``v0_other_marks``.
- ``v0_nearest_fill``: ``identity`` with its neutral fill replaced by the nearest tracked value of each joint; a joint never tracked keeps the neutral. Every frame still marked valid.

**Partner** (``pair_partners``). Clips adjacent by length within a signer, so length does not stand in for content; no pair from one video, which How2Sign's ids expose
(``<video>_<n>-<m>-rgb_front``; an id that does not is counted). An odd clip joins its nearest pair as a three-cycle. Never itself. The report gives the pairs' length ratios, cycles, and any
pair across signers or within a video.

**Per form,** ``tok_probes.form_panel``: BLEU, likelihood minus blank, loop share, length, wrist, hands, tracked share, share of the gap, paired against real, blank and ``comparators_of``.
``reader_input`` says what v1 receives against real: marks added and removed, coordinate differences on frames both mark (arm points apart from hand-local ones), and how far the clip's
shoulder origin and unit moved, since ``keypoints2d.normalise`` takes them from the marked frames. ``fold_share`` is the form's own marked hand-frames whose hand points all sit on the wrist
(``from_state`` does not check the wrist rotation, so an all-zero one would do that). 252 c02 train clips show none in real. ``marks_as_real`` is the clips whose marks equal real's.
``reads`` gives what the declaration reads, on likelihood minus blank and on BLEU: A = v0_blank_real_marks − v0_blank_other_marks, B = v0_other_motion − v0_other_motion_other_marks,
real − identity, p2 = (v0_other_marks − identity) / (real − identity), p3 = (v0_nearest_fill − identity) / (real − identity), each with a bootstrap interval. This script holds no rule:
``scripts/tok_v0_rule.py`` applies one, and the job never calls it.

**Path check.** ``identity`` is read again and must meet the probes' stored read (a SHA-pinned file, the canary's limits): a miss voids the read.

    python scripts/tok_v0.py preflight --out-dir W --final-key K --void-prefix V
    python scripts/tok_v0.py read --out-dir W/forms --instrument-cache W/i
    python scripts/tok_v0.py canary --out-dir W/forms --void-prefix V
    python scripts/tok_v0.py path-check --out-dir W/forms --void-prefix V
    python scripts/tok_v0.py merge --out-dir W/forms --output-file W/report.json
    python scripts/tok_v0.py check --report W/report.json

The job is ``runpod/fleet/tok_v0.sh``.
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import replace
import functools
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402

PROBE = "tok-v0"
JOB_FILE = "runpod/fleet/tok_v0.sh"
FPS = L.FPS
MASKED = ("body", "left_hand", "right_hand")            # what v1 reads (``keypoints2d.from_state``)
ANCHORS = ("real", "blank", "identity")
V0_FORMS = ("v0_blank_real_marks", "v0_blank_other_marks", "v0_other_motion", "v0_other_marks", "v0_random_marks", "v0_nearest_fill", "v0_other_motion_other_marks")
FORMS = ANCHORS + V0_FORMS
FORM_ID = {name: i for i, name in enumerate(FORMS)}
V0_SEED = 20261102
VIDEO_ID = re.compile(r"^(.+?)_\d+-\d+-rgb_front$")        # How2Sign's clip ids
FOLD_FLAG = .005                                        # the report flags a mean fold share above this; it decides nothing
PROBES_IDENTITY = f"{L.O}/tok-probes/forms/form-identity.json"   # the probes' stored read of identity on D20
PROBES_IDENTITY_SHA = "24958febc663ab00f984f8f24bea791e3e5a0904a191311e49a213e4f58e0356"
BOOT = 2000
DISCLOSURES = dict(
    normalise="State forms cannot separate a mark's flag from its effect on normalise: the shoulder origin, shoulder width and palm width are medians over marked frames, so changing marks "
              "moves every point a little as well (reader_input).",
    fill="The fill comparison carries that normalisation change: v0_nearest_fill and identity take their medians over filled frames holding different values.",
    removing_marks="Removing a mark also removes that frame's real values: v0_other_marks and v0_random_marks hide real motion wherever they unmark a tracked frame, as well as changing the flag.",
    partner="Partners are adjacent by length within a signer, so wrong-sentence forms keep the signer and about the length; the time scale still changes the tempo slightly.",
    folds="An earlier report of folded hands at 0-5.6% of frames was wrong: that count took tracked forearms without checking the hand's mark. 252 c02 train clips show none by the rule "
          "or by direct geometry among 61,324 marked hand-frames (the wrist rotation is missing on 21.4% of hand-frames, none of them marked).")
COMPARATORS = dict(identity=("real",), v0_blank_real_marks=("v0_blank_other_marks",), v0_blank_other_marks=(), v0_other_motion=("v0_other_motion_other_marks", "v0_blank_real_marks"),
                   v0_other_marks=("v0_random_marks", "identity"), v0_random_marks=("identity",), v0_nearest_fill=("identity",))
HAND_WRISTS = (6, 27)                                    # each hand's wrist point: placed by the arm, not hand-local
NOT_DECIDED = "report only: the declaration's reading is its own; no rule, bar or verdict is computed here"


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    try:
        return provenance.code_digest(extra_files=(JOB_FILE,))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


# ---- marks and values ---------------------------------------------------------------------------------------------------

def with_marks(values, marks):
    """``marks``' state, with the read channels' values taken from ``values`` on the frames ``marks`` marks; zero elsewhere, as an untracked frame stores it."""
    channels = dict(marks.channels)
    for name in MASKED:
        c, v = marks.channels[name], values.channels[name]
        if c.values.shape != v.values.shape:
            raise SystemExit(f"{name}: values {v.values.shape} and marks {c.values.shape} differ")
        channels[name] = replace(c, values=np.where(c.valid[..., None], v.values, 0.).astype(c.values.dtype))
    return replace(marks, channels=channels)


def nearest_index(valid: np.ndarray) -> np.ndarray:
    """(T,) for one joint: each frame's nearest valid frame, the earlier on a tie; itself where none is valid."""
    seen = np.flatnonzero(valid)
    if not len(seen):
        return np.arange(len(valid))
    right = np.clip(np.searchsorted(seen, np.arange(len(valid))), 0, len(seen) - 1)
    left = np.clip(right - 1, 0, len(seen) - 1)
    frames = np.arange(len(valid))
    return np.where(np.abs(frames - seen[left]) <= np.abs(seen[right] - frames), seen[left], seen[right])


def filled(values, tracked=None, neutral=None):
    """``values`` with every read channel valid: each joint's untracked frames (by ``tracked``, default its own marks) hold its nearest tracked value; a joint never tracked holds ``neutral``."""
    tracked, neutral = tracked or values, neutral or values
    channels = dict(values.channels)
    for name in MASKED:
        c, seen, base = values.channels[name], tracked.channels[name].valid, neutral.channels[name].values
        out = c.values.copy()
        for j in range(c.values.shape[1]):
            index = nearest_index(seen[:, j])
            out[:, j] = c.values[index, j] if seen[:, j].any() else base[:, j]
        channels[name] = replace(c, values=out, valid=np.ones_like(c.valid), inference_method=c.inference_method or "tok_v0 nearest tracked value")
    return replace(values, channels=channels)


def neutral_pose(state, tokenizer):
    """The path's own neutral: ``identity`` of the clip with every channel removed. A constant pose, valid everywhere."""
    from evaluate_text_to_motion import blanked
    return P.decoded_form("identity", blanked([state])[0], tokenizer, None)


def stretched_marks(partner, frames: int):
    """The partner's marks at another length, by nearest frame: its pattern, not interpolated marks."""
    from signtranslator.generation.compositional import take_frames
    return take_frames(partner, np.rint(np.linspace(0, len(partner.timestamps) - 1, frames)).astype(np.int64))


def shuffled_runs(state, rng):
    """``state``'s frames with its runs of identical read-channel patterns in a random order: each joint's tracked rate and each run's length stay, the order does not."""
    from signtranslator.generation.compositional import take_frames
    pattern = np.concatenate([state.channels[n].valid for n in MASKED], axis=1)
    starts = np.flatnonzero(np.r_[True, np.any(pattern[1:] != pattern[:-1], axis=1)])
    runs = np.split(np.arange(len(pattern)), starts[1:])
    order = rng.permutation(len(runs))
    return take_frames(state, np.concatenate([runs[i] for i in order]))


def v0_rng(clip: int, form: str) -> np.random.Generator:
    return np.random.default_rng([V0_SEED, clip, FORM_ID[form]])


def video_of(sign: str) -> str | None:
    match = VIDEO_ID.match(sign)
    return match[1] if match else None


def pair_partners(records, frames) -> tuple[list[int], dict]:
    """(partner per clip, what the pairing did). Within a signer, by length: each clip takes the next unpaired one of another video; a clip left over joins the pair nearest in length that holds
    neither its video nor itself, as a three-cycle. Only a signer with one clip goes across signers (counted). Deterministic: length, then id."""
    n = len(records)
    if n < 2:
        raise ValueError("pairing needs at least two clips")
    videos = [video_of(r["sign"]) for r in records]
    groups = collections.defaultdict(list)
    for i, r in enumerate(records):
        groups[r.get("signer_component")].append(i)
    partner, same_video, cross_signer, cycles = [-1] * n, 0, 0, 0

    def apart(a: int, b: int) -> bool:
        return videos[a] is None or videos[a] != videos[b]

    for group in groups.values():
        order = sorted(group, key=lambda i: (frames[i], records[i]["sign"]))
        pairs, left, open_ = [], [], list(order)
        while open_:
            a = open_.pop(0)
            b = next((x for x in open_ if apart(a, x)), None)
            if b is None:
                left.append(a)
            else:
                open_.remove(b)
                pairs.append((a, b))
                partner[a], partner[b] = b, a
        for a in left:
            usable = [p for p in pairs if apart(a, p[0]) and apart(a, p[1])] or pairs
            if not usable:
                continue
            x, y = min(usable, key=lambda p: (abs(frames[p[0]] - frames[a]), p[0]))
            same_video += sum(not apart(a, z) for z in (x, y))
            partner[a], partner[x], partner[y] = x, y, a            # a -> x -> y -> a
            cycles += 1
    for a in [i for i in range(n) if partner[i] < 0]:               # a signer alone in the list
        partner[a] = min((i for i in range(n) if i != a), key=lambda i: (abs(frames[i] - frames[a]), records[i]["sign"]))
        cross_signer += 1
    ratios = np.array([frames[partner[i]] / max(frames[i], 1) for i in range(n)], np.float64)
    return partner, dict(rule="adjacent by length within a signer, no pair from one video", clips=n, video_ids_exposed=sum(v is not None for v in videos), same_video=same_video, cross_signer=cross_signer, cycles=cycles,
                         length_ratio=dict(p10=round(float(np.percentile(ratios, 10)), 4), p50=round(float(np.percentile(ratios, 50)), 4), p90=round(float(np.percentile(ratios, 90)), 4),
                                           outside_25_percent=round(float(np.mean(np.abs(ratios - 1) > .25)), 4)))


def never_tracked_records(store, k: int, *, scan: int = 300) -> list[dict]:
    """``k`` c02 train clips with a hand never tracked, from a fixed draw, read a chunk at a time: a smoke must meet them (about 1 clip in 40)."""
    pool = L.train_records(L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA), ("c02",))
    draw = random.Random(V0_SEED).sample(pool, min(scan, len(pool)))
    L.check_train_records(draw, ("c02",))
    found = []
    for start in range(0, len(draw), 32):
        chunk = draw[start:start + 32]
        for record, state in zip(chunk, L.load_states(store, chunk)):
            if len(found) < k and any(not state.channels[h].valid.any() for h in ("left_hand", "right_hand")):
                found.append(record)
        if len(found) == k:
            return found
    raise SystemExit(f"{len(found)} of {k} clips with a never-tracked hand in {len(draw)} c02 train clips")


def v0_form(name: str, state, partner, clip: int, tokenizer):
    """One clip's form."""
    from signtranslator.generation.compositional import time_scale
    frames = len(state.timestamps)
    if name == "v0_nearest_fill":
        path = P.decoded_form("identity", state, tokenizer, None)
        return filled(path, tracked=state)
    neutral = neutral_pose(state, tokenizer)
    own = filled(state, neutral=neutral)
    if name == "v0_blank_real_marks":
        return with_marks(neutral, state)
    if name == "v0_blank_other_marks":
        return with_marks(neutral, stretched_marks(partner, frames))
    if name in ("v0_other_motion", "v0_other_motion_other_marks"):
        motion = time_scale(filled(partner, neutral=neutral_pose(partner, tokenizer)), frames)
        return with_marks(motion, state if name == "v0_other_motion" else stretched_marks(partner, frames))
    if name == "v0_other_marks":
        return with_marks(own, stretched_marks(partner, frames))
    if name == "v0_random_marks":
        return with_marks(own, shuffled_runs(state, v0_rng(clip, name)))
    raise SystemExit(f"unknown form {name}")


def build_form(name: str, refs, tokenizer, partner) -> list:
    from evaluate_text_to_motion import blanked
    if name == "real":
        return list(refs)
    if name == "blank":
        return blanked(refs)
    if name == "identity":
        return [P.decoded_form("identity", s, tokenizer, None) for s in refs]
    if name not in FORM_ID:
        raise SystemExit(f"unknown form {name}")
    return [v0_form(name, s, refs[partner[i]], i, tokenizer) for i, s in enumerate(refs)]


# ---- what v1 receives -----------------------------------------------------------------------------------------------------

def _frame(points, valid):
    both = valid[:, 0] & valid[:, 1]
    if not both.any():
        return None, None
    p = points.astype(np.float64)
    return np.median((p[both, 0] + p[both, 1]) / 2, axis=0), float(np.median(np.linalg.norm(p[both, 0] - p[both, 1], axis=-1)))


def fold_share(state) -> float:
    """The share of this form's hand-frames (two per frame) that are marked and whose 20 hand points all sit on the wrist."""
    from signtranslator.data.keypoints2d import from_state
    points, valid = from_state(state)
    folded = 0
    for a in HAND_WRISTS:
        folded += int((valid[:, a] & (np.abs(points[:, a + 1:a + 21] - points[:, a:a + 1]).max(axis=(1, 2)) < 1e-6)).sum())
    return round(folded / max(2 * len(points), 1), 6)


def marks_as_real(real, form) -> bool:
    return all(np.array_equal(real.channels[n].valid, form.channels[n].valid) for n in MASKED)


def reader_input_differences(real, form) -> dict:
    """What v1 receives from ``form`` against ``real`` (``translator_items``: ``from_state``, then ``normalise``, then x, y and a mark per point)."""
    from signtranslator.data.keypoints2d import from_state, normalise
    raw_r, valid_r = from_state(real)
    raw_f, valid_f = from_state(form)
    p_r, p_f = normalise(raw_r, valid_r, y_down=False), normalise(raw_f, valid_f, y_down=False)
    both = valid_r & valid_f
    arm = np.zeros(both.shape[1], bool)
    arm[list(range(6)) + list(HAND_WRISTS)] = True
    gap = np.abs(p_f - p_r).sum(-1)
    mid_r, width_r = _frame(raw_r, valid_r)
    mid_f, width_f = _frame(raw_f, valid_f)
    moved = None if mid_r is None or mid_f is None or width_r == 0 else (float(np.linalg.norm(mid_f - mid_r) / width_r), float(width_f / width_r))

    def mean(mask):
        return round(float(gap[mask].mean()), 6) if mask.any() else None
    return dict(marks_added=round(float((valid_f & ~valid_r).mean()), 5), marks_removed=round(float((valid_r & ~valid_f).mean()), 5),
                arm_difference_on_both=mean(both & arm), hand_local_difference_on_both=mean(both & ~arm),
                origin_shift_in_shoulder_widths=None if moved is None else round(moved[0], 6), unit_ratio=None if moved is None else round(moved[1], 6))


def summarise_reader_input(rows) -> dict:
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows if r[key] is not None]
        out[key] = round(float(np.mean(values)), 6) if values else None
    return out


# ---- commands -----------------------------------------------------------------------------------------------------------------

def run_preflight(args) -> int:
    """Before anything is spent: the finished read is not stored, and this code's read was not voided."""
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
    unknown = [f for f in names if f not in FORM_ID]
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
        records = list(records) + [r for r in never_tracked_records(store, args.smoke_never_tracked) if r["sign"] not in have]
        described = dict(described, clips=len(records), never_tracked=args.smoke_never_tracked)
    P.check_read_list(records, declared)
    texts, signs = [r["text"] for r in records], [r["sign"] for r in records]
    refs = L.load_states(store, records)
    tokenizer = L.load_tokenizer(store)
    reader = P.load_reader(store, args)
    candidates = M.retrieval_candidates(len(texts), 1, seed=0)
    digest, place = code_digest(), P.environment()
    partner, pairing = pair_partners(records, [len(s.timestamps) for s in refs])
    for name in names:
        started = time.time()
        states = build_form(name, refs, tokenizer, partner)
        built = time.time()
        raw = reader.read(states, texts, candidates, group=args.nll_group)
        report = dict(kind=f"{PROBE}-form", report_only=True, form=name, declared_read=declared, list=described, code_digest=digest, environment=place,
                      instrument=reader.info, tokenizer=dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA, gains=None), signs=signs, references=texts,
                      partner=partner, pairing=pairing, hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(),
                      seconds_over_reference=[round(len(s.timestamps) / max(len(r.timestamps), 1), 5) for s, r in zip(states, refs)],
                      motion=L.motion_statistics(states, refs), hands=P.hand_statistics(states, refs), tracked=P.tracked_statistics(states),
                      reader_input=[reader_input_differences(r, s) for r, s in zip(refs, states)], fold_share=[fold_share(s) for s in states],
                      marks_as_real=[marks_as_real(r, s) for r, s in zip(refs, states)],
                      build_seconds=round(built - started, 2), read_seconds=round(time.time() - built, 2), peak_rss_bytes=P.peak_rss_bytes(),
                      gpu_peak_reserved_bytes=P.gpu_peak_reserved_bytes())
        P.write_report_once(out_dir / f"form-{name}.json", report)
        print(json.dumps({name: dict(build_seconds=report["build_seconds"], read_seconds=report["read_seconds"], peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2),
                                     gpu_reserved_bytes=report["gpu_peak_reserved_bytes"])}), flush=True)
        del states, raw
    return 0


def declared_reads(reads: dict) -> dict | None:
    """What the declaration reads, as facts, on likelihood minus blank (its primary metric) and on BLEU, jointly resampled over clips (n 2,000, seed 0). None when a form is missing."""
    forms = ("real", "identity", "v0_blank_real_marks", "v0_blank_other_marks", "v0_other_motion", "v0_other_motion_other_marks", "v0_other_marks", "v0_nearest_fill")
    if any(f not in reads for f in forms) or "blank" not in reads:
        return None
    refs = reads["real"]["references"]
    idx = np.random.default_rng(0).integers(0, len(refs), (BOOT, len(refs)))
    blank = np.asarray(reads["blank"]["nll"])[:, 0]
    dnll = {f: M.delta_nll(blank, np.asarray(reads[f]["nll"])[:, 0]) for f in forms}
    stats = {f: M.bleu_statistics(reads[f]["hypotheses"], refs) for f in forms}

    def values(metric: str) -> tuple[dict, dict]:
        if metric == "likelihood_minus_blank":
            return {f: float(dnll[f].mean()) for f in forms}, {f: dnll[f][idx].mean(1) for f in forms}
        return {f: float(M.bleu_from_statistics(stats[f].sum(0))) for f in forms}, {f: M.bleu_from_statistics(stats[f][idx].sum(1)) for f in forms}

    def block(metric: str) -> dict:
        point, draws = values(metric)

        def difference(a: str, b: str) -> dict:
            return dict(point=round(point[a] - point[b], 5), ci95=P._quantiles(draws[a] - draws[b]))

        def position(form: str) -> dict:
            gap = point["real"] - point["identity"]
            with np.errstate(divide="ignore", invalid="ignore"):
                resampled = (draws[form] - draws["identity"]) / (draws["real"] - draws["identity"])
            return dict(point=round((point[form] - point["identity"]) / gap, 4) if gap else None, ci95=P._quantiles(resampled))
        return dict(a=difference("v0_blank_real_marks", "v0_blank_other_marks"), b=difference("v0_other_motion", "v0_other_motion_other_marks"), real_minus_identity=difference("real", "identity"),
                    p2=position("v0_other_marks"), p3=position("v0_nearest_fill"))
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
    if len({json.dumps(r["partner"]) for r in reads.values()}) != 1:
        raise SystemExit("forms read different partners")
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
    blank_nll = np.asarray(reads["blank"]["nll"])[:, 0]
    real_dnll = M.delta_nll(blank_nll, np.asarray(reads["real"]["nll"])[:, 0])
    panels = {}
    for name in names:
        panels[name] = P.form_panel(reads, name, tuple(c for c in COMPARATORS.get(name, ()) if c in reads))
        panels[name]["reader_input"] = summarise_reader_input(reads[name]["reader_input"])
        panels[name]["fold_share"] = M.bootstrap_mean(reads[name]["fold_share"])
        panels[name]["clips_with_real_marks"] = round(float(np.mean(reads[name]["marks_as_real"])), 4)
        panels[name]["likelihood_ratio_of_means"] = M.ratio_of_means(M.delta_nll(blank_nll, np.asarray(reads[name]["nll"])[:, 0]), real_dnll, n=BOOT, seed=0)
    folds = {name: panels[name]["fold_share"]["mean"] for name in names}
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, declared_read=first["declared_read"], list=first["list"], instrument=first["instrument"],
                  tokenizer=first["tokenizer"], clips=len(first["signs"]), code_digest=digest, environment=place,
                  rerun=json.loads((out_dir / "rerun.json").read_bytes()) if (out_dir / "rerun.json").exists() else None,
                  smoke=None if first["declared_read"] else "SMOKE on c02 train clips: these numbers are plumbing, not a read", forms=panels, partner=first["partner"], pairing=first["pairing"],
                  disclosures=DISCLOSURES, reads=declared_reads(reads), fold_flag=dict(limit=FOLD_FLAG, real=folds["real"], flagged=folds["real"] > FOLD_FLAG, forms_over=sorted(n for n, v in folds.items() if v > FOLD_FLAG)),
                  hypotheses={name: reads[name]["hypotheses"] for name in names},
                  checks=dict(blank_identical_to_real=panels["blank"]["identical_to_real"], canary=canary, path_check=path_check),
                  resources={name: dict(peak_rss_bytes=reads[name]["peak_rss_bytes"], gpu_peak_reserved_bytes=reads[name]["gpu_peak_reserved_bytes"],
                                        build_seconds=reads[name]["build_seconds"], read_seconds=reads[name]["read_seconds"]) for name in names})
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, forms=len(panels))), flush=True)
    return 0


def run_check(args) -> int:
    """Exit 3 on broken plumbing: not report_only, blank reading as real, or a declared read without its passed canary and path check. It judges no form."""
    report = json.loads(Path(args.report).read_text())
    checks, problems = report["checks"], []
    if report.get("report_only") is not True:
        problems.append("the report does not say report_only")
    if checks["blank_identical_to_real"] >= 1:
        problems.append("blank reads exactly as real")
    if report.get("declared_read") and not (checks.get("canary") or {}).get("reproduced"):
        problems.append("a declared read without its passed canary")
    if report.get("declared_read") and not (checks.get("path_check") or {}).get("reproduced"):
        problems.append("a declared read without its passed path check")
    print(json.dumps(dict(problems=problems)), flush=True)
    for line in problems:
        print(f"ALERT {line}", flush=True)
    return 3 if problems else 0


def run_path_check(args) -> int:
    """``identity`` against the probes' stored read: the same path, the canary's limits. Exit 3, voiding the read, on a miss. A smoke checks the pin only, and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    raw = store.get(PROBES_IDENTITY)
    P.check_sha(raw, PROBES_IDENTITY_SHA, PROBES_IDENTITY)
    stored = json.loads(raw)
    out = Path(args.out_dir)
    read = json.loads((out / "form-identity.json").read_bytes())
    if not read["declared_read"]:
        print(json.dumps(dict(path_check="not applicable: a smoke reads train clips; the pin is checked")), flush=True)
        return 0
    if read["signs"] != stored["signs"] or read["references"] != stored["references"]:
        raise SystemExit("the stored identity read other clips or another order")
    identical = float(np.mean([a == b for a, b in zip(read["hypotheses"], stored["hypotheses"])]))
    gap = float(np.mean(np.abs(np.asarray(read["nll"])[:, 0] - np.asarray(stored["nll"])[:, 0])))
    misses = [why for why, bad in (("hypotheses", identical < P.CANARY["identical"]), ("NLL", gap > P.CANARY["nll"])) if bad]
    result = dict(identical=round(identical, 4), nll_mean_absolute_difference=round(gap, 6), limits=dict(identical=P.CANARY["identical"], nll=P.CANARY["nll"]), misses=misses, reproduced=not misses,
                  stored=PROBES_IDENTITY, stored_sha256=PROBES_IDENTITY_SHA, stored_environment=stored.get("environment"), code_digest=read["code_digest"], environment=read["environment"])
    if not (out / "path_check.json").exists():
        P.write_report_once(out / "path_check.json", result)
    print(json.dumps(result), flush=True)
    if not misses:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{read['code_digest']}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-void", code_digest=read["code_digest"], path_check=result), indent=1, sort_keys=True).encode())
    print(f"ALERT identity does not reproduce the probes' stored read: {misses}. The read is void: no retry; a rerun needs new code and a recorded cause", flush=True)
    return 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
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
    read.add_argument("--smoke-train", type=int, default=0, help="read this many c02 train clips, not D20")
    read.add_argument("--smoke-longest", type=int, default=0, help="a smoke's list also holds this many of c02 train's longest clips")
    read.add_argument("--smoke-never-tracked", type=int, default=0, help="a smoke's list also holds this many c02 train clips with a hand never tracked")
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    canary = sub.add_parser("canary", help="tok_probes' canary: real and blank against the pinned stored report; exit 3, voiding the read, otherwise")
    canary.add_argument("--store", default="r2")
    canary.add_argument("--out-dir", required=True)
    canary.add_argument("--void-prefix", required=True)
    path = sub.add_parser("path-check", help="identity against the probes' stored read; exit 3, voiding the read, otherwise")
    path.add_argument("--store", default="r2")
    path.add_argument("--out-dir", required=True)
    path.add_argument("--void-prefix", required=True)
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
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return dict(preflight=run_preflight, read=run_read, canary=P.run_canary, merge=run_merge, **{"path-check": run_path_check}, check=run_check, upload=P.run_upload)[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
