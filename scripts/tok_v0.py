"""V0: do real's validity marks carry content, or does v1 only mind what marking every frame valid does? (Report only; nothing trains, nothing is declared by this code.)

The tokenizer probes read ``round_trip_masked`` and ``round_trip`` far apart, and ``identity`` (no network at all) between them. Before that reads as "the tokenizer is fine",
this asks whether the marks themselves tell v1 about the sentence. D20 only, instrument v1, through ``tok_probes.py``'s reads, canary, digest and panels, write-once under
``tok-probes-v0/``. No option names another list or instrument. ``--smoke-train N`` reads c02 train clips and says so.

**Forms.** ``real``, ``blank``, ``identity`` (as ``tok_probes``), and:

- ``v0_blank_real_marks``: a constant neutral pose, with this clip's validity. All it can tell v1 is the marks. **Control:** ``v0_blank_other_marks``, the same pose with the partner's
  marks, stretched to this clip's length by nearest frame. (This control is not on the task's list: without it, blank against this form also moves zeros to a pose, so it could
  not show that the pattern is this sentence's.)
- ``v0_other_motion``: the partner's motion, time-scaled to this clip's length, with this clip's validity.
- ``v0_other_marks``: this clip's values with the partner's marks. A frame the partner marks and this clip did not track holds this clip's nearest tracked value.
- ``v0_random_marks``: this clip's values with its own marks reordered: its runs of identical frame patterns in a random order (seeded per clip and form). Each joint's tracked rate
  is exact and run lengths stay, so random flicker does not stand in for a wrong pattern. Filled as ``v0_other_marks``.
- ``v0_nearest_fill``: ``identity`` with its neutral fill replaced by the nearest tracked value of each joint; a joint never tracked keeps the neutral. Every frame still marked valid.

**Partner.** The mismatch rule in use: ``sentence_exposure_diagnostic.derangement(N, seed + 1)`` over the list's positions, seed 0 as ``round15_compose_eval.py``. So never itself.

**Per form,** ``tok_probes.form_panel``: BLEU, likelihood minus blank, loop share, length, wrist, hands, tracked share, share of the gap, paired against real, blank and ``comparators_of``.
``reader_input`` says what v1 receives against real: marks added and removed, coordinate differences on frames both mark (arm points apart from hand-local ones), and how far the clip's
shoulder origin and unit moved, since ``keypoints2d.normalise`` takes them from the marked frames.

    python scripts/tok_v0.py preflight --out-dir W --final-key K --void-prefix V
    python scripts/tok_v0.py read --out-dir W/forms --instrument-cache W/i
    python scripts/tok_v0.py canary --out-dir W/forms --void-prefix V
    python scripts/tok_v0.py merge --out-dir W/forms --output-file W/report.json
    python scripts/tok_v0.py check --report W/report.json

The job is ``runpod/fleet/tok_v0.sh``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import functools
import json
from pathlib import Path
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
V0_FORMS = ("v0_blank_real_marks", "v0_blank_other_marks", "v0_other_motion", "v0_other_marks", "v0_random_marks", "v0_nearest_fill")
FORMS = ANCHORS + V0_FORMS
FORM_ID = {name: i for i, name in enumerate(FORMS)}
V0_SEED = 20261102
MISMATCH_SEED = 0 + 1                                   # round15_compose_eval: --seed 0, derangement(n, seed + 1)
COMPARATORS = dict(identity=("real",), v0_blank_real_marks=("v0_blank_other_marks",), v0_blank_other_marks=(), v0_other_motion=("v0_blank_real_marks",),
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
        channels[name] = replace(c, values=out, valid=np.ones_like(c.valid))
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


def partner_of(n: int) -> list[int]:
    from sentence_exposure_diagnostic import derangement
    return list(derangement(n, MISMATCH_SEED))


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
    if name == "v0_other_motion":
        return with_marks(time_scale(filled(partner, neutral=neutral_pose(partner, tokenizer)), frames), state)
    if name == "v0_other_marks":
        return with_marks(own, stretched_marks(partner, frames))
    if name == "v0_random_marks":
        return with_marks(own, shuffled_runs(state, v0_rng(clip, name)))
    raise SystemExit(f"unknown form {name}")


def build_form(name: str, refs, tokenizer) -> list:
    from evaluate_text_to_motion import blanked
    if name == "real":
        return list(refs)
    if name == "blank":
        return blanked(refs)
    if name == "identity":
        return [P.decoded_form("identity", s, tokenizer, None) for s in refs]
    if name not in FORM_ID:
        raise SystemExit(f"unknown form {name}")
    partner = partner_of(len(refs))
    return [v0_form(name, s, refs[partner[i]], i, tokenizer) for i, s in enumerate(refs)]


# ---- what v1 receives -----------------------------------------------------------------------------------------------------

def _frame(points, valid):
    both = valid[:, 0] & valid[:, 1]
    if not both.any():
        return None, None
    p = points.astype(np.float64)
    return np.median((p[both, 0] + p[both, 1]) / 2, axis=0), float(np.median(np.linalg.norm(p[both, 0] - p[both, 1], axis=-1)))


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
    if args.smoke_longest and not args.smoke_train:
        raise SystemExit("--smoke-longest belongs to a smoke")
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
    P.check_read_list(records, declared)
    texts, signs = [r["text"] for r in records], [r["sign"] for r in records]
    refs = L.load_states(store, records)
    tokenizer = L.load_tokenizer(store)
    reader = P.load_reader(store, args)
    candidates = M.retrieval_candidates(len(texts), 1, seed=0)
    digest, place = code_digest(), P.environment()
    for name in names:
        started = time.time()
        states = build_form(name, refs, tokenizer)
        built = time.time()
        raw = reader.read(states, texts, candidates, group=args.nll_group)
        report = dict(kind=f"{PROBE}-form", report_only=True, form=name, declared_read=declared, list=described, code_digest=digest, environment=place,
                      instrument=reader.info, tokenizer=dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA, gains=None), signs=signs, references=texts,
                      partner=partner_of(len(refs)), hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(),
                      seconds_over_reference=[round(len(s.timestamps) / max(len(r.timestamps), 1), 5) for s, r in zip(states, refs)],
                      motion=L.motion_statistics(states, refs), hands=P.hand_statistics(states, refs), tracked=P.tracked_statistics(states),
                      reader_input=[reader_input_differences(r, s) for r, s in zip(refs, states)],
                      build_seconds=round(built - started, 2), read_seconds=round(time.time() - built, 2), peak_rss_bytes=P.peak_rss_bytes(),
                      gpu_peak_reserved_bytes=P.gpu_peak_reserved_bytes())
        P.write_report_once(out_dir / f"form-{name}.json", report)
        print(json.dumps({name: dict(build_seconds=report["build_seconds"], read_seconds=report["read_seconds"], peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2),
                                     gpu_reserved_bytes=report["gpu_peak_reserved_bytes"])}), flush=True)
        del states, raw
    return 0


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
    canary = None
    if first["declared_read"]:
        path = out_dir / "canary.json"
        canary = json.loads(path.read_bytes()) if path.exists() else None
        if canary is None or canary.get("reproduced") is not True or canary.get("code_digest") != digest or canary.get("environment") != place:
            raise SystemExit(f"{path}: no passed canary of this code and environment; the read is void")
    panels = {}
    for name in names:
        panels[name] = P.form_panel(reads, name, tuple(c for c in COMPARATORS.get(name, ()) if c in reads))
        panels[name]["reader_input"] = summarise_reader_input(reads[name]["reader_input"])
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, declared_read=first["declared_read"], list=first["list"], instrument=first["instrument"],
                  tokenizer=first["tokenizer"], clips=len(first["signs"]), code_digest=digest, environment=place,
                  rerun=json.loads((out_dir / "rerun.json").read_bytes()) if (out_dir / "rerun.json").exists() else None,
                  smoke=None if first["declared_read"] else "SMOKE on c02 train clips: these numbers are plumbing, not a read", forms=panels, partner=first["partner"],
                  hypotheses={name: reads[name]["hypotheses"] for name in names},
                  checks=dict(blank_identical_to_real=panels["blank"]["identical_to_real"], canary=canary),
                  resources={name: dict(peak_rss_bytes=reads[name]["peak_rss_bytes"], gpu_peak_reserved_bytes=reads[name]["gpu_peak_reserved_bytes"],
                                        build_seconds=reads[name]["build_seconds"], read_seconds=reads[name]["read_seconds"]) for name in names})
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, forms=len(panels))), flush=True)
    return 0


def run_check(args) -> int:
    """Exit 3 on broken plumbing: not report_only, blank reading as real, or a declared read without its passed canary. It judges no form."""
    report = json.loads(Path(args.report).read_text())
    checks, problems = report["checks"], []
    if report.get("report_only") is not True:
        problems.append("the report does not say report_only")
    if checks["blank_identical_to_real"] >= 1:
        problems.append("blank reads exactly as real")
    if report.get("declared_read") and not (checks.get("canary") or {}).get("reproduced"):
        problems.append("a declared read without its passed canary")
    print(json.dumps(dict(problems=problems)), flush=True)
    for line in problems:
        print(f"ALERT {line}", flush=True)
    return 3 if problems else 0


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
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    canary = sub.add_parser("canary", help="tok_probes' canary: real and blank against the pinned stored report; exit 3, voiding the read, otherwise")
    canary.add_argument("--store", default="r2")
    canary.add_argument("--out-dir", required=True)
    canary.add_argument("--void-prefix", required=True)
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
    return dict(preflight=run_preflight, read=run_read, canary=P.run_canary, merge=run_merge, check=run_check, upload=P.run_upload)[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
