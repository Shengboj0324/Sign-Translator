"""The UWB lifter gate's downstream row: the frozen back-translator's read of lifted states against our states.

Declared (plan "UWB lifter: declared selection and gate", 2026-10-30 03:34 UTC, amendment 1 03:59, owner 04:02):
"back-translator read of lifted states >= 90% of its read of our states on paired cues, and clearly above blank".

**Pairs.** The gate report (``uwb_lifter_gate.py --out``; a local path, a store key, or ``KEY@SHA256``) keeps, per
evaluated pair, ``video_id``, ``clip_id``, ``sample_id`` (our cue), ``cue_start``, ``iou``, ``fps`` (the gate's estimate
for that video), ``zip``, ``member_sha256``, the chosen ``alignment`` (lag) and ``aligned_frames``; it does not keep our
state key, its SHA-256 or the cue text. Each pair is therefore **reconstructed** from the same inputs the gate read:
our report ``<youtube prefix>/videos/<video>.json`` and the UWB translations file. ``uwb_lifter_gate.pair_clips`` is
rerun at the pair's fps and must return the same (clip, cue, IoU), the cue's start must equal ``cue_start``, the
video's fps must equal the report's fps row, the UWB member's SHA-256 must equal ``member_sha256`` and the frame map
must hold ``aligned_frames`` frames; any difference refuses the pair (and is counted), never silently re-pairs it.

**Sample.** Every evaluated pair when there are at most ``--pairs`` (default 2000); otherwise a seeded subsample
(``--seed``, default 0): one pair per video first (videos in a seeded order, one seeded pick each), then, if fewer
videos than ``--pairs``, a seeded draw from the remaining pairs.

**Per pair**, three states, all read by the frozen instrument in use (``back-translator-kp2d-uwb-ft-v1``, its
``best.json`` checkpoint, through ``score_back_translation.load_translator`` / ``translate`` after ``at_rate(., 24)``,
exactly as the pilots read motion), against the **same reference**, our cue's English text:

1. **lifted**: the UWB clip converted by ``convert_uwb_clips.convert_member`` itself (its worker state set up as a
   conversion run would: the xy2 bundle, the gate's visibility rule, ``generator`` flags, the converter's face and
   head defaults, ``fps_source`` ``estimated`` at the gate's fps), cut to the UWB frames the gate aligned
   (``uwb_lifter_gate.frame_map`` with the pair's lag): the frame range ``[j_min, j_max]``;
2. **ours**: our state of the cue cut to the frames those UWB frames were matched to: ``[i_min, i_max]`` (the same
   time window, within half a frame);
3. **blank**: ``evaluate_text_to_motion.blanked`` of (2): every channel removed, the same duration.

**Statistics.** Corpus BLEU (the repo's ``motion_to_text.corpus_bleu``, which every pilot and helper uses; the repo has
no sacrebleu) of each read; the ratio lifted / ours; a paired bootstrap **clustered by video** (``--bootstrap`` 2,000
resamples of videos with replacement, seed 0; the BLEU of a resample is computed exactly from summed per-segment
n-gram statistics) for lifted - ours, lifted - blank and ours - blank; per-pair exact-match rates (each read's
hypothesis equal to the reference, and the reads' hypotheses equal to each other); the same reads split by the UWB
partition of the clip (the instrument was pretrained on UWB's train split: a contamination diagnostic, not gated).

**Verdict** (per read set): ``pass`` iff ratio >= 0.9 and the lifted - blank lower bound > 0, provided ours - blank's
lower bound > 0; when the instrument does not read our states above blank (or reads them at 0) it is
``not-evaluable``, never pass. Values are compared unrounded.

**Contamination rule** (review of d2591fa): the instrument was pretrained on UWB's train split, so the row passes only
if it passes on the pooled sample **and** on UWB's dev partition alone (every evaluated pair of
``YT.translations.dev.json``, labelled ``monitor`` by ``convert_uwb_clips.translations``), which needs >= 200 dev pairs
from >= 50 videos (fewer: not-evaluable). More than 10% refused pairs (of the pooled and dev samples together) make the
row not-evaluable.

Workers: a process pool sized from ``os.sched_getaffinity`` and the cgroup memory limit (never ``nproc``, which
honours OMP_NUM_THREADS=1); the instrument runs in the main process on ``--device``.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

UWB_PREFIX = "acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1"
BUNDLE_KEY = f"{UWB_PREFIX}/lifter-v1/files/world-regressors-xy2-v1.pt"
BUNDLE_SHA256 = "02a90e4ff333202d556bee62b12216ffca85761e45dafb011144fcbc1bd82dde"
OUTPUT_KEY = f"{UWB_PREFIX}/lifter-v1/gate-rerun/downstream-v1.json"
INSTRUMENT_RUN = "back-translator-kp2d-uwb-ft-v1"
INSTRUMENT_MANIFEST = "004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd"   # as round17_diagnostics
READ_FPS = 24.                     # the pilots resample every state they read to 24 fps (at_rate) before translate
RATIO_BAR = .9
DEV_PARTITION = "monitor"          # UWB's YT.translations.dev.json (convert_uwb_clips.translations labels it monitor)
MIN_DEV_PAIRS, MIN_DEV_VIDEOS = 200, 50
MAX_REFUSED_SHARE = .10
READS = ("lifted", "ours", "blank")
COMPARISONS = (("lifted", "ours"), ("lifted", "blank"), ("ours", "blank"))
MAX_ORDER = 4
_WORKER: dict = {}


class Mismatch(ValueError):
    """The pair cannot be reconstructed exactly as the gate evaluated it."""


# ----------------------------------------------------------------------------------------------------------------------
# Inputs

def worker_count(per_worker_gb: float = 2., cap: int = 30, reserve: int = 2) -> int:
    """Processes from the CPUs this process may run on and the cgroup memory limit (``per_worker_gb`` each); never
    ``nproc`` (it honours OMP_NUM_THREADS=1)."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:                                    # macOS: no affinity masks
        cpus = os.cpu_count() or 1
    try:
        memory = int(Path("/sys/fs/cgroup/memory.max").read_text().strip()) / 2 ** 30
    except (OSError, ValueError):                             # no cgroup v2 limit ("max") or not Linux
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    return max(1, min(cap, cpus - reserve, int(memory // per_worker_gb)))


def read_spec(store, spec: str) -> tuple[bytes, str]:
    """(bytes, SHA-256) of a local path, a store key, or ``KEY@SHA256`` (the SHA is then checked)."""
    if Path(spec).exists():
        payload = Path(spec).read_bytes()
    else:
        key, _, sha = spec.rpartition("@") if "@" in spec else (spec, "", "")
        payload = store.get(key)
        if sha and hashlib.sha256(payload).hexdigest() != sha:
            raise SystemExit(f"{key}: content SHA-256 differs from the pinned one")
    return payload, hashlib.sha256(payload).hexdigest()


def sample_pairs(pairs: list[dict], limit: int, seed: int = 0) -> list[dict]:
    """Every evaluated pair when at most ``limit`` (0: no limit); else one pair per video first (videos in a seeded
    order, a seeded pick in each), then a seeded draw from the rest. Sorted by (video, clip)."""
    ok = sorted((p for p in pairs if p.get("status") == "evaluated"), key=lambda p: (p["video_id"], p["clip_id"]))
    if not limit or len(ok) <= limit:
        return ok
    rng = random.Random(seed)
    by_video = collections.defaultdict(list)
    for p in ok:
        by_video[p["video_id"]].append(p)
    videos = sorted(by_video)
    rng.shuffle(videos)
    chosen = [rng.choice(by_video[v]) for v in videos][:limit]
    if len(chosen) < limit:
        taken = {p["clip_id"] for p in chosen}
        rest = [p for p in ok if p["clip_id"] not in taken]
        chosen += rng.sample(rest, limit - len(chosen))
    return sorted(chosen, key=lambda p: (p["video_id"], p["clip_id"]))


def reconstruct(pair: dict, our_report: dict, uwb_entry: dict, fps_row: dict | None) -> dict:
    """The pair with our cue's state key, SHA-256, text and end and the clip's frame range, proved equal to the gate's
    pairing (``Mismatch`` otherwise)."""
    import uwb_lifter_gate as gate
    if our_report.get("status") != "converted":
        raise Mismatch("our report is not converted")
    if fps_row is None or fps_row.get("fps") != pair["fps"]:
        raise Mismatch(f"fps {pair['fps']} differs from the report's fps row {fps_row and fps_row.get('fps')}")
    clips = gate.uwb_clips_of(uwb_entry)
    found = [(c, q, iou) for c, q, iou in gate.pair_clips(clips, our_report["segments"], pair["fps"])
             if c["clip_id"] == pair["clip_id"]]
    if not found:
        raise Mismatch("the clip is not paired at the gate's fps")
    clip, cue, iou = found[0]
    if cue["sample_id"] != pair["sample_id"] or iou != pair["iou"] or cue["start"] != pair["cue_start"]:
        raise Mismatch(f"repaired to {cue['sample_id']} (IoU {iou}, start {cue['start']}), the gate had "
                       f"{pair['sample_id']} (IoU {pair['iou']}, start {pair['cue_start']})")
    if not cue.get("text"):
        raise Mismatch("the cue has no text")
    return dict(pair, state_key=cue["state_key"], state_sha256=cue["state_sha256"], text=cue["text"],
                cue_end=cue["end"], first=clip["first"], last=clip["last"])


def frames_of(state, lo: int, hi: int):
    """``state`` restricted to frames ``lo..hi`` (inclusive)."""
    sl = slice(int(lo), int(hi) + 1)
    channels = {n: replace(c, values=c.values[sl].copy(), valid=c.valid[sl].copy(), observed=c.observed[sl].copy(),
                           confidence=c.confidence[sl].copy()) for n, c in state.channels.items()}
    return replace(state, timestamps=state.timestamps[sl].copy(), channels=channels)


def aligned_windows(ours, lifted, uwb_clip, fps: float, lag: int):
    """(our window, lifted window, (i, j)): the gate's frame map of the pair (``uwb_lifter_gate.frame_map``) and each
    state cut to the range of its matched frames."""
    import uwb_lifter_gate as gate
    i, j = gate.frame_map(ours.timestamps, uwb_clip.first, uwb_clip.frames, fps, lag)
    if len(i) < 2:
        raise Mismatch(f"{len(i)} aligned frames")
    return frames_of(ours, i.min(), i.max()), frames_of(lifted, j.min(), j.max()), (i, j)


# ----------------------------------------------------------------------------------------------------------------------
# Conversion workers (convert_uwb_clips.convert_member itself)

def converter_args(visibility: str, face: str, head: str, flags: str = "generator", min_frames: int = 8):
    """The converter's settings for this check (its own defaults where the gate declares nothing)."""
    return SimpleNamespace(fps=30., min_frames=min_frames, world="learned", visibility=visibility, face=face, head=head,
                           flags=flags, out_prefix="uwb-downstream-check", eval_index_sha256=None)


def configure_converter(args, fps_table: dict, regressors, blendshapes=None, resources=None) -> None:
    """``convert_uwb_clips``' worker state as a conversion run sets it, minus text, leak and overlap lookups (they only
    annotate the record, never the state)."""
    import convert_uwb_clips as cu
    cu._WORKER.update(args=args, fps=dict(fps_table), regressors=regressors, blendshapes=blendshapes,
                      resources=resources, texts={}, overlap=dict(videos={}, clips={}), eval_index=None)


def _init(settings: dict, fps_table: dict) -> None:
    import torch
    torch.set_num_threads(1)
    from train_text_to_motion import make_store
    from signtranslator.data_engineering.world_regressor import WorldRegressors
    store = make_store(settings["store"])
    spec = settings["regressors"]
    payload = Path(spec).read_bytes() if Path(spec).exists() else store.get(spec)
    regressors = WorldRegressors.load(payload, expected_sha256=settings["regressors_sha256"])
    resources = blendshapes = None
    if settings["face"] == "blendshapes" or settings["head"] == "affine":
        from signtranslator.data_engineering.semlex_pose import BlendshapeModel, load_face_resources
        resources = load_face_resources(Path(settings["models_dir"]) / "face_landmarker.task")
        blendshapes = BlendshapeModel(resources) if settings["face"] == "blendshapes" else None
    configure_converter(converter_args(settings["visibility"], settings["face"], settings["head"]), fps_table,
                        regressors, blendshapes, resources)
    _WORKER.update(settings=settings, store=store, archives={})


def lift_pair(job: dict, member: bytes, ours) -> dict:
    """(lifted window, our window) of one reconstructed pair, from the member's bytes and our state."""
    import convert_uwb_clips as cu
    from signtranslator.data_engineering import uwb_keypoints as uk
    from signtranslator.pose.multichannel import MultichannelMotion
    if hashlib.sha256(member).hexdigest() != job["member_sha256"]:
        raise Mismatch("the UWB member's SHA-256 differs from the gate's")
    record, data = cu.convert_member(job["clip_id"] + ".json", member, job["zip"])
    if data is None:
        raise Mismatch(f"conversion refused: {record.get('reason')}")
    if record["fps"] != job["fps"] or record["fps_source"] != "estimated":
        raise Mismatch(f"converted at {record['fps']} ({record['fps_source']}), the gate at {job['fps']}")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.npz"
        path.write_bytes(data)
        lifted = MultichannelMotion.load(path, expected_sha256=record["state_sha256"])
    clip = uk.parse_clip(member, job["clip_id"])
    lag = int(job["alignment"]["lag"])
    ours_w, lifted_w, (i, j) = aligned_windows(ours, lifted, clip, job["fps"], lag)
    if len(i) != job["aligned_frames"]:
        raise Mismatch(f"{len(i)} aligned frames, the gate had {job['aligned_frames']}")
    return dict(status="ok", clip_id=job["clip_id"], lifted=lifted_w, ours=ours_w, aligned_frames=int(len(i)),
                lifted_frames=len(lifted_w.timestamps), our_frames=len(ours_w.timestamps), lag=lag,
                seconds=round(float(ours_w.timestamps[-1] - ours_w.timestamps[0]), 4),
                lifted_state_sha256=record["state_sha256"])


def _lift(job: dict) -> dict:
    import uwb_lifter_gate as gate
    from evaluate_text_to_motion import load_state
    from signtranslator.data_engineering import uwb_keypoints as uk
    settings, store = _WORKER["settings"], _WORKER["store"]
    try:
        archive = _WORKER["archives"].get(job["zip"])
        if archive is None:
            archive = _WORKER["archives"][job["zip"]] = gate.open_member_zip(store, f"{settings['uwb_prefix']}/{job['zip']}")
        member = archive.read(job["clip_id"] + ".json")
        ours = load_state(store, job["state_key"], job["state_sha256"])
        return lift_pair(job, member, ours)
    except (Mismatch, uk.Refusal, ValueError, KeyError) as error:
        return dict(status="refused", clip_id=job["clip_id"], reason=f"{type(error).__name__}: {error}"[:300])


# ----------------------------------------------------------------------------------------------------------------------
# BLEU from summed segment statistics (identical to motion_to_text.corpus_bleu)

def segment_stats(hypothesis: str, reference: str) -> np.ndarray:
    """(2 * MAX_ORDER + 2,) clipped matches and totals per order, hypothesis and reference lengths: what
    ``corpus_bleu`` sums over segments."""
    from signtranslator.generation.motion_to_text import words
    h, r = words(hypothesis), words(reference)
    out = np.zeros(2 * MAX_ORDER + 2, dtype=np.int64)
    for n in range(1, MAX_ORDER + 1):
        hc = collections.Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        rc = collections.Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        out[n - 1] = sum(min(c, rc[g]) for g, c in hc.items())
        out[MAX_ORDER + n - 1] = max(len(h) - n + 1, 0)
    out[-2], out[-1] = len(h), len(r)
    return out


def bleu_from(stats: np.ndarray) -> float:
    """``corpus_bleu``'s value (0-100) from summed segment statistics."""
    matches, totals = stats[:MAX_ORDER], stats[MAX_ORDER:2 * MAX_ORDER]
    hyp_len, ref_len = stats[-2], stats[-1]
    precisions = [m / t if t else 0. for m, t in zip(matches, totals)]
    if hyp_len == 0 or min(precisions) == 0:
        return 0.
    bp = 1. if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100 * bp * math.exp(sum(math.log(p) for p in precisions) / MAX_ORDER)


def clustered_bootstrap(stats: dict, videos: list[str], *, resamples: int, seed: int = 0) -> dict:
    """Per read: BLEU and its 95% interval; per comparison (a - b): the paired difference and its interval. Videos are
    resampled with replacement (the same draws for every read), each contributing all its pairs."""
    keys = sorted(set(videos))
    owner = np.array([keys.index(v) for v in videos]) if keys else np.zeros(0, int)
    per_video = {r: np.zeros((len(keys), 2 * MAX_ORDER + 2), dtype=np.int64) for r in stats}
    for r, s in stats.items():
        np.add.at(per_video[r], owner, s)
    point = {r: bleu_from(per_video[r].sum(0)) for r in stats}
    draws = {r: [] for r in stats}
    rng = np.random.default_rng(seed)
    for _ in range(resamples if len(keys) >= 2 else 0):
        counts = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))
        for r in stats:
            draws[r].append(bleu_from(counts @ per_video[r]))

    def interval(values):                     # unrounded: the verdicts compare them to their bars
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if len(values) else None
    reads = {r: dict(bleu=float(point[r]), ci95=interval(draws[r])) for r in stats}
    diffs = {}
    for a, b in COMPARISONS:
        if a in stats and b in stats:
            d = np.asarray(draws[a]) - np.asarray(draws[b])
            diffs[f"{a}_minus_{b}"] = dict(difference=float(point[a] - point[b]), ci95=interval(d))
    return dict(reads=reads, differences=diffs, videos=len(keys), resamples=resamples if len(keys) >= 2 else 0,
                seed=seed)


def verdict(boot: dict, bar: float = RATIO_BAR) -> dict:
    """The declared downstream row: pass / fail / not-evaluable (module docstring)."""
    reads, diffs = boot["reads"], boot["differences"]
    ours, lifted = reads["ours"]["bleu"], reads["lifted"]["bleu"]
    sanity = diffs["ours_minus_blank"]["ci95"]
    above = diffs["lifted_minus_blank"]["ci95"]
    ratio = None if not ours else lifted / ours                    # unrounded against the bar
    out = dict(ratio_lifted_over_ours=ratio, ratio_bar=bar, lifted_minus_blank_lower=above and above[0],
               ours_minus_blank_lower=sanity and sanity[0])
    if not ours or sanity is None or sanity[0] <= 0:
        return dict(out, status="not-evaluable",
                    reason="the instrument does not read our states clearly above blank (ours - blank lower bound "
                           f"{sanity and sanity[0]}), so the row cannot be judged")
    ok_ratio, ok_blank = ratio >= bar, above is not None and above[0] > 0
    return dict(out, ratio_ok=bool(ok_ratio), clearly_above_blank=bool(ok_blank),
                status="pass" if ok_ratio and ok_blank else "fail")


def combined_verdict(pooled: dict, dev: dict | None, *, dev_pairs: int, dev_videos: int, refused: int,
                     sampled: int) -> dict:
    """The row's verdict: pass only if the pooled read **and** the UWB dev-partition read (the instrument never trained
    on it; >= MIN_DEV_PAIRS pairs from >= MIN_DEV_VIDEOS videos) both pass; not-evaluable when more than
    MAX_REFUSED_SHARE of the sampled pairs were refused, or the dev read is too small, or either read is
    not-evaluable; fail when either evaluable read fails."""
    share = refused / sampled if sampled else 1.
    out = dict(pooled=pooled["status"], dev=dev and dev["status"], dev_pairs=dev_pairs, dev_videos=dev_videos,
               min_dev=dict(pairs=MIN_DEV_PAIRS, videos=MIN_DEV_VIDEOS), refused=refused, sampled=sampled,
               refused_share=share, max_refused_share=MAX_REFUSED_SHARE)
    if share > MAX_REFUSED_SHARE:
        return dict(out, status="not-evaluable", reason=f"{refused} of {sampled} sampled pairs refused")
    statuses = [pooled["status"]] + ([dev["status"]] if dev else [])
    if "fail" in statuses:
        return dict(out, status="fail", reason="a read fails: " + ", ".join(
            name for name, v in (("pooled", pooled), ("dev", dev)) if v and v["status"] == "fail"))
    if dev is None or dev_pairs < MIN_DEV_PAIRS or dev_videos < MIN_DEV_VIDEOS:
        return dict(out, status="not-evaluable", reason=f"UWB dev partition: {dev_pairs} pairs from {dev_videos} "
                                                        f"videos (needs {MIN_DEV_PAIRS} from {MIN_DEV_VIDEOS})")
    if "not-evaluable" in statuses:
        return dict(out, status="not-evaluable", reason="a read is not evaluable (ours not above blank)")
    return dict(out, status="pass", reason="pooled and dev both pass")


def exact_matches(hypotheses: dict, references: list[str]) -> dict:
    """Per read, the share of hypotheses equal to the reference (case and spacing as BLEU tokenises), and the share of
    pairs where two reads produced the same hypothesis."""
    from signtranslator.generation.motion_to_text import words
    n = len(references)
    out = {f"{r}_equals_reference": round(sum(words(h) == words(t) for h, t in zip(hypotheses[r], references)) / n, 4)
           for r in hypotheses} if n else {}
    for a, b in COMPARISONS:
        if n and a in hypotheses and b in hypotheses:
            out[f"{a}_equals_{b}"] = round(sum(x == y for x, y in zip(hypotheses[a], hypotheses[b])) / n, 4)
    return out


def score(rows: list[dict], hypotheses: dict, *, resamples: int, seed: int = 0) -> dict:
    """BLEU, bootstrap, verdict and exact matches of translated rows (``hypotheses``: read -> list aligned to rows)."""
    references = [r["text"] for r in rows]
    stats = {read: np.array([segment_stats(h, t) for h, t in zip(hyps, references)]).reshape(-1, 2 * MAX_ORDER + 2)
             for read, hyps in hypotheses.items()}
    boot = clustered_bootstrap(stats, [r["video_id"] for r in rows], resamples=resamples, seed=seed)
    return dict(pairs=len(rows), **boot, verdict=verdict(boot), exact_match=exact_matches(hypotheses, references))


# ----------------------------------------------------------------------------------------------------------------------
# The instrument

class _PinnedStore:
    """A store view that records the SHA-256 of the one key it serves (the instrument's checkpoint)."""

    def __init__(self, store, key: str):
        self.store, self.key, self.sha256 = store, key, None

    def get(self, key: str) -> bytes:
        payload = self.store.get(key)
        if key == self.key:
            self.sha256 = hashlib.sha256(payload).hexdigest()
        return payload


def load_instrument(store, run: str, manifest_sha256: str, device: str, expected_sha256: str | None = None):
    """(translate function, checkpoint key, its SHA-256): the run's best checkpoint through ``load_translator``."""
    import torch
    from score_back_translation import load_translator, translate
    key = json.loads(store.get(f"training-runs/{run}/best.json"))["checkpoint_key"]
    pinned = _PinnedStore(store, key)
    model, kind = load_translator(pinned, key, manifest_sha256)
    if expected_sha256 and pinned.sha256 != expected_sha256:
        raise SystemExit(f"{key}: SHA-256 {pinned.sha256} differs from the pinned {expected_sha256}")
    model = model.to(torch.device(device))
    return (lambda states: list(translate(model, kind, states))), key, pinned.sha256


def read_all(rows: list[dict], translate_fn) -> dict:
    """read -> hypotheses: lifted, ours and blank (ours blanked) windows, each at READ_FPS, as the pilots read."""
    from evaluate_text_to_motion import at_rate, blanked
    ours = [r["ours"] for r in rows]
    states = dict(lifted=[r["lifted"] for r in rows], ours=ours, blank=blanked(ours))
    return {read: translate_fn([at_rate(s, READ_FPS) for s in states[read]]) for read in READS}


# ----------------------------------------------------------------------------------------------------------------------

def main(argv=None, *, translate_fn=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--gate-report", required=True, help="local path, store key or KEY@SHA256 of the gate report")
    p.add_argument("--uwb-prefix", help="default: the gate report's")
    p.add_argument("--youtube-prefix", help="default: the gate report's")
    p.add_argument("--regressors", default=BUNDLE_KEY)
    p.add_argument("--regressors-sha256", default=BUNDLE_SHA256)
    p.add_argument("--face", choices=("blendshapes", "none"), default="blendshapes", help="the converter's default")
    p.add_argument("--head", choices=("none", "affine"), default="none", help="the converter's default")
    p.add_argument("--models-dir", default=os.path.expanduser("~/.sign-translator-ops/models"))
    p.add_argument("--instrument-run", default=INSTRUMENT_RUN)
    p.add_argument("--instrument-manifest-sha256", default=INSTRUMENT_MANIFEST)
    p.add_argument("--instrument-sha256", help="pin the checkpoint's SHA-256 (recorded either way)")
    p.add_argument("--pairs", type=int, default=2000, help="at most this many pairs (0: all)")
    p.add_argument("--dev-pairs", type=int, default=0, help="at most this many UWB dev-partition pairs (0: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--workers", type=int, default=0, help="0: from sched_getaffinity and cgroup memory")
    p.add_argument("--out", required=True, help="local JSON report")
    p.add_argument("--output-key", help=f"also store the report here (refused if it exists), e.g. {OUTPUT_KEY}")
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists; a rerun writes elsewhere")
    if args.device == "cuda" and translate_fn is None:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda: no CUDA device")
    payload, gate_sha = read_spec(store, args.gate_report)
    report = json.loads(payload)
    gs = report["settings"]
    if report.get("regressors_sha256") != args.regressors_sha256:
        raise SystemExit(f"the gate ran bundle {report.get('regressors_sha256')}, not {args.regressors_sha256}")
    uwb_prefix, youtube_prefix = args.uwb_prefix or gs["uwb_prefix"], args.youtube_prefix or gs["youtube_prefix"]
    from convert_uwb_clips import translations
    partitions = {clip: part for clip, (_, part) in translations(store, uwb_prefix).items()}
    evaluated = [p for p in report["pairs"] if p.get("status") == "evaluated"]
    sample = sample_pairs(evaluated, args.pairs, args.seed)
    dev_sample = sample_pairs([p for p in evaluated if partitions.get(p["clip_id"]) == DEV_PARTITION],
                              args.dev_pairs, args.seed)
    pooled_ids, dev_ids = {p["clip_id"] for p in sample}, {p["clip_id"] for p in dev_sample}
    union = sorted({p["clip_id"]: p for p in sample + dev_sample}.values(), key=lambda p: (p["video_id"], p["clip_id"]))
    print(json.dumps(dict(evaluated=len(evaluated), sampled=len(sample), dev_sampled=len(dev_sample), union=len(union),
                          videos=len({p["video_id"] for p in sample}))), flush=True)
    uwb = json.loads(store.get(f"{uwb_prefix}/{gs['translations']}"))
    jobs, refused = [], []
    for video in sorted({p["video_id"] for p in union}):
        ours_report = json.loads(store.get(f"{youtube_prefix}/videos/{video}.json"))
        for pair in (p for p in union if p["video_id"] == video):
            try:
                jobs.append(reconstruct(pair, ours_report, uwb.get(video, {}), report["fps"]["videos"].get(video)))
            except Mismatch as error:
                refused.append(dict(clip_id=pair["clip_id"], stage="reconstruct", reason=str(error)[:300]))
    del uwb
    settings = dict(store=args.store, uwb_prefix=uwb_prefix, regressors=args.regressors,
                    regressors_sha256=args.regressors_sha256, visibility=gs["visibility"], face=args.face,
                    head=args.head, models_dir=args.models_dir)
    fps_table = {j["video_id"]: j["fps"] for j in jobs}
    workers = args.workers or worker_count()
    started, lifted = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init, initargs=(settings, fps_table)) as pool:
        for result in pool.map(_lift, jobs, chunksize=1):
            lifted.append(result)
            if len(lifted) % 250 == 0 or len(lifted) == len(jobs):
                print(f"lifted {len(lifted)}/{len(jobs)} ({time.monotonic() - started:.0f} s)", flush=True)
    rows = []
    for job, result in zip(jobs, lifted):
        if result["status"] != "ok":
            refused.append(dict(clip_id=job["clip_id"], stage="lift", reason=result["reason"]))
            continue
        rows.append(dict(job, **{k: v for k, v in result.items() if k != "status"},
                         uwb_partition=partitions.get(job["clip_id"])))
    if not rows:
        raise SystemExit("no pair could be reconstructed and lifted")
    if translate_fn is None:
        translate_fn, instrument_key, instrument_sha = load_instrument(
            store, args.instrument_run, args.instrument_manifest_sha256, args.device, args.instrument_sha256)
    else:
        instrument_key = instrument_sha = "injected"
    hypotheses = read_all(rows, translate_fn)

    def subset(keep: list[int]) -> dict:
        sub = score([rows[k] for k in keep], {read: [h[k] for k in keep] for read, h in hypotheses.items()},
                    resamples=args.bootstrap, seed=args.seed)
        return dict(sub, videos_scored=len({rows[k]["video_id"] for k in keep}))
    pooled = subset([k for k, r in enumerate(rows) if r["clip_id"] in pooled_ids])
    dev_keep = [k for k, r in enumerate(rows) if r["clip_id"] in dev_ids]
    dev = subset(dev_keep) if dev_keep else None
    by_partition = {}
    pooled_rows = [k for k, r in enumerate(rows) if r["clip_id"] in pooled_ids]
    for part in sorted({str(rows[k]["uwb_partition"]) for k in pooled_rows}):
        sub = subset([k for k in pooled_rows if str(rows[k]["uwb_partition"]) == part])
        by_partition[part] = {k: sub[k] for k in ("pairs", "reads", "differences", "verdict")}
    verdict_all = combined_verdict(pooled["verdict"], dev and dev["verdict"], dev_pairs=len(dev_keep),
                                   dev_videos=len({rows[k]["video_id"] for k in dev_keep}), refused=len(refused),
                                   sampled=len(union))
    out = dict(
        row="Downstream: back-translator read of lifted states >= 90% of its read of our states on paired cues, and "
            "clearly above blank",
        gate_report=args.gate_report, gate_report_sha256=gate_sha, regressors_sha256=args.regressors_sha256,
        visibility=gs["visibility"], bootstrap=args.bootstrap, seed=args.seed,
        instrument=dict(run=args.instrument_run, checkpoint_key=instrument_key, checkpoint_sha256=instrument_sha,
                        read_fps=READ_FPS),
        converter=dict(function="convert_uwb_clips.convert_member", world="learned", flags="generator", face=args.face,
                       head=args.head, fps_source="estimated (the gate's per-video fps)"),
        sample=dict(rule=f"pooled: all evaluated pairs if <= {args.pairs}, else one per video first (seed {args.seed}); "
                         f"dev: every evaluated pair of UWB's dev partition ({DEV_PARTITION}) "
                         f"{'(at most ' + str(args.dev_pairs) + ')' if args.dev_pairs else ''}",
                    sampled=len(sample), dev_sampled=len(dev_sample), union=len(union), reconstructed=len(jobs),
                    scored=len(rows), refused=len(refused)),
        refused=refused, **{k: v for k, v in pooled.items() if k not in ("verdict", "seed")}, verdict_pooled=pooled["verdict"],
        dev=dev and {k: dev[k] for k in ("pairs", "videos_scored", "reads", "differences", "verdict", "exact_match")},
        verdict=verdict_all, by_uwb_partition=by_partition,
        bleu="signtranslator.generation.motion_to_text.corpus_bleu (the repo's helper; summed segment statistics in "
             "the bootstrap); values unrounded",
        per_pair=[dict(video_id=r["video_id"], clip_id=r["clip_id"], sample_id=r["sample_id"], text=r["text"],
                       lag=r["lag"], aligned_frames=r["aligned_frames"], seconds=r["seconds"],
                       uwb_partition=r["uwb_partition"], lifted_state_sha256=r["lifted_state_sha256"],
                       pooled=r["clip_id"] in pooled_ids, dev=r["clip_id"] in dev_ids,
                       **{read: hypotheses[read][k] for read in READS}) for k, r in enumerate(rows)])
    text = json.dumps(out, indent=1, sort_keys=True)
    Path(args.out).write_text(text)
    if args.output_key:
        store.put(args.output_key, text.encode())
    print(json.dumps({k: out[k] for k in ("sample", "reads", "differences", "verdict_pooled", "verdict", "exact_match")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
