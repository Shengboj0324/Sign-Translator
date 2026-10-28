"""Extract Sem-Lex's original ``.webm`` videos (val, train) to canonical states with our own pipeline (CPU pods, sharded).

``val.tar.gz`` and ``train.tar.gz`` of the verified mirror hold the original webcam videos (``./<split>/<id>.webm``,
``semlex_archive_inventory.py``), so these splits need no adapter: they go through exactly the extraction and solver
of our other corpora, under the generator convention (docs/NEXT_PHASE_PLAN.md, 2026-10-28 11:13 UTC). The test split
(pose files only) is the held-out evaluation set and is refused here (``convert_semlex_poses.py`` converts it).

**Pipeline.** One producer streams the split's archive from the object store member by member (sequential ranged reads
through a buffered reader; the whole archive is read, nothing is seeked). The archive is read to its very end
(``convert_semlex_poses.read_tar``: every gzip member with its CRC, past end-of-archive blocks; a short or malformed read
is an error, never a silent end). Members of this shard (``crc32(video_id) % shards``) go through a bounded queue to a
pool of worker processes; the producer waits while more than ``2 x workers`` clips are pending, so memory stays bounded
whatever the archive's order. Each worker verifies the pinned MediaPipe models and
probes the runtime once, then per clip writes the member's bytes to a temporary file and runs
``video_landmarks.extract_clip(video, models, roi=True)``.

**Write-once outputs.**

- Raw landmarks ``<landmarks prefix>/<split>/<video_id>.npz`` (``save_raw_landmarks``): every MediaPipe output, so a
  later solver change is a re-solve, not a re-extraction, plus one array ``extractor_inference_method`` naming the
  MediaPipe version and model hashes that produced them (a state solved from them, on any machine, carries that
  method).
- The state ``<states prefix>/<split>/<video_id>.npz``, solved from the **stored** landmarks (re-loaded with
  ``load_raw_landmarks``), so the stored landmarks reproduce it exactly. Generator convention:
  ``canonical_from_landmarks(..., upright_chest=True, upright_root=True)`` with MediaPipe's own visibility and **no**
  in-image elbow gate; ``inference_method`` states this (``GENERATOR_METHOD``, the same text as the YouTube arm
  recovery).

Both are conditional PUTs (``put_once``; an existing key is never replaced). A rerun resumes: a clip whose landmarks are
stored is not extracted again (MediaPipe is not bit-reproducible across machines); its state is solved from them and is
``already_present`` if it equals the stored state (same bytes, or the same content to 1e-6), ``exists_different``
otherwise, or ``resumed`` when only the landmarks had been written. Production: ``states-video-v1/`` and
``landmarks-video-v1/`` under the Sem-Lex prefix. Smoke runs (``--limit`` or ``--match``) write under
``adapter-dev/smoke/`` only.

**Status taxonomy** (per clip, as ``convert_semlex_poses.py``): ``converted``, ``resumed``, ``already_present``;
``refused`` (a data property: undecodable video, fewer than 8 frames, no person tracked, no metadata row of this split,
conflicting metadata durations, a repeated member name (the first copy is used), member over the size cap); ``error`` (store or I/O failures, retried
with backoff in-run and still failing: run the shard again); ``failed`` (anything unexpected, a corrupt write or an
undecodable stored object; not retried); ``exists_different``. A worker process that dies (memory, a crash in the
runtime) or a pool that stops making progress for ``--stall-seconds`` is replaced and its pending clips resubmitted; a
clip pending in two such breakdowns is run alone at the end, and is ``failed`` if it breaks a one-worker pool too;
more than 20 breakdowns, or 5 in a row without a finished clip, stop the run.

**Report** (one per run, ``<states prefix>/reports/<split>-shard-XXX-of-YYY-<UTC time>.json``): per clip the frames,
measured frame rate and frame size against the metadata duration, forearm, upper-arm and hand tracked fractions per side,
the elbow-in-image fraction per side, chest validity, ``upright_root_applied``, keys and hashes, timings, status and
reason; status counts, summary distributions, throughput, and completeness of the shard and of the whole split against
the pinned known gaps of ``semlex_known_missing.json``. A full run also hashes every archive byte it streamed and checks
it against the pinned SHA-256.

**Exit codes** (the first that applies): 3 when a clip ``failed`` or a key holds another state, or the streamed archive's
hash differs; 2 when a clip ended in ``error`` (run the shard again: it resumes); 4 when metadata ids of this shard are
missing from the archive beyond the pinned known gaps; 0 otherwise (refusals and known gaps are reported, not failures).
Exit 1 means the run stopped before writing its report: a refusal at start (archive, pins, metadata), an archive stream
that still failed after its retries, or a pool that broke ``Pool.FRUITLESS`` times without finishing a clip (the
environment); what was written stays, and a rerun resumes.

Archive contents are untrusted: member names must match ``<split>/<id>.webm`` (an optional ``./``; ids of at most 64
letters, digits, ``_`` or ``-``), only regular files are read, sizes are capped, member names never become paths (each
clip is written to a fixed name in a fresh temporary directory), and the archive and metadata must be the pinned,
verified acquisitions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import convert_semlex_poses as poses  # noqa: E402
from convert_semlex_poses import (  # noqa: E402
    BASE, METADATA_KEY, METADATA_SHA256, SMOKE_BASE, StoreCorruption, check_read_through, load_metadata, make_store,
    put_once, read_tar, same_state, shard_of, split_duration, transient)
from acquire_youtube_asl import ARM_RECOVERY_METHOD  # noqa: E402

SPLITS = ("val", "train")
STATES_DIR = "states-video-v1"
LANDMARKS_DIR = "landmarks-video-v1"
SOURCE_ID = "semlex_video_estimate"
EXTRACTOR = "semlex-video-v1"
GENERATOR_METHOD = ARM_RECOVERY_METHOD   # upright chest + upright root, MediaPipe visibility, no elbow gate, stored landmarks
MEMBER = re.compile(r"\A(?:\./)?(val|train)/([A-Za-z0-9_-]{1,64})\.webm\Z")   # \Z: no trailing newline
MAX_MEMBER_BYTES = 64 << 20               # the largest of the first 100 val members is 2.4 MB (clips of at most 9 s)
MIN_FRAMES = 8
VIDEO_REFUSALS = ("the file has no video stream", "no frames in the requested window",   # extract_clip's own refusals
                  "video presentation timestamps do not strictly increase", "blendshape categories changed")
LANDMARKS_METHOD = "extractor_inference_method"   # an extra array in the raw landmarks: the MediaPipe that made them
CONVENTION = dict(upright_chest=True, upright_root=True, visibility="mediapipe", elbow_in_image=False,
                  roi=True, hand_scale=.9, face_scale=.9)
_WORKER: dict = {}


class Refusal(Exception):
    """A property of the clip's data: reported as ``refused``, never retried."""


# ---------------------------------------------------------------------------------------------------------- archive

class HashingReader(io.RawIOBase):
    """A forward-only raw stream that hashes every byte read through it."""

    def __init__(self, raw):
        self.raw, self.digest, self.bytes = raw, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        n = self.raw.readinto(buffer)
        if n:
            self.digest.update(memoryview(buffer)[:n])
            self.bytes += n
        return n

    def close(self):
        try:
            self.raw.close()
        finally:
            super().close()


def open_archive(store, key: str, buffer: int = 8 << 20) -> tuple[io.BufferedReader, HashingReader]:
    """A sequential, buffered, hashing read-only stream of one object (ranged GETs with retries on R2)."""
    if hasattr(store, "s3"):
        from extract_video_states import StoreRangeFile
        raw = StoreRangeFile(store.s3, store.bucket, key)
    else:
        raw = io.FileIO(Path(store.root) / key, "rb")
    hashing = HashingReader(raw)
    return io.BufferedReader(hashing, buffer), hashing


def members(stream, split: str, unexpected: dict):
    """(index, video_id, payload or None) of the archive's ``<split>/<id>.webm`` members, in archive order; payload is
    None for a member over the size cap (its bytes are skipped, never read into memory). Other entries are counted in
    ``unexpected`` (the first 50 listed); directories are ignored. The archive is read to its end (``read_tar``: every
    gzip member, past end-of-archive blocks); a short or malformed read raises ``ArchiveError``, and a complete one
    records ``unexpected["read_through"]``."""
    index = 0
    with read_tar(stream) as archive:
        for member in archive:
            if member.isdir():
                continue
            match = MEMBER.match(member.name)
            if not member.isfile() or match is None or match.group(1) != split:
                unexpected["count"] += 1
                if len(unexpected["first"]) < 50:
                    unexpected["first"].append(dict(name=member.name[:200], size=int(member.size),
                                                    type=member.type.decode() if isinstance(member.type, bytes)
                                                    else str(member.type)))
                continue
            if member.size > MAX_MEMBER_BYTES:
                yield index, match.group(2), None
            else:
                yield index, match.group(2), archive.extractfile(member).read()
            index += 1
        unexpected["read_through"] = check_read_through(archive)


def pinned_split(split: str, archive_sha256: str) -> dict:
    """The pinned inventory entry of the split's archive; the archive must be pinned as ``.webm`` videos."""
    if not poses.KNOWN_MISSING.exists():
        raise SystemExit("semlex_known_missing.json is missing")
    pinned = json.loads(poses.KNOWN_MISSING.read_text())["splits"].get(split)
    if not pinned or pinned.get("archive_sha256") != archive_sha256:
        raise SystemExit(f"{split}.tar.gz with this SHA-256 is not pinned in semlex_known_missing.json")
    if pinned.get("format") != "webm":
        raise SystemExit(f"{split}.tar.gz is pinned as {pinned.get('format')!r} files, not .webm videos")
    return pinned


# ----------------------------------------------------------------------------------------------------------- worker

def _init(store_spec: str, models_dir: str | None, models=None):
    """Once per worker process: the store, the pinned models (hash-checked) and a runtime probe."""
    from signtranslator.data_engineering import video_landmarks as vl
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    if models is None:
        models = vl.verify_models(Path(models_dir))
        vl.probe_runtime(models)            # an environment failure stops the job; it is not a clip refusal
    _WORKER.update(store=make_store(store_spec), models=models, method=vl.inference_method() + GENERATOR_METHOD)


def extract(payload: bytes, models) -> "object":
    """MediaPipe landmarks of one video given as bytes (a fixed name in a fresh temporary directory)."""
    import av
    from signtranslator.data_engineering.video_landmarks import extract_clip
    with tempfile.TemporaryDirectory() as temporary:
        video = Path(temporary) / "clip.webm"
        video.write_bytes(payload)
        try:
            return extract_clip(video, models, roi=True)
        except (ValueError, av.error.FFmpegError) as error:
            if isinstance(error, MemoryError):                       # av.error.MemoryError: the machine, not the clip
                raise
            if not isinstance(error, av.error.FFmpegError) and not str(error).startswith(VIDEO_REFUSALS):
                raise                                                # another ValueError is a bug: failed, not refused
            raise Refusal(f"video: {type(error).__name__}: {error}"[:300]) from error


def tag_landmarks(raw: bytes, method: str) -> bytes:
    """``save_raw_landmarks`` bytes with the extractor's inference method added as one more array (``load_raw_landmarks``
    reads only its own arrays, so the format stays readable as it is)."""
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if LANDMARKS_METHOD in arrays:
        raise ValueError("raw landmarks already tagged")
    arrays[LANDMARKS_METHOD] = np.array(method)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def landmarks_method(raw: bytes) -> str | None:
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return str(archive[LANDMARKS_METHOD]) if LANDMARKS_METHOD in archive.files else None


def check_clip(clip) -> None:
    if len(clip.times) < MIN_FRAMES:
        raise Refusal(f"too few frames ({len(clip.times)} < {MIN_FRAMES})")
    if not np.asarray(clip.pose_found).any():
        raise Refusal("no person tracked on any frame")


def solve(clip, video_id: str, source_sha256: str, method: str):
    from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks
    return canonical_from_landmarks(clip, sample_id=video_id, source_id=SOURCE_ID, source_sha256=source_sha256,
                                    clock_id=f"{SOURCE_ID}:{video_id}:pts", inference_method=method,
                                    upright_chest=True, upright_root=True)


def clip_quality(clip, state, report: dict, duration_ms: float) -> dict:
    """Frames, measured rate and size against the metadata duration; tracked fractions per side; elbow in image."""
    from signtranslator.data_engineering.video_landmarks import elbows_in_image
    times = np.asarray(clip.times, dtype=np.float64)
    frames = len(times)
    steps = np.diff(times)
    span = float(times[-1] - times[0])
    median_step = float(np.median(steps)) if len(steps) else 0.
    measured_s = span + median_step                            # duration covered by the frames
    body = state.channels["body"]

    def fraction(mask) -> float:
        return round(float(np.mean(mask)), 4)
    joint = {label: body.valid[:, j] for j, label in enumerate(body.labels)}
    pose = np.asarray(clip.pose_found, bool)
    inside = elbows_in_image(clip.pose_image)
    hand = {s: state.channels[f"{s.lower()}_hand"].valid.any(axis=1) for s in ("Left", "Right")}
    return dict(
        frames=frames, fps=round((frames - 1) / span, 3) if span > 0 else None,
        fps_median=round(1. / median_step, 3) if median_step > 0 else None,
        max_gap_s=round(float(steps.max()), 4) if len(steps) else None,
        measured_s=round(measured_s, 4), metadata_duration_ms=duration_ms,
        duration_ratio=round(measured_s * 1000. / duration_ms, 4) if duration_ms > 0 else None,
        image_size=None if clip.image_size is None else list(clip.image_size),
        tracked=dict(pose=fraction(pose), face=fraction(clip.face_found),
                     forearm={s: fraction(joint[f"{s}Forearm"]) for s in ("Left", "Right")},
                     upper_arm={s: fraction(joint[f"{s}UpperArm"]) for s in ("Left", "Right")},
                     hand={s: fraction(hand[s]) for s in ("Left", "Right")},
                     elbow_in_image={s: fraction(inside[s] & pose) for s in ("Left", "Right")},
                     chest=fraction(joint["Chest"])),
        upright_root_applied=bool(report.get("upright_root_applied")),
        available_channels=sorted(n for n, c in state.channels.items() if c.valid.any()))


def _process_once(job) -> dict:
    index, video_id, split, payload, duration_ms, states, landmarks = job
    from signtranslator.data_engineering.video_landmarks import load_raw_landmarks, save_raw_landmarks
    store = _WORKER["store"]
    state_key, raw_key = f"{states}/{split}/{video_id}.npz", f"{landmarks}/{split}/{video_id}.npz"
    source_sha256 = hashlib.sha256(payload).hexdigest()
    base = dict(index=index, video_id=video_id, split=split, source_sha256=source_sha256, member_bytes=len(payload))
    started = time.monotonic()
    extract_s = None
    if store.exists(raw_key):                       # resumed: never re-extract (MediaPipe differs across machines)
        raw, raw_outcome = store.get(raw_key), "stored"
    else:
        try:
            clip = extract(payload, _WORKER["models"])
            check_clip(clip)
        except Refusal as error:
            return dict(base, status="refused", reason=str(error)[:300])
        extract_s = time.monotonic() - started
        raw = tag_landmarks(save_raw_landmarks(clip), _WORKER["method"])
        raw_outcome = put_once(store, raw_key, raw)
        if raw_outcome != "written":                # another run stored this clip's landmarks first: use those
            raw = store.get(raw_key)
    try:
        clip = load_raw_landmarks(raw)              # solve from exactly the stored landmarks (reproducible)
    except Exception as error:  # noqa: BLE001 - a stored object that does not decode is corrupt, not transient
        raise StoreCorruption(f"stored landmarks {raw_key} do not decode: {type(error).__name__}: {error}") from error
    try:
        check_clip(clip)
    except Refusal as error:
        return dict(base, status="refused", reason=f"stored landmarks: {error}"[:300], landmarks_key=raw_key)
    method = landmarks_method(raw)                  # the MediaPipe that extracted them, wherever this solve runs
    if method is None:
        raise StoreCorruption(f"stored landmarks {raw_key} carry no {LANDMARKS_METHOD}")
    state, report = solve(clip, video_id, source_sha256, method)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "state.npz"
        digest = state.save(path)
        outcome = put_once(store, state_key, path.read_bytes(), equivalent=same_state)
    extra = {}
    if outcome != "written":
        stored = hashlib.sha256(store.get(state_key)).hexdigest()
        extra = dict(stored_sha256=stored, this_run_sha256=digest)
        if outcome == "already_present":
            digest = stored                         # the record points at the object that is there
    status = outcome if outcome != "written" else ("converted" if raw_outcome == "written" else "resumed")
    return dict(base, status=status, state_key=state_key,
                state_sha256=digest if outcome != "exists_different" else None, **extra,
                landmarks_key=raw_key, landmarks_sha256=hashlib.sha256(raw).hexdigest(), landmarks_outcome=raw_outcome,
                extract_s=None if extract_s is None else round(extract_s, 3),
                seconds=round(time.monotonic() - started, 3), **clip_quality(clip, state, report, duration_ms))


def _process(job, attempts: int = 4) -> dict:
    """One clip. Store and I/O failures are retried and, if they persist, reported as ``error`` (the next run retries
    them); refusals were returned by ``_process_once``; anything else is ``failed``, not retried."""
    import random
    for attempt in range(attempts):
        try:
            return _process_once(job)
        except Exception as error:  # noqa: BLE001 - classified below
            reason = f"{type(error).__name__}: {error}"[:300]
            if not transient(error):
                return dict(index=job[0], video_id=job[1], split=job[2], status="failed", reason=reason)
            if attempt == attempts - 1:
                return dict(index=job[0], video_id=job[1], split=job[2], status="error", reason=reason)
            time.sleep(min(60., 2. * 2 ** attempt) * (.5 + random.random()))


# ---------------------------------------------------------------------------------------------------------- the pool

class Pool:
    """A process pool that survives dead workers and stalls: the clips a broken pool had finished are kept, its other
    pending clips are resubmitted to a new one; a clip pending in two breakdowns is quarantined and run alone at the end
    (``failed`` if it breaks a one-worker pool as well). More than ``max_restarts`` breakdowns, or ``FRUITLESS`` in a
    row without a finished clip, stop the run (exit 1; a rerun resumes): memory stays bounded."""

    FRUITLESS = 5          # consecutive breakdowns without a finished clip: the environment, not a clip (stop the job)

    def __init__(self, workers: int, initargs: tuple, stall_seconds: float, max_restarts: int = 20, *,
                 function=None, initializer=None):
        self.workers, self.initargs, self.stall, self.max_restarts = workers, initargs, stall_seconds, max_restarts
        self.function, self.initializer = function or _process, initializer or _init
        self.context = multiprocessing.get_context("spawn")
        self.pending: dict = {}                    # future -> job
        self.crashes: dict = {}                    # job index -> breakdowns it was pending in
        self.quarantine: list = []
        self.restarts, self.breakdowns, self.fruitless = 0, [], 0
        self.ready: list = []                      # results harvested from a broken pool
        self.executor = self._new(workers)

    def _new(self, workers: int) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=workers, mp_context=self.context, initializer=self.initializer,
                                   initargs=self.initargs)

    def submit(self, job) -> None:
        try:
            future = self.executor.submit(self.function, job)
        except BrokenProcessPool:                      # the pool broke while the producer was reading
            self._break("a worker process died")
            self.submit(job)
            return
        self.pending[future] = job

    def collect(self, limit: int) -> list:
        """Results of finished clips, waiting while more than ``limit`` are pending."""
        out = []
        while len(self.pending) > limit or self.ready:
            out += self.ready
            self.ready = []
            if len(self.pending) <= limit:
                break
            done, _ = wait(list(self.pending), timeout=self.stall, return_when=FIRST_COMPLETED)
            if not done:
                self._break(f"no clip finished in {self.stall:.0f} s")
                continue
            broken = False
            for future in done:
                try:
                    out.append(future.result())
                    del self.pending[future]
                    self.fruitless = 0
                except BrokenProcessPool:
                    broken = True
            if broken:
                self._break("a worker process died")
        return out

    def _break(self, why: str) -> None:
        jobs = []
        for future, job in self.pending.items():         # keep what finished cleanly before the pool broke
            if future.done() and not future.cancelled() and future.exception() is None:
                self.ready.append(future.result())
                self.fruitless = 0
            else:
                jobs.append(job)
        self.breakdowns.append(dict(reason=why, pending=len(jobs), at=time.strftime("%H:%M:%SZ", time.gmtime())))
        print(f"POOL BREAKDOWN ({why}): {len(jobs)} pending clips resubmitted", flush=True)
        self._stop(self.executor)
        self.fruitless += 1
        if self.fruitless >= self.FRUITLESS:
            raise SystemExit(f"the worker pool broke {self.fruitless} times in a row without finishing a clip "
                             f"(last: {why}): the environment, not a clip")
        if self.restarts >= self.max_restarts:
            raise SystemExit(f"the worker pool broke {self.restarts + 1} times (last: {why}): stopped; a rerun resumes")
        self.pending = {}
        self.restarts += 1
        self.executor = self._new(self.workers)
        for job in jobs:
            self.crashes[job[0]] = self.crashes.get(job[0], 0) + 1
            if self.crashes[job[0]] >= 2:
                self.quarantine.append(job)
            else:
                self.submit(job)

    @staticmethod
    def _stop(executor) -> None:
        """Kill the executor's workers (a stalled pool's are still running) and shut it down without waiting."""
        for process in list((getattr(executor, "_processes", None) or {}).values()):
            try:
                process.kill()
            except Exception:  # noqa: BLE001 - already gone
                pass
        executor.shutdown(wait=False, cancel_futures=True)

    def finish(self) -> list:
        out = self.collect(0)
        self.executor.shutdown()
        for job in self.quarantine:                    # each alone, in its own one-worker pool
            executor = self._new(1)
            future = executor.submit(self.function, job)
            try:
                result = future.result(timeout=self.stall)
                out.append(dict(result, quarantined=True))
            except (BrokenProcessPool, TimeoutError) as error:
                out.append(dict(index=job[0], video_id=job[1], split=job[2], status="failed", quarantined=True,
                                reason=f"the worker process died or stalled on this clip, alone too "
                                       f"({type(error).__name__})"))
            self._stop(executor)
        return out


class Watchdog:
    """Whole-process watchdog (2026-10-28: a pass hung for 20 min in a blocked archive read, where the pool's own stall
    check never runs). ``beat()`` marks progress (a member read, a clip finished, a trailing chunk hashed); when none
    happens for ``limit`` seconds, ``fire`` runs (kill the workers, exit 1: the pod loop reruns the pass, which resumes
    from the stored landmarks). The limit exceeds the pool's own stall check, so that recovery is tried first."""

    def __init__(self, limit: float, fire, clock=time.monotonic):
        self.limit, self.fire, self.clock = limit, fire, clock
        self.last = clock()

    def beat(self) -> None:
        self.last = self.clock()

    def expired(self) -> bool:
        return self.clock() - self.last > self.limit

    def start(self, every: float = 30., sleep=time.sleep) -> None:   # sleep bound at import: tests patch time.sleep
        import threading

        def loop():
            while True:
                try:
                    sleep(every)
                    if self.expired():
                        self.fire()
                        return
                except Exception:  # noqa: BLE001 - the watchdog must never raise into the interpreter
                    return
        threading.Thread(target=loop, daemon=True, name="watchdog").start()


# ------------------------------------------------------------------------------------------------------------- run

def prefixes(smoke: bool) -> tuple[str, str]:
    root = SMOKE_BASE if smoke else BASE
    return f"{root}/{STATES_DIR}", f"{root}/{LANDMARKS_DIR}"


def summary(results: list) -> dict:
    """Distributions over the clips with a state (median and 10th/90th percentiles)."""
    ok = [r for r in results if "tracked" in r]

    def q(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return dict(p10=round(float(np.percentile(values, 10)), 4), median=round(float(np.median(values)), 4),
                    p90=round(float(np.percentile(values, 90)), 4), mean=round(float(np.mean(values)), 4))
    out = dict(clips=len(ok), frames=q([r["frames"] for r in ok]), fps=q([r["fps"] for r in ok]),
               duration_ratio=q([r["duration_ratio"] for r in ok]),
               image_sizes={str(k): sum(str(r["image_size"]) == str(k) for r in ok)
                            for k in sorted({str(r["image_size"]) for r in ok})},
               upright_root_applied=sum(r["upright_root_applied"] for r in ok),
               pose=q([r["tracked"]["pose"] for r in ok]), face=q([r["tracked"]["face"] for r in ok]),
               chest=q([r["tracked"]["chest"] for r in ok]))
    for name in ("forearm", "upper_arm", "hand", "elbow_in_image"):
        out[name] = {s: q([r["tracked"][name][s] for r in ok]) for s in ("Left", "Right")}
    return out


def run(args, *, models=None) -> dict:
    store = make_store(args.store)
    archive_key = args.archive_key or f"{BASE}/data/{args.split}.tar.gz"
    provenance = json.loads(store.get(archive_key + ".provenance.json"))
    if provenance.get("sha256") != args.archive_sha256 or not provenance.get("verified_by_r2_reread"):
        raise SystemExit("the archive is not the verified acquisition named by --archive-sha256")
    pinned = pinned_split(args.split, args.archive_sha256)
    smoke = bool(args.limit or args.match)
    if not smoke and pinned.get("missing") is None:
        raise SystemExit(f"the known gaps of {args.split} are not pinned (semlex_archive_inventory.py): a full run "
                         "could not judge its completeness")
    known = set(pinned.get("missing") or ())
    metadata = load_metadata(store, args.metadata_key, args.metadata_sha256)
    states, landmarks = prefixes(smoke)
    in_split = {v for v, rows in metadata.items() if any(r["split"] == args.split for r in rows)}
    expected = {v for v in in_split if shard_of(v, args.shards) == args.shard}
    from signtranslator.data_engineering.video_landmarks import MODEL_SHA256, inference_method
    method = inference_method() + GENERATOR_METHOD
    inline = args.workers <= 0
    if inline:
        _init(args.store, args.models_dir, models)
        pool = None
    else:
        pool = Pool(args.workers, (args.store, args.models_dir), args.stall_seconds)
    results, seen, split_seen, unexpected = [], set(), set(), dict(count=0, first=[])
    total = submitted = 0

    def fire():
        print(f"WATCHDOG: no member read and no clip finished in {watchdog.limit:.0f} s: killing the workers and "
              "exiting 1 (the pass is rerun and resumes)", flush=True)
        if pool is not None:
            pool._stop(pool.executor)
        os._exit(1)
    watchdog = Watchdog(args.stall_seconds + 600., fire)
    watchdog.start()
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")      # microseconds: one report per run
    clock = time.monotonic()
    stream_sha256 = stream_bytes = trailing = None
    try:
        stream, hashing = open_archive(store, archive_key)
        with stream:
            complete = True
            for index, video_id, payload in members(stream, args.split, unexpected):
                watchdog.beat()
                total += 1
                split_seen.add(video_id)
                if shard_of(video_id, args.shards) != args.shard:
                    continue
                if args.match and not re.search(args.match, video_id):
                    continue
                if video_id in seen:
                    results.append(dict(index=index, video_id=video_id, split=args.split, status="refused",
                                        reason="duplicate member"))
                    continue
                seen.add(video_id)
                if payload is None:
                    results.append(dict(index=index, video_id=video_id, split=args.split, status="refused",
                                        reason=f"member over the size cap ({MAX_MEMBER_BYTES} bytes)"))
                    continue
                duration, reason = split_duration(metadata.get(video_id, []), args.split)
                if duration is None:
                    results.append(dict(index=index, video_id=video_id, split=args.split, status="refused",
                                        reason=reason))
                    continue
                job = (index, video_id, args.split, payload, duration, states, landmarks)
                submitted += 1
                if inline:
                    results.append(_process(job))
                else:
                    pool.submit(job)
                    results += pool.collect(2 * args.workers)
                    watchdog.beat()
                if submitted % 500 == 0:
                    print(f"{submitted} clips of this shard submitted, {total} members read, "
                          f"{len(results)} done, {time.monotonic() - clock:.0f} s", flush=True)
                if args.limit and len(seen) >= args.limit:
                    complete = False
                    break
            if complete and not smoke:
                trailing = 0
                while chunk := stream.read(1 << 20):   # anything after the last gzip member: hash every byte
                    trailing += len(chunk)
                    watchdog.beat()
                stream_sha256, stream_bytes = hashing.digest.hexdigest(), hashing.bytes
        if pool is not None:
            watchdog.limit = float("inf")          # finish() bounds each quarantined clip by the pool's own stall time
            results += pool.finish()
    finally:
        if pool is not None:
            pool._stop(pool.executor)
    wall = time.monotonic() - clock
    results.sort(key=lambda r: r["index"])
    statuses = sorted({r["status"] for r in results})
    extracted = [r for r in results if r.get("extract_s") is not None]
    workers = max(args.workers, 1)
    report = dict(
        schema_version=1, extractor=EXTRACTOR, convention=CONVENTION, inference_method=method,
        models={n: d for n, d in sorted(MODEL_SHA256.items())}, source_id=SOURCE_ID,
        split=args.split, shard=args.shard, shards=args.shards, smoke=smoke, limit=args.limit, match=args.match,
        archive_key=archive_key, archive_sha256=args.archive_sha256, metadata_key=args.metadata_key,
        metadata_sha256=args.metadata_sha256, states_prefix=states, landmarks_prefix=landmarks,
        started=started, finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        members_read=total, clips=len(results),
        status_counts={s: sum(r["status"] == s for r in results) for s in statuses},
        archive_stream=dict(bytes=stream_bytes, sha256=stream_sha256,
                            matches_pinned=None if stream_sha256 is None else stream_sha256 == args.archive_sha256,
                            read_through=unexpected.get("read_through"), trailing_bytes=trailing),
        completeness=dict(poses.completeness(expected, seen, known, smoke=smoke),
                          split=None if smoke else dict(
                              expected=len(in_split), members=len(split_seen),
                              missing_count=len(in_split - split_seen),
                              known_missing_count=len((in_split - split_seen) & known),
                              unexpected_missing=sorted(in_split - split_seen - known)[:1000],
                              unexpected_missing_count=len(in_split - split_seen - known))),
        unexpected_members=unexpected["first"], unexpected_member_count=unexpected["count"],
        timing=dict(wall_s=round(wall, 1), workers=workers, extracted=len(extracted),
                    clips_per_minute=round(len(extracted) / wall * 60., 2) if wall > 0 else None,
                    clips_per_minute_per_worker=round(len(extracted) / wall * 60. / workers, 3) if wall > 0 else None,
                    extract_s=summary_q([r["extract_s"] for r in extracted]),
                    seconds_per_video_second=summary_q([r["extract_s"] / r["measured_s"] for r in extracted
                                                        if r.get("measured_s")])),
        pool=None if pool is None else dict(restarts=pool.restarts, breakdowns=pool.breakdowns,
                                            quarantined=len(pool.quarantine)),
        summary=summary(results), results=results)
    key = f"{states}/reports/{args.split}-shard-{args.shard:03d}-of-{args.shards:03d}-{started.replace(':', '')}.json"
    if put_once(store, key, json.dumps(report, sort_keys=True).encode()) != "written":
        raise SystemExit(f"report key {key} exists")
    report["report_key"] = key
    return report


def summary_q(values) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(p10=round(float(np.percentile(values, 10)), 3), median=round(float(np.median(values)), 3),
                p90=round(float(np.percentile(values, 90)), 3), mean=round(float(np.mean(values)), 3))


EXIT_CODES = {0: "done (refusals and pinned known gaps are reported, not failures)",
              2: "store or I/O errors remain: run the shard again (it resumes)",
              3: "a clip failed (unexpected error, corrupt write or stored object), a key holds another state, or the "
                 "streamed archive's hash differs: investigate",
              4: "metadata ids missing from the archive beyond the pinned known gaps: investigate the archive"}


def exit_code(report: dict) -> int:
    if report.get("archive_stream", {}).get("matches_pinned") is False:
        return 3
    return poses.exit_code(report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--archive-sha256", required=True, help="the verified hash recorded at acquisition")
    parser.add_argument("--archive-key", help=f"default {BASE}/data/<split>.tar.gz")
    parser.add_argument("--metadata-key", default=METADATA_KEY)
    parser.add_argument("--metadata-sha256", default=METADATA_SHA256)
    parser.add_argument("--store", default="r2", help="'r2' or 'dir:/path' (tests)")
    parser.add_argument("--models-dir", help="the pinned MediaPipe Tasks models (video_landmarks.MODEL_SHA256)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="0: extract in this process")
    parser.add_argument("--limit", type=int, help="stop after this many members of the shard, refused ones included "
                                                  "(smoke runs)")
    parser.add_argument("--match", help="only video ids matching this regular expression (smoke runs)")
    parser.add_argument("--stall-seconds", type=float, default=900.,
                        help="replace the pool when no clip finishes for this long")
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        raise SystemExit("shard must be in [0, shards)")
    if not re.fullmatch(r"[0-9a-f]{64}", args.archive_sha256):
        raise SystemExit("--archive-sha256 must be a SHA-256 hex digest")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.models_dir is None:
        raise SystemExit("--models-dir is required")
    report = run(args)
    print(json.dumps(dict(report_key=report["report_key"], status_counts=report["status_counts"],
                          archive_stream=report["archive_stream"], timing=report["timing"],
                          completeness={k: v for k, v in report["completeness"].items() if k != "missing"})))
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
