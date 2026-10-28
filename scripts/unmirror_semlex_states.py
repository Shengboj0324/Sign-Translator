"""Un-mirror Sem-Lex's video states: re-solve each from its stored raw landmarks flipped horizontally (CPU pods, sharded).

The declared handedness test (docs/NEXT_PHASE_PLAN.md, round 14 data, 2026-10-28 18:01 UTC) judged Sem-Lex's webcam
recordings mirrored. For every clip with a final state in ``extract_semlex_videos.py``'s reports (the manifest
builder's rule: full passes, the latest final result per clip, every shard of the split required), this job:

1. reads the clip's raw landmarks (``landmarks-video-v1/``, hashed against the report) and the MediaPipe that made
   them (``extractor_inference_method``);
2. flips them (``video_landmarks.flip_landmarks``: image x -> 1 - x, world x -> -x, pose left/right indices and hand
   slots swapped, blendshape left/right pairs swapped, face matrix reflected);
3. re-solves under the generator convention (``extract_semlex_videos.solve``: upright chest + upright root, MediaPipe
   visibility, no elbow gate), the inference method naming the flip;
4. **checks** the result against ``pose.mirror.mirror_state`` of the stored state (hashed against the report): the two
   are equal by construction (``tests/test_flip_landmarks.py``), so any channel whose validity differs or whose values
   differ by more than ``CHECK_ATOL`` makes the clip ``failed`` and nothing is written for it;
5. writes the state once to ``states-video-v1-unmirrored/<split>/<video_id>.npz`` (``put_once``; an existing equal
   state is ``already_present``, another one ``exists_different``).

The flip is a reflection of MediaPipe's estimates on the mirrored video, not a re-estimation on flipped pixels, so the
estimators' side-specific behaviour on the mirrored image is carried over (as with ``mirror_state``); the stored
landmarks make it a re-solve, not a re-extraction.

**Report** (one per run, ``<prefix>/reports/<split>-shard-XXX-of-YYY-<UTC time>.json``, write-once): per clip the
status, keys and hashes (source state, landmarks, new state), the check's largest deviation, and the forearm and hand
tracked fractions per side after the flip; status counts; the extraction reports read. Statuses: ``converted``,
``already_present``; ``error`` (store or I/O failures retried in-run; run the shard again); ``failed`` (the check, a
corrupt stored object, anything unexpected); ``exists_different``. Dead workers and stalls are handled by the
extractor's pool; a whole-process watchdog kills a run that makes no progress for ``--stall-seconds`` + 600 s
(exit 1; a rerun resumes, existing states are ``already_present``).

**Exit codes:** 3 a clip failed or a key holds another state; 2 store errors remain (rerun); 0 done; 1 the run stopped
before writing its report. Smoke runs (``--limit``) write under ``adapter-dev/smoke/`` only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from convert_semlex_poses import (  # noqa: E402
    BASE, SMOKE_BASE, StoreCorruption, make_store, put_once, same_state, shard_of, transient)

JOB = "semlex-unmirror-v1"
STATES_DIR = "states-video-v1-unmirrored"
SPLITS = ("val", "train")
CHECK_ATOL = 1e-5
UNMIRROR_METHOD = ("; un-mirrored: the stored raw landmarks flipped horizontally before the solve (image x -> 1 - x, "
                   "world x -> -x, pose left/right swapped, hand slots swapped, blendshape left/right swapped, face "
                   "matrix reflected): the recording was judged mirrored (round 14 handedness test)")
_WORKER: dict = {}


def output_prefix(smoke: bool) -> str:
    return f"{SMOKE_BASE if smoke else BASE}/{STATES_DIR}"


def mirror_check(new, stored) -> tuple[float, list[str]]:
    """(largest deviation, channels whose validity differs) between the re-solved state and ``mirror_state`` of the
    stored one, over every channel the stored state has valid (values compared where both are valid)."""
    from signtranslator.pose.mirror import mirror_state
    want = mirror_state(stored, sample_id=new.sample_id)
    worst, invalid = 0., []
    for name, channel in want.channels.items():
        got = new.channels[name]
        if not channel.valid.any() and not got.valid.any():
            continue
        if got.labels != channel.labels or got.valid.shape != channel.valid.shape or \
                not np.array_equal(got.valid, channel.valid):
            invalid.append(name)
            continue
        worst = max(worst, float(np.abs(got.values[got.valid] - channel.values[channel.valid]).max()))
    return worst, invalid


def _init(store_spec: str) -> None:
    import warnings
    warnings.simplefilter("error")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    _WORKER["store"] = make_store(store_spec)


def _load_state(payload: bytes, sha: str):
    from signtranslator.pose.multichannel import MultichannelMotion
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "s.npz"
        path.write_bytes(payload)
        return MultichannelMotion.load(path, expected_sha256=sha)


def _process_once(job) -> dict:
    index, video_id, split, source_sha256, state_key, state_sha256, landmarks_key, landmarks_sha256, prefix = job
    from extract_semlex_videos import landmarks_method, solve
    from signtranslator.data_engineering.video_landmarks import flip_landmarks, load_raw_landmarks
    store = _WORKER["store"]
    started = time.monotonic()
    base = dict(index=index, video_id=video_id, split=split, source_state_key=state_key,
                source_state_sha256=state_sha256, landmarks_key=landmarks_key, landmarks_sha256=landmarks_sha256)
    raw = store.get(landmarks_key)
    if hashlib.sha256(raw).hexdigest() != landmarks_sha256:
        raise StoreCorruption(f"{landmarks_key}: bytes differ from the extraction report")
    stored_bytes = store.get(state_key)
    if hashlib.sha256(stored_bytes).hexdigest() != state_sha256:
        raise StoreCorruption(f"{state_key}: bytes differ from the extraction report")
    try:
        clip = load_raw_landmarks(raw)
    except Exception as error:  # noqa: BLE001 - a stored object that does not decode is corrupt
        raise StoreCorruption(f"{landmarks_key} does not decode: {type(error).__name__}: {error}") from error
    method = landmarks_method(raw)
    if method is None:
        raise StoreCorruption(f"{landmarks_key} carries no extractor inference method")
    state, report = solve(flip_landmarks(clip), video_id, source_sha256, method + UNMIRROR_METHOD)
    worst, invalid = mirror_check(state, _load_state(stored_bytes, state_sha256))
    if invalid or worst > CHECK_ATOL:
        return dict(base, status="failed", reason=f"differs from mirror_state of the stored state: validity of "
                                                  f"{invalid}, largest deviation {worst:.3g}", check_max_abs=worst)
    key = f"{prefix}/{split}/{video_id}.npz"
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "state.npz"
        digest = state.save(path)
        outcome = put_once(store, key, path.read_bytes(), equivalent=same_state)
    extra = {}
    if outcome != "written":
        stored = hashlib.sha256(store.get(key)).hexdigest()
        extra = dict(stored_sha256=stored, this_run_sha256=digest)
        if outcome == "already_present":
            digest = stored
    body = state.channels["body"]
    labels = list(body.labels)

    def fraction(mask):
        return round(float(np.mean(mask)), 4)
    tracked = dict(forearm={s: fraction(body.valid[:, labels.index(f"{s}Forearm")]) for s in ("Left", "Right")},
                   hand={s: fraction(state.channels[f"{s.lower()}_hand"].valid.any(1)) for s in ("Left", "Right")})
    return dict(base, status="converted" if outcome == "written" else outcome, state_key=key,
                state_sha256=digest if outcome != "exists_different" else None, **extra, check_max_abs=worst,
                frames=int(len(state.timestamps)), tracked=tracked,
                available_channels=sorted(n for n, c in state.channels.items() if c.valid.any()),
                upright_root_applied=bool(report.get("upright_root_applied")),
                seconds=round(time.monotonic() - started, 3))


def _process(job, attempts: int = 4) -> dict:
    """One clip; store and I/O failures retried, then ``error``; anything else ``failed``."""
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


def jobs_of(store, args, prefix: str) -> tuple[list, dict]:
    """The clips of this shard with a final extraction state, as jobs, and the extraction reports read."""
    from build_semlex_manifest import FINAL_STATES, VIDEO_PREFIX, video_results
    results, read = video_results(store, VIDEO_PREFIX, (args.split,), require_all_shards=not args.limit)
    out = []
    for (video_id, split), r in sorted(results.items()):
        if r.get("status") not in FINAL_STATES or not r.get("state_sha256"):
            continue
        if shard_of(video_id, args.shards) != args.shard or (args.match and not re.search(args.match, video_id)):
            continue
        if not r.get("landmarks_key") or not r.get("landmarks_sha256"):
            raise SystemExit(f"{video_id}: the extraction report names no raw landmarks")
        out.append((len(out), video_id, split, r.get("source_sha256"), r["state_key"], r["state_sha256"],
                    r["landmarks_key"], r["landmarks_sha256"], prefix))
        if args.limit and len(out) >= args.limit:
            break
    return out, read


def run(args) -> dict:
    from extract_semlex_videos import Pool, Watchdog, summary_q
    store = make_store(args.store)
    smoke = bool(args.limit or args.match)
    prefix = output_prefix(smoke)
    jobs, read = jobs_of(store, args, prefix)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    clock = time.monotonic()
    results, pool = [], None
    if args.workers <= 0:
        _init(args.store)
        results = [_process(j) for j in jobs]
    else:
        pool = Pool(args.workers, (args.store,), args.stall_seconds, function=_process, initializer=_init)

        def fire():
            print(f"WATCHDOG: no clip finished in {watchdog.limit:.0f} s: killing the workers and exiting 1 (a rerun "
                  "resumes)", flush=True)
            pool._stop(pool.executor)
            os._exit(1)
        watchdog = Watchdog(args.stall_seconds + 600., fire)
        watchdog.start()
        try:
            for k, job in enumerate(jobs):
                pool.submit(job)
                done = pool.collect(2 * args.workers)
                if done:
                    watchdog.beat()
                results += done
                if (k + 1) % 2000 == 0:
                    print(f"{k + 1}/{len(jobs)} submitted, {len(results)} done, {time.monotonic() - clock:.0f} s",
                          flush=True)
            watchdog.limit = float("inf")          # finish() bounds each quarantined clip by the pool's stall time
            results += pool.finish()
        finally:
            pool._stop(pool.executor)
    wall = time.monotonic() - clock
    results.sort(key=lambda r: r["index"])
    statuses = sorted({r["status"] for r in results})
    checks = [r["check_max_abs"] for r in results if r.get("check_max_abs") is not None]
    report = dict(schema_version=1, job=JOB, split=args.split, shard=args.shard, shards=args.shards, smoke=smoke,
                  limit=args.limit, match=args.match, states_prefix=prefix, method_suffix=UNMIRROR_METHOD,
                  check_atol=CHECK_ATOL, started=started, finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  clips=len(results), expected=len(jobs),
                  status_counts={s: sum(r["status"] == s for r in results) for s in statuses},
                  check_max_abs=max(checks) if checks else None,
                  extraction_reports=dict(used=read["used"], ignored=read["ignored"], shards=read["shards"]),
                  timing=dict(wall_s=round(wall, 1), workers=max(args.workers, 1),
                              seconds=summary_q([r.get("seconds") for r in results])),
                  pool=None if pool is None else dict(restarts=pool.restarts, breakdowns=pool.breakdowns,
                                                      quarantined=len(pool.quarantine)),
                  results=results)
    key = f"{prefix}/reports/{args.split}-shard-{args.shard:03d}-of-{args.shards:03d}-{started.replace(':', '')}.json"
    if put_once(store, key, json.dumps(report, sort_keys=True).encode()) != "written":
        raise SystemExit(f"report key {key} exists")
    report["report_key"] = key
    return report


def exit_code(report: dict) -> int:
    counts = report["status_counts"]
    if counts.get("failed") or counts.get("exists_different"):
        return 3
    if counts.get("error") or report["clips"] != report["expected"]:
        return 2
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--store", default="r2", help="'r2' or 'dir:/path' (tests)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="0: in this process")
    p.add_argument("--limit", type=int, help="smoke: the first N clips of the shard (writes under the smoke prefix)")
    p.add_argument("--match", help="smoke: only video ids matching this regular expression")
    p.add_argument("--stall-seconds", type=float, default=900.)
    args = p.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        raise SystemExit("shard must be in [0, shards)")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    return args


def main(argv=None) -> int:
    import warnings
    warnings.simplefilter("error")
    report = run(parse_args(argv))
    print(json.dumps(dict(report_key=report["report_key"], status_counts=report["status_counts"],
                          expected=report["expected"], check_max_abs=report["check_max_abs"],
                          timing=report["timing"])))
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
