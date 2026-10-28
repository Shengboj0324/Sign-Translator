"""Convert Sem-Lex Holistic pose files to canonical states (CPU pods, sharded; adapter ``semlex_pose``).

Streams one split's ``.tar.gz`` from the object store member by member (sequential ranged reads through a buffered
reader; nothing but the buffer and the current member is held), converts every member of this shard with
:func:`signtranslator.data_engineering.semlex_pose.convert_array`, and writes each state **write-once** to
``<prefix>/<split>/<video_id>.npz`` (a conditional PUT; an existing key is never replaced: identical bytes are
reported ``already_present``, different bytes ``exists_different``). Shards are ``crc32(video_id) % shards``, so
every shard reads the whole archive but converts only its own members. The archive must be the verified acquisition
(``--archive-sha256`` equal to its provenance record, re-read in R2) and the metadata CSV must have the pinned hash.

One JSON report per run: ``<prefix>/reports/<split>-shard-XXX-of-YYY-<UTC time>.json`` with status counts, per-clip
results (state key and hash, source hash, frames, rate, aspect estimate, tracked fractions, refusal reasons), and the
shard's completeness against the metadata (ids of this split and shard that the archive lacks). Exit codes: 2 when any
clip ended in ``error`` (a store or I/O failure, retried in-run; run the shard again and it resumes); 3 when any clip
``failed`` (an unexpected error, not retried), a state key holds different bytes (``exists_different``) or metadata
ids of this shard are missing from the archive.

**Path.** By default the pinned learned path: regressed world points and visibility from the pinned
``world_regressor`` bundle (``WORLD_REGRESSORS``), solved under the generator convention (upright chest and upright
root, no elbow gate), states under the bundle-named ``states-<adapter>-<sha12>/``. Another bundle needs an explicit
``--prefix``. ``--pseudo-world`` selects the geometric pseudo-world instead (the inside-image visibility rule, the same
convention, ``states-<pseudo adapter>/``). Smoke runs (``--limit`` or ``--match``) write under
``adapter-dev/smoke/`` and may not use a production prefix.

Data files are untrusted: member names must match ``<split>/<id>.npy``, sizes are capped, ``.npy`` is read with
``allow_pickle=False``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = "acquisition/2026-10-17-v1/sem-lex-mirror-v1"
METADATA_KEY = f"{BASE}/data/semlex_metadata.csv"
METADATA_SHA256 = "f4250b2877e738f028e4b9517922952a350ec8952e59d53ceabdd778938fc3c3"   # acquire_semlex_mirror.py
SMOKE_BASE = f"{BASE}/adapter-dev/smoke"
CONVENTION_FLAGS = "generator"
SPLITS = ("test", "val", "train")
MAX_MEMBER_BYTES = 16 << 20
MEMBER = re.compile(r"^(test|val|train)/([A-Za-z0-9_-]{1,64})\.npy$")
_WORKER: dict = {}


def make_store(spec: str):
    from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store
    if spec == "r2":
        return R2Store(os.environ)
    if spec.startswith("dir:"):
        return DirectoryStore(Path(spec[4:]))
    raise ValueError("store must be 'r2' or 'dir:/path'")


def open_stream(store, key: str, buffer: int = 8 << 20):
    """A sequential, buffered read-only stream of one object (ranged GETs with retries on R2)."""
    if hasattr(store, "s3"):
        from extract_video_states import StoreRangeFile
        return io.BufferedReader(StoreRangeFile(store.s3, store.bucket, key), buffer)
    return open(Path(store.root) / key, "rb")


def put_once(store, key: str, payload: bytes) -> str:
    """Create ``key`` only if absent: ``written``, ``already_present`` (same bytes) or ``exists_different``."""
    digest = hashlib.sha256(payload).hexdigest()
    if hasattr(store, "s3"):
        from botocore.exceptions import ClientError
        try:
            store._call(lambda: store.s3.put_object(Bucket=store.bucket, Key=key, Body=payload,
                                                    Metadata={"sha256": digest}, IfNoneMatch="*"))
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in ("PreconditionFailed", "412"):
                raise
            return "already_present" if hashlib.sha256(store.get(key)).hexdigest() == digest else "exists_different"
        if hashlib.sha256(store.get(key)).hexdigest() != digest:
            raise IOError(f"R2 re-read mismatch for {key}")
        return "written"
    path = Path(store.root) / key
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return "already_present" if hashlib.sha256(path.read_bytes()).hexdigest() == digest else "exists_different"
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
    return "written"


def shard_of(video_id: str, shards: int) -> int:
    return zlib.crc32(video_id.encode()) % shards


def load_metadata(store, key: str, expected_sha256: str) -> dict[str, dict]:
    payload = store.get(key)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise SystemExit("the metadata CSV is not the pinned acquisition")
    rows = {}
    for row in csv.DictReader(io.StringIO(payload.decode("utf-8"))):
        if row["video_id"] in rows:
            raise SystemExit(f"duplicate video_id {row['video_id']} in the metadata")
        rows[row["video_id"]] = row
    return rows


def _init(store_spec: str, models_dir: str | None, regressors_key: str | None = None,
          regressors_sha256: str | None = None, resources=None, blendshape_model=None, regressors=None):
    from signtranslator.data_engineering.semlex_pose import BlendshapeModel, load_face_resources
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    store = make_store(store_spec)
    resources = resources or load_face_resources(Path(models_dir) / "face_landmarker.task")
    if regressors is None and regressors_key:
        from signtranslator.data_engineering.world_regressor import WorldRegressors
        regressors = WorldRegressors.load(store.get(regressors_key), expected_sha256=regressors_sha256)
    path = dict(regressors=regressors, learned_visibility=True) if regressors is not None else {}
    _WORKER.update(store=store, resources=resources, model=blendshape_model or BlendshapeModel(resources),
                   learned=dict(path, flags=CONVENTION_FLAGS))


def tracked(state, report: dict) -> dict:
    frames = max(report["frames"], 1)
    out = dict(pose=report["pose_frames"] / frames, face=report["face_frames"] / frames,
               left_hand=report["left_hand_frames"] / frames, right_hand=report["right_hand_frames"] / frames,
               **{k: v / frames for k, v in report["arm_frames"].items()},
               **{f"{side}ElbowInImage": v / frames for side, v in report.get("elbow_in_image_frames", {}).items()})
    return {k: round(float(v), 4) for k, v in out.items()}


def _process_once(job) -> dict:
    video_id, split, payload, duration_ms, prefix = job
    from signtranslator.data_engineering.semlex_pose import Refusal, convert_array, load_pose_file
    store = _WORKER["store"]
    key = f"{prefix}/{split}/{video_id}.npz"
    source_sha256 = hashlib.sha256(payload).hexdigest()
    base = dict(video_id=video_id, split=split, source_sha256=source_sha256)
    if store.exists(key):
        digest = hashlib.sha256(store.get(key)).hexdigest()
        return dict(base, status="already_present", state_key=key, state_sha256=digest)
    try:
        array = load_pose_file(payload)
        state, report = convert_array(array, duration_ms, _WORKER["resources"], video_id=video_id,
                                      source_sha256=source_sha256, blendshape_model=_WORKER["model"],
                                      **_WORKER["learned"])
    except Refusal as error:
        return dict(base, status="refused", reason=str(error)[:300])
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "state.npz"
        digest = state.save(path)
        outcome = put_once(store, key, path.read_bytes())
    status = {"written": "converted"}.get(outcome, outcome)
    return dict(base, status=status, state_key=key, state_sha256=digest, frames=report["frames"], fps=report["fps"],
                aspect=report["aspect"], tracked=tracked(state, report), hand_assignment=report["hand_assignment"],
                face_without_crop=report["face_without_crop"], world=report["world"], flags=report["flags"],
                adapter=report["adapter"], regressors_sha256=report.get("regressors_sha256"),
                available_channels=sorted(n for n, c in state.channels.items() if c.valid.any()))


def transient(error: BaseException) -> bool:
    """A store or I/O failure that another attempt may not meet (botocore, connection, timeout, OS-level I/O).
    Data errors never reach here as such: unreadable pose files are ``Refusal`` in ``load_pose_file``."""
    try:
        from botocore.exceptions import BotoCoreError, ClientError
        store_errors = (BotoCoreError, ClientError)
    except ImportError:
        store_errors = ()
    return isinstance(error, (ConnectionError, TimeoutError, OSError, *store_errors))


def _process(job, attempts: int = 4) -> dict:
    """One clip. Store and I/O failures are retried and, if they persist, reported as ``error`` (the next run retries
    them); anything else is ``failed``, a terminal status that is not retried."""
    import random
    for attempt in range(attempts):
        try:
            return _process_once(job)
        except Exception as error:  # noqa: BLE001 - classified below; refusals were returned above
            reason = f"{type(error).__name__}: {error}"[:300]
            if not transient(error):
                return dict(video_id=job[0], split=job[1], status="failed", reason=reason)
            if attempt == attempts - 1:
                return dict(video_id=job[0], split=job[1], status="error", reason=reason)
            time.sleep(min(60., 2. * 2 ** attempt) * (.5 + random.random()))


def members(stream, split: str, unexpected: list):
    """(video_id, payload) of the archive's ``<split>/<id>.npy`` members, in archive order."""
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            match = MEMBER.match(member.name)
            if not member.isfile() or match is None or match.group(1) != split or member.size > MAX_MEMBER_BYTES:
                if len(unexpected) < 50:
                    unexpected.append(dict(name=member.name[:200], size=int(member.size), type=member.type.decode()
                                           if isinstance(member.type, bytes) else str(member.type)))
                continue
            handle = archive.extractfile(member)
            yield match.group(2), handle.read()


def run(args, *, resources=None, blendshape_model=None, regressors=None) -> dict:
    store = make_store(args.store)
    archive_key = args.archive_key or f"{BASE}/data/{args.split}.tar.gz"
    provenance = json.loads(store.get(archive_key + ".provenance.json"))
    if provenance.get("sha256") != args.archive_sha256 or not provenance.get("verified_by_r2_reread"):
        raise SystemExit("the archive is not the verified acquisition named by --archive-sha256")
    metadata = load_metadata(store, args.metadata_key, args.metadata_sha256)
    prefix = state_prefix(args, regressors)
    expected = {v for v, row in metadata.items() if row["split"] == args.split and shard_of(v, args.shards) == args.shard}
    inline = args.workers <= 0
    initargs = (args.store, args.models_dir, args.regressors_key, args.regressors_sha256, resources, blendshape_model,
                regressors)
    pool = None if inline else ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=initargs[:4])
    if inline:
        _init(*initargs)
    results, pending, seen, unexpected, total = [], [], set(), [], 0
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")      # microseconds: one report per run

    def drain(limit: int):
        while len(pending) > limit:
            results.append(pending.pop(0).result())
    try:
        with open_stream(store, archive_key) as stream:
            for video_id, payload in members(stream, args.split, unexpected):
                total += 1
                if shard_of(video_id, args.shards) != args.shard:
                    continue
                if args.match and not re.search(args.match, video_id):
                    continue
                if video_id in seen:
                    results.append(dict(video_id=video_id, split=args.split, status="refused", reason="duplicate member"))
                    continue
                seen.add(video_id)
                row = metadata.get(video_id)
                if row is None or row["split"] != args.split:
                    results.append(dict(video_id=video_id, split=args.split, status="refused",
                                        reason="not in the metadata" if row is None else f"metadata split {row['split']}"))
                    continue
                job = (video_id, args.split, payload, float(row["duration"]), prefix)
                if inline:
                    results.append(_process(job))
                else:
                    pending.append(pool.submit(_process, job))
                    drain(4 * args.workers)
                if args.limit and len(seen) >= args.limit:
                    break
                if len(seen) % 500 == 0:
                    print(f"{len(seen)} clips of this shard, {total} members read", flush=True)
            drain(0)
    finally:
        if pool is not None:
            pool.shutdown()
    statuses = sorted({r["status"] for r in results})
    from signtranslator.data_engineering.semlex_pose import ADAPTER, LEARNED_ADAPTER
    learned = not args.pseudo_world
    report = dict(schema_version=2, adapter=LEARNED_ADAPTER if learned else ADAPTER, flags=CONVENTION_FLAGS,
                  split=args.split, shard=args.shard,
                  shards=args.shards, archive_key=archive_key, archive_sha256=args.archive_sha256,
                  metadata_key=args.metadata_key, metadata_sha256=args.metadata_sha256, prefix=prefix,
                  regressors_key=args.regressors_key if learned else None,
                  regressors_sha256=(regressors.sha256 if regressors is not None else args.regressors_sha256)
                  if learned else None,
                  started=started, finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  members_read=total, clips=len(results),
                  status_counts={s: sum(r["status"] == s for r in results) for s in statuses},
                  completeness=dict(expected_in_shard=len(expected), seen_in_shard=len(seen & expected),
                                    missing=sorted(expected - seen)[:1000] if not args.limit and not args.match else None,
                                    missing_count=len(expected - seen) if not args.limit and not args.match else None),
                  unexpected_members=unexpected, results=results)
    key = f"{prefix}/reports/{args.split}-shard-{args.shard:03d}-of-{args.shards:03d}-{started.replace(':', '')}.json"
    if put_once(store, key, json.dumps(report, sort_keys=True).encode()) != "written":
        raise SystemExit(f"report key {key} exists")
    report["report_key"] = key
    return report


def production_prefix(adapter: str, sha256: str | None) -> str:
    return f"{BASE}/states-{adapter}" + (f"-{sha256[:12]}" if sha256 else "")


def state_prefix(args, regressors=None) -> str:
    """Where the states go: bundle-named for the learned path; smoke runs under adapter-dev/smoke/; a bundle other
    than the pinned one, or a production prefix for a smoke run, is refused."""
    from signtranslator.data_engineering.semlex_pose import ADAPTER, LEARNED_ADAPTER
    from signtranslator.data_engineering.world_regressor import WORLD_REGRESSORS
    if args.pseudo_world:
        sha, adapter = None, ADAPTER
    else:
        sha, adapter = (regressors.sha256 if regressors is not None else args.regressors_sha256), LEARNED_ADAPTER
    production = production_prefix(adapter, sha)
    smoke = bool(args.limit or args.match)
    pinned = args.pseudo_world or (sha == WORLD_REGRESSORS["sha256"]
                                   and (regressors is not None or args.regressors_key == WORLD_REGRESSORS["key"]))
    if args.prefix:
        prefix = args.prefix.rstrip("/")
        if smoke and prefix.startswith(f"{BASE}/states-"):
            raise SystemExit("smoke runs (--limit, --match) may not write to a production states prefix")
        return prefix
    if not pinned:
        raise SystemExit("a regressor bundle other than the pinned one needs an explicit --prefix")
    return production.replace(f"{BASE}/", f"{SMOKE_BASE}/", 1) if smoke else production


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--archive-sha256", required=True, help="the verified hash recorded at acquisition")
    parser.add_argument("--archive-key", help=f"default {BASE}/data/<split>.tar.gz")
    parser.add_argument("--metadata-key", default=METADATA_KEY)
    parser.add_argument("--metadata-sha256", default=METADATA_SHA256)
    parser.add_argument("--prefix", help="default: states-<adapter>-<sha12> (smoke runs: under adapter-dev/smoke/)")
    parser.add_argument("--regressors-key", help="world_regressor bundle in the store (default: the pinned one)")
    parser.add_argument("--regressors-sha256", help="its SHA-256 (default: the pinned one)")
    parser.add_argument("--pseudo-world", action="store_true",
                        help="the geometric pseudo-world instead of the learned regressors")
    parser.add_argument("--store", default="r2", help="'r2' or 'dir:/path' (tests)")
    parser.add_argument("--models-dir", help="holds face_landmarker.task (pinned hash)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="0: convert in this process")
    parser.add_argument("--limit", type=int, help="stop after this many clips of the shard (smoke runs)")
    parser.add_argument("--match", help="only video ids matching this regular expression")
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        raise SystemExit("shard must be in [0, shards)")
    if not re.fullmatch(r"[0-9a-f]{64}", args.archive_sha256):
        raise SystemExit("--archive-sha256 must be a SHA-256 hex digest")
    from signtranslator.data_engineering.world_regressor import WORLD_REGRESSORS
    if args.pseudo_world:
        if args.regressors_key or args.regressors_sha256:
            raise SystemExit("--pseudo-world takes no regressor bundle")
    else:
        if (args.regressors_key is None) != (args.regressors_sha256 is None):
            raise SystemExit("--regressors-key and --regressors-sha256 go together")
        if args.regressors_key is None:
            args.regressors_key, args.regressors_sha256 = WORLD_REGRESSORS["key"], WORLD_REGRESSORS["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", args.regressors_sha256):
            raise SystemExit("--regressors-sha256 must be a SHA-256 hex digest")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.models_dir is None:
        raise SystemExit("--models-dir is required")
    report = run(args)
    print(json.dumps(dict(report_key=report["report_key"], status_counts=report["status_counts"],
                          completeness={k: v for k, v in report["completeness"].items() if k != "missing"})))
    return exit_code(report)


def exit_code(report: dict) -> int:
    counts = report["status_counts"]
    if counts.get("failed") or counts.get("exists_different") or report["completeness"].get("missing_count"):
        return 3
    return 2 if counts.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
