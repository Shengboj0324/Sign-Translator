"""Convert Sem-Lex Holistic pose files to canonical states (CPU pods, sharded; adapter ``semlex_pose``).

Streams one split's ``.tar.gz`` from the object store member by member (sequential ranged reads through a buffered
reader; nothing but the buffer and the current member is held), converts every member of this shard with
:func:`signtranslator.data_engineering.semlex_pose.convert_array`, and writes each state **write-once** to
``<prefix>/<split>/<video_id>.npz`` (a conditional PUT; an existing key is never replaced). Every clip is converted,
also when its key exists: the stored object is ``already_present`` only if it is this run's state (the same bytes, or
the same content to 1e-6 when float noise across machines changes the bytes), and ``exists_different`` otherwise. A
write whose re-read differs is ``failed`` (``StoreCorruption``, never retried as transient). Shards are
``crc32(video_id) % shards``, so
every shard reads the whole archive but converts only its own members. The archive must be the verified acquisition
(``--archive-sha256`` equal to its provenance record, re-read in R2) and the metadata CSV must have the pinned hash.

One JSON report per run: ``<prefix>/reports/<split>-shard-XXX-of-YYY-<UTC time>.json`` with status counts, per-clip
results (state key and hash, source hash, frames, rate, aspect estimate, tracked fractions, refusal reasons), and the
shard's completeness against the metadata (ids of this split and shard that the archive lacks, split into the pinned
known gaps of ``semlex_known_missing.json`` and any others).

**Archives** (``semlex_archive_inventory.py`` -> ``semlex_known_missing.json``, 2026-10-28): only ``test.tar.gz``
holds Holistic ``.npy`` pose files (15,559; 101 metadata ids of the test split have none: the pinned known gaps, and
no member lacks a metadata row). ``val.tar.gz`` and ``train.tar.gz`` hold ``.webm`` videos; the converter refuses
those splits at start (they need a video path, not this pose-file adapter).

**Metadata.** The pinned CSV has 91,148 rows but 88,174 video ids: 2,974 extra rows repeat 2,854 ids (the "2,974" of
the review is this count of repeated rows, not archive members without a row). 1,540 of them
repeat the row exactly; 1,314 differ (995 are listed in both train and val, others differ in label or label type). The
converter needs only the clip's duration: it uses the rows of the archive's split and refuses a clip whose rows there
disagree on duration. Labels and partitions of repeated ids are the manifest builder's business. Members with no row
of their archive's split (none, or only another split's) are refused and counted; that is expected, not a failure.

**Exit codes** (the first that applies): 3 when any clip ``failed`` (an unexpected error or a corrupt write, not
retried) or a key holds another state (``exists_different``); 2 when any clip ended in ``error`` (a store or I/O
failure, retried in-run: run the shard again, it resumes); 4 when metadata ids of this shard are missing from the
archive beyond the pinned known gaps; 0 otherwise (known gaps and refusals are reported, not failures).

**Path.** By default the pinned learned path: regressed world points and visibility from the pinned
``world_regressor`` bundle (``WORLD_REGRESSORS``), solved under the generator convention (upright chest and upright
root, no elbow gate), states under the bundle-named ``states-<adapter>-<sha12>/``. Another bundle needs an explicit
``--prefix``. ``--pseudo-world`` selects the geometric pseudo-world instead (the inside-image visibility rule, the same
convention, ``states-<pseudo adapter>/``). Smoke runs (``--limit`` or ``--match``) write under
``adapter-dev/smoke/`` and may not use a production prefix. A ``--prefix`` under ``states-`` must be the exact
production name of the pinned bundle (or of the pseudo-world adapter): no other bundle may write there.

Data files are untrusted: member names must match ``<split>/<id>.npy``, sizes are capped, ``.npy`` is read with
``allow_pickle=False``.
"""
from __future__ import annotations

import argparse
import contextlib
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
KNOWN_MISSING = Path(__file__).resolve().parent / "semlex_known_missing.json"
_WORKER: dict = {}


class StoreCorruption(Exception):
    """A written object does not read back as written: not transient, never retried."""


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


def same_state(stored: bytes, payload: bytes, atol: float = 1e-6) -> str | None:
    """``"bytes"`` when equal, ``"content"`` when both decode to the same state (masks, labels and metadata exactly,
    values to ``atol``), None otherwise (including a stored object that does not decode)."""
    if hashlib.sha256(stored).hexdigest() == hashlib.sha256(payload).hexdigest():
        return "bytes"
    import numpy as np
    from signtranslator.pose.multichannel import MultichannelMotion
    states = []
    try:
        for blob in (stored, payload):
            with tempfile.TemporaryDirectory() as t:
                path = Path(t) / "s.npz"
                path.write_bytes(blob)
                states.append(MultichannelMotion.load(path, expected_sha256=hashlib.sha256(blob).hexdigest()))
    except Exception:  # noqa: BLE001 - an undecodable stored object is simply not this state
        return None
    a, b = states
    if (a.sample_id, a.clock_id) != (b.sample_id, b.clock_id) or not np.array_equal(a.timestamps, b.timestamps):
        return None
    for name, x in a.channels.items():
        y = b.channels[name]
        if (x.labels, x.coordinate_frame, x.units, x.source_id, x.source_sha256, x.convention, x.inference_method) != \
                (y.labels, y.coordinate_frame, y.units, y.source_id, y.source_sha256, y.convention, y.inference_method):
            return None
        if not (np.array_equal(x.valid, y.valid) and np.array_equal(x.observed, y.observed)
                and np.allclose(x.values, y.values, rtol=0., atol=atol)
                and np.allclose(x.confidence, y.confidence, rtol=0., atol=atol)):
            return None
    return "content"


def put_once(store, key: str, payload: bytes, *, equivalent=None) -> str:
    """Create ``key`` only if absent: ``written``; else ``already_present`` when the stored object is this payload
    (same bytes, or ``equivalent(stored, payload)`` true) and ``exists_different`` otherwise. A write that does not
    read back as written raises ``StoreCorruption``."""
    digest = hashlib.sha256(payload).hexdigest()

    def existing(stored: bytes) -> str:
        if hashlib.sha256(stored).hexdigest() == digest or (equivalent is not None and equivalent(stored, payload)):
            return "already_present"
        return "exists_different"
    if hasattr(store, "s3"):
        from botocore.exceptions import ClientError
        try:
            store._call(lambda: store.s3.put_object(Bucket=store.bucket, Key=key, Body=payload,
                                                    Metadata={"sha256": digest}, IfNoneMatch="*"))
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in ("PreconditionFailed", "412"):
                raise
            return existing(store.get(key))
        if hashlib.sha256(store.get(key)).hexdigest() != digest:
            raise StoreCorruption(f"R2 re-read mismatch for {key}")
        return "written"
    path = Path(store.root) / key
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return existing(path.read_bytes())
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise StoreCorruption(f"re-read mismatch for {key}")
    return "written"


def shard_of(video_id: str, shards: int) -> int:
    return zlib.crc32(video_id.encode()) % shards


def load_metadata(store, key: str, expected_sha256: str) -> dict[str, list[dict]]:
    """Per video id, its distinct metadata rows (the unnamed row-index column dropped). The pinned CSV repeats 2,854
    ids; repeated identical rows collapse to one, differing ones are all kept for the caller to judge."""
    payload = store.get(key)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise SystemExit("the metadata CSV is not the pinned acquisition")
    rows: dict[str, list[dict]] = {}
    for row in csv.DictReader(io.StringIO(payload.decode("utf-8"))):
        row = {k: v for k, v in row.items() if k}
        kept = rows.setdefault(row["video_id"], [])
        if row not in kept:
            kept.append(row)
    return rows


def split_duration(rows: list[dict], split: str) -> tuple[float | None, str | None]:
    """(duration in ms, None) from the rows of ``split``, or (None, refusal reason)."""
    here = [r for r in rows if r["split"] == split]
    if not here:
        return None, "not in the metadata" if not rows else f"metadata split {rows[0]['split']}"
    durations = {r["duration"] for r in here}
    if len(durations) > 1:
        return None, f"conflicting metadata durations {sorted(durations)}"
    return float(here[0]["duration"]), None


def archive_format(split: str, archive_sha256: str) -> str | None:
    """The pinned member format of a split's archive (``npy``, ``webm``, ...), or None when not pinned for it."""
    if not KNOWN_MISSING.exists():
        return None
    pinned = json.loads(KNOWN_MISSING.read_text())["splits"].get(split)
    return pinned.get("format") if pinned and pinned["archive_sha256"] == archive_sha256 else None


def known_missing(split: str, archive_sha256: str) -> set[str]:
    """The pinned known gaps of a split (metadata ids the verified archive lacks), when the archive matches."""
    if not KNOWN_MISSING.exists():
        return set()
    pinned = json.loads(KNOWN_MISSING.read_text())["splits"].get(split)
    if not pinned or pinned["archive_sha256"] != archive_sha256:
        return set()
    return set(pinned.get("missing") or ())


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
        outcome = put_once(store, key, path.read_bytes(), equivalent=same_state)
    status = {"written": "converted"}.get(outcome, outcome)
    extra = {}
    if outcome != "written":
        stored = hashlib.sha256(store.get(key)).hexdigest()
        extra = dict(stored_sha256=stored, this_run_sha256=digest)
        if outcome == "already_present":
            digest = stored                               # the record points at the object that is there
    return dict(base, status=status, state_key=key, state_sha256=digest if outcome != "exists_different" else None,
                **extra, frames=report["frames"], fps=report["fps"],
                aspect=report["aspect"], tracked=tracked(state, report), hand_assignment=report["hand_assignment"],
                face_without_crop=report["face_without_crop"], world=report["world"], flags=report["flags"],
                adapter=report["adapter"], regressors_sha256=report.get("regressors_sha256"),
                available_channels=sorted(n for n, c in state.channels.items() if c.valid.any()))


def transient(error: BaseException) -> bool:
    """A store or I/O failure that another attempt may not meet (botocore, connection, timeout, OS-level I/O).
    Data errors never reach here as such: unreadable pose files are ``Refusal`` in ``load_pose_file``, and a corrupt
    write is ``StoreCorruption``, which is not transient."""
    if isinstance(error, StoreCorruption):
        return False
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


class ArchiveError(Exception):
    """The archive was not read to its end as a well-formed tar: never a silent short read."""


class _Counted(io.RawIOBase):
    """A read-only view of the decompressed stream that counts its bytes."""

    def __init__(self, raw):
        self.raw, self.bytes = raw, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.raw.read(len(buffer))
        buffer[:len(data)] = data
        self.bytes += len(data)
        return len(data)


class _ReadThrough(tarfile.TarFile):
    """A stream reader that records every block ``ignore_zeros`` makes it skip."""

    def _dbg(self, level, msg):
        self.__dict__.setdefault("skipped", []).append(str(msg))
        super()._dbg(level, msg)


@contextlib.contextmanager
def read_tar(stream, bufsize: int = 1 << 20):
    """A sequential tar reader over a ``.tar.gz`` stream that does not stop short silently. ``tarfile``'s ``r|gz``
    stops at the first end-of-archive marker (two tars joined), at the end of the first gzip member, and at a corrupt
    header, all without an error. Here ``gzip.GzipFile`` decodes every gzip member and checks each CRC, ``ignore_zeros``
    reads past end-of-archive blocks, and :func:`check_read_through` (after the last member) refuses anything but
    zero blocks skipped, the decompressed stream read to its end and a block-aligned length."""
    import gzip
    counted = _Counted(gzip.GzipFile(fileobj=stream, mode="rb"))
    with _ReadThrough.open(fileobj=counted, mode="r|", ignore_zeros=True, bufsize=bufsize) as archive:
        archive.counted = counted
        yield archive


def check_read_through(archive) -> dict:
    """After the last member of :func:`read_tar`: raise ``ArchiveError`` unless the read reached the end."""
    rest = archive.counted.read(1 << 16)
    skipped = archive.__dict__.get("skipped", [])
    invalid = [m for m in skipped if not m.endswith("end of file header")]
    if rest or invalid or archive.counted.bytes % tarfile.BLOCKSIZE:
        raise ArchiveError(f"the archive was not read through: {len(rest)} bytes left, invalid blocks {invalid[:5]}, "
                           f"{archive.counted.bytes} decompressed bytes")
    return dict(decompressed_bytes=archive.counted.bytes, zero_blocks=len(skipped))


def members(stream, split: str, unexpected: list):
    """(video_id, payload) of the archive's ``<split>/<id>.npy`` members, in archive order (read to the end: an archive
    that stops short raises ``ArchiveError``)."""
    with read_tar(stream) as archive:
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
        check_read_through(archive)


def run(args, *, resources=None, blendshape_model=None, regressors=None) -> dict:
    store = make_store(args.store)
    archive_key = args.archive_key or f"{BASE}/data/{args.split}.tar.gz"
    provenance = json.loads(store.get(archive_key + ".provenance.json"))
    if provenance.get("sha256") != args.archive_sha256 or not provenance.get("verified_by_r2_reread"):
        raise SystemExit("the archive is not the verified acquisition named by --archive-sha256")
    pinned_format = archive_format(args.split, args.archive_sha256)
    if pinned_format not in (None, "npy"):
        raise SystemExit(f"{args.split}.tar.gz holds {pinned_format} files (semlex_known_missing.json), not Holistic "
                         ".npy pose files: this converter cannot convert that split")
    metadata = load_metadata(store, args.metadata_key, args.metadata_sha256)
    prefix = state_prefix(args, regressors)
    expected = {v for v, rows in metadata.items() if any(r["split"] == args.split for r in rows)
                and shard_of(v, args.shards) == args.shard}
    known = known_missing(args.split, args.archive_sha256)
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
                duration, reason = split_duration(metadata.get(video_id, []), args.split)
                if duration is None:
                    results.append(dict(video_id=video_id, split=args.split, status="refused", reason=reason))
                    continue
                job = (video_id, args.split, payload, duration, prefix)
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
                  completeness=completeness(expected, seen, known, smoke=bool(args.limit or args.match)),
                  unexpected_members=unexpected, results=results)
    key = f"{prefix}/reports/{args.split}-shard-{args.shard:03d}-of-{args.shards:03d}-{started.replace(':', '')}.json"
    if put_once(store, key, json.dumps(report, sort_keys=True).encode()) != "written":
        raise SystemExit(f"report key {key} exists")
    report["report_key"] = key
    return report


def completeness(expected: set, seen: set, known: set, *, smoke: bool) -> dict:
    """Metadata ids of this shard the archive lacks, split into the pinned known gaps and unexpected ones (a smoke run
    reads part of the archive, so it judges nothing)."""
    missing = expected - seen
    out = dict(expected_in_shard=len(expected), seen_in_shard=len(seen & expected))
    if smoke:
        return dict(out, missing=None, missing_count=None, known_missing_count=None, unexpected_missing_count=None)
    return dict(out, missing=sorted(missing)[:1000], missing_count=len(missing),
                known_missing_count=len(missing & known), unexpected_missing=sorted(missing - known)[:1000],
                unexpected_missing_count=len(missing - known))


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
        if prefix.startswith(f"{BASE}/states-"):
            if smoke:
                raise SystemExit("smoke runs (--limit, --match) may not write to a production states prefix")
            if not pinned or prefix != production:
                raise SystemExit(f"a states- prefix must be the pinned path's exact production name {production}")
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


EXIT_CODES = {0: "done (refusals and pinned known gaps are reported, not failures)",
              2: "store or I/O errors remain: run the shard again (it resumes)",
              3: "a clip failed (unexpected error or corrupt write) or a key holds another state: investigate",
              4: "metadata ids missing from the archive beyond the pinned known gaps: investigate the archive"}


def exit_code(report: dict) -> int:
    counts = report["status_counts"]
    if counts.get("failed") or counts.get("exists_different"):
        return 3
    if counts.get("error"):
        return 2
    if report["completeness"].get("unexpected_missing_count"):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
