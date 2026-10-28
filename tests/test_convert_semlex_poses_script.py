"""Sem-Lex converter: streamed members, write-once states, shards, refusals and the shard report (fake store)."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import convert_semlex_poses as job  # noqa: E402
from semlex_synthetic import FakeBlendshapes, fake_resources, semlex_array  # noqa: E402

from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402

BASE = "acquisition/2026-10-17-v1/sem-lex-mirror-v1"
IDS = ["clipAAAAAAAAAAAAAAA1", "clipAAAAAAAAAAAAAAA2", "clipAAAAAAAAAAAAAAA3", "clipAAAAAAAAAAAAAAA4"]


def npy(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


class OneWay(io.RawIOBase):
    """A forward-only stream: the converter must not seek in the archive."""

    def __init__(self, payload):
        self._data, self._at = payload, 0

    def readable(self):
        return True

    def seekable(self):
        return False

    def readinto(self, buffer):
        chunk = self._data[self._at:self._at + min(len(buffer), 4096)]
        buffer[:len(chunk)] = chunk
        self._at += len(chunk)
        return len(chunk)


def build(root: Path, *, extra_members=()):
    """Store with metadata, a test archive (4 good clips, 1 malformed, 1 unknown id, odd names) and provenance."""
    rows = [dict(video_id=v, signer_id="7", duration=str(1000. * 8 / 30), split="test", label_type="asllex",
                 label="cat") for v in IDS + ["clipBADSHAPExxxxxxxx", "clipMISSINGFROMTAR00"]]
    rows.append(dict(video_id="clipVALSPLITxxxxxxxx", signer_id="8", duration="300", split="val", label_type="asllex",
                     label="dog"))
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    meta = text.getvalue().encode()
    (root / BASE / "data").mkdir(parents=True)
    (root / BASE / "data/semlex_metadata.csv").write_bytes(meta)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        def add(name, payload):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        for k, v in enumerate(IDS):
            add(f"test/{v}.npy", npy(semlex_array((640, 480), seed=k + 1)[0].astype(np.float16)))
        add("test/clipBADSHAPExxxxxxxx.npy", npy(np.zeros((8, 500, 3), np.float16)))
        add("test/clipNOTINMETADATA000.npy", npy(np.zeros((8, 553, 3), np.float16)))
        add("../escape.npy", b"x")
        add("test/sub/clip.npy", b"x")
        for name, payload in extra_members:
            add(name, payload)
    payload = archive.getvalue()
    (root / BASE / "data/test.tar.gz").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (root / BASE / "data/test.tar.gz.provenance.json").write_text(json.dumps(dict(sha256=digest,
                                                                                verified_by_r2_reread=True)))
    return digest, hashlib.sha256(meta).hexdigest()


def run(root, digest, meta_sha, *extra):
    args = job.parse_args(["--split", "test", "--archive-sha256", digest, "--metadata-sha256", meta_sha,
                           "--store", f"dir:{root}", "--workers", "0", *extra])
    return job.run(args, resources=fake_resources(), blendshape_model=FakeBlendshapes())


def test_converts_streams_and_reports(tmp_path, monkeypatch):
    digest, meta_sha = build(tmp_path)
    real_open = job.open_stream

    def one_way(store, key):
        with real_open(store, key) as stream:
            return io.BufferedReader(OneWay(stream.read()), 1024)
    monkeypatch.setattr(job, "open_stream", one_way)
    report = run(tmp_path, digest, meta_sha)
    by = {r["video_id"]: r for r in report["results"]}
    assert report["status_counts"] == {"converted": 4, "refused": 2}
    assert "shape" in by["clipBADSHAPExxxxxxxx"]["reason"] and by["clipNOTINMETADATA000"]["reason"] == "not in the metadata"
    assert {m["name"] for m in report["unexpected_members"]} == {"../escape.npy", "test/sub/clip.npy"}
    assert report["completeness"] == dict(expected_in_shard=6, seen_in_shard=5, missing=["clipMISSINGFROMTAR00"],
                                          missing_count=1)
    for v in IDS:
        r = by[v]
        path = tmp_path / r["state_key"]
        assert r["state_key"] == f"{BASE}/states-v1/test/{v}.npz"
        state = MultichannelMotion.load(path, expected_sha256=r["state_sha256"])
        assert state.sample_id == v and len(state.timestamps) == 8
        assert r["aspect"]["snapped_to"] == "4:3" and r["frames"] == 8
        assert 0 <= r["tracked"]["right_hand"] <= 1 and "body" in r["available_channels"]
        member = npy(semlex_array((640, 480), seed=IDS.index(v) + 1)[0].astype(np.float16))
        assert r["source_sha256"] == hashlib.sha256(member).hexdigest()
    stored = json.loads((tmp_path / report["report_key"]).read_text())
    assert stored["status_counts"] == report["status_counts"] and stored["archive_sha256"] == digest
    assert report["report_key"].startswith(f"{BASE}/states-v1/reports/test-shard-000-of-001-")


def test_states_are_write_once(tmp_path):
    digest, meta_sha = build(tmp_path)
    first = run(tmp_path, digest, meta_sha)
    key = next(r["state_key"] for r in first["results"] if r["video_id"] == IDS[0])
    before = (tmp_path / key).read_bytes()
    second = run(tmp_path, digest, meta_sha)
    assert second["status_counts"] == {"already_present": 4, "refused": 2}
    assert (tmp_path / key).read_bytes() == before
    assert second["report_key"] != first["report_key"]
    other = tmp_path / f"{BASE}/states-v1/test/{IDS[1]}.npz"
    other.write_bytes(b"not the state")                         # a different object already at the key
    assert job.put_once(job.make_store(f"dir:{tmp_path}"), str(other.relative_to(tmp_path)), before) == "exists_different"
    assert other.read_bytes() == b"not the state"


def test_shards_partition_the_clips(tmp_path):
    digest, meta_sha = build(tmp_path)
    seen = []
    for shard in range(3):
        report = run(tmp_path, digest, meta_sha, "--shard", str(shard), "--shards", "3")
        ids = [r["video_id"] for r in report["results"]]
        assert all(job.shard_of(v, 3) == shard for v in ids)
        seen += ids
    assert sorted(seen) == sorted(IDS + ["clipBADSHAPExxxxxxxx", "clipNOTINMETADATA000"])


def test_unverified_archive_or_metadata_is_refused(tmp_path):
    digest, meta_sha = build(tmp_path)
    with pytest.raises(SystemExit, match="verified acquisition"):
        run(tmp_path, "0" * 64, meta_sha)
    with pytest.raises(SystemExit, match="pinned acquisition"):
        run(tmp_path, digest, "1" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        job.parse_args(["--split", "test", "--archive-sha256", "abc"])
    with pytest.raises(SystemExit, match="shard"):
        job.parse_args(["--split", "test", "--archive-sha256", digest, "--shard", "2", "--shards", "2"])


def test_duplicate_and_wrong_split_members_are_refused(tmp_path):
    duplicate = npy(semlex_array((640, 480), seed=1)[0].astype(np.float16))
    digest, meta_sha = build(tmp_path, extra_members=[(f"test/{IDS[0]}.npy", duplicate),
                                                      ("test/clipVALSPLITxxxxxxxx.npy", duplicate)])
    report = run(tmp_path, digest, meta_sha)
    reasons = [r.get("reason") for r in report["results"] if r["status"] == "refused"]
    assert "duplicate member" in reasons and "metadata split val" in reasons


def test_a_transient_failure_is_retried_then_reported_as_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def flaky(j):
        calls.append(j)
        if len(calls) < 2:
            raise ConnectionError("reset")
        return dict(video_id=j[0], status="converted")
    monkeypatch.setattr(job, "_process_once", flaky)
    assert job._process(("v", "test", b"", 1., "p"))["status"] == "converted" and len(calls) == 2
    monkeypatch.setattr(job, "_process_once", lambda j: (_ for _ in ()).throw(TimeoutError("slow")))
    assert job._process(("v", "test", b"", 1., "p")) == dict(video_id="v", split="test", status="error",
                                                            reason="TimeoutError: slow")


def test_the_learned_path_loads_a_pinned_bundle_from_the_store(tmp_path):
    import torch
    from signtranslator.data_engineering import world_regressor as wr
    digest, meta_sha = build(tmp_path)
    torch.manual_seed(0)
    bundle = wr.WorldRegressors(
        hand={s: (wr.MLP(60, 63, width=16, depth=1).eval(), np.zeros(60, np.float32), np.ones(60, np.float32))
              for s in ("Left", "Right")},
        pose=(wr.MLP(79, 108, width=16, depth=1).eval(), np.zeros(79, np.float32), np.ones(79, np.float32)))
    payload = bundle.save()
    key = f"{BASE}/adapter-dev/bundle.pt"
    (tmp_path / key).parent.mkdir(parents=True)
    (tmp_path / key).write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    with pytest.raises(SystemExit, match="regressors-sha256"):
        job.parse_args(["--split", "test", "--archive-sha256", digest, "--regressors-key", key])
    with pytest.raises(ValueError, match="SHA-256"):
        job._init(f"dir:{tmp_path}", None, key, "0" * 64, fake_resources(), FakeBlendshapes())
    job._init(f"dir:{tmp_path}", None, key, sha, fake_resources(), FakeBlendshapes())
    assert job._WORKER["learned"]["regressors"].sha256 == sha and job._WORKER["learned"]["flags"] == "upright-chest"
    args = job.parse_args(["--split", "test", "--archive-sha256", digest, "--metadata-sha256", meta_sha,
                           "--store", f"dir:{tmp_path}", "--workers", "0", "--regressors-key", key,
                           "--regressors-sha256", sha])
    report = job.run(args, resources=fake_resources(), blendshape_model=FakeBlendshapes())
    converted = [r for r in report["results"] if r["status"] == "converted"]
    assert len(converted) == 4 and all(r["state_key"].startswith(f"{BASE}/states-learned-v1/test/") for r in converted)
    assert all(r["world"] == f"learned:{sha}" and r["flags"] == "upright-chest" for r in converted)
    assert report["regressors_sha256"] == sha
