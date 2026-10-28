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
PSEUDO = f"{BASE}/states-semlex-holistic-pseudo-world-v1"
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


def run(root, digest, meta_sha, *extra, pseudo=True, regressors=None):
    args = job.parse_args(["--split", "test", "--archive-sha256", digest, "--metadata-sha256", meta_sha,
                           "--store", f"dir:{root}", "--workers", "0", *(["--pseudo-world"] if pseudo else []), *extra])
    return job.run(args, resources=fake_resources(), blendshape_model=FakeBlendshapes(), regressors=regressors)


def tiny_bundle(seed=0):
    import torch
    from signtranslator.data_engineering import world_regressor as wr
    torch.manual_seed(seed)
    pose = wr.MLP(79, 108, width=16, depth=1).eval()
    with torch.no_grad():
        pose.net[-1].bias[75:] = 6.                    # every point "visible" by the visibility head
    return wr.WorldRegressors(
        hand={s: (wr.MLP(60, 63, width=16, depth=1).eval(), np.zeros(60, np.float32), np.ones(60, np.float32))
              for s in ("Left", "Right")},
        pose=(pose, np.zeros(79, np.float32), np.ones(79, np.float32)))


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
        assert r["state_key"] == f"{PSEUDO}/test/{v}.npz"
        state = MultichannelMotion.load(path, expected_sha256=r["state_sha256"])
        assert state.sample_id == v and len(state.timestamps) == 8
        assert r["aspect"]["snapped_to"] == "4:3" and r["frames"] == 8
        assert 0 <= r["tracked"]["right_hand"] <= 1 and "body" in r["available_channels"]
        member = npy(semlex_array((640, 480), seed=IDS.index(v) + 1)[0].astype(np.float16))
        assert r["source_sha256"] == hashlib.sha256(member).hexdigest()
    stored = json.loads((tmp_path / report["report_key"]).read_text())
    assert stored["status_counts"] == report["status_counts"] and stored["archive_sha256"] == digest
    assert report["report_key"].startswith(f"{PSEUDO}/reports/test-shard-000-of-001-")
    assert stored["adapter"] == "semlex-holistic-pseudo-world-v1" and stored["flags"] == "generator"
    assert stored["regressors_sha256"] is None


def test_states_are_write_once(tmp_path):
    digest, meta_sha = build(tmp_path)
    first = run(tmp_path, digest, meta_sha)
    key = next(r["state_key"] for r in first["results"] if r["video_id"] == IDS[0])
    before = (tmp_path / key).read_bytes()
    second = run(tmp_path, digest, meta_sha)
    assert second["status_counts"] == {"already_present": 4, "refused": 2}
    assert (tmp_path / key).read_bytes() == before
    assert second["report_key"] != first["report_key"]
    other = tmp_path / f"{PSEUDO}/test/{IDS[1]}.npz"
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


def test_store_failures_are_retried_and_anything_else_is_terminal(monkeypatch):
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
    calls.clear()

    def bug(j):
        calls.append(j)
        raise KeyError("a bug, not the store")
    monkeypatch.setattr(job, "_process_once", bug)
    result = job._process(("v", "test", b"", 1., "p"))
    assert result["status"] == "failed" and len(calls) == 1               # not retried


def test_truncated_or_corrupt_pose_files_are_refusals(tmp_path):
    from signtranslator.data_engineering.semlex_pose import Refusal, load_pose_file
    whole = npy(np.zeros((4, 553, 3), np.float16))
    for payload in (whole[:60], whole[:-10], b"", b"\x93NUMPY garbage"):
        with pytest.raises(Refusal, match="unreadable"):
            load_pose_file(payload)
    truncated = whole[:-100]
    digest, meta_sha = build(tmp_path, extra_members=[("test/clipMISSINGFROMTAR00.npy", truncated)])
    report = run(tmp_path, digest, meta_sha)
    by = {r["video_id"]: r for r in report["results"]}
    assert by["clipMISSINGFROMTAR00"]["status"] == "refused" and "unreadable" in by["clipMISSINGFROMTAR00"]["reason"]


def test_exit_codes():
    ok = dict(status_counts={"converted": 3}, completeness=dict(missing_count=0))
    assert job.exit_code(ok) == 0
    assert job.exit_code(dict(ok, status_counts={"error": 1})) == 2
    assert job.exit_code(dict(ok, status_counts={"failed": 1})) == 3
    assert job.exit_code(dict(ok, status_counts={"exists_different": 1, "error": 1})) == 3
    assert job.exit_code(dict(ok, completeness=dict(missing_count=2))) == 3
    assert job.exit_code(dict(ok, completeness=dict(missing_count=None))) == 0     # a smoke run checks no completeness


def test_the_default_is_the_pinned_bundle_and_prefixes_are_bundle_named(tmp_path):
    from signtranslator.data_engineering.world_regressor import WORLD_REGRESSORS
    digest = "a" * 64
    args = job.parse_args(["--split", "test", "--archive-sha256", digest])
    assert (args.regressors_key, args.regressors_sha256) == (WORLD_REGRESSORS["key"], WORLD_REGRESSORS["sha256"])
    assert job.state_prefix(args) == f"{BASE}/states-semlex-holistic-learned-world-v2-{WORLD_REGRESSORS['sha256'][:12]}"
    smoke = job.parse_args(["--split", "test", "--archive-sha256", digest, "--limit", "5"])
    assert job.state_prefix(smoke).startswith(f"{BASE}/adapter-dev/smoke/states-semlex-holistic-learned-world-v2-")
    with pytest.raises(SystemExit, match="production"):
        job.state_prefix(job.parse_args(["--split", "test", "--archive-sha256", digest, "--limit", "5", "--prefix",
                                         job.state_prefix(args)]))
    other = job.parse_args(["--split", "test", "--archive-sha256", digest, "--regressors-key", "k",
                            "--regressors-sha256", "b" * 64])
    with pytest.raises(SystemExit, match="pinned"):
        job.state_prefix(other)
    other.prefix = f"{BASE}/adapter-dev/experiment"
    assert job.state_prefix(other) == f"{BASE}/adapter-dev/experiment"
    with pytest.raises(SystemExit, match="go together"):
        job.parse_args(["--split", "test", "--archive-sha256", digest, "--regressors-key", "k"])
    with pytest.raises(SystemExit, match="no regressor"):
        job.parse_args(["--split", "test", "--archive-sha256", digest, "--pseudo-world", "--regressors-key", "k",
                        "--regressors-sha256", "b" * 64])
    assert job.state_prefix(job.parse_args(["--split", "test", "--archive-sha256", digest, "--pseudo-world"])) == PSEUDO


def test_the_learned_path_loads_its_bundle_and_uses_the_generator_convention(tmp_path):
    digest, meta_sha = build(tmp_path)
    payload = tiny_bundle().save()
    key = f"{BASE}/adapter-dev/bundle.pt"
    (tmp_path / key).parent.mkdir(parents=True)
    (tmp_path / key).write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="SHA-256"):
        job._init(f"dir:{tmp_path}", None, key, "0" * 64, fake_resources(), FakeBlendshapes())
    job._init(f"dir:{tmp_path}", None, key, sha, fake_resources(), FakeBlendshapes())
    assert job._WORKER["learned"]["regressors"].sha256 == sha and job._WORKER["learned"]["flags"] == "generator"
    with pytest.raises(SystemExit, match="pinned"):                     # not the pinned bundle and no --prefix
        run(tmp_path, digest, meta_sha, "--regressors-key", key, "--regressors-sha256", sha, pseudo=False)
    prefix = f"{BASE}/adapter-dev/tiny"
    report = run(tmp_path, digest, meta_sha, "--regressors-key", key, "--regressors-sha256", sha, "--prefix", prefix,
                 pseudo=False)
    converted = [r for r in report["results"] if r["status"] == "converted"]
    assert len(converted) == 4 and all(r["state_key"].startswith(f"{prefix}/test/") for r in converted)
    assert all(r["world"] == f"learned:{sha}" and r["flags"] == "generator" and r["regressors_sha256"] == sha
               and r["adapter"] == "semlex-holistic-learned-world-v2" for r in converted)
    assert report["regressors_sha256"] == sha and report["adapter"] == "semlex-holistic-learned-world-v2"
    from signtranslator.pose.multichannel import MultichannelMotion
    state = MultichannelMotion.load(tmp_path / converted[0]["state_key"], expected_sha256=converted[0]["state_sha256"])
    method = state.channels["face"].inference_method
    assert sha in method and "upright root" in method and "no elbow gate" in method and "visibility head" in method
    labels = state.channels["body"].labels
    assert state.channels["body"].valid[:, labels.index("Pelvis")].all()          # the root is stored (generator)
