"""Sem-Lex video extractor: streamed members, untrusted names, write-once landmarks and states solved from the stored
landmarks, resume, refusals, the error taxonomy, the resilient pool, and the shard report (fake store, fake tracker)."""
import csv
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import convert_semlex_poses as poses  # noqa: E402
import extract_semlex_videos as job  # noqa: E402

from signtranslator.data_engineering import video_landmarks as vl  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402

BASE = "acquisition/2026-10-17-v1/sem-lex-mirror-v1"
STATES = f"{BASE}/states-video-v1"
LANDMARKS = f"{BASE}/landmarks-video-v1"
GOOD = ["vidAAAAAAAAAAAAAAAA1", "vidAAAAAAAAAAAAAAAA2", "vidAAAAAAAAAAAAAAAA3"]
NAMES = ("_neutral", "eyeBlinkLeft", "eyeBlinkRight", *(f"bs{i}" for i in range(49)))


def fake_clip(frames: int, seed: int, *, pose=True) -> vl.ClipLandmarks:
    """A chest-up webcam clip: hips invisible, both hands near the wrists, the left elbow outside the image on the
    first half of the frames; irregular timestamps at about 30 fps."""
    rng = np.random.default_rng(seed)
    y_up = rng.normal(scale=.03, size=(frames, 33, 3))
    anchors = {0: [0., .62, .08], 11: [.19, .45, 0.], 12: [-.19, .45, 0.], 13: [.30, .22, .08], 14: [-.31, .20, .12],
               15: [.20, .30, .30], 16: [-.18, .32, .28], 23: [.11, -.10, 0.], 24: [-.11, -.10, 0.]}
    for k, v in anchors.items():
        y_up[:, k] = np.asarray(v) + .01 * np.arange(frames)[:, None] * rng.normal(size=3)
    visibility = np.ones((frames, 33))
    visibility[:, 23:] = 0.                                       # no hips: the upright root applies
    image = np.full((frames, 33, 2), .5)
    image[:, 15], image[:, 16] = [.7, .6], [.3, .6]
    image[:, 13], image[:, 14] = [.8, .9], [.2, .9]
    image[: frames // 2, 13, 1] = 1.2                             # the left elbow below the frame
    hands_world = rng.normal(scale=.03, size=(frames, 2, 21, 3))
    hands_world[:, :, 9] += [0., -.08, 0.]
    hands_image = np.zeros((frames, 2, 21, 2))
    hands_image[:, 0, 0], hands_image[:, 1, 0] = [.31, .61], [.69, .59]
    found = np.ones((frames, 2), bool)
    found[::3, 1] = False                                         # the left hand lost on every third frame
    times = np.cumsum(np.r_[0., rng.uniform(.02, .046, frames - 1)])
    matrix = np.tile(np.eye(4), (frames, 1, 1))
    pose_found = np.full(frames, pose)
    return vl.ClipLandmarks(times, y_up @ vl.Y_UP, visibility * pose, image, pose_found, hands_world, hands_image,
                            found & pose, np.tile(np.linspace(0, 1, 52), (frames, 1)), NAMES, matrix,
                            np.full(frames, pose), roi=True, image_size=(640, 480))


CALLS = []


def fake_extract_clip(video, models, *, roi=True, **kw):
    """Stands in for MediaPipe: the member bytes name the clip to return."""
    assert roi is True and Path(video).name == "clip.webm" and not kw
    payload = Path(video).read_bytes()
    CALLS.append(payload)
    kind, *rest = payload.decode().split(":")
    if kind == "corrupt":
        raise ValueError("the file has no video stream")
    if kind == "crash":
        raise RuntimeError("tracker exploded")
    frames, seed = int(rest[0]), int(rest[1])
    return fake_clip(frames, seed, pose=kind != "nopose")


def webm(kind: str, frames: int = 30, seed: int = 1) -> bytes:
    return f"{kind}:{frames}:{seed}".encode()


def build(root: Path, *, members=None, pinned_missing=("vidMISSINGFROMTAR00",), layout="one", pin_digest=None):
    """Store with metadata, a val archive and its provenance and pins. ``layout``: one tar in one gzip member, ``joined``
    (two tars concatenated, then gzipped) or ``multi`` (two gzip members); ``pin_digest`` pins another SHA-256 than
    the archive's own (a stream that differs from its pin)."""
    rows = [dict(video_id=v, signer_id="7", duration="1000.0", split="val", label_type="asllex", label="cat")
            for v in GOOD + ["vidCORRUPTxxxxxxxxxx", "vidSHORTxxxxxxxxxxxx", "vidNOPOSExxxxxxxxxxx",
                             "vidMISSINGFROMTAR00", "vidBIGxxxxxxxxxxxxxx", "vidDUPxxxxxxxxxxxxxx"]]
    rows.append(dict(video_id="vidTRAINONLYxxxxxxxx", signer_id="8", duration="500", split="train",
                     label_type="asllex", label="dog"))
    rows.append(dict(rows[0]))                                    # an exact repeat
    rows.append(dict(rows[1], split="train"))                     # listed in train as well: val's row decides
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    meta = text.getvalue().encode()
    (root / BASE / "data").mkdir(parents=True, exist_ok=True)
    (root / BASE / "data/semlex_metadata.csv").write_bytes(meta)
    if members is None:
        members = [*((f"./val/{v}.webm", webm("good", 30, k + 1)) for k, v in enumerate(GOOD)),
                   ("./val/vidCORRUPTxxxxxxxxxx.webm", b"corrupt"),
                   ("./val/vidSHORTxxxxxxxxxxxx.webm", webm("good", 5)),
                   ("./val/vidNOPOSExxxxxxxxxxx.webm", webm("nopose")),
                   ("./val/vidBIGxxxxxxxxxxxxxx.webm", b"x" * 5000),
                   ("./val/vidDUPxxxxxxxxxxxxxx.webm", webm("good", 12, 9)),
                   ("./val/vidDUPxxxxxxxxxxxxxx.webm", webm("good", 12, 9)),
                   ("./val/vidNOTINMETADATA000.webm", webm("good")),
                   ("./val/vidTRAINONLYxxxxxxxx.webm", webm("good")),
                   ("../escape.webm", b"x"), ("./val/sub/clip.webm", b"x"), ("./val/clip.npy", b"x"),
                   ("./train/vidAAAAAAAAAAAAAAAA1.webm", b"x"), ("./val/" + "a" * 65 + ".webm", b"x"),
                   ("symlink", None)]
    def tar(part):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            tar.addfile(_dir("./val"))
            for name, payload in part:
                if payload is None:
                    info = tarfile.TarInfo("./val/vidLINKxxxxxxxxxxxxx.webm")
                    info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                    tar.addfile(info)
                    continue
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        return archive.getvalue()
    import gzip
    half = len(members) // 2
    payload = {"one": lambda: gzip.compress(tar(members)),
               "joined": lambda: gzip.compress(tar(members[:half]) + tar(members[half:])),
               "multi": lambda: gzip.compress(tar(members[:half])) + gzip.compress(tar(members[half:]))}[layout]()
    (root / BASE / "data/val.tar.gz").write_bytes(payload)
    digest = pin_digest or hashlib.sha256(payload).hexdigest()
    (root / BASE / "data/val.tar.gz.provenance.json").write_text(json.dumps(dict(sha256=digest,
                                                                               verified_by_r2_reread=True)))
    pinned = root / "known.json"
    pinned.write_text(json.dumps(dict(splits=dict(val=dict(archive_sha256=digest, format="webm",
                                                           missing=None if pinned_missing is None
                                                           else list(pinned_missing))))))
    return digest, hashlib.sha256(meta).hexdigest(), pinned


def _dir(name):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    return info


@pytest.fixture
def env(tmp_path, monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(vl, "extract_clip", fake_extract_clip)
    monkeypatch.setattr(job, "MAX_MEMBER_BYTES", 4096)
    digest, meta_sha, pinned = build(tmp_path)
    monkeypatch.setattr(poses, "KNOWN_MISSING", pinned)
    return tmp_path, digest, meta_sha


def run(root, digest, meta_sha, *extra):
    args = job.parse_args(["--split", "val", "--archive-sha256", digest, "--metadata-sha256", meta_sha,
                           "--store", f"dir:{root}", "--workers", "0", *extra])
    return job.run(args, models={})


def test_extracts_streams_and_reports(env):
    root, digest, meta_sha = env
    report = run(root, digest, meta_sha)
    by = {r["video_id"]: r for r in report["results"]}
    assert report["status_counts"] == {"converted": 4, "refused": 7}
    reasons = {v: r.get("reason", "") for v, r in by.items()}
    assert "no video stream" in reasons["vidCORRUPTxxxxxxxxxx"] and "too few frames" in reasons["vidSHORTxxxxxxxxxxxx"]
    assert "no person" in reasons["vidNOPOSExxxxxxxxxxx"] and "size cap" in reasons["vidBIGxxxxxxxxxxxxxx"]
    assert reasons["vidNOTINMETADATA000"] == "not in the metadata"
    assert reasons["vidTRAINONLYxxxxxxxx"] == "metadata split train"
    assert [r["status"] for r in report["results"] if r["video_id"] == "vidDUPxxxxxxxxxxxxxx"] == ["converted", "refused"]
    assert {m["name"] for m in report["unexpected_members"]} == {
        "../escape.webm", "./val/sub/clip.webm", "./val/clip.npy", "./train/vidAAAAAAAAAAAAAAAA1.webm",
        "./val/" + "a" * 65 + ".webm", "./val/vidLINKxxxxxxxxxxxxx.webm"}
    assert report["unexpected_member_count"] == 6
    assert not any(p.name == "escape.webm" for p in root.rglob("*"))   # member names never become paths
    # the whole archive was streamed and hashed against the pinned SHA-256
    stream = report["archive_stream"]
    assert stream["bytes"] == (root / BASE / "data/val.tar.gz").stat().st_size and stream["sha256"] == digest
    assert stream["matches_pinned"] and stream["trailing_bytes"] == 0
    assert stream["read_through"]["decompressed_bytes"] % 512 == 0 and stream["read_through"]["zero_blocks"] >= 2
    completeness = report["completeness"]
    assert completeness["expected_in_shard"] == 9 and completeness["missing"] == ["vidMISSINGFROMTAR00"]
    assert completeness["known_missing_count"] == 1 and completeness["unexpected_missing_count"] == 0
    assert completeness["split"]["unexpected_missing_count"] == 0 and completeness["split"]["known_missing_count"] == 1
    assert job.exit_code(report) == 0
    for k, v in enumerate(GOOD):
        r = by[v]
        assert r["state_key"] == f"{STATES}/val/{v}.npz" and r["landmarks_key"] == f"{LANDMARKS}/val/{v}.npz"
        assert r["source_sha256"] == hashlib.sha256(webm("good", 30, k + 1)).hexdigest()
        state = MultichannelMotion.load(root / r["state_key"], expected_sha256=r["state_sha256"])
        raw = (root / r["landmarks_key"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == r["landmarks_sha256"]
        # the state is exactly the solve of the stored landmarks (reproducible from them)
        again, _ = job.solve(vl.load_raw_landmarks(raw), v, r["source_sha256"], vl.inference_method() + job.GENERATOR_METHOD)
        with_path = root / f"again-{v}.npz"
        assert again.save(with_path) == r["state_sha256"]
        assert state.sample_id == v and len(state.timestamps) == 30 and r["frames"] == 30
        body = state.channels["body"]
        assert "no in-image elbow gate" in body.inference_method and "upright root" in body.inference_method
        assert body.source_sha256 == r["source_sha256"] and body.source_id == job.SOURCE_ID
        assert r["upright_root_applied"] and r["tracked"]["chest"] == 1.
        assert r["tracked"]["elbow_in_image"] == {"Left": .5, "Right": 1.}
        assert r["tracked"]["forearm"] == {"Left": 1., "Right": 1.}             # no elbow gate: arms kept outside
        assert r["tracked"]["hand"] == {"Left": round(20 / 30, 4), "Right": 1.}
        assert r["image_size"] == [640, 480] and 21 < r["fps"] < 50 and r["metadata_duration_ms"] == 1000.
        assert r["duration_ratio"] == pytest.approx(r["measured_s"], abs=1e-3)
        assert r["status"] == "converted" and r["landmarks_outcome"] == "written" and r["extract_s"] is not None
    stored = json.loads((root / report["report_key"]).read_text())
    assert report["report_key"].startswith(f"{STATES}/reports/val-shard-000-of-001-")
    assert stored["convention"] == dict(upright_chest=True, upright_root=True, visibility="mediapipe",
                                        elbow_in_image=False, roi=True, hand_scale=.9, face_scale=.9)
    assert stored["inference_method"].endswith(job.GENERATOR_METHOD) and stored["smoke"] is False
    assert stored["summary"]["clips"] == 4 and stored["timing"]["extracted"] == 4


def test_a_rerun_resumes_from_the_stored_landmarks_and_never_overwrites(env):
    root, digest, meta_sha = env
    first = run(root, digest, meta_sha)
    CALLS.clear()
    second = run(root, digest, meta_sha)
    assert second["status_counts"] == {"already_present": 4, "refused": 7}
    assert all(p.decode().split(":")[0] != "good" or int(p.decode().split(":")[1]) < 8 for p in CALLS)   # no re-extraction
    a = {r["video_id"]: r for r in first["results"] if r["status"] == "converted"}
    for r in second["results"]:
        if r["status"] == "already_present":
            assert r["state_sha256"] == a[r["video_id"]]["state_sha256"] and r["landmarks_outcome"] == "stored"
    # an interrupted clip (landmarks stored, no state) is solved from its landmarks
    (root / f"{STATES}/val/{GOOD[0]}.npz").unlink()
    third = {r["video_id"]: r for r in run(root, digest, meta_sha)["results"]}
    assert third[GOOD[0]]["status"] == "resumed" and third[GOOD[0]]["state_sha256"] == a[GOOD[0]]["state_sha256"]
    # a key holding another state is never replaced
    other = root / f"{STATES}/val/{GOOD[1]}.npz"
    (root / f"{STATES}/val/{GOOD[2]}.npz").replace(other)
    report = run(root, digest, meta_sha)
    by = {r["video_id"]: r for r in report["results"]}
    assert by[GOOD[1]]["status"] == "exists_different" and by[GOOD[1]]["state_sha256"] is None
    assert hashlib.sha256(other.read_bytes()).hexdigest() == a[GOOD[2]]["state_sha256"]
    assert job.exit_code(report) == 3


def test_stored_landmarks_that_do_not_decode_are_failed_not_retried(env, monkeypatch):
    root, digest, meta_sha = env
    run(root, digest, meta_sha, "--match", "AAA1$")
    raw = root / f"{BASE}/adapter-dev/smoke/landmarks-video-v1/val/{GOOD[0]}.npz"
    (root / f"{BASE}/adapter-dev/smoke/states-video-v1/val/{GOOD[0]}.npz").unlink()
    raw.write_bytes(b"not an npz")
    monkeypatch.setattr("time.sleep", lambda s: pytest.fail("a corrupt object must not be retried"))
    report = run(root, digest, meta_sha, "--match", "AAA1$")
    (r,) = report["results"]
    assert r["status"] == "failed" and "StoreCorruption" in r["reason"] and job.exit_code(report) == 3


def test_store_errors_are_retried_then_error_and_tracker_bugs_are_failed(env, monkeypatch):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    root, digest, meta_sha = env
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    real = DirectoryStore.exists

    def flaky(self, key):
        if "/val/vidAAAAAAAAAAAAAAAA2" in key:
            raise ConnectionError("connection reset")
        return real(self, key)
    monkeypatch.setattr(DirectoryStore, "exists", flaky)
    report = run(root, digest, meta_sha, "--match", "AAA[12]$")
    by = {r["video_id"]: r for r in report["results"]}
    assert by[GOOD[0]]["status"] == "converted" and by[GOOD[1]]["status"] == "error" and len(sleeps) == 3
    assert job.exit_code(dict(report, completeness=dict(unexpected_missing_count=0))) == 2
    members = [("./val/vidAAAAAAAAAAAAAAAA1.webm", b"crash:30:1")]
    digest, meta_sha, pinned = build(root / "b", members=members)
    monkeypatch.setattr(poses, "KNOWN_MISSING", pinned)
    report = run(root / "b", digest, meta_sha, "--limit", "1")
    assert report["results"][0]["status"] == "failed" and "tracker exploded" in report["results"][0]["reason"]


def test_smoke_runs_are_isolated_and_stop_reading_early(env):
    root, digest, meta_sha = env
    report = run(root, digest, meta_sha, "--limit", "2")
    assert report["smoke"] and report["members_read"] == 2 and report["status_counts"] == {"converted": 2}
    assert report["archive_stream"]["sha256"] is None and report["completeness"]["missing"] is None
    assert report["completeness"]["split"] is None and job.exit_code(report) == 0
    written = [str(p.relative_to(root)) for p in root.rglob("*.npz")] + [report["report_key"]]
    assert all(k.startswith(f"{BASE}/adapter-dev/smoke/") for k in written) and len(written) == 5
    assert not (root / STATES).exists() and not (root / LANDMARKS).exists()


def test_refusals_at_start(tmp_path, monkeypatch):
    digest, meta_sha, pinned = build(tmp_path, pinned_missing=None)
    monkeypatch.setattr(poses, "KNOWN_MISSING", pinned)
    with pytest.raises(SystemExit, match="known gaps of val are not pinned"):
        run(tmp_path, digest, meta_sha)                         # a full run must be able to judge completeness
    with pytest.raises(SystemExit, match="verified acquisition"):
        run(tmp_path, "0" * 64, meta_sha, "--limit", "1")
    pinned.write_text(json.dumps(dict(splits=dict(val=dict(archive_sha256=digest, format="npy", missing=[])))))
    with pytest.raises(SystemExit, match="not .webm videos"):
        run(tmp_path, digest, meta_sha, "--limit", "1")
    pinned.write_text(json.dumps(dict(splits=dict(val=dict(archive_sha256="1" * 64, format="webm", missing=[])))))
    with pytest.raises(SystemExit, match="not pinned"):
        run(tmp_path, digest, meta_sha, "--limit", "1")
    with pytest.raises(SystemExit):
        job.parse_args(["--split", "test", "--archive-sha256", digest])     # the test split is pose files
    with pytest.raises(SystemExit, match="metadata CSV"):
        pinned.write_text(json.dumps(dict(splits=dict(val=dict(archive_sha256=digest, format="webm", missing=[])))))
        run(tmp_path, digest, "2" * 64, "--limit", "1")
    with pytest.raises(SystemExit, match="shard"):
        job.parse_args(["--split", "val", "--archive-sha256", digest, "--shard", "3", "--shards", "3"])


def test_unexpected_gaps_exit_4_and_a_wrong_stream_exits_3(tmp_path, monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(vl, "extract_clip", fake_extract_clip)
    monkeypatch.setattr(job, "MAX_MEMBER_BYTES", 4096)
    digest, meta_sha, pinned = build(tmp_path, pinned_missing=())
    monkeypatch.setattr(poses, "KNOWN_MISSING", pinned)
    report = run(tmp_path, digest, meta_sha)
    assert report["completeness"]["unexpected_missing"] == ["vidMISSINGFROMTAR00"] and job.exit_code(report) == 4
    assert job.exit_code(dict(report, archive_stream=dict(matches_pinned=False))) == 3
    assert set(job.EXIT_CODES) == {0, 2, 3, 4}


def test_shards_partition_the_clips(env):
    root, digest, meta_sha = env
    done = []
    for shard in range(3):
        report = run(root, digest, meta_sha, "--shard", str(shard), "--shards", "3")
        done += [r["video_id"] for r in report["results"] if r["status"] == "converted"]
        assert all(poses.shard_of(r["video_id"], 3) == shard for r in report["results"])
    assert sorted(done) == sorted(GOOD + ["vidDUPxxxxxxxxxxxxxx"])


def test_member_names():
    ok = ["./val/abc.webm", "val/abc_DEF-1.webm", "./train/" + "a" * 64 + ".webm"]
    bad = ["../val/a.webm", "./val/a/b.webm", "./val/a.webm.exe", "./val/" + "a" * 65 + ".webm", "/val/a.webm",
           "./val/ä.webm", "./val/.webm", "./test/a.webm", "./val/a.WEBM", "./val/a.webm\n"]
    assert all(job.MEMBER.match(n) for n in ok) and not any(job.MEMBER.match(n) for n in bad)


def test_the_declared_generator_convention():
    import acquire_youtube_asl
    assert job.GENERATOR_METHOD == acquire_youtube_asl.ARM_RECOVERY_METHOD          # one convention, one text
    for phrase in ("upright prior", "upright root", "arms by MediaPipe visibility (no in-image elbow gate)",
                   "solved from the stored raw landmarks"):
        assert phrase in job.GENERATOR_METHOD


# ------------------------------------------------------------------------------------------- the resilient pool

def _fake_init(*_):
    pass


def _fake_work(item):
    index, name = item[0], item[1]
    if name == "die":
        os._exit(7)                                   # a worker killed mid-clip (out of memory, a runtime crash)
    if name == "hang":
        time.sleep(600)
    return dict(index=index, video_id=name, split="val", status="converted", pid=os.getpid())


def test_the_pool_replaces_dead_and_stalled_workers_and_quarantines_the_culprits():
    pool = job.Pool(2, (), 6., function=_fake_work, initializer=_fake_init)
    out = []
    try:
        for k, name in enumerate(["a", "b", "die", "c"]):        # a dead worker: twice in the pool, then alone
            pool.submit((k, name, "val"))
            out += pool.collect(4)
        out += pool.collect(0)
        assert {b["reason"] for b in pool.breakdowns} == {"a worker process died"}
        for k, name in enumerate(["d", "hang", "e"], start=4):   # a stalled worker: no clip finishes for 6 s
            pool.submit((k, name, "val"))
        out += pool.finish()
    finally:
        pool.executor.shutdown(wait=False, cancel_futures=True)
    by = {r["video_id"]: r for r in out}
    assert sorted(by) == ["a", "b", "c", "d", "die", "e", "hang"] and len(out) == 7
    assert by["die"]["status"] == "failed" and by["die"]["quarantined"] and "died or stalled" in by["die"]["reason"]
    assert by["hang"]["status"] == "failed" and by["hang"]["quarantined"]
    assert all(by[n]["status"] == "converted" for n in "abcde")
    assert pool.restarts >= 2 and {b["reason"].split(" ")[0] for b in pool.breakdowns} >= {"a", "no"}


def _failing_init(*_):
    raise RuntimeError("the MediaPipe runtime cannot create landmarkers")


def test_a_pool_that_cannot_start_stops_the_job_instead_of_failing_every_clip():
    pool = job.Pool(1, (), 30., function=_fake_work, initializer=_failing_init)
    try:
        with pytest.raises(SystemExit, match="the environment, not a clip"):
            for k in range(40):                                # as the producer: submit, then collect
                pool.submit((k, f"c{k}", "val"))
                pool.collect(2)
            pool.finish()
        assert pool.fruitless == job.Pool.FRUITLESS
    finally:
        pool._stop(pool.executor)


def test_decoder_errors_are_refusals_but_memory_errors_are_not(monkeypatch):
    import av

    def raises(error):
        def extract_clip(video, models, **kw):
            raise error
        return extract_clip
    monkeypatch.setattr(vl, "extract_clip", raises(av.error.InvalidDataError(1094995529, "Invalid data")))
    with pytest.raises(job.Refusal, match="InvalidDataError"):
        job.extract(b"x", {})
    monkeypatch.setattr(vl, "extract_clip", raises(av.error.MemoryError(12, "Cannot allocate memory")))
    with pytest.raises(MemoryError):
        job.extract(b"x", {})
    assert not job.transient(MemoryError()) and job.transient(ConnectionError())


def test_the_real_video_archives_are_pinned_with_their_gaps():
    val = job.pinned_split("val", "6eca70a5761f2bfeea5d4f58b1ed34431f2d1be39a20947f01a27e4a22516b90")
    train = job.pinned_split("train", "fa1890fb79312f50eb1d5f60ef71a3fd52cb57342eb43a54c82bd814d7778547")
    assert len(val["missing"]) == 14 and len(train["missing"]) == 3557
    with pytest.raises(SystemExit, match="not .webm"):
        job.pinned_split("test", "729397ee389fe603e497393f1d38660d5331e059bfc96d29b033c0e8a7924ae5")
    fleet = (HERE.parent / "runpod/fleet/extract_semlex_videos.sh").read_text()
    assert val["archive_sha256"] in fleet and train["archive_sha256"] in fleet


@pytest.mark.parametrize("layout", ["joined", "multi"])
def test_joined_and_multi_member_archives_are_read_to_the_end(tmp_path, monkeypatch, layout):
    CALLS.clear()
    monkeypatch.setattr(vl, "extract_clip", fake_extract_clip)
    monkeypatch.setattr(job, "MAX_MEMBER_BYTES", 4096)
    digest, meta_sha, pinned = build(tmp_path, layout=layout)
    monkeypatch.setattr(poses, "KNOWN_MISSING", pinned)
    report = run(tmp_path, digest, meta_sha)
    assert report["status_counts"] == {"converted": 4, "refused": 7} and report["unexpected_member_count"] == 6
    assert report["completeness"]["unexpected_missing_count"] == 0 and report["archive_stream"]["matches_pinned"]
    assert job.exit_code(report) == 0


def test_a_short_or_malformed_archive_is_an_error_not_a_silent_end(tmp_path, monkeypatch):
    import gzip
    digest, meta_sha, pinned = build(tmp_path, members=[(f"./val/{v}.webm", webm("good", 30, 1)) for v in GOOD])
    good = (tmp_path / BASE / "data/val.tar.gz").read_bytes()
    plain = bytearray(gzip.decompress(good))
    plain[512 * 3 + 148] ^= 0x11                       # the second member's header checksum: a corrupt header
    assert len(list(job.members(io.BufferedReader(io.BytesIO(good)), "val", dict(count=0, first=[])))) == 3
    for broken in (good[:-20], gzip.compress(bytes(plain)), good + b"trailing junk"):
        with pytest.raises((poses.ArchiveError, EOFError, gzip.BadGzipFile)):
            list(job.members(io.BufferedReader(io.BytesIO(broken)), "val", dict(count=0, first=[])))


def test_a_stream_that_differs_from_its_pin_exits_3(tmp_path, monkeypatch):
    monkeypatch.setattr(vl, "extract_clip", fake_extract_clip)
    monkeypatch.setattr(job, "MAX_MEMBER_BYTES", 4096)
    digest, meta_sha, pinned = build(tmp_path, pin_digest="e" * 64)
    monkeypatch.setattr(poses, "KNOWN_MISSING", pinned)
    report = run(tmp_path, "e" * 64, meta_sha)
    assert report["archive_stream"]["matches_pinned"] is False and job.exit_code(report) == 3


def test_resumed_states_carry_the_method_of_the_mediapipe_that_made_the_landmarks(env, monkeypatch):
    root, digest, meta_sha = env
    first = {r["video_id"]: r for r in run(root, digest, meta_sha, "--match", "AAA1$")["results"]}
    raw = (root / first[GOOD[0]]["landmarks_key"]).read_bytes()
    original = job.landmarks_method(raw)
    assert original == vl.inference_method() + job.GENERATOR_METHOD
    assert len(vl.load_raw_landmarks(raw).times) == 30                       # the format stays readable as it was
    (root / first[GOOD[0]]["state_key"]).unlink()
    monkeypatch.setattr(vl, "inference_method", lambda: "monocular estimate: another mediapipe")
    again = run(root, digest, meta_sha, "--match", "AAA1$")["results"][0]
    assert again["status"] == "resumed" and again["state_sha256"] == first[GOOD[0]]["state_sha256"]
    state = MultichannelMotion.load(root / again["state_key"], expected_sha256=again["state_sha256"])
    assert state.channels["body"].inference_method == original
    with pytest.raises(ValueError):
        job.tag_landmarks(raw, "twice")


def test_only_extract_clips_own_value_errors_are_refusals(monkeypatch):
    def bug(video, models, **kw):
        raise ValueError("operands could not be broadcast together")
    monkeypatch.setattr(vl, "extract_clip", bug)
    with pytest.raises(ValueError, match="broadcast"):
        job.extract(b"x", {})

    def empty(video, models, **kw):
        raise ValueError("no frames in the requested window")
    monkeypatch.setattr(vl, "extract_clip", empty)
    with pytest.raises(job.Refusal):
        job.extract(b"x", {})


def test_more_breakdowns_than_allowed_stop_the_run():
    pool = job.Pool(1, (), 30., max_restarts=1, function=_fake_work, initializer=_fake_init)
    try:
        with pytest.raises(SystemExit, match="a rerun resumes"):
            for k, name in enumerate(["a", "die", "b", "c", "die", "d", "e", "die"]):
                pool.submit((k, name, "val"))
                pool.collect(0)
    finally:
        pool._stop(pool.executor)


MODELS = Path(os.path.expanduser("~/.sign-translator-ops/models"))


@pytest.mark.skipif(not all((MODELS / n).exists() for n in vl.MODEL_SHA256), reason="needs the pinned MediaPipe models")
def test_a_real_pool_run_end_to_end(tmp_path, monkeypatch):
    """Spawned workers with the real models: a synthetic video without a person and an undecodable member are
    refused, through the real pool, its back-pressure and its report."""
    import av
    video = tmp_path / "noise.webm"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libvpx-vp9", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        rng = np.random.default_rng(0)
        for _ in range(10):
            frame = av.VideoFrame.from_ndarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    members = [(f"./val/{v}.webm", video.read_bytes()) for v in GOOD] + [("./val/vidCORRUPTxxxxxxxxxx.webm", b"junk")]
    digest, meta_sha, pinned = build(tmp_path / "s", members=members, pinned_missing=None)
    monkeypatch.setattr(poses, "KNOWN_MISSING", pinned)
    args = job.parse_args(["--split", "val", "--archive-sha256", digest, "--metadata-sha256", meta_sha,
                           "--store", f"dir:{tmp_path / 's'}", "--workers", "2", "--models-dir", str(MODELS),
                           "--limit", "4"])
    report = job.run(args)
    by = {r["video_id"]: r for r in report["results"]}
    assert report["status_counts"] == {"refused": 4} and report["pool"]["restarts"] == 0
    assert "no person" in by[GOOD[0]]["reason"] and "video" in by["vidCORRUPTxxxxxxxxxx"]["reason"]


def test_watchdog_fires_only_after_its_limit_without_a_beat():
    import extract_semlex_videos as ex
    now = [0.]
    fired = []
    dog = ex.Watchdog(100., lambda: fired.append(1), clock=lambda: now[0])
    now[0] = 99.
    assert not dog.expired()
    dog.beat()
    now[0] = 198.
    assert not dog.expired()                     # 99 s after the beat
    now[0] = 200.5
    assert dog.expired()
    dog.limit = float("inf")
    assert not dog.expired()
