"""Take-5 recovery script end to end on synthetic recordings; triangulation accepts the re-cut."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recut_how2sign_take5 as script  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from tests.test_de_how2sign_recut import cameras, views  # noqa: E402

OFFSET = 9.25
FRAMES = 420
CLIPS = {0: (30, .74, 90), 1: (150, 3.68, 100), 2: (290, 7.23, 80)}    # sentence: (true raw start, CSV START, frames)


def video(frames: np.ndarray) -> bytes:
    """MP4 (30 fps) of grey frames."""
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        stream.options = {"qscale": "2"}
        for image in frames:
            rgb = np.repeat(image[..., None], 3, axis=2).astype(np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")))
        container.mux(stream.encode())
    return buffer.getvalue()


def images(count, seed):
    """A bright blob on a random walk, so every frame differs from its neighbours."""
    walk = np.cumsum(np.random.default_rng(seed).normal(scale=2., size=(count, 2)), axis=0)
    y, x = np.mgrid[0:72, 0:128]
    return np.stack([60 + 150 * np.exp(-((y - 16 - w[0] % 40) ** 2 + (x - 20 - w[1] % 88) ** 2) / 60.) for w in walk])


def put_verified(store, key, payload):
    store.put(key, payload)
    store.put(key + ".provenance.json", json.dumps(dict(sha256=hashlib.sha256(payload).hexdigest(),
                                                        verified_by_r2_reread=True)).encode())
    return hashlib.sha256(payload).hexdigest()


def zipped(members: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def corpus(tmp_path, side_seed=0):
    store = DirectoryStore(tmp_path)
    raw_front = images(FRAMES, 1)
    p1, _ = cameras()
    front_points, front_conf = views(np.arange(FRAMES), p1)
    clips, rows = {}, ["VIDEO_ID\tVIDEO_NAME\tSENTENCE_ID\tSENTENCE_NAME\tSTART\tEND\tSENTENCE"]
    for sentence, (start, csv_start, count) in CLIPS.items():
        name = f"vid_{sentence}-5-rgb_front"
        clips[f"raw_videos/{name}.mp4"] = clip = video(raw_front[start:start + count])
        buffer = io.BytesIO()
        np.savez(buffer, times=np.arange(count) / 30., points=front_points[start:start + count].astype(np.float32),
                 confidence=front_conf[start:start + count].astype(np.float16),
                 clip_sha256=np.array(hashlib.sha256(clip).hexdigest()))
        store.put(f"front/keypoints/{name}.npz", buffer.getvalue())
        rows.append(f"vid\tvid-5-rgb_front\tvid_{sentence}\t{name}\t{csv_start}\t{csv_start + count / 40}\ttext")
    zip_sha = put_verified(store, "pub/val_rgb_front_clips.zip", zipped(clips))
    put_verified(store, "pub/how2sign_val.csv", "\n".join(rows).encode())
    put_verified(store, "raw/val_raw_videos/shard_001.zip", zipped({"vid-5-rgb_front.mp4": video(raw_front)}))
    put_verified(store, "raw/val_side_raw_videos/shard_001.zip",
                 zipped({"vid-5-rgb_side.mp4": video(images(FRAMES, 2))}))
    return store, zip_sha


class FakeSapiens:
    """Stands in for Sapiens on the raw lateral recording: the scene seen by camera 2, ``OFFSET`` frames late."""

    side_seed = 0

    def __init__(self, path, sha256, batch, flip):
        self.path, self.sha256, self.flip = "fake", "f" * 64, flip

    def __call__(self, payload, first, stop):
        _, p2 = cameras()
        points, confidence = views(np.arange(FRAMES) - OFFSET, p2, seed=self.side_seed)
        confidence[:first], confidence[stop:] = 0., 0.
        return np.arange(FRAMES) / 30., points.astype(np.float32), confidence.astype(np.float16), min(stop, FRAMES)


def run(tmp_path, zip_sha, prefix="out"):
    return script.main(["--store", f"dir:{tmp_path}", "--front-keypoints", "front",
                        "--front-zip", f"pub/val_rgb_front_clips.zip={zip_sha}", "--sentences", "pub/how2sign_val.csv",
                        "--raw-prefix", "raw", "--prefix", prefix, "--max-offset", "20"])


def test_take5_session_is_placed_synchronised_and_recut(tmp_path, monkeypatch):
    store, zip_sha = corpus(tmp_path)
    monkeypatch.setattr(script, "Sapiens", FakeSapiens)
    assert run(tmp_path, zip_sha) == 0
    record = json.loads(store.get("out/sessions/vid-5.json"))
    assert record["accepted"], record["reason"]
    for sentence, (start, _, count) in CLIPS.items():
        clip = record["front_clips"][f"vid_{sentence}-5-rgb_front"]
        assert clip["placement"]["status"] == "placed" and clip["placement"]["start"] == start
    assert record["front_clips"]["vid_1-5-rgb_front"]["placement"]["nominal"] == 148     # 40 x 3.68, rounded up
    offset = record["offset"]
    assert abs(offset["offset_frames"] - OFFSET) < .1 and offset["sampson_px"] < 1.
    assert abs(offset["offset_seconds"] - OFFSET / 30) < .01 and offset["curve"]
    assert record["raw_side"]["member"] == "vid-5-rgb_side.mp4" and len(record["raw_side"]["member_sha256"]) == 64
    assert record["sentences"]["pub/how2sign_val.csv"] == hashlib.sha256(store.get("pub/how2sign_val.csv")).hexdigest()
    _, p2 = cameras()
    for sentence, (start, _, count) in CLIPS.items():
        output = record["outputs"][f"vid_{sentence}-5-rgb_side"]
        payload = store.get(output["key"])
        assert hashlib.sha256(payload).hexdigest() == output["sha256"]
        with np.load(io.BytesIO(payload)) as data:
            assert len(data["times"]) == count and int(data["raw_start"]) == start
            truth, _ = views(start + np.arange(count), p2, noise=0.)
            joints = data["confidence"][0] > 0
            error = np.linalg.norm(data["points"][:, joints] - truth[:, joints], axis=-1)
            assert np.median(error) < 1.
    report = json.loads(store.get("out/reports/shard-000-of-001.json"))
    assert report["accepted"] == 1
    # Resumable: a second run reads the session record instead of recomputing it.
    monkeypatch.setattr(script, "recover_session", None)
    assert run(tmp_path, zip_sha) == 0


def test_a_session_whose_side_recording_does_not_match_is_refused(tmp_path, monkeypatch):
    store, zip_sha = corpus(tmp_path)
    monkeypatch.setattr(FakeSapiens, "side_seed", 7)
    monkeypatch.setattr(script, "Sapiens", FakeSapiens)
    assert run(tmp_path, zip_sha) == 0
    record = json.loads(store.get("out/sessions/vid-5.json"))
    assert not record["accepted"] and record["reason"] and "outputs" not in record
    assert not store.list("out/keypoints/")


def test_a_raw_zip_still_being_acquired_is_not_used(tmp_path, monkeypatch):
    store, zip_sha = corpus(tmp_path)
    (tmp_path / "raw/val_side_raw_videos/shard_001.zip.provenance.json").unlink()   # mid-acquisition
    monkeypatch.setattr(script, "Sapiens", FakeSapiens)
    assert run(tmp_path, zip_sha) == 0
    assert not store.exists("out/sessions/vid-5.json")      # no write-once record from a partial input set
    report = json.loads(store.get("out/reports/shard-000-of-001.json"))
    assert report["sessions"][0]["reason"].startswith("raw recordings not acquired")


def test_a_session_with_an_unusable_source_is_refused_and_the_pass_continues(tmp_path, monkeypatch):
    store, zip_sha = corpus(tmp_path)
    monkeypatch.setattr(script, "Sapiens", FakeSapiens)

    def variable_rate(times, what):
        raise ValueError(f"the raw {what} recording is not constant 30 fps: 3 of 299 frame steps off")
    monkeypatch.setattr(script, "check_rate", variable_rate)
    assert run(tmp_path, zip_sha) == 0
    record = json.loads(store.get("out/sessions/vid-5.json"))
    assert not record["accepted"] and record["reason"].startswith("the raw frontal recording is not constant")
    assert json.loads(store.get("out/reports/shard-000-of-001.json"))["accepted"] == 0


def test_a_wrong_front_zip_hash_is_refused(tmp_path):
    _, zip_sha = corpus(tmp_path)
    try:
        run(tmp_path, "0" * 64)
        raise AssertionError("an unpinned frontal zip must be refused")
    except SystemExit as error:
        assert "not the verified acquisition" in str(error)


def test_sapiens_streams_frames_in_batches_and_keeps_their_order():
    import torch
    from sapiens_keypoints_runpod import frames_keypoints

    def model(inputs):                     # peak column grows with the frame's brightness
        heat = torch.zeros(len(inputs), 308, 256, 192)
        for i, image in enumerate(inputs):
            heat[i, :, 128, int(40 + 20 * (image.mean() + 2))] = 1.
        return heat

    frames = (np.full((72, 128, 3), 40 * k, np.uint8) for k in range(5))
    points, confidence = frames_keypoints(frames, model, torch.device("cpu"), batch=2, flip=False)
    assert points.shape == (5, 308, 2) and confidence.shape == (5, 308)
    assert np.all(np.diff(points[:, 0, 0]) > 0) and np.all(confidence == 1)


def test_clip_claims_block_a_second_job_but_expire(tmp_path, monkeypatch):
    import json, time
    from sapiens_keypoints_runpod import claim
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path)
    store.put("c/x", json.dumps(dict(time=time.time(), host="another-pod")).encode())
    assert not claim(store, "c/x")                               # another job holds it
    store.put("c/x", json.dumps(dict(time=time.time() - 3600, host="dead")).encode())
    assert claim(store, "c/x")                                   # a stale claim is taken over


def test_losing_a_claim_race_skips_instead_of_crashing():
    from sapiens_keypoints_runpod import claim
    class Racing:
        def exists(self, key): return False
        def put(self, key, payload): raise IOError("R2 re-read mismatch")
    assert claim(Racing(), "c/x") is False
