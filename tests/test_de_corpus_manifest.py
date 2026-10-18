"""Corpus manifest: admission rules, canonical bytes, verified bounded cache, masking."""
import hashlib

import numpy as np
import pytest

from signtranslator.data_engineering.corpus_manifest import (
    ContentCache, build_corpus_manifest, load_manifest, load_sample, manifest_sha256, mask_channels,
    performer_split, usable_channels, write_manifest,
)
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion

SHA = "b" * 64


def state(sample="s"):
    frames = 4
    channels = {}
    for name, (kind, width, units) in CHANNELS.items():
        values = np.zeros((frames, 1, width))
        if kind == "rotation_6d_columns":
            values[..., [0, 4]] = 1.
        elif kind == "unit_vector":
            values[..., 2] = 1.
        valid = np.ones((frames, 1), bool)
        channels[name] = MotionChannel(values, valid, np.zeros_like(valid), np.ones((frames, 1)), (name,),
                                       "f", units, "src", "a" * 64, "c", inference_method="solve")
    return MultichannelMotion(np.arange(frames, dtype=np.float64) / 24, "clock", sample, channels)


def record(sign, *, marker="passed", left="validated", right="failed", face="converted", performer="p1",
           sha=SHA):
    return dict(sign=sign, performer_sha256=performer,
                skeleton=dict(state_key=f"k/{sign}", state_sha256=sha, marker_validation=marker,
                              available_channels=["root_translation", "body", "head_rotation",
                                                  "left_hand", "right_hand"]),
                hands=dict(Left=dict(status=left), Right=dict(status=right)),
                face=dict(status=face, state_key=f"f/{sign}", state_sha256=sha),
                gloss=dict(status="unique_stem"))


def test_admission_rules_follow_the_evidence():
    assert usable_channels(record("a"), "skeleton") == ("root_translation", "body", "left_hand",
                                                         "head_rotation")
    assert usable_channels(record("a", marker="failed"), "skeleton") == ()
    assert usable_channels(record("a", left="no_glove", right="validated"), "skeleton") == (
        "root_translation", "body", "right_hand", "head_rotation")
    assert usable_channels(record("a"), "face") == ("face", "left_gaze", "right_gaze", "blink")
    assert usable_channels(record("a", face="refused"), "face") == ()
    salvaged = record("a")
    salvaged["skeleton"]["available_channels"] = ["body", "head_rotation", "left_hand", "right_hand"]
    assert "root_translation" not in usable_channels(salvaged, "skeleton")


def test_manifest_is_order_independent_canonical_and_write_once(tmp_path):
    split = dict(kind="skeleton")
    one = build_corpus_manifest(corpus_id="c", inputs={"x": SHA}, records=[record("b"), record("a")], split=split)
    two = build_corpus_manifest(corpus_id="c", inputs={"x": SHA}, records=[record("a"), record("b")], split=split)
    assert manifest_sha256(one) == manifest_sha256(two) and one["counts"]["left_hand_validated"] == 2
    digest = write_manifest(one, tmp_path / "m.json")
    assert digest == manifest_sha256(one) and load_manifest(tmp_path / "m.json", expected_sha256=digest) == one
    with pytest.raises(FileExistsError):
        write_manifest(one, tmp_path / "m.json")
    tampered = tmp_path / "t.json"
    one["records"][0]["usable"]["skeleton"].append("face")
    write_manifest(one, tampered)
    with pytest.raises(ValueError, match="disagree"):
        load_manifest(tampered, expected_sha256=manifest_sha256(one))


def test_manifest_rejects_duplicates_and_malformed_hashes():
    with pytest.raises(ValueError, match="unique"):
        build_corpus_manifest(corpus_id="c", inputs={"x": SHA}, records=[record("a"), record("a")], split={})
    with pytest.raises(ValueError, match="malformed"):
        build_corpus_manifest(corpus_id="c", inputs={"x": SHA}, records=[record("a", sha="xyz")], split={})


def test_cache_verifies_every_byte_and_stays_within_its_bound(tmp_path):
    blobs = {f"k{i}": bytes([i]) * 100 for i in range(5)}
    shas = {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}
    cache = ContentCache(tmp_path / "c", max_bytes=250)
    for key in blobs:
        assert cache.get(key, shas[key], blobs.__getitem__) == blobs[key]
        assert cache.size() <= 250
    assert cache.fetches == 5
    cache.get("k4", shas["k4"], blobs.__getitem__)
    assert cache.fetches == 5  # served from disk after re-verification
    (tmp_path / "c" / shas["k4"]).write_bytes(b"corrupt")
    assert cache.get("k4", shas["k4"], blobs.__getitem__) == blobs["k4"] and cache.fetches == 6
    with pytest.raises(ValueError, match="do not match"):
        cache.get("k0", shas["k1"], blobs.__getitem__)


def test_load_sample_masks_unadmitted_channels(tmp_path):
    path = tmp_path / "s.npz"
    digest = state().save(path)
    payload = path.read_bytes()
    rec = record("a", sha=digest)
    loaded = load_sample(rec, "skeleton", cache=ContentCache(tmp_path / "c", 10 ** 6), fetch=lambda k: payload)
    assert loaded.channels["left_hand"].valid.all() and not loaded.channels["right_hand"].valid.any()
    assert not loaded.channels["face"].valid.any()
    assert not loaded.channels["right_hand"].values.any()
    with pytest.raises(ValueError, match="no usable"):
        load_sample(record("a", marker="failed", sha=digest), "skeleton",
                    cache=ContentCache(tmp_path / "d", 10 ** 6), fetch=lambda k: payload)
    masked = mask_channels(state(), {"body"})
    masked.validate()
    assert [n for n, c in masked.channels.items() if c.valid.any()] == ["body"]


def test_performer_split_is_descriptive_below_three_groups():
    records = [record("a"), record("b"), record("c", performer="p2"), record("d", performer="p3", marker="x")]
    split = performer_split(records)
    assert split["train_performer_sha256"] == "p1" and split["test_performer_sha256"] == ["p2"]
    assert split["independent_units"] == 2 and not split["confirmatory"]
    with pytest.raises(ValueError):
        performer_split([record("a")])
