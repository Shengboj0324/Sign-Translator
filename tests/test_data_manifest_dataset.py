"""Manifest dataset: partitions by performer, admitted channels only, collatable."""
import pytest

from signtranslator.data.manifest_dataset import ManifestMotionDataset
from signtranslator.data.multichannel import collate_multichannel
from signtranslator.data_engineering.corpus_manifest import ContentCache, build_corpus_manifest
from tests.test_de_corpus_manifest import record, state


@pytest.fixture
def setup(tmp_path):
    blobs = {}
    records = []
    for sign, performer, left in (("a", "p1", "validated"), ("b", "p1", "failed"), ("c", "p2", "validated")):
        path = tmp_path / f"{sign}.npz"
        digest = state(sign).save(path)
        blobs[f"k/{sign}"] = path.read_bytes()
        records.append(record(sign, performer=performer, left=left, sha=digest))
    split = dict(train_performer_sha256="p1", test_performer_sha256=["p2"])
    manifest = build_corpus_manifest(corpus_id="c", inputs={"x": "b" * 64}, records=records, split=split)
    return manifest, ContentCache(tmp_path / "cache", 10 ** 7), blobs.__getitem__


def test_partitions_are_disjoint_by_performer(setup):
    manifest, cache, fetch = setup
    train = ManifestMotionDataset(manifest, "skeleton", partition="train", cache=cache, fetch=fetch)
    test = ManifestMotionDataset(manifest, "skeleton", partition="test", cache=cache, fetch=fetch)
    assert [r["sign"] for r in train.records] == ["a", "b"] and [r["sign"] for r in test.records] == ["c"]
    batch = collate_multichannel([train[0], train[1]])
    assert batch.channels["left_hand"].valid[0].all() and not batch.channels["left_hand"].valid[1].any()


def test_required_channels_filter_samples(setup):
    manifest, cache, fetch = setup
    data = ManifestMotionDataset(manifest, "skeleton", partition="all", cache=cache, fetch=fetch,
                                 require_channels=("left_hand",))
    assert [r["sign"] for r in data.records] == ["a", "c"]
    with pytest.raises(ValueError):
        ManifestMotionDataset(manifest, "skeleton", partition="val", cache=cache, fetch=fetch)


def test_overlapping_split_is_refused(setup):
    manifest, cache, fetch = setup
    manifest = dict(manifest, split=dict(train_performer_sha256="p1", test_performer_sha256=["p1"]))
    with pytest.raises(ValueError, match="both partitions"):
        ManifestMotionDataset(manifest, "skeleton", partition="train", cache=cache, fetch=fetch)
