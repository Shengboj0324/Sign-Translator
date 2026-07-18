"""Tests for the training-readiness audit.

Each failure mode is constructed on disk and the auditor must catch it — in
particular split leakage, which would otherwise make held-out metrics
meaningless.
"""

import json
import numpy as np
import pytest

from signtranslator.data.corpus import CorpusSpec, generate_corpus, load_manifest
from signtranslator.data.readiness import assess_corpus


def _make(tmp_path, train=64, val=24, K=6, L=3, seed=0):
    spec = CorpusSpec.build(num_concepts=K, seq_len=L, num_joints=27,
                            in_channels=3, num_frames=16)
    generate_corpus(str(tmp_path), spec=spec, counts={"train": train, "val": val},
                    seed=seed)
    return spec


def _rewrite(tmp_path, split, **arrays):
    path = tmp_path / f"{split}.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data.update(arrays)
    np.savez_compressed(path, **data)


def test_healthy_corpus_is_ready(tmp_path):
    _make(tmp_path)
    rep = assess_corpus(str(tmp_path))
    assert rep.passed, rep.summary()
    assert "READY: YES" in rep.summary()


def test_detects_insufficient_samples(tmp_path):
    _make(tmp_path, train=8, val=4)
    rep = assess_corpus(str(tmp_path), min_train_samples=32)
    assert not rep.passed
    assert any(c.name == "sample_count" and not c.passed for c in rep.checks)


def test_detects_split_leakage(tmp_path):
    """Copy train samples into val: held-out metrics would be inflated."""
    _make(tmp_path, train=64, val=24)
    with np.load(tmp_path / "train.npz") as z:
        train_pose = z["pose"]
    with np.load(tmp_path / "val.npz") as z:
        val_pose = z["pose"].copy()
    val_pose[:5] = train_pose[:5]                      # inject leakage
    _rewrite(tmp_path, "val", pose=val_pose)

    rep = assess_corpus(str(tmp_path))
    leak = [c for c in rep.checks if c.name == "no_split_leakage"][0]
    assert not leak.passed
    assert rep.stats["val_leakage"] == 5
    assert not rep.passed


def test_detects_missing_class_in_val(tmp_path):
    spec = _make(tmp_path, train=64, val=24, K=6)
    with np.load(tmp_path / "val.npz") as z:
        concepts = z["concepts"].copy()
    concepts[concepts == 5] = 0                        # class 5 never in val
    _rewrite(tmp_path, "val", concepts=concepts)

    rep = assess_corpus(str(tmp_path))
    cov = [c for c in rep.checks if c.name == "class_coverage_val"][0]
    assert not cov.passed and "missing" in cov.detail


def test_detects_class_imbalance(tmp_path):
    _make(tmp_path, train=64, val=24, K=6)
    with np.load(tmp_path / "train.npz") as z:
        concepts = z["concepts"].copy()
    concepts[:] = 0                                    # collapse to one class
    concepts[0, 0] = 1                                 # keep class 1 barely present
    _rewrite(tmp_path, "train", concepts=concepts)

    rep = assess_corpus(str(tmp_path))
    assert not rep.passed
    names = {c.name for c in rep.checks if not c.passed}
    assert "class_balance" in names or "class_coverage_train" in names


def test_detects_ctc_infeasible_lengths(tmp_path):
    """Target longer than the number of frames makes CTC degenerate."""
    spec = _make(tmp_path, train=64, val=24, K=6, L=3)
    manifest = load_manifest(str(tmp_path))
    manifest["spec"]["num_frames"] = 2                 # fewer frames than targets
    with open(tmp_path / "manifest.json", "w") as f:
        json.dump(manifest, f)
    rep = assess_corpus(str(tmp_path), run_quality=False)
    ctc = [c for c in rep.checks if c.name == "ctc_length_feasible"][0]
    assert not ctc.passed


def test_detects_corrupt_pose_quality(tmp_path):
    _make(tmp_path, train=64, val=24)
    with np.load(tmp_path / "train.npz") as z:
        pose = z["pose"].copy()
    pose[0, 0, 3, 5] = np.nan
    _rewrite(tmp_path, "train", pose=pose)
    rep = assess_corpus(str(tmp_path))
    q = [c for c in rep.checks if c.name == "pose_quality"][0]
    assert not q.passed and "NaN" in q.detail


def test_report_records_useful_stats(tmp_path):
    _make(tmp_path, train=64, val=24)
    rep = assess_corpus(str(tmp_path))
    for key in ("train_samples", "val_samples", "imbalance_ratio",
                "normalized_mean", "normalized_var"):
        assert key in rep.stats
    assert rep.stats["train_samples"] == 64
