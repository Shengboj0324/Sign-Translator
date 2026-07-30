"""Tests for on-disk corpus generation, ingestion, and collation."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from signtranslator.data.corpus import (
    CorpusSpec, generate_corpus, validate_corpus, load_manifest,
    SignDataset, collate_corpus, CONTENT_OFFSET, BOS, EOS,
)


@pytest.fixture
def corpus(tmp_path):
    spec = CorpusSpec.build(num_concepts=8, seq_len=4, num_joints=27,
                            in_channels=3, num_frames=16)
    generate_corpus(str(tmp_path), spec=spec, counts={"train": 40, "val": 12}, seed=0)
    return str(tmp_path), spec


def test_generate_and_validate(corpus):
    path, spec = corpus
    got = validate_corpus(path)               # raises on any inconsistency
    assert got.num_concepts == spec.num_concepts
    manifest = load_manifest(path)
    assert manifest["splits"] == {"train": 40, "val": 12}
    assert len(manifest["perm"]) == spec.num_concepts


def test_generation_refuses_non_empty_directory(tmp_path):
    (tmp_path / "valuable.txt").write_text("real data")
    with pytest.raises(FileExistsError, match="refusing"):
        generate_corpus(str(tmp_path))
    assert (tmp_path / "valuable.txt").read_text() == "real data"


def test_generation_overwrite_requires_explicit_opt_in(tmp_path):
    generate_corpus(str(tmp_path))
    with pytest.raises(FileExistsError, match="overwrite=True"):
        generate_corpus(str(tmp_path))
    generate_corpus(str(tmp_path), overwrite=True)


def test_dataset_shapes(corpus):
    path, spec = corpus
    ds = SignDataset(path, "train")
    assert len(ds) == 40
    item = ds[0]
    assert item["pose"].shape == (spec.in_channels, spec.num_frames, spec.num_joints)
    assert item["concepts"].shape == item["src_concepts"].shape


def test_collate_derives_consistent_branch_tensors(corpus):
    path, spec = corpus
    loader = DataLoader(SignDataset(path, "train"), batch_size=8,
                        collate_fn=collate_corpus)
    b = next(iter(loader))
    n = b["pose"].shape[0]
    # gloss_seq is BOS + content + EOS
    assert (b["gloss_seq"][:, 0] == BOS).all()
    for i in range(n):
        L = len(b["concepts"][i])
        assert b["gloss_seq"][i, 1 + L] == EOS
        # gloss_tokens == concepts + offset
        assert torch.equal(b["gloss_tokens"][i, :L], b["concepts"][i] + CONTENT_OFFSET)
    # CTC targets are concept+1 (blank-free), concatenated with matching lengths
    assert int(b["ctc_lengths"].sum()) == b["ctc_targets"].numel()
    assert b["ctc_targets"].min() >= 1


def test_src_is_a_consistent_bijection_of_gloss(corpus):
    """Across the whole corpus, each spoken token maps to exactly one gloss token
    (the fixed cipher), which is what makes the planner task learnable."""
    path, _ = corpus
    loader = DataLoader(SignDataset(path, "train"), batch_size=16,
                        collate_fn=collate_corpus)
    mapping = {}
    for b in loader:
        for i in range(b["pose"].shape[0]):
            L = len(b["concepts"][i])
            for s_tok, g_tok in zip(b["src"][i, :L].tolist(), b["gloss_tokens"][i, :L].tolist()):
                if s_tok in mapping:
                    assert mapping[s_tok] == g_tok      # deterministic
                mapping[s_tok] = g_tok
    # Bijection: distinct spoken tokens map to distinct gloss tokens.
    assert len(set(mapping.values())) == len(mapping)


def test_validate_detects_corruption(corpus, tmp_path):
    path, spec = corpus
    # Corrupt the train split: wrong pose shape.
    np.savez_compressed(f"{path}/train.npz",
                        pose=np.zeros((40, 3, 8, 27), dtype=np.float32),  # wrong T
                        concepts=np.zeros((40, 4), dtype=np.int64),
                        src_concepts=np.zeros((40, 4), dtype=np.int64),
                        lengths=np.full(40, 4, dtype=np.int64))
    with pytest.raises(ValueError):
        validate_corpus(path)
