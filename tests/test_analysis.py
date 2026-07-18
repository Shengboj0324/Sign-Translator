"""Tests for the analysis/reporting stage and manifold integration."""

import torch
from torch.utils.data import DataLoader

from signtranslator import ModelConfig, DiffusionConfig
from signtranslator.models import BidirectionalSignTranslator
from signtranslator.data.corpus import (
    CorpusSpec, generate_corpus, SignDataset, collate_corpus,
)
from signtranslator.analysis import analyze, AnalysisReport, DEFAULT_THRESHOLDS
from signtranslator.analysis.report import DIAGNOSTIC_THRESHOLDS


def _model_and_loader(tmp_path):
    spec = CorpusSpec.build(num_concepts=6, seq_len=3, num_joints=27,
                            in_channels=3, num_frames=16)
    generate_corpus(str(tmp_path), spec=spec, counts={"train": 24, "val": 16}, seed=0)
    mcfg = ModelConfig(num_joints=27, num_frames=16, stgcn_channels=(16, 32),
                       text_embed_dim=32, text_layers=2, text_heads=2, latent_dim=32,
                       speech_input_dim=spec.speech_dim)
    dcfg = DiffusionConfig(num_timesteps=30, denoiser_dim=32, denoiser_layers=2,
                           denoiser_heads=2)
    model = BidirectionalSignTranslator(mcfg, dcfg, src_vocab=spec.src_vocab,
                                        gloss_vocab=spec.gloss_vocab,
                                        num_glosses=spec.num_glosses, planner_layers=2)
    val = DataLoader(SignDataset(str(tmp_path), "val"), batch_size=16,
                     collate_fn=collate_corpus)
    return model, val


def test_analyze_produces_full_report(tmp_path):
    model, val = _model_and_loader(tmp_path)
    report = analyze(model, val, ddim_steps=3, cycle_subset=8)
    assert isinstance(report, AnalysisReport)
    for key in ("recognition_wer", "planner_token_accuracy", "recall_at_1",
                "generation_val_loss", "cycle_consistency_wer"):
        assert key in report.metrics
    assert isinstance(report.passed, bool)
    # Cycle-consistency is now a full acceptance gate: generated motion must be
    # faithful enough for the recogniser to read it back.
    assert "cycle_consistency_wer" in report.gating
    assert set(DEFAULT_THRESHOLDS.keys()) == report.gating
    assert "OVERALL" in report.summary()


def test_metrics_are_in_valid_ranges(tmp_path):
    model, val = _model_and_loader(tmp_path)
    report = analyze(model, val, ddim_steps=3, cycle_subset=8)
    assert 0.0 <= report.metrics["planner_token_accuracy"] <= 1.0
    assert 0.0 <= report.metrics["recall_at_1"] <= 1.0
    assert report.metrics["recognition_wer"] >= 0.0
    assert report.metrics["generation_val_loss"] >= 0.0


# ---- manifold integration into the bidirectional model --------------------
def test_embeddings_are_unit_norm(tmp_path):
    model, val = _model_and_loader(tmp_path)
    b = next(iter(val))
    zm = model.embed_motion(b["pose"])
    zl = model.embed_gloss(b["gloss_tokens"])
    assert torch.allclose(zm.norm(dim=-1), torch.ones(zm.shape[0]), atol=1e-5)
    assert torch.allclose(zl.norm(dim=-1), torch.ones(zl.shape[0]), atol=1e-5)


def test_training_step_includes_alignment(tmp_path):
    model, val = _model_and_loader(tmp_path)
    losses = model.training_step(next(iter(val)))
    assert "alignment" in losses and torch.isfinite(losses["alignment"])


def test_shared_encode_pooled_equals_encoder_pooled(tmp_path):
    """The shared pass's pooled embedding must equal the encoder's clip
    embedding (time-mean of per-frame == global time+joint mean)."""
    model, val = _model_and_loader(tmp_path)
    model.eval()  # freeze batchnorm statistics for an exact comparison
    pose = next(iter(val))["pose"]
    with torch.no_grad():
        _, pooled = model._encode_pose_shared(pose)
        direct = model.recognizer.encoder(pose)
    assert torch.allclose(pooled, direct, atol=1e-5)
