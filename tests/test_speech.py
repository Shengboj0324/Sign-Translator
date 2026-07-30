"""Tests for the acoustic branch (audio features -> spoken tokens -> sign)."""

import pytest
import torch
from torch.utils.data import DataLoader

from signtranslator import ModelConfig, DiffusionConfig
from signtranslator.models import (
    BidirectionalSignTranslator, SpeechRecognizer, TranslationAbstainedError,
)
from signtranslator.data.corpus import (
    CorpusSpec, generate_corpus, SignDataset, collate_corpus, validate_corpus,
)


def test_speech_recognizer_shapes_and_subsampling():
    rec = SpeechRecognizer(input_dim=40, num_tokens=10, hidden_dim=32,
                           num_layers=2, num_heads=2, subsample=2)
    x = torch.randn(4, 64, 40)
    lp = rec(x)
    assert lp.shape == (4, 32, 11)              # T/2 frames, 10 tokens + blank
    assert torch.allclose(lp.exp().sum(-1), torch.ones(4, 32), atol=1e-5)


def test_output_lengths_match_actual_frames():
    for sub in (1, 2, 4):
        rec = SpeechRecognizer(input_dim=40, num_tokens=6, hidden_dim=32,
                               num_layers=1, num_heads=2, subsample=sub)
        x = torch.randn(2, 64, 40)
        t_out = rec(x).shape[1]
        pred = rec.output_lengths(torch.tensor([64, 64]))
        assert int(pred[0]) == t_out, f"subsample={sub}"


def test_invalid_subsample_rejected():
    with pytest.raises(ValueError):
        SpeechRecognizer(input_dim=40, num_tokens=6, subsample=3)


def test_speech_ctc_overfits_a_fixed_batch():
    torch.manual_seed(0)
    rec = SpeechRecognizer(input_dim=40, num_tokens=6, hidden_dim=64,
                           num_layers=2, num_heads=2)
    x = torch.randn(4, 64, 40)
    targets = torch.randint(1, 7, (4, 3))
    lengths = torch.full((4,), 3, dtype=torch.long)
    opt = torch.optim.Adam(rec.parameters(), lr=5e-3)
    first = None
    for _ in range(40):
        loss = rec.loss(x, targets, lengths)
        assert torch.isfinite(loss)
        first = first if first is not None else loss.detach().item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.detach().item() < first * 0.6


# ---- corpus integration ----------------------------------------------------
@pytest.fixture
def corpus(tmp_path):
    spec = CorpusSpec.build(num_concepts=6, seq_len=3, num_joints=27,
                            in_channels=3, num_frames=16, speech_frames=32,
                            speech_dim=40)
    generate_corpus(str(tmp_path), spec=spec, counts={"train": 32, "val": 16}, seed=0)
    return str(tmp_path), spec


def test_corpus_carries_speech_features(corpus):
    path, spec = corpus
    validate_corpus(path)                       # validates speech shape too
    loader = DataLoader(SignDataset(path, "train"), batch_size=8,
                        collate_fn=collate_corpus)
    b = next(iter(loader))
    assert b["speech"].shape == (8, spec.speech_frames, spec.speech_dim)
    assert torch.isfinite(b["speech"]).all()
    # Speech CTC targets follow the SPOKEN order/ids, not the gloss ids.
    assert int(b["speech_ctc_lengths"].sum()) == b["speech_ctc_targets"].numel()
    assert b["speech_ctc_targets"].min() >= 1


def test_speech_targets_are_the_ciphered_ids(corpus):
    """speech targets = src_concepts+1; gloss targets = concepts+1."""
    path, _ = corpus
    ds = SignDataset(path, "train")
    batch = collate_corpus([ds[0], ds[1]])
    for i, item in enumerate((ds[0], ds[1])):
        n = len(item["concepts"])
        sp = batch["speech_ctc_targets"][i * n:(i + 1) * n]
        assert torch.equal(sp, item["src_concepts"] + 1)


def _model(spec):
    mcfg = ModelConfig(num_joints=27, num_frames=16, stgcn_channels=(16, 32),
                       text_embed_dim=32, text_layers=2, text_heads=2,
                       latent_dim=32, speech_input_dim=spec.speech_dim)
    dcfg = DiffusionConfig(num_timesteps=30, denoiser_dim=32, denoiser_layers=2,
                           denoiser_heads=2)
    return BidirectionalSignTranslator(mcfg, dcfg, src_vocab=spec.src_vocab,
                                       gloss_vocab=spec.gloss_vocab,
                                       num_glosses=spec.num_glosses,
                                       planner_layers=2)


def test_training_step_includes_speech_branch(corpus):
    path, spec = corpus
    model = _model(spec)
    loader = DataLoader(SignDataset(path, "train"), batch_size=8,
                        collate_fn=collate_corpus)
    losses = model.training_step(next(iter(loader)))
    assert "speech" in losses and torch.isfinite(losses["speech"])
    assert torch.isfinite(losses["total"])


def test_speech_gradients_reach_speech_recognizer_only(corpus):
    path, spec = corpus
    model = _model(spec)
    loader = DataLoader(SignDataset(path, "train"), batch_size=8,
                        collate_fn=collate_corpus)
    b = next(iter(loader))
    model.speech_loss(b["speech"], b["speech_ctc_targets"],
                      b["speech_ctc_lengths"]).backward()
    assert model.speech_recognizer.classifier.weight.grad.abs().sum() > 0
    # Must not touch the visual recogniser.
    assert (model.recognizer.classifier.weight.grad is None
            or model.recognizer.classifier.weight.grad.abs().sum() == 0)


def test_full_audio_to_sign_path(corpus):
    """audio features -> spoken tokens -> gloss -> 3D motion."""
    path, spec = corpus
    torch.manual_seed(0)
    model = _model(spec)
    loader = DataLoader(SignDataset(path, "val"), batch_size=4,
                        collate_fn=collate_corpus)
    b = next(iter(loader))
    out = model.translate_audio_to_sign(b["speech"], num_frames=12, ddim_steps=3)
    assert out["motion"].shape == (4, 3, 12, 27)
    assert torch.isfinite(out["motion"]).all()
    assert len(out["spoken_tokens"]) == 4
    assert len(out["gloss"]) == 4


def test_empty_decoder_output_abstains_instead_of_generating_nan(corpus):
    _, spec = corpus
    model = _model(spec)
    model.planner.greedy_decode = lambda src, max_len: [[] for _ in range(src.shape[0])]
    src = torch.ones(2, 3, dtype=torch.long)
    with pytest.raises(TranslationAbstainedError, match="no gloss evidence"):
        model.translate_speech_to_sign(src, num_frames=8, ddim_steps=2)
