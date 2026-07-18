"""Tests for the generation-fidelity mechanisms.

These cover the changes that took generated motion from unrecognisable
(cycle-consistency WER ~0.92) to recognised as accurately as ground truth
(~0.08): pose standardisation, high-noise timestep emphasis, and the
generator-only fine-tuning stage that cannot damage the manifold.
"""

import torch
from torch.utils.data import DataLoader

from signtranslator import ModelConfig, DiffusionConfig, TrainerConfig
from signtranslator.models import BidirectionalSignTranslator, CrossModalDenoiser
from signtranslator.models.diffusion import GaussianMotionDiffusion
from signtranslator.data.corpus import (
    CorpusSpec, generate_corpus, SignDataset, collate_corpus, PoseStandardizer,
    load_manifest,
)
from signtranslator.training import Trainer


# ---- pose standardisation -------------------------------------------------
def test_standardizer_roundtrip_and_stats(tmp_path):
    spec = CorpusSpec.build(num_concepts=6, seq_len=3, num_joints=27,
                            in_channels=3, num_frames=16)
    generate_corpus(str(tmp_path), spec=spec, counts={"train": 64, "val": 16}, seed=0)
    manifest = load_manifest(str(tmp_path))
    std = PoseStandardizer.from_manifest(manifest)

    raw = SignDataset(str(tmp_path), "train", normalize=False)
    norm = SignDataset(str(tmp_path), "train", normalize=True)
    # Normalised train data is ~zero-mean / unit-variance (diffusion assumption).
    assert abs(float(norm.pose.mean())) < 0.05
    assert abs(float(norm.pose.var()) - 1.0) < 0.15
    # Exact invertibility.
    assert torch.allclose(std.denormalize(norm.pose), raw.pose, atol=1e-4)


def test_normalization_stats_come_from_train_only(tmp_path):
    """Statistics must be computed on train and reused for val (no leakage)."""
    spec = CorpusSpec.build(num_concepts=6, seq_len=3, num_joints=27,
                            in_channels=3, num_frames=16)
    generate_corpus(str(tmp_path), spec=spec, counts={"train": 64, "val": 16}, seed=0)
    manifest = load_manifest(str(tmp_path))
    mean = torch.tensor(manifest["pose_mean"])
    raw_train = SignDataset(str(tmp_path), "train", normalize=False).pose
    expected = raw_train.mean(dim=(0, 2), keepdim=True)[0]
    assert torch.allclose(mean, expected, atol=1e-4)
    # Val uses the same (train-derived) standardizer object.
    val = SignDataset(str(tmp_path), "val")
    assert torch.allclose(val.standardizer.mean, mean, atol=1e-6)


# ---- high-noise timestep emphasis ----------------------------------------
def _diff(**kw):
    net = CrossModalDenoiser(num_joints=4, in_channels=3, context_dim=8,
                             hidden_dim=32, num_layers=2, num_heads=2)
    return GaussianMotionDiffusion(net, num_timesteps=100, **kw)


def test_uniform_sampling_by_default():
    d = _diff()
    t = d.sample_timesteps(20000, torch.device("cpu"))
    frac_high = float((t >= 85).float().mean())
    assert abs(frac_high - 0.15) < 0.02          # uniform => 15% above t=85


def test_high_t_emphasis_shifts_distribution():
    d = _diff(high_t_frac=0.65, high_t_start=0.85)
    t = d.sample_timesteps(20000, torch.device("cpu"))
    frac_high = float((t >= 85).float().mean())
    # 0.65 drawn from [85,100) plus 0.35*0.15 from the uniform part.
    assert abs(frac_high - (0.65 + 0.35 * 0.15)) < 0.03
    assert int(t.max()) < 100 and int(t.min()) >= 0


def test_high_t_params_validated():
    for bad in ({"high_t_frac": 1.5}, {"high_t_start": 1.0}):
        try:
            _diff(**bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


# ---- generator fine-tuning isolation --------------------------------------
def _setup(tmp_path):
    spec = CorpusSpec.build(num_concepts=6, seq_len=3, num_joints=27,
                            in_channels=3, num_frames=16)
    generate_corpus(str(tmp_path), spec=spec, counts={"train": 32, "val": 16}, seed=0)
    mcfg = ModelConfig(num_joints=27, num_frames=16, stgcn_channels=(16, 32),
                       text_embed_dim=32, text_layers=2, text_heads=2, latent_dim=32)
    dcfg = DiffusionConfig(num_timesteps=30, denoiser_dim=32, denoiser_layers=2,
                           denoiser_heads=2)
    model = BidirectionalSignTranslator(mcfg, dcfg, src_vocab=spec.src_vocab,
                                        gloss_vocab=spec.gloss_vocab,
                                        num_glosses=spec.num_glosses, planner_layers=2)
    loader = DataLoader(SignDataset(str(tmp_path), "train"), batch_size=16,
                        shuffle=False, collate_fn=collate_corpus, drop_last=True)
    return model, loader


def test_finetune_generation_reduces_generation_loss(tmp_path):
    torch.manual_seed(0)
    model, loader = _setup(tmp_path)
    hist = Trainer.finetune_generation(model, loader, loader, epochs=8, lr=2e-3)
    assert hist["val_generation"][-1] < hist["val_generation"][0]


def test_finetune_generation_leaves_manifold_untouched(tmp_path):
    """The fine-tune must not modify the recogniser or the manifold encoder --
    otherwise retrieval silently collapses."""
    torch.manual_seed(0)
    model, loader = _setup(tmp_path)
    before_gloss = model.gloss_encoder.token_emb.weight.detach().clone()
    before_recog = model.recognizer.classifier.weight.detach().clone()
    before_align = model.aligner.motion_head.net[0].weight.detach().clone()
    before_cond = model.cond_encoder.token_emb.weight.detach().clone()

    Trainer.finetune_generation(model, loader, epochs=4, lr=2e-3)

    assert torch.equal(model.gloss_encoder.token_emb.weight, before_gloss)
    assert torch.equal(model.recognizer.classifier.weight, before_recog)
    assert torch.equal(model.aligner.motion_head.net[0].weight, before_align)
    # The generator-private encoder *is* trained.
    assert not torch.equal(model.cond_encoder.token_emb.weight, before_cond)
