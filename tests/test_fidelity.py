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


# ---- W2 affine inverse and observed-support contracts ----------------------
def test_standardizer_masked_inverse_has_independent_values_and_jacobian():
    mean = torch.tensor([[[10., -2.]], [[3., 7.]]], dtype=torch.float64)
    scale = torch.tensor([[[2., 4.]], [[5., 10.]]], dtype=torch.float64)
    transform = PoseStandardizer(mean, scale)
    raw = torch.tensor([[[12., 2.], [float('nan'), 6.]],
                        [[8., 27.], [float('inf'), 37.]]],
                       dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True, True], [False, True]])
    expected = torch.tensor([[[1., 1.], [0., 2.]], [[1., 2.], [0., 3.]]],
                            dtype=torch.float64)
    normalized = transform.normalize(raw, valid)
    torch.testing.assert_close(normalized, expected, rtol=0, atol=0)
    restored = transform.denormalize(normalized, valid)
    supported = valid.unsqueeze(0).expand_as(raw)
    torch.testing.assert_close(restored[supported], raw[supported], rtol=0, atol=0)
    assert torch.count_nonzero(restored[~supported]) == 0
    gradient, = torch.autograd.grad(normalized.sum(), raw)
    torch.testing.assert_close(gradient, torch.where(supported, 1 / scale, 0))
    assert torch.isfinite(gradient).all()


def test_standardizer_inverse_sanitizes_missing_values_before_arithmetic():
    transform = PoseStandardizer(torch.ones(2, 1, 2), torch.ones(2, 1, 2) * 3)
    z = torch.full((3, 2, 4, 2), float('nan'), requires_grad=True)
    mask = torch.zeros(3, 4, 2, dtype=torch.bool)
    restored = transform.denormalize(z, mask)
    assert torch.count_nonzero(restored) == 0
    restored.sum().backward()
    assert torch.count_nonzero(z.grad) == 0


def test_standardizer_preserves_precision_and_batch_padding_invariance():
    transform = PoseStandardizer(torch.ones(2, 1, 2), torch.ones(2, 1, 2) * 3)
    raw = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2)
    single = transform.normalize(raw)
    padded = torch.full((2, 2, 5, 2), float('nan'), dtype=torch.float64)
    padded[:, :, :3] = raw
    valid = torch.zeros(2, 5, 2, dtype=torch.bool)
    valid[:, :3] = True
    batched = transform.normalize(padded, valid)
    assert batched.dtype == torch.float64
    torch.testing.assert_close(batched[0, :, :3], single, rtol=0, atol=0)
    torch.testing.assert_close(transform.denormalize(single), raw, rtol=1e-14, atol=1e-14)
    assert torch.count_nonzero(batched[:, :, 3:]) == 0


def test_standardizer_rejects_malformed_statistics_inputs_and_mutations():
    import pytest
    base = torch.ones(2, 1, 2)
    for mean, std in ((base, base * 0), (base, -base),
                      (base, base * float('nan')), (base, base * float('inf')),
                      (base, torch.ones(2, 2, 2)),
                      (torch.ones(1, 1, 2), base)):
        with pytest.raises(ValueError):
            PoseStandardizer(mean, std)
    with pytest.raises(TypeError):
        PoseStandardizer(base.long(), base)
    transform = PoseStandardizer(base.clone(), base.clone())
    for pose in (torch.ones(2, 2), torch.ones(2, 0, 2), torch.ones(1, 2, 2),
                 torch.ones(2, 2, 3), torch.full((2, 2, 2), float('nan'))):
        with pytest.raises(ValueError):
            transform.normalize(pose)
    with pytest.raises(TypeError):
        transform.normalize(torch.ones(2, 2, 2, dtype=torch.int64))
    with pytest.raises(ValueError):
        transform.normalize(torch.ones(2, 2, 2), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(TypeError):
        transform.denormalize(torch.ones(2, 2, 2), torch.ones(2, 2))
    transform.std[0, 0, 0] = 0
    with pytest.raises(ValueError):
        transform.denormalize(torch.ones(2, 2, 2))


def test_standardizer_rejects_unrepresentable_stats_and_affine_overflow():
    import pytest
    mean = torch.ones(1, 1, 1, dtype=torch.float64)
    transform = PoseStandardizer(mean, mean * 1e-300)
    with pytest.raises(ValueError, match='representable'):
        transform.normalize(torch.ones(1, 1, 1, dtype=torch.float32))
    transform = PoseStandardizer(mean * 0, mean * 1e300)
    with pytest.raises(ValueError, match='overflowed'):
        transform.denormalize(mean * 1e100)
    transform = PoseStandardizer(mean * 0, mean * 1e-300)
    with pytest.raises(ValueError, match='overflowed'):
        transform.normalize(mean * 1e100)
