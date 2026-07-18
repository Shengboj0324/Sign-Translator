"""Tests for the LR schedule, the unified Trainer, and checkpointing."""

import math

import torch
from torch.utils.data import DataLoader

from signtranslator import ModelConfig, DiffusionConfig, TrainerConfig
from signtranslator.models import BidirectionalSignTranslator
from signtranslator.data.corpus import (
    CorpusSpec, generate_corpus, SignDataset, collate_corpus,
)
from signtranslator.training import Trainer, cosine_warmup_lambda


def test_cosine_warmup_schedule_shape():
    fn = cosine_warmup_lambda(total_steps=100, warmup_steps=10, min_lr_frac=0.05)
    assert abs(fn(0) - 0.1) < 1e-9            # first step = 1/warmup
    assert abs(fn(9) - 1.0) < 1e-9            # peak at end of warmup
    assert fn(50) < 1.0 and fn(50) > 0.05     # decaying
    assert abs(fn(99) - 0.05) < 1e-2          # floor at the end
    # Monotone non-increasing through the decay region.
    decay = [fn(s) for s in range(10, 100)]
    assert all(decay[i] >= decay[i + 1] - 1e-9 for i in range(len(decay) - 1))


def _tiny_setup(tmp_path):
    spec = CorpusSpec.build(num_concepts=8, seq_len=3, num_joints=27,
                            in_channels=3, num_frames=16)
    generate_corpus(str(tmp_path), spec=spec, counts={"train": 48, "val": 16}, seed=0)
    mcfg = ModelConfig(num_joints=27, num_frames=16, stgcn_channels=(16, 32),
                       text_embed_dim=32, text_layers=2, text_heads=2, latent_dim=32,
                       speech_input_dim=spec.speech_dim)
    dcfg = DiffusionConfig(num_timesteps=40, denoiser_dim=32, denoiser_layers=2,
                           denoiser_heads=2)
    model = BidirectionalSignTranslator(mcfg, dcfg, src_vocab=spec.src_vocab,
                                        gloss_vocab=spec.gloss_vocab,
                                        num_glosses=spec.num_glosses, planner_layers=2)
    train = DataLoader(SignDataset(str(tmp_path), "train"), batch_size=16,
                       shuffle=True, collate_fn=collate_corpus, drop_last=True)
    val = DataLoader(SignDataset(str(tmp_path), "val"), batch_size=16,
                     collate_fn=collate_corpus)
    return model, train, val


def test_trainer_reduces_total_loss(tmp_path):
    torch.manual_seed(0)
    model, train, val = _tiny_setup(tmp_path)
    cfg = TrainerConfig(epochs=6, batch_size=16, lr=3e-3, seed=0)
    trainer = Trainer(model, cfg, train, val)
    history = trainer.fit(verbose=False)
    assert history["val_total"][-1] < history["val_total"][0]
    # Recognition (CTC) and planner should each improve too.
    assert history["train_recognition"][-1] < history["train_recognition"][0]
    assert history["train_planner"][-1] < history["train_planner"][0]


def test_checkpoint_roundtrip(tmp_path):
    model, train, val = _tiny_setup(tmp_path)
    cfg = TrainerConfig(epochs=1, batch_size=16, lr=3e-3, seed=0)
    trainer = Trainer(model, cfg, train, val)
    trainer.fit(verbose=False)
    ckpt = str(tmp_path / "m.pt")
    trainer.save(ckpt)

    # Fresh model + trainer, load, compare a parameter.
    model2, train2, val2 = _tiny_setup(tmp_path)
    cfg2 = TrainerConfig(epochs=1, batch_size=16, seed=0)
    trainer2 = Trainer(model2, cfg2, train2, val2)
    trainer2.load(ckpt)
    a = model.recognizer.classifier.weight
    b = model2.recognizer.classifier.weight
    assert torch.allclose(a, b, atol=1e-6)


def test_loss_weights_applied_to_total(tmp_path):
    model, train, _ = _tiny_setup(tmp_path)
    batch = next(iter(train))
    unweighted = model.training_step(batch)
    zeroed = model.training_step(batch, weights={"generation": 0.0, "alignment": 0.0,
                                                 "planner": 0.0, "recognition": 0.0,
                                                 "speech": 0.0})
    assert float(zeroed["total"]) == 0.0
    assert float(unweighted["total"]) > 0.0
