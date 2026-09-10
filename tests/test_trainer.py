"""Tests for the LR schedule, the unified Trainer, and checkpointing."""

import math
import json

import torch
from torch.utils.data import DataLoader

from signtranslator import ModelConfig, DiffusionConfig, TrainerConfig
from signtranslator.models import BidirectionalSignTranslator
from signtranslator.data.corpus import (
    CorpusSpec, generate_corpus, SignDataset, collate_corpus,
)
from signtranslator.training import Trainer, checkpoint_paths, cosine_warmup_lambda


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
    if not (tmp_path / "manifest.json").exists():
        generate_corpus(str(tmp_path), spec=spec,
                        counts={"train": 48, "val": 16}, seed=0)
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
    trainer2.load(ckpt, mode="weights")
    a = model.recognizer.classifier.weight
    b = model2.recognizer.classifier.weight
    assert torch.allclose(a, b, atol=1e-6)
    manifest = json.loads((tmp_path / "m.pt.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["kind"] == "last"
    assert manifest["model_contract"]["configs"]


def test_validation_is_repeatable_and_does_not_advance_training_rng(tmp_path):
    torch.manual_seed(7)
    model, train, val = _tiny_setup(tmp_path)
    trainer = Trainer(model, TrainerConfig(epochs=1, batch_size=16, seed=19),
                      train, val)
    trainer.model.train()
    before = torch.get_rng_state().clone()
    first = trainer.validate()
    after_first = torch.get_rng_state().clone()
    second = trainer.validate()
    after_second = torch.get_rng_state().clone()
    assert first == second
    assert torch.equal(before, after_first)
    assert torch.equal(before, after_second)
    assert trainer.model.training is True


def test_exact_resume_matches_uninterrupted_training_bit_for_bit(tmp_path):
    corpus = tmp_path / "corpus"

    torch.manual_seed(23)
    full_model, full_train, full_val = _tiny_setup(corpus)
    full_cfg = TrainerConfig(
        epochs=4, batch_size=16, lr=3e-3, seed=11,
        ckpt_path=str(tmp_path / "full.pt"))
    full = Trainer(full_model, full_cfg, full_train, full_val)
    full.fit(verbose=False)

    torch.manual_seed(23)
    partial_model, partial_train, partial_val = _tiny_setup(corpus)
    partial_cfg = TrainerConfig(
        epochs=4, batch_size=16, lr=3e-3, seed=11,
        ckpt_path=str(tmp_path / "resumed.pt"))
    partial = Trainer(partial_model, partial_cfg, partial_train, partial_val)
    partial.fit(verbose=False, max_epochs=2)

    resumed_model, resumed_train, resumed_val = _tiny_setup(corpus)
    resumed = Trainer(resumed_model, partial_cfg, resumed_train, resumed_val)
    resumed.load(checkpoint_paths(partial_cfg.ckpt_path)["last"], mode="resume")
    resumed.fit(verbose=False)

    assert resumed.completed_epochs == full.completed_epochs == 4
    assert resumed.global_step == full.global_step
    assert resumed.history == full.history
    for name, expected in full.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[name]), name
    resumed_opt = resumed.opt.state_dict()
    full_opt = full.opt.state_dict()
    assert resumed_opt["param_groups"] == full_opt["param_groups"]
    assert resumed_opt["state"].keys() == full_opt["state"].keys()
    for parameter in full_opt["state"]:
        assert resumed_opt["state"][parameter].keys() == full_opt["state"][parameter].keys()
        for field, expected in full_opt["state"][parameter].items():
            actual = resumed_opt["state"][parameter][field]
            if torch.is_tensor(expected):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected
    assert resumed.sched.state_dict() == full.sched.state_dict()


def test_resume_rejects_configuration_drift_and_byte_tampering(tmp_path):
    model, train, val = _tiny_setup(tmp_path / "corpus")
    cfg = TrainerConfig(
        epochs=2, batch_size=16, lr=3e-3, seed=5,
        ckpt_path=str(tmp_path / "run.pt"))
    trainer = Trainer(model, cfg, train, val)
    trainer.fit(max_epochs=1)
    last = checkpoint_paths(cfg.ckpt_path)["last"]

    changed_model, changed_train, changed_val = _tiny_setup(tmp_path / "corpus")
    changed_cfg = TrainerConfig(
        epochs=2, batch_size=16, lr=1e-3, seed=5,
        ckpt_path=str(tmp_path / "run.pt"))
    changed = Trainer(changed_model, changed_cfg, changed_train, changed_val)
    try:
        changed.load(last, mode="resume")
    except ValueError as error:
        assert "configuration" in str(error)
    else:
        raise AssertionError("resume accepted an optimization-changing configuration")

    last.write_bytes(last.read_bytes() + b"tamper")
    try:
        trainer.load(last, mode="resume")
    except ValueError as error:
        assert "manifest" in str(error)
    else:
        raise AssertionError("resume accepted a tampered checkpoint")


def test_loss_weights_applied_to_total(tmp_path):
    model, train, _ = _tiny_setup(tmp_path)
    batch = next(iter(train))
    unweighted = model.training_step(batch)
    zeroed = model.training_step(batch, weights={"generation": 0.0, "alignment": 0.0,
                                                 "planner": 0.0, "recognition": 0.0,
                                                 "speech": 0.0})
    assert zeroed["total"].detach().item() == 0.0
    assert unweighted["total"].detach().item() > 0.0
