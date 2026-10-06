"""Production API integration on fictional admitted evidence, not ASL acceptance."""
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from signtranslator.config import TrainerConfig
from signtranslator.data.governed_text import PLAINTEXT_ENCODING
from signtranslator.governed_run import run_governed_planner
from signtranslator.planning.loci import LocusAlphabet
from signtranslator.planning.text_loci import LocusTextConfig
from signtranslator.training import checkpoint_paths
from test_loci import PAYLOAD
from test_text_loci import WEIGHTS
from test_text_sir_labels import setup


def fixture(tmp_path):
    vocab, corpus = setup(tmp_path, configurations=[{}, {}, {}, {'split': 'val'}, {'split': 'test'}],
                          convention_payload=PAYLOAD, loci=(0, None, 1))
    alphabet = LocusAlphabet(vocab.convention, PAYLOAD)
    model = LocusTextConfig(embedding_dim=8, hidden_dim=16, max_bytes=64, max_events=4,
                           lexicon_sha256=vocab.lexicon.sha256, convention_sha256=vocab.convention.sha256,
                           declared_encoding=PLAINTEXT_ENCODING, timing_scale_seconds=.5, locus_count=2)
    config = TrainerConfig(epochs=2, batch_size=2, loss_weights=WEIGHTS,
                           selection_metric='sir_sequence', seed=41)
    return corpus, vocab, alphabet, model, config


def invoke(values, **kwargs):
    corpus, vocab, alphabet, model, config = values
    return run_governed_planner(corpus, vocab, alphabet, model_config=model, trainer_config=config,
                                validation=True, shuffle=True, **kwargs)


def test_seed_controls_initialization_and_training_and_restores_caller_rng(tmp_path):
    values = fixture(tmp_path)
    torch.manual_seed(999)
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    first = invoke(values)
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()
    np.testing.assert_array_equal(before_numpy[1], np.random.get_state()[1])
    assert before_numpy[0] == np.random.get_state()[0] and before_numpy[2:] == np.random.get_state()[2:]
    torch.manual_seed(123)
    second = invoke(values)
    assert first.trainer.history == second.trainer.history
    assert first.trainer.optimizer_exposure == second.trainer.optimizer_exposure
    assert all(torch.equal(value, second.trainer.model.state_dict()[name])
               for name, value in first.trainer.model.state_dict().items())
    assert first.trainer.global_step == 4
    assert [len(r['sample_ids']) for r in first.trainer.optimizer_exposure] == [2, 1, 2, 1]
    assert not first.phase_exit_approved
    assert not first.exposure_audit.to_dict()['target_cell_audit']['contradictions']
    assert all(first.trainer.model.model_cfg == row.trainer.model.model_cfg for row in (first, second))


def test_shuffled_partial_run_exact_resume_matches_uninterrupted(tmp_path):
    values = fixture(tmp_path)
    full = invoke(values)
    config = replace(values[-1], ckpt_path=str(tmp_path / 'run.pt'))
    configured = (*values[:-1], config)
    partial = invoke(configured, max_epochs=1)
    assert partial.trainer.completed_epochs == 1
    paths = checkpoint_paths(config.ckpt_path)
    assert paths['last'].exists() and paths['best'].exists()
    resumed = invoke(configured, resume_from=paths['last'])
    assert resumed.trainer.history == full.trainer.history
    assert resumed.trainer.optimizer_exposure == full.trainer.optimizer_exposure
    assert all(torch.equal(value, resumed.trainer.model.state_dict()[name])
               for name, value in full.trainer.model.state_dict().items())


@pytest.mark.parametrize('fault', ['bytes', 'events', 'binding', 'weights', 'selection', 'source'])
def test_preflight_rejects_unusable_inputs_before_model_creation(tmp_path, monkeypatch, fault):
    values = fixture(tmp_path)
    corpus, vocab, alphabet, model, config = values
    if fault == 'bytes':
        model = replace(model, max_bytes=1)
    elif fault == 'events':
        model = replace(model, max_events=2)
    elif fault == 'binding':
        model = replace(model, lexicon_sha256='f' * 64)
    elif fault == 'weights':
        config = replace(config, loss_weights={'sir_sequence': 1.})
    elif fault == 'selection':
        config = replace(config, selection_metric='referent_equality')
    else:
        corpus._records[0].transcript_path.write_bytes(b'changed')
    def forbid(*args, **kwargs):
        raise AssertionError('model allocated before failed preflight')
    monkeypatch.setattr('signtranslator.governed_run.LocusTextSIRModel', forbid)
    with pytest.raises((ValueError, PermissionError)):
        invoke((corpus, vocab, alphabet, model, config))


def test_reserved_test_bytes_are_not_opened_and_validation_is_explicit(tmp_path):
    corpus, vocab, alphabet, model, config = fixture(tmp_path)
    corpus._records[-1].transcript_path.write_bytes(b'reserved partition must not be read')
    run = run_governed_planner(corpus, vocab, alphabet, model_config=model, trainer_config=config,
                               validation=False, shuffle=False, max_epochs=1)
    assert run.validation_support is None and run.trainer.val_loader is None
    assert not any(key.startswith('val_') for key in run.trainer.history)
    assert run.training_support.to_dict()['sample_count'] == 3


def test_failed_resume_restores_caller_rng(tmp_path):
    values = fixture(tmp_path)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="non-symlink regular file"):
        invoke(values, resume_from=tmp_path / 'missing.pt')
    assert torch.equal(before, torch.get_rng_state())
