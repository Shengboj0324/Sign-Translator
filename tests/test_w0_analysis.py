"""W0 evaluation adversaries: ragged support, insertions, caps and RNG isolation."""
import random

import numpy as np
import pytest
import torch

from test_analysis import _model_and_loader
from signtranslator.analysis import analyze
from signtranslator.analysis.observations import observations
from signtranslator.data.corpus import collate_corpus


def _ragged(tmp_path):
    model, loader = _model_and_loader(tmp_path)
    samples = [loader.dataset[i] for i in range(3)]
    for i, sample in enumerate(samples):
        sample['concepts'] = sample['concepts'][:i+1]
        sample['src_concepts'] = sample['src_concepts'][:i+1]
        sample['motion_length'] = 9 + i * 3
        sample['pose'] = sample['pose'][:, :sample['motion_length']]
        sample['speech_length'] = 10 + i * 4
        sample['speech'] = sample['speech'][:sample['speech_length']]
    return model, samples


def test_ragged_analysis_is_batch_partition_and_padding_invariant(tmp_path):
    model, samples = _ragged(tmp_path)
    combined = collate_corpus(samples)
    # Padding must not reach a convolution, pooled embedding or acoustic decoder.
    for i, sample in enumerate(samples):
        combined['pose'][i, :, sample['motion_length']:] = float('nan')
        combined['speech'][i, sample['speech_length']:] = float('nan')
    single = [collate_corpus([s]) for s in samples]
    a = analyze(model, [combined], cycle_subset=2, ddim_steps=2, seed=81)
    b = analyze(model, single, cycle_subset=2, ddim_steps=2, seed=81)
    assert a.metrics == b.metrics
    assert a.protocol['observations'] == 3
    assert a.protocol['speech_observations'] == 3


def test_analysis_restores_rng_and_individual_module_modes_on_success_and_failure(tmp_path):
    model, samples = _ragged(tmp_path)
    model.train()
    model.planner.eval()
    modes = [m.training for m in model.modules()]
    torch.manual_seed(49); random.seed(49); np.random.seed(49)
    ts, ps, ns = torch.get_rng_state(), random.getstate(), np.random.get_state()
    for loader in ([collate_corpus(samples)], []):
        if loader:
            analyze(model, loader, cycle_subset=1, ddim_steps=2)
        else:
            with pytest.raises(ValueError, match='nonempty'):
                analyze(model, loader)
        assert torch.equal(ts, torch.get_rng_state())
        assert ps == random.getstate()
        now = np.random.get_state()
        assert ns[0] == now[0] and np.array_equal(ns[1], now[1]) and ns[2:] == now[2:]
        assert modes == [m.training for m in model.modules()]


@pytest.mark.parametrize('extra,ended', [(True, True), (False, False), (False, True)])
def test_planner_insertions_and_unterminated_outputs_cannot_pass(tmp_path, monkeypatch, extra, ended):
    model, loader = _model_and_loader(tmp_path)
    batch = collate_corpus([loader.dataset[0]])
    reference = batch['gloss_tokens'][0].tolist()
    monkeypatch.setattr(model.planner, 'greedy_decode',
                        lambda *a, **k: ([reference + ([3] if extra else [])], [ended]))
    report = analyze(model, [batch], thresholds={'planner_token_accuracy':1.0},
                     cycle_subset=1, ddim_steps=2)
    assert report.checks['planner_token_accuracy'] == (not extra and ended)
    assert report.metrics['planner_wer'] == (1/len(reference) if extra else 0)
    assert report.metrics['planner_exact_match'] == float(not extra and ended)


def test_long_reference_is_scored_and_capacity_is_explicit(tmp_path, monkeypatch):
    model, loader = _model_and_loader(tmp_path)
    sample = loader.dataset[0]
    sample['concepts'] = torch.arange(10) % 6
    sample['src_concepts'] = sample['concepts'].clone()
    sample.pop('speech'); sample.pop('speech_length')
    batch = collate_corpus([sample])
    reference = batch['gloss_tokens'][0].tolist()
    seen = []
    def decode(src, max_len, return_status):
        seen.append(max_len)
        return [reference], [True]
    monkeypatch.setattr(model.planner, 'greedy_decode', decode)
    report = analyze(model, [batch], cycle_subset=1, ddim_steps=2, planner_max_len=16)
    assert seen == [16] and report.metrics['planner_exact_match'] == 1
    with pytest.raises(ValueError, match='cap'):
        analyze(model, [batch], planner_max_len=8)


def test_observation_target_slicing_uses_speech_not_gloss_lengths(tmp_path):
    _, samples = _ragged(tmp_path)
    samples[0]['src_concepts'] = torch.tensor([1, 2, 3, 4])
    batch = collate_corpus(samples)
    rows = list(observations(batch))
    assert rows[0]['speech_ctc_targets'].tolist() == [2, 3, 4, 5]
    assert len(rows[0]['ctc_targets']) == 1
    assert rows[1]['speech_ctc_targets'].tolist() == (samples[1]['src_concepts']+1).tolist()


@pytest.mark.parametrize('cap', [0, -1, 513, True, 1.5])
def test_planner_invalid_decode_capacity(tmp_path, cap):
    model, loader = _model_and_loader(tmp_path)
    with pytest.raises(ValueError, match='capacity'):
        model.planner.greedy_decode(next(iter(loader))['src'], max_len=cap)


def test_validation_macro_observations_are_partition_invariant(tmp_path):
    from signtranslator.training import Trainer
    from signtranslator import TrainerConfig
    model, samples = _ragged(tmp_path)
    from torch.utils.data import DataLoader
    one = DataLoader(samples, batch_size=3, collate_fn=collate_corpus)
    split = DataLoader(samples, batch_size=2, collate_fn=collate_corpus)
    trainer = Trainer(model, TrainerConfig(epochs=1), one, one)
    first = trainer.validate()
    trainer.val_loader = split
    second = trainer.validate()
    assert first == second
    # Independent per-observation oracle for the declared generation estimand.
    from signtranslator.reproducibility import isolated_deterministic_rng
    from signtranslator.training.trainer import VALIDATION_SEED_OFFSET
    model.eval()
    with torch.no_grad(), isolated_deterministic_rng(VALIDATION_SEED_OFFSET):
        losses = [float(model.training_step(row)['generation'])
                  for row in observations(next(iter(one)))]
    # Separate float64 evaluation paths may differ by a final rounding bit.
    assert first['generation'] == pytest.approx(sum(losses)/3, rel=1e-12, abs=1e-12)


def test_checkpoint_selection_uses_declared_branch_not_weighted_total(tmp_path, monkeypatch):
    from test_trainer import _tiny_setup
    from signtranslator import TrainerConfig
    from signtranslator.training import Trainer
    model, train, val = _tiny_setup(tmp_path)
    trainer = Trainer(model, TrainerConfig(epochs=2, selection_metric='generation'), train, val)
    results = iter([{'generation':1., 'total':100.}, {'generation':2., 'total':0.}])
    monkeypatch.setattr(trainer, 'validate', lambda: next(results))
    monkeypatch.setattr(trainer, 'train_epoch', lambda: {'total':0.})
    trainer.fit()
    assert trainer.best_val == 1.
    assert trainer.best_model_state is not None


def test_empty_analysis_report_never_passes():
    from signtranslator.analysis import AnalysisReport
    assert not AnalysisReport({}, {}).passed


def test_pipeline_analyzes_best_artifact_and_retains_final_weights(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from signtranslator.data.corpus import CorpusSpec, generate_corpus
    from signtranslator.training import checkpoint_paths
    from signtranslator.reproducibility import sha256_file
    import signtranslator.run as entry
    corpus = tmp_path/'corpus'
    generate_corpus(str(corpus), CorpusSpec.build(4, 2, 27, 3, 8),
                    counts={'train':8, 'val':4})
    captured = {}
    def inspect(model, loader, **kwargs):
        captured['state'] = {k:v.clone() for k,v in model.state_dict().items()}
        captured.update(kwargs)
        return SimpleNamespace(protocol={})
    monkeypatch.setattr(entry, 'analyze', inspect)
    prefix = tmp_path/'weights.pt'
    result = entry.run_pipeline(str(corpus), epochs=2, batch_size=4, diff_timesteps=10,
                                ckpt_path=str(prefix), require_ready=False, verbose=False)
    paths = checkpoint_paths(prefix)
    best = torch.load(paths['best'], weights_only=False)
    last = torch.load(paths['last'], weights_only=False)
    for key in best['model']:
        assert torch.equal(captured['state'][key], best['model'][key])
        assert torch.equal(result['model'].state_dict()[key], last['model'][key])
    assert sha256_file(paths['best']) in captured['checkpoint_identity']
    assert result['report'].protocol['selection_metric'] == 'generation'
    assert result['trainer'].global_step == last['training_state']['global_step']
    assert result['trainer'].history == last['training_state']['history']
    result['trainer'].save(tmp_path / 'after-analysis.pt')
