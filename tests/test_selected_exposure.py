"""Selected weights must not be attributed later optimizer-call declarations."""
from dataclasses import replace
import hashlib

import pytest

from signtranslator.governed_run import diagnose_governed_planner, load_governed_planner
from signtranslator.reproducibility import canonical_json_bytes
from signtranslator.training import checkpoint_paths
from signtranslator.training.trainer import Trainer
from test_governed_run import fixture, invoke


def diagnose(run, state):
    return diagnose_governed_planner(run, view='train', model_state=state,
        sample_indices=(0, 1), permutation=(1, 0), seed=9, max_samples=2).to_dict()


@pytest.mark.parametrize('cadence', [1, 2])
def test_best_boundary_excludes_later_exposure_and_retains_first_tie(tmp_path, monkeypatch, cadence):
    values = fixture(tmp_path)
    config = replace(values[-1], epochs=3 * cadence, val_every=cadence,
                     ckpt_path=str(tmp_path / 'run.pt'))
    values = (*values[:-1], config)
    actual_validate = Trainer.validate
    metrics = iter((1., 2., 1.))
    def controlled_selection(self):
        result = actual_validate(self)
        result[self.cfg.selection_metric] = next(metrics)
        return result
    monkeypatch.setattr(Trainer, 'validate', controlled_selection)
    run = invoke(values, max_epochs=cadence)
    early = diagnose(run, 'current')
    early_exposure = run.trainer.exposure_report()
    run.trainer.fit()  # Only the validation selection trajectory is controlled.
    late = diagnose(run, 'best_validation')
    current = diagnose(run, 'current')
    assert late['schema_version'] == 3
    assert late['completed_epochs'] == 3 * cadence
    assert late['global_step'] == 6 * cadence
    assert late['selected_training_boundary'] == dict(completed_epochs=cadence,
        global_step=2 * cadence, basis='first-minimum-validation-row')
    assert late['selected_boundary_exposure'] == early_exposure.to_dict()
    assert late['selected_boundary_exposure_sha256'] == early_exposure.sha256
    assert late['training_exposure_sha256'] == run.trainer.exposure_report().sha256
    assert late['selected_boundary_exposure_sha256'] != late['training_exposure_sha256']
    assert late['selected_relation_exposure'] == early['selected_relation_exposure']
    assert late['intervention'] == early['intervention']
    assert current['selected_training_boundary'] == dict(completed_epochs=3 * cadence,
        global_step=6 * cadence, basis='current-committed-cursor')
    assert current['selected_boundary_exposure_sha256'] == current['training_exposure_sha256']
    assert hashlib.sha256(canonical_json_bytes(late['selected_boundary_exposure'])).hexdigest() == late['selected_boundary_exposure_sha256']
    corpus, vocab, alphabet, model, _ = values
    loaded = load_governed_planner(corpus, vocab, alphabet, model_config=model,
        trainer_config=config, validation=True, shuffle=True,
        checkpoint_path=checkpoint_paths(config.ckpt_path)['last'])
    assert diagnose(loaded, 'best_validation') == late


@pytest.mark.parametrize('fault', ['best_value', 'epoch_rows', 'support_rows'])
def test_inconsistent_selection_history_is_refused_before_model_copy(tmp_path, monkeypatch, fault):
    run = invoke(fixture(tmp_path))
    if fault == 'best_value':
        run.trainer.best_val += 1
    elif fault == 'epoch_rows':
        run.trainer.history['val_sir_sequence_epoch'][0] = 2
    else:
        run.trainer.history['train_support_sir_sequence'][0] = 0
    def forbid(*args, **kwargs):
        raise AssertionError('invalid history reached model copy')
    monkeypatch.setattr('signtranslator.governed_run.deepcopy', forbid)
    with pytest.raises(ValueError, match='history'):
        diagnose(run, 'best_validation')


def test_selection_history_mutation_during_diagnostics_refuses_report(tmp_path, monkeypatch):
    import signtranslator.governed_run as module
    run = invoke(fixture(tmp_path))
    original = module.compare_source_intervention
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        run.trainer.history['val_sir_sequence'][0] += .5
        return result
    monkeypatch.setattr(module, 'compare_source_intervention', mutate)
    with pytest.raises(ValueError, match='selection history changed'):
        diagnose(run, 'current')


def test_missing_best_sentinel_cannot_change_to_nan_during_current_diagnostics(tmp_path, monkeypatch):
    import signtranslator.governed_run as module
    corpus, vocab, alphabet, model, config = fixture(tmp_path)
    run = module.run_governed_planner(corpus, vocab, alphabet, model_config=model,
        trainer_config=config, validation=False, shuffle=True)
    original = module.compare_source_intervention
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        run.trainer.best_val = float('nan')
        return result
    monkeypatch.setattr(module, 'compare_source_intervention', mutate)
    with pytest.raises(ValueError, match='selection history changed'):
        diagnose(run, 'current')
