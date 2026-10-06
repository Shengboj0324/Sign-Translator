import pytest
import torch
from torch import nn

from signtranslator.inference_context import preserving_eval_mode
from signtranslator.planning.source_intervention import compare_source_intervention
from test_source_intervention import fixture


def modes(model):
    return tuple(module.training for module in model.modules())


@pytest.mark.parametrize('method', ['generate', 'generate_temporal', 'generate_relational',
                                    'generate_referents', 'generate_loci', 'intervention'])
@pytest.mark.parametrize('root_training', [True, False])
def test_nested_generation_preserves_mixed_modes_on_success_and_failure(tmp_path, monkeypatch,
                                                                     method, root_training):
    model, view = fixture(tmp_path)
    model.train(root_training)
    model.encoder.train(not root_training)
    model.classifier.train(not root_training)
    before = modes(model)
    observed = []
    actual = model._source

    def source(*args, **kwargs):
        observed.append(modes(model))
        return actual(*args, **kwargs)

    monkeypatch.setattr(model, '_source', source)

    def invoke():
        if method == 'intervention':
            return compare_source_intervention(model, view, permutation=(1, 0), seed=9, max_samples=2)
        kwargs = {} if method == 'generate' else {'origins_seconds': torch.tensor([0.], dtype=torch.float64)}
        return getattr(model, method)(torch.tensor([[66]]), torch.tensor([1]), **kwargs)

    invoke()
    assert modes(model) == before
    assert observed and all(not any(state) for state in observed)

    def fail(*args, **kwargs):
        assert not any(modes(model))
        raise RuntimeError('nested source failure')

    monkeypatch.setattr(model, '_source', fail)
    with pytest.raises(RuntimeError, match='nested source failure'):
        invoke()
    assert modes(model) == before


def test_partial_eval_entry_failure_restores_existing_flags():
    class EntryFailure(nn.Sequential):
        def train(self, mode=True):
            super().train(mode)
            if not mode:
                raise RuntimeError('entry failure')
            return self

    model = EntryFailure(nn.Linear(2, 2), nn.Dropout())
    model[0].eval()
    before = modes(model)
    with pytest.raises(RuntimeError, match='entry failure'):
        with preserving_eval_mode(model):
            pytest.fail('entry should fail before yielding')
    assert modes(model) == before
