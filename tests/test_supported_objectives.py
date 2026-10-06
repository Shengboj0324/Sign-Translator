"""Population objective versus conditional branch metrics, including absent support."""
import math

import pytest
import torch

from signtranslator.training.objectives import SupportedObjective, SupportedTerm, ObjectiveAccumulator


def objective(a, b, support, population):
    return SupportedObjective({'a': SupportedTerm(torch.tensor(a, dtype=torch.float64), population),
                               'b': SupportedTerm(torch.tensor(b, dtype=torch.float64), support)},
                              {'a': 1., 'b': 2.}, population)


def test_population_normalization_and_report_are_partition_invariant():
    # Three examples: a losses [1,2,3], b losses [unknown,4,unknown].
    combined = objective(6, 4, 1, 3)
    assert combined.total().item() == pytest.approx(14/3)
    assert combined['a'].item() == 2. and combined['b'].item() == 4.
    accumulator = ObjectiveAccumulator()
    accumulator.add(objective(3, 4, 1, 2))
    accumulator.add(objective(3, 0, 0, 1))
    assert accumulator.result() == pytest.approx({'a': 2., 'b': 4., 'total': 14/3})
    assert accumulator.support == {'a': 3, 'b': 1}
    assert accumulator.population == 3


def test_unavailable_branch_has_no_metric_or_autograd_path():
    a = torch.nn.Parameter(torch.tensor(2.))
    b = torch.nn.Parameter(torch.tensor(3.))
    result = SupportedObjective({'a': SupportedTerm(a.square(), 1),
                                 'b': SupportedTerm(b.square()*0, 0)}, {'a': 1., 'b': 1.}, 1)
    result.total().backward()
    assert a.grad.item() == 4. and b.grad is None
    with pytest.raises(KeyError, match='unavailable'):
        result['b']
    accumulator = ObjectiveAccumulator()
    accumulator.add(result)
    assert 'b' not in accumulator.result() and accumulator.support['b'] == 0


@pytest.mark.parametrize('count,value', [(True, 0.), (-1, 0.), (2, 0.), (0, 1.),
                                       (1, -1.), (1, math.inf), (1, math.nan)])
def test_invalid_support_or_numerators_fail(count, value):
    with pytest.raises(ValueError):
        SupportedObjective({'a': SupportedTerm(torch.tensor(value), count)}, {'a': 1.}, 1)


def test_no_observed_objective_and_mutated_tensors_do_not_report_success():
    result = SupportedObjective({'a': SupportedTerm(torch.tensor(0.), 0)}, {'a': 1.}, 1)
    with pytest.raises(ValueError, match='unavailable'):
        result.total()
    accumulator = ObjectiveAccumulator()
    accumulator.add(result)
    with pytest.raises(ValueError, match='unavailable'):
        accumulator.result()
    good = objective(1., 1., 1, 1)
    good.terms['a'].example_loss_sum.fill_(float('nan'))
    with pytest.raises(ValueError):
        good.total()
