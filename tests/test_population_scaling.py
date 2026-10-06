"""Forward/backward range checks and independent analytic derivatives."""
from fractions import Fraction

import pytest
import torch

from signtranslator.training.objectives import SupportedObjective, SupportedTerm
from signtranslator.training.scaling import weighted_population_scalar


@pytest.mark.parametrize('dtype,tiny', [(torch.float32, 2.**-149), (torch.float64, 2.**-1074)])
def test_population_weight_cancellation_preserves_smallest_subnormal(dtype, tiny):
    value = torch.tensor(tiny, dtype=dtype, requires_grad=True)
    objective = SupportedObjective({'a': SupportedTerm(value, 1)}, {'a': 2.}, 2)
    loss = objective.total()
    assert loss.item() == tiny
    loss.backward()
    assert value.grad.item() == 1.


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_scaled_forward_and_incoming_gradient_avoid_intermediate_range_loss(dtype):
    maximum = torch.finfo(dtype).max
    # The tiny coefficient is not representable in float32 but its product is.
    weight = 2.**-160 if dtype == torch.float32 else 2.**-1074
    population = 2
    value = torch.tensor(maximum, dtype=dtype, requires_grad=True)
    result = weighted_population_scalar(value, weight, population)
    expected = float(Fraction(maximum) * Fraction(weight) / population)
    assert result.item() == torch.tensor(expected, dtype=dtype).item()
    result.backward(torch.tensor(maximum, dtype=dtype))
    assert value.grad.item() == torch.tensor(expected, dtype=dtype).item()


def test_large_weight_small_numerator_and_zero_remain_finite():
    for value in (0., 2.**-1074):
        tensor = torch.tensor(value, dtype=torch.float64)
        result = weighted_population_scalar(tensor, 2.**1023, 2)
        assert result.item() == float(Fraction(value) * Fraction(2.**1023) / 2)


def test_first_and_second_order_derivatives():
    value = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)
    fn = lambda x: weighted_population_scalar(x.square(), .7, 3)
    assert torch.autograd.gradcheck(fn, (value,))
    assert torch.autograd.gradgradcheck(fn, (value,))
    first, = torch.autograd.grad(fn(value), value, create_graph=True)
    second, = torch.autograd.grad(first, value)
    assert first.item() == pytest.approx(2 * 1.25 * .7 / 3)
    assert second.item() == pytest.approx(2 * .7 / 3)


def test_true_final_overflow_is_still_rejected():
    value = torch.tensor(torch.finfo(torch.float64).max, dtype=torch.float64)
    objective = SupportedObjective({'a': SupportedTerm(value, 1)}, {'a': 2.}, 1)
    with pytest.raises(ValueError, match='objective overflow'):
        objective.total()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_fraction_reference_across_binary_ranges(dtype):
    import math
    info = torch.finfo(dtype)
    tiny = 2.**(-149 if dtype == torch.float32 else -1074)
    for value in (0., tiny, info.tiny, 1., info.max):
        for weight in (.1, .7, 2.**-1074, 2.**1023):
            for population in (1, 3, 7, 2**64):
                exact = Fraction(value) * Fraction(weight) / population
                try:
                    rounded = float(exact)
                except OverflowError:
                    rounded = math.inf
                expected = torch.tensor(rounded, dtype=dtype)
                observed = weighted_population_scalar(torch.tensor(value, dtype=dtype), weight, population)
                if not bool(torch.isfinite(expected)):
                    assert observed.item() == expected.item()
                else:
                    toward = torch.tensor(0. if expected.item() else 1., dtype=dtype)
                    ulp = abs(expected.item() - torch.nextafter(expected, toward).item())
                    # Mantissa conversion/product still round; no exact-rounding claim.
                    assert abs(observed.item() - expected.item()) <= 2 * ulp


def test_negative_incoming_gradient_uses_the_same_stable_scale():
    value = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    result = weighted_population_scalar(value, 2.**-1074, 2)
    result.backward(torch.tensor(-torch.finfo(torch.float64).max, dtype=torch.float64))
    expected = float(-Fraction(torch.finfo(torch.float64).max) * Fraction(2.**-1074) / 2)
    assert value.grad.item() == expected


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
@pytest.mark.parametrize('value,weight,population', [(1.25, .7, 3), (2.**-149, 2., 2)])
def test_mps_scalar_fallback_preserves_values_and_gradients(value, weight, population):
    x = torch.tensor(value, dtype=torch.float32, device='mps', requires_grad=True)
    y = weighted_population_scalar(x, weight, population)
    y.backward()
    assert y.device.type == 'mps' and x.grad.device.type == 'mps'
    expected = torch.tensor(float(Fraction(value)*Fraction(weight)/population), dtype=torch.float32)
    assert y.item() == expected.item()
    assert x.grad.item() == torch.tensor(weight/population,dtype=torch.float32).item()


@pytest.mark.parametrize('dtype,tiny', [(torch.float32,2.**-149),(torch.float64,2.**-1074)])
def test_multiple_subnormal_contributions_are_combined_before_rounding(dtype,tiny):
    a = torch.tensor(tiny,dtype=dtype,requires_grad=True)
    b = torch.tensor(tiny,dtype=dtype,requires_grad=True)
    loss = SupportedObjective({'a':SupportedTerm(a,1),'b':SupportedTerm(b,1)},
                              {'a':1.,'b':1.},2).total()
    assert loss.item() == tiny
    loss.backward()
    assert a.grad.item() == b.grad.item() == .5


def test_joint_loss_mixed_dtype_zero_branch_and_higher_derivatives():
    from signtranslator.training.scaling import weighted_population_sum
    x = torch.tensor(1.25,dtype=torch.float64,requires_grad=True)
    fn = lambda a: weighted_population_sum((a.square(),a**3),(.7,.2),3)
    assert torch.autograd.gradcheck(fn,(x,))
    assert torch.autograd.gradgradcheck(fn,(x,))
    zero = torch.tensor(0.,dtype=torch.float32,requires_grad=True)
    tiny = torch.tensor(2.**-1074,dtype=torch.float64,requires_grad=True)
    result = weighted_population_sum((zero,tiny),(2.**1023,1.),1)
    assert result.dtype == torch.float64 and result.item() == 2.**-1074


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_mps_joint_sum_preserves_combined_subnormal():
    a = torch.tensor(2.**-149,dtype=torch.float32,device='mps',requires_grad=True)
    b = torch.tensor(2.**-149,dtype=torch.float32,device='mps',requires_grad=True)
    loss = SupportedObjective({'a':SupportedTerm(a,1),'b':SupportedTerm(b,1)},
                              {'a':1.,'b':1.},2).total()
    loss.backward()
    assert loss.item() == 2.**-149 and a.grad.item() == b.grad.item() == .5


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('support', [2**64, 2**128, 2**1024, 2**8192])
def test_branch_mean_large_support_matches_exact_reporting_and_gradient(dtype, support):
    from signtranslator.training.objectives import ObjectiveAccumulator

    maximum = torch.finfo(dtype).max
    value = torch.tensor(maximum, dtype=dtype, requires_grad=True)
    objective = SupportedObjective({'a': SupportedTerm(value, support)}, {'a': 1}, support)
    mean = objective['a']
    exact = Fraction(maximum) / support
    expected = torch.tensor(float(exact), dtype=dtype)
    assert mean.dtype == dtype and mean.item() == expected.item()
    assert objective.total().item() == mean.item()
    # A large incoming derivative makes some otherwise unrepresentable inverse
    # counts produce representable gradients; do not materialize 1/support first.
    mean.backward(torch.tensor(maximum, dtype=dtype))
    assert value.grad.item() == expected.item()
    accumulator = ObjectiveAccumulator()
    accumulator.add(objective)
    assert accumulator.result()['a'] == float(exact)


def test_branch_mean_uses_support_not_full_population_or_branch_weight():
    value = torch.tensor(18., dtype=torch.float64, requires_grad=True)
    objective = SupportedObjective({'a': SupportedTerm(value, 3)}, {'a': 7}, 10)
    assert objective['a'].item() == 6.
    assert objective.total().item() == pytest.approx(12.6)
    objective['a'].backward()
    assert value.grad.item() == pytest.approx(1 / 3)


def test_branch_mean_preserves_higher_order_derivatives():
    value = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)

    def branch_mean(x):
        return SupportedObjective({'a': SupportedTerm(x.square(), 3)}, {'a': 7}, 10)['a']

    assert torch.autograd.gradcheck(branch_mean, (value,))
    assert torch.autograd.gradgradcheck(branch_mean, (value,))
