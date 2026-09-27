"""Exact independent-unit resolution bounds; no asymptotic approximations."""
import pytest
from fractions import Fraction
from signtranslator.eval_framework.design import paired_sign_resolution
from signtranslator.eval_framework.statistics import sign_test_pvalue


def test_resolution_matches_independent_exact_sign_oracle():
    for n in range(1,30):
        result=paired_sign_resolution(n)
        assert float(Fraction(result.minimum_p_numerator,result.minimum_p_denominator))==sign_test_pvalue([1.]*n,[0.]*n)


def test_six_components_cannot_support_two_endpoint_family_at_five_percent():
    one=paired_sign_resolution(6)
    two=paired_sign_resolution(6,primary_endpoints=2)
    assert one.resolution_possible and one.minimum_units_for_resolution==6
    assert not two.resolution_possible and two.minimum_units_for_resolution==7
    assert (two.minimum_p_numerator,two.minimum_p_denominator)==(1,32)
    heldout=paired_sign_resolution(4)
    assert not heldout.resolution_possible
    assert (heldout.minimum_p_numerator,heldout.minimum_p_denominator)==(1,8)
    assert not heldout.power_established


def test_boundary_and_no_evidence():
    assert paired_sign_resolution(4,family_alpha=.125).resolution_possible
    assert not paired_sign_resolution(0).resolution_possible


@pytest.mark.parametrize('kwargs',[{'independent_units':True},{'independent_units':-1},
 {'independent_units':1.5},{'independent_units':10001},{'independent_units':2,'primary_endpoints':0},
 {'independent_units':2,'family_alpha':float('nan')},{'independent_units':2,'family_alpha':1.}])
def test_invalid_design_controls(kwargs):
    with pytest.raises(ValueError):paired_sign_resolution(**kwargs)
