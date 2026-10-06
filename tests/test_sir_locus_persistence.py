from itertools import permutations

import pytest

from signtranslator.grammar.sir import EventKind, SIREvent, SIRGraph, sir_to_dict, validate_sir
from test_candidate_graph import fixture, run


def graph(assignments):
    return SIRGraph([SIREvent(i, EventKind.MANUAL, 10, float(i), float(i + 1), referent=ref, locus=locus)
                     for i, (ref, locus) in enumerate(assignments)])


@pytest.mark.parametrize('capacity', [None, 3])
def test_persistence_and_collision_do_not_depend_on_supplied_capacity(capacity):
    for assignments, code in [([(0, 0), (0, 1)], 'referent_locus_changed'),
                              ([(0, 0), (1, 0)], 'locus_collision')]:
        g = graph(assignments)
        assert code in validate_sir(g, num_loci=capacity)
        with pytest.raises(ValueError, match=code):
            sir_to_dict(g)


def test_unknowns_do_not_erase_or_invent_assignments_and_order_does_not_matter():
    assignments = [(0, 0), (None, 0), (0, None), (0, 1), (1, 0)]
    for ordering in permutations(assignments):
        violations = validate_sir(graph(ordering))
        assert 'locus_collision' in violations and 'referent_locus_changed' in violations
    assert validate_sir(graph([(0, 0), (None, 0), (0, None), (1, 1), (0, 0)]), num_loci=2) == []
    assert validate_sir(graph([(None, 0), (None, 1)])) == []


def test_numeric_invalidity_does_not_alias_boolean_ids_or_mask_range_errors():
    assert 'invalid_referent' in validate_sir(graph([(True, 0)]))
    assert 'invalid_locus' in validate_sir(graph([(0, True)]))
    violations = validate_sir(graph([(0, 0), (0, 3)]), num_loci=2)
    assert 'locus_out_of_range' in violations and 'referent_locus_changed' in violations


def test_direct_candidate_assembly_cannot_bypass_persistence_solver():
    vocabulary, candidate, decisions = fixture()
    result = run(vocabulary, candidate, decisions, referents=(0, 0), loci=(0, 1))
    assert result.graph_json is None and 'referent_locus_changed' in result.violations
