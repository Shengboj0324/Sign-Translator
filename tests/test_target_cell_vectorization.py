"""Vector extraction preserves logical coordinate order, values and refusal."""
from itertools import product

import pytest
import torch

from signtranslator.training.target_cells import selected_target_cells, selected_continuous_target_cells


@pytest.mark.parametrize('dtype', [torch.bool, torch.int64, torch.float64])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'strided', 'expanded'])
@pytest.mark.parametrize('selection', ['none', 'all', 'mixed'])
def test_vector_extraction_matches_independent_logical_index_oracle(dtype, layout, selection):
    values = torch.arange(2 * 6 * 4).reshape(2, 6, 4)
    known = values.remainder(3) != 0
    if selection != 'mixed':
        known.fill_(selection == 'all')
    if dtype == torch.bool:
        values = values.remainder(2).bool()
    elif dtype == torch.float64:
        values = values.double() / 7 - 5
    if layout == 'transpose':
        values, known = values.transpose(1, 2), known.transpose(1, 2)
    elif layout == 'strided':
        values, known = values[:, ::2], known[:, ::2]
    elif layout == 'expanded':
        values, known = values[:1].expand(3, -1, -1), known[:1].expand(3, -1, -1)
    continuous = dtype == torch.float64
    result = (selected_continuous_target_cells(known, values, axes=('a', 'b'), unit='seconds')
              if continuous else selected_target_cells(known, values, axes=('a', 'b'), class_count=100))
    cast = float if continuous else int
    expected = tuple(tuple(index + (cast(values[(row,) + index]),)
                           for index in product(*(range(size) for size in known.shape[1:]))
                           if bool(known[(row,) + index])) for row in range(known.shape[0]))
    assert result.examples == expected


def test_float64_bit_patterns_and_unknown_nonfinite_values_are_preserved():
    values = torch.tensor([[-0., 5e-324, -5e-324, 1e308, float('nan')]], dtype=torch.float64)
    result = selected_continuous_target_cells(torch.tensor([[True, True, True, True, False]]),
                                              values, axes=('event',), unit='seconds')
    actual = torch.tensor([cell[-1] for cell in result.examples[0]], dtype=torch.float64)
    assert torch.equal(actual.view(torch.int64), values[0, :4].view(torch.int64))
    with pytest.raises(ValueError, match='finite float'):
        selected_continuous_target_cells(torch.ones_like(values, dtype=torch.bool), values,
                                         axes=('event',), unit='seconds')


def test_integer_values_are_not_rounded_through_float():
    value = 2**63 - 1
    result = selected_target_cells(torch.tensor([[True]]), torch.tensor([[value]]),
                                   axes=('event',), class_count=value + 1)
    assert result.examples == (((0, value),),)
