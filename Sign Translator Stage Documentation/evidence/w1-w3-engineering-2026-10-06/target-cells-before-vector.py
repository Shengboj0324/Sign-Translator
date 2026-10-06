"""Immutable, loss-local target declarations, distinct from gradient evidence."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class TargetCells:
    axes: tuple[str, ...]
    class_count: int
    examples: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self):
        if (type(self.axes) is not tuple or not self.axes
                or any(type(axis) is not str or not axis.isidentifier() for axis in self.axes)
                or len(set(self.axes)) != len(self.axes)
                or type(self.class_count) is not int or self.class_count <= 0
                or type(self.examples) is not tuple):
            raise ValueError('invalid target-cell contract')
        for cells in self.examples:
            if type(cells) is not tuple:
                raise ValueError('immutable target-cell rows required')
            previous = None
            for cell in cells:
                if (type(cell) is not tuple or len(cell) != len(self.axes) + 1
                        or any(type(value) is not int or value < 0 for value in cell)
                        or cell[-1] >= self.class_count
                        or (previous is not None and cell[:-1] <= previous)):
                    raise ValueError('invalid, duplicate or unordered target cell')
                previous = cell[:-1]

    def to_dict(self):
        return dict(axes=list(self.axes), class_count=self.class_count,
                    examples=[[list(cell) for cell in cells] for cells in self.examples])

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict) or set(value) != {'axes', 'class_count', 'examples'}
                or not isinstance(value['axes'], list) or not isinstance(value['examples'], list)
                or any(not isinstance(cells, list) or any(not isinstance(cell, list) for cell in cells)
                       for cells in value['examples'])):
            raise ValueError('invalid serialized target cells')
        return cls(tuple(value['axes']), value['class_count'],
                   tuple(tuple(tuple(cell) for cell in cells) for cells in value['examples']))


@dataclass(frozen=True)
class ContinuousTargetCells:
    """Float64 target values in a declared unit, never categorical class IDs."""
    axes: tuple[str, ...]
    unit: str
    examples: tuple[tuple[tuple, ...], ...]

    def __post_init__(self):
        # Reuse coordinate/ordering validation without coercing real targets.
        if (type(self.examples) is not tuple or type(self.unit) is not str or not self.unit
                or any(type(cells) is not tuple for cells in self.examples)):
            raise ValueError('invalid continuous target-cell contract')
        for cells in self.examples:
            for cell in cells:
                if (type(cell) is not tuple or not cell or type(cell[-1]) is not float
                        or not math.isfinite(cell[-1])):
                    raise ValueError('continuous target cells require finite float values')
        TargetCells(self.axes, 1, tuple(tuple(cell[:-1] + (0,) for cell in cells)
                                      for cells in self.examples))

    def to_dict(self):
        return dict(axes=list(self.axes), unit=self.unit,
                    examples=[[list(cell) for cell in cells] for cells in self.examples])

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict) or set(value) != {'axes', 'unit', 'examples'}
                or not isinstance(value['axes'], list) or not isinstance(value['examples'], list)
                or any(not isinstance(cells, list) or any(not isinstance(cell, list) for cell in cells)
                       for cells in value['examples'])):
            raise ValueError('invalid serialized continuous target cells')
        return cls(tuple(value['axes']), value['unit'],
                   tuple(tuple(tuple(cell) for cell in cells) for cells in value['examples']))


def parse_target_cells(value):
    if isinstance(value, dict) and 'unit' in value:
        return ContinuousTargetCells.from_dict(value)
    return TargetCells.from_dict(value)


def selected_continuous_target_cells(known, values, *, axes, unit):
    if (known.dtype != torch.bool or known.shape != values.shape
            or known.ndim != len(axes) + 1 or values.dtype != torch.float64):
        raise ValueError('typed float64 continuous targets required')
    mask, targets = known.detach().cpu(), values.detach().cpu()
    rows = []
    for row, target in zip(mask, targets, strict=True):
        rows.append(tuple(tuple(index) + (float(target[tuple(index)]),)
                          for index in row.nonzero(as_tuple=False).tolist()))
    return ContinuousTargetCells(axes, unit, tuple(rows))


def selected_target_cells(known, labels, *, axes, class_count):
    """Snapshot exactly the known coordinates used by a loss reduction."""
    if (known.dtype != torch.bool or known.shape != labels.shape
            or known.ndim != len(axes) + 1
            or labels.dtype not in (torch.bool, torch.int64)):
        raise ValueError('typed target-cell mask and labels required')
    mask, values = known.detach().cpu(), labels.detach().cpu()
    rows = []
    for row, targets in zip(mask, values, strict=True):
        coordinates = row.nonzero(as_tuple=False).tolist()
        rows.append(tuple(tuple(index) + (int(targets[tuple(index)]),) for index in coordinates))
    return TargetCells(axes, class_count, tuple(rows))
