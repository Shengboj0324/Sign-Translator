"""Explicit support for joint objectives with partially observed branches.

Optimization averages weighted observed contributions over the full batch.
Reporting averages each branch over its supported examples only. A zero support
count is unavailable, never a measured zero loss or a reason to drop the sample.
"""
from __future__ import annotations

from fractions import Fraction
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import torch

from .scaling import weighted_population_scalar, weighted_population_sum


@dataclass(frozen=True)
class SupportedTerm:
    example_loss_sum: torch.Tensor
    supported_examples: int
    support_mask: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class SupportedObjective:
    terms: Mapping[str, SupportedTerm]
    weights: Mapping[str, float]
    population_size: int

    def __post_init__(self):
        object.__setattr__(self, 'terms', MappingProxyType(dict(self.terms)))
        object.__setattr__(self, 'weights', MappingProxyType(dict(self.weights)))
        self.validate()

    def validate(self):
        if type(self.population_size) is not int or self.population_size <= 0:
            raise ValueError('objective population size must be positive integer')
        if not self.terms or self.terms.keys() != self.weights.keys():
            raise ValueError('objective terms and weights must have identical nonempty keys')
        devices = set()
        for name, term in self.terms.items():
            if (not isinstance(name, str) or not name.isascii() or not name.isidentifier()
                    or name in {'total', 'lr'} or name.startswith('support_') or name.endswith('_epoch')):
                raise ValueError('invalid or reserved objective branch name')
            if not isinstance(term, SupportedTerm):
                raise ValueError('typed supported objective terms required')
            count, value = term.supported_examples, term.example_loss_sum
            if type(count) is not int or not 0 <= count <= self.population_size:
                raise ValueError('branch support must be an integer within the population')
            if term.support_mask is not None and (
                    type(term.support_mask) is not tuple
                    or len(term.support_mask) != self.population_size
                    or any(type(flag) is not bool for flag in term.support_mask)
                    or sum(term.support_mask) != count):
                raise ValueError('support mask must bind every example and match support count')
            if (not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64)
                    or value.ndim != 0 or not bool(torch.isfinite(value)) or bool(value < 0)):
                raise ValueError('branch loss sum must be a finite nonnegative float scalar')
            if count == 0 and bool(value != 0):
                raise ValueError('unsupported branch must have an exact zero bookkeeping sum')
            devices.add(value.device)
            weight = self.weights[name]
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
                raise ValueError('objective weights must be finite and positive')
        if len(devices) != 1:
            raise ValueError('objective branches must share one device')

    def total(self) -> torch.Tensor:
        self.validate()  # tensor storage remains mutable even in a frozen dataclass
        active = [(term.example_loss_sum, self.weights[name])
                  for name, term in self.terms.items() if term.supported_examples > 0]
        if not active:
            raise ValueError('joint objective unavailable: no supported branches')
        # Do not connect an entirely unsupported branch to the autograd graph.
        # Otherwise zero gradients can still trigger Adam momentum/weight decay.
        result = weighted_population_sum(tuple(value for value, _ in active),
                                         tuple(weight for _, weight in active), self.population_size)
        if not bool(torch.isfinite(result)):
            raise ValueError('joint objective overflow')
        return result

    def __getitem__(self, name: str) -> torch.Tensor:
        if name == 'total':
            return self.total()
        self.validate()
        term = self.terms[name]
        if term.supported_examples == 0:
            raise KeyError(f'{name} is unavailable: zero supported examples')
        return weighted_population_scalar(term.example_loss_sum, 1, term.supported_examples)


class ObjectiveAccumulator:
    """Epoch numerators and denominators; no average of partial-batch means."""

    def __init__(self):
        self.sums: dict[str, list[float]] = {}
        self.support: dict[str, int] = {}
        self.population = 0
        self.weights = None

    def add(self, objective: SupportedObjective):
        objective.validate()
        if self.weights is None:
            self.weights = dict(objective.weights)
            self.sums = {name: [] for name in objective.terms}
            self.support = {name: 0 for name in objective.terms}
        elif self.weights != objective.weights or self.sums.keys() != objective.terms.keys():
            raise ValueError('objective branches/weights changed within epoch')
        for name, term in objective.terms.items():
            self.sums[name].append(float(term.example_loss_sum.detach()))
            self.support[name] += term.supported_examples
        self.population += objective.population_size

    def result(self) -> dict[str, float]:
        if self.population == 0:
            raise ValueError('cannot report an empty objective population')
        if not any(self.support.values()):
            raise ValueError('joint objective unavailable: no supported branches')
        try:
            # Reporting is outside autograd. Exact binary-rational accumulation
            # avoids overflowing a numerator whose normalized mean is finite,
            # and avoids rounding away tiny contributions before weighting.
            sums = {name: sum(map(Fraction, parts), Fraction())
                    for name, parts in self.sums.items()}
            means = {name: float(value / self.support[name]) for name, value in sums.items()
                     if self.support[name] > 0}
            means['total'] = float(sum((Fraction(self.weights[name]) * value / self.population
                                        for name, value in sums.items()), Fraction()))
        except OverflowError as error:
            raise FloatingPointError('epoch objective aggregation overflow') from error
        if any(not math.isfinite(value) for value in means.values()):
            raise FloatingPointError('nonfinite epoch objective aggregation')
        return means
