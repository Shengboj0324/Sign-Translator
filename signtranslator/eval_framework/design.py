"""Finite-sample resolution checks for a declared paired sign-test design.

A resolution bound is not power, an independence proof, or scientific approval.
It cannot be improved by relabelling dependent clips as independent replicates.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict
from fractions import Fraction
import math
from numbers import Integral,Real


@dataclass(frozen=True)
class SignTestResolution:
    independent_units: int
    primary_endpoints: int
    family_alpha: float
    minimum_p_numerator: int
    minimum_p_denominator: int
    minimum_units_for_resolution: int
    resolution_possible: bool
    power_established: bool = False

    def to_dict(self):
        return asdict(self)


def paired_sign_resolution(independent_units: int, *, primary_endpoints: int = 1,
                           family_alpha: float = .05) -> SignTestResolution:
    """Best possible two-sided exact sign-test p-value: min(1, 2/2**n).

    Assumes all units are non-tied and in the same direction. Comparisons use
    exact rational arithmetic against the float alpha used by the runtime.
    n=0 has no evidence and is never attainable. A limit bounds integer memory.
    """
    for name,value,minimum in [('independent_units',independent_units,0),
                               ('primary_endpoints',primary_endpoints,1)]:
        if isinstance(value,bool) or not isinstance(value,Integral) or not minimum <= value <= 10000:
            raise ValueError(f'{name} must be an integer in [{minimum},10000]')
    if (isinstance(family_alpha,bool) or not isinstance(family_alpha,Real)
            or not math.isfinite(family_alpha) or not 0 < family_alpha < 1):
        raise ValueError('family_alpha must be finite and strictly between zero and one')
    n,m=int(independent_units),int(primary_endpoints)
    threshold=Fraction.from_float(float(family_alpha))/m
    minimum=Fraction(1,1 << max(n-1,0))
    needed=1
    while Fraction(1,1 << (needed-1)) > threshold:
        needed+=1
    return SignTestResolution(n,m,float(family_alpha),minimum.numerator,minimum.denominator,
                             needed,n>0 and minimum<=threshold)
