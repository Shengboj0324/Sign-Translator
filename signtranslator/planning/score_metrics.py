"""Numerical reductions shared by nonnegative diagnostic scores."""
import math


def nonnegative_mean(values):
    """Mean of finite nonnegative values without overflowing their raw sum.

    Callers construct/validate the score domain. An empty collection is
    unavailable; normalization preserves equal subnormal values without dividing
    each tiny contribution by the population size before accumulation.
    """
    if not values:
        return None
    scale = max(values)
    if scale == 0:
        return 0.
    result = scale * (math.fsum(value / scale for value in values) / len(values))
    if not math.isfinite(result):
        raise ValueError('diagnostic mean overflow')
    return result
