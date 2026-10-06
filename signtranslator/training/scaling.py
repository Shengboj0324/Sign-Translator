"""Range-safe differentiable scaling of validated floating loss scalars."""
from fractions import Fraction

import torch


class _BinaryScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, mantissa, exponent):
        ctx.mantissa, ctx.exponent = mantissa, exponent
        # This runtime lacks MPS frexp. Move only this scalar operation to CPU;
        # the custom backward uses the same path and returns to the input device.
        work = value.to('cpu') if value.device.type == 'mps' else value
        fraction, power = torch.frexp(work)
        return torch.ldexp(fraction * mantissa, power + exponent).to(value.device)

    @staticmethod
    def backward(ctx, gradient):
        # Use the same stable linear map instead of materializing a coefficient
        # that can underflow/overflow before multiplication by the incoming grad.
        # Applying this function also preserves higher-order differentiation.
        return _BinaryScale.apply(gradient, ctx.mantissa, ctx.exponent), None, None


def weighted_population_scalar(value, weight, population):
    """Scale a validated float32/64 scalar by positive weight/population.

    The exact binary-rational coefficient is normalized before conversion to a
    bounded floating mantissa. Final tensor arithmetic still rounds in its dtype;
    this prevents intermediate range loss, not all floating-point rounding.
    """
    mantissa, exponent = _coefficient_parts(weight, population)
    return _BinaryScale.apply(value, mantissa, exponent)


def _coefficient_parts(weight, population):
    coefficient = Fraction(weight) / population
    numerator, denominator = coefficient.numerator, coefficient.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    mantissa = (Fraction(numerator, denominator << exponent) if exponent >= 0 else
                Fraction(numerator << -exponent, denominator))
    if mantissa >= 1:
        mantissa /= 2
        exponent += 1
    # Below this exponent every finite float64 product rounds to zero. Clamping
    # avoids integer exponent overflow for arbitrarily large Python populations.
    exponent = max(exponent, -4096)
    return float(mantissa), exponent


class _WeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coefficients, *values):
        ctx.coefficients = coefficients
        ctx.dtypes = tuple(value.dtype for value in values)
        dtype = values[0].dtype
        for value in values[1:]:
            dtype = torch.promote_types(dtype, value.dtype)
        device = values[0].device
        work_device = 'cpu' if device.type == 'mps' else device
        terms = []
        powers = []
        for value, (mantissa, exponent) in zip(values, coefficients):
            fraction, power = torch.frexp(value.to(device=work_device, dtype=dtype))
            fraction, power = fraction * mantissa, power + exponent
            terms.append((fraction, power))
            if bool(fraction != 0):
                powers.append(int(power))
        if not powers:
            return torch.zeros((), dtype=dtype, device=device)
        common = max(powers)
        # All contributions are nonnegative. Normalize before summation so
        # separately subnormal terms can combine into a representable result.
        total = torch.zeros((), dtype=dtype, device=work_device)
        for fraction, power in terms:
            total = total + torch.ldexp(fraction, power - common)
        return torch.ldexp(total, torch.tensor(common, device=work_device)).to(device)

    @staticmethod
    def backward(ctx, gradient):
        return (None, *(_BinaryScale.apply(gradient, mantissa, exponent).to(dtype)
                        for (mantissa, exponent), dtype in zip(ctx.coefficients, ctx.dtypes)))


def weighted_population_sum(values, weights, population):
    """Range-safe sum of validated nonnegative branch loss contributions."""
    coefficients = tuple(_coefficient_parts(weight, population) for weight in weights)
    return _WeightedSum.apply(coefficients, *values)
