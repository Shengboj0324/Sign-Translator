"""Adversarial tests for masking + interpolation certificate (Doc-11, stage 11a)."""

import numpy as np
import pytest

from signtranslator.pretraining.masking import (
    random_point_mask, span_mask, part_mask, semantic_boundary_mask, mask_ratio,
    mask_interpolation_error_floor, worst_masked_floor, typical_masked_floor,
    is_certified_hard, linear_interpolate_reconstruction,
)


def test_span_mask_is_contiguous():
    m = span_mask(T=20, P=1, span_len=5, num_spans=1, seed=1)
    idx = np.where(m[:, 0])[0]
    assert idx.size == 5
    assert np.all(np.diff(idx) == 1)              # contiguous run


def test_part_mask_masks_whole_stream():
    m = part_mask(T=10, P=4, parts_to_mask=[1, 3])
    assert m[:, 1].all() and m[:, 3].all()
    assert not m[:, 0].any() and not m[:, 2].any()


def test_semantic_boundary_mask_radius():
    m = semantic_boundary_mask(T=12, P=1, boundaries=[5], radius=2)
    assert np.where(m[:, 0])[0].tolist() == [3, 4, 5, 6, 7]


def test_ratio_matches():
    m = random_point_mask(50, 4, ratio=0.6, seed=0)
    assert abs(mask_ratio(m) - 0.6) < 0.05


def test_floor_exact_for_quadratic_signal():
    # A quadratic x=c t^2 has interpolation error EXACTLY 0.5*x''*a*b = c*a*b.
    c = 0.3
    T = 15
    t = np.arange(T, dtype=np.float64)
    signal = (c * t * t)[:, None]                 # (T, 1)
    m = np.zeros((T, 1), dtype=bool)
    m[7, 0] = True                                 # single masked point, a=b=1
    floor = mask_interpolation_error_floor(m, curvature=2 * c)  # x''=2c
    recon = linear_interpolate_reconstruction(signal, m)
    actual_err = abs(recon[7, 0] - signal[7, 0])
    assert np.isclose(floor[7, 0], actual_err, atol=1e-12)
    assert np.isclose(floor[7, 0], c * 1 * 1, atol=1e-12)


def test_span_interior_floor_grows_quadratically():
    # random single point: a=b=1 -> floor ~ c; span interior: a*b ~ (L/2)^2 -> larger.
    c = 0.5
    T = 41
    single = np.zeros((T, 1), dtype=bool); single[20, 0] = True
    span = np.zeros((T, 1), dtype=bool); span[11:30, 0] = True  # length 19
    f_single = worst_masked_floor(single, curvature=2 * c)
    f_span = worst_masked_floor(span, curvature=2 * c)
    assert f_single == pytest.approx(c)            # a=b=1
    assert f_span > 20 * f_single                  # span is far harder


def test_full_part_mask_has_infinite_floor():
    # a fully-masked part has no temporal neighbour => interpolation impossible.
    m = part_mask(T=10, P=2, parts_to_mask=[0])
    floor = mask_interpolation_error_floor(m)
    assert np.isinf(floor[:, 0]).all()
    assert is_certified_hard(m, threshold=1e9)     # exceeds any finite threshold


def test_random_point_mask_is_typically_easy_relative_to_span():
    # The document's claim is about the TYPICAL masked token (adjacent frames
    # interpolate). Random masking leaves most tokens with a=b=1 (small median
    # floor); a span raises the typical floor. (worst-case can be +inf for either
    # when an edge token is masked, so compare the robust median.)
    T, P = 60, 1
    rnd = random_point_mask(T, P, ratio=0.3, seed=2)
    spn = span_mask(T, P, span_len=18, num_spans=1, seed=2)
    assert typical_masked_floor(spn) > 10 * typical_masked_floor(rnd)
    assert typical_masked_floor(rnd) < 2.0        # small: neighbours stay close


def test_isolated_random_point_has_unit_floor():
    # A genuinely isolated masked point (both neighbours adjacent) floors at 0.5.
    T = 11
    m = np.zeros((T, 1), dtype=bool)
    m[5, 0] = True
    assert typical_masked_floor(m) == pytest.approx(0.5, abs=1e-12)  # a=b=1


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        random_point_mask(10, 2, ratio=1.5)
    with pytest.raises(ValueError):
        span_mask(10, 1, span_len=0, num_spans=1)
    with pytest.raises(ValueError):
        part_mask(10, 2, parts_to_mask=[5])
