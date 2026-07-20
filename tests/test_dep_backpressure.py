"""Adversarial tests for backpressure (Doc-13, stage 13c)."""

import pytest

from signtranslator.deployment.backpressure import (
    simulate_queue, backlog_growth_per_step, bounded_queue_latency,
    should_apply_backpressure,
)


def test_overload_without_backpressure_diverges():
    # lambda=3 > mu=1: backlog grows ~ (lambda-mu) per step -> unbounded.
    short = simulate_queue(3.0, 1.0, capacity=5.0, steps=10, backpressure=False)
    long = simulate_queue(3.0, 1.0, capacity=5.0, steps=100, backpressure=False)
    assert long.peak_occupancy > short.peak_occupancy      # keeps growing
    assert long.peak_occupancy == pytest.approx((3.0 - 1.0) * 100, rel=0.05)


def test_overload_with_backpressure_is_bounded():
    sim = simulate_queue(3.0, 1.0, capacity=5.0, steps=200, backpressure=True)
    assert sim.peak_occupancy <= 5.0 + 1e-9                # never exceeds capacity
    assert sim.total_blocked > 0                           # excess was slowed/blocked


def test_underload_stays_small_either_way():
    a = simulate_queue(0.5, 1.0, capacity=5.0, steps=100, backpressure=False)
    b = simulate_queue(0.5, 1.0, capacity=5.0, steps=100, backpressure=True)
    assert a.peak_occupancy <= 1.0 and b.peak_occupancy <= 1.0
    assert b.total_blocked == 0.0                          # nothing blocked


def test_backlog_growth_rate():
    assert backlog_growth_per_step(3.0, 1.0) == 2.0
    assert backlog_growth_per_step(0.5, 1.0) == 0.0        # stable regime


def test_bounded_latency_is_capacity_over_rate():
    assert bounded_queue_latency(5.0, 2.0) == pytest.approx(2.5)


def test_backpressure_trigger_watermark():
    assert should_apply_backpressure(4.6, 5.0)             # >= 0.9*5
    assert not should_apply_backpressure(4.0, 5.0)


def test_bad_parameters_rejected():
    with pytest.raises(ValueError):
        simulate_queue(1.0, 0.0, 5.0, 10, backpressure=True)   # mu <= 0
    with pytest.raises(ValueError):
        bounded_queue_latency(5.0, 0.0)
