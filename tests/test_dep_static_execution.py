"""Adversarial tests for shape guard + optimization order (Doc-13, stage 13g)."""

import pytest
import torch

from signtranslator.deployment.static_execution import (
    StaticShapeExecutor, ShapeInstabilityError,
    OptimizationStep, OptimizationPlan, OptimizationOrderError,
)


def test_static_executor_replays_captured_shape():
    ex = StaticShapeExecutor(lambda x: x * 2)
    ex.capture(torch.zeros(2, 3))
    out = ex.replay(torch.ones(2, 3))
    assert torch.equal(out, torch.full((2, 3), 2.0))


def test_static_executor_rejects_new_shape():
    ex = StaticShapeExecutor(lambda x: x + 1)
    ex.capture(torch.zeros(2, 3))
    with pytest.raises(ShapeInstabilityError):
        ex.replay(torch.zeros(2, 4))               # unstable shape -> hard error


def test_replay_before_capture_raises():
    ex = StaticShapeExecutor(lambda x: x)
    with pytest.raises(ShapeInstabilityError):
        ex.replay(torch.zeros(1))


def test_distill_requires_quality_baseline():
    plan = OptimizationPlan()
    with pytest.raises(OptimizationOrderError):
        plan.apply(OptimizationStep.DISTILL_QUANTIZE)   # before baseline
    plan.establish_baseline()
    plan.apply(OptimizationStep.DISTILL_QUANTIZE)       # allowed now
    assert OptimizationStep.DISTILL_QUANTIZE in plan.completed


def test_int8_requires_fp16_first():
    plan = OptimizationPlan()
    ok, reason = plan.can_apply(OptimizationStep.TENSORRT_INT8)
    assert not ok and "FP16" in reason
    plan.apply(OptimizationStep.TENSORRT_FP16)
    plan.apply(OptimizationStep.TENSORRT_INT8)          # now allowed
    assert OptimizationStep.TENSORRT_INT8 in plan.completed


def test_cuda_graphs_require_stable_shapes_first():
    plan = OptimizationPlan()
    with pytest.raises(OptimizationOrderError):
        plan.apply(OptimizationStep.CUDA_GRAPHS)        # before chunked attention
    plan.apply(OptimizationStep.CHUNKED_ATTENTION)
    plan.apply(OptimizationStep.CUDA_GRAPHS)
    assert OptimizationStep.CUDA_GRAPHS in plan.completed


def test_documented_order_is_monotone():
    assert list(OptimizationStep) == sorted(OptimizationStep)
