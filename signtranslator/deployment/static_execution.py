"""Shape-stability guard + optimization-order gate (Doc-13 §7).

A captured static-buffer / CUDA-Graph executor is valid only for STABLE shapes;
replaying it on a different shape raises. Distillation/quantization of the planner
and few-step diffusion are permitted only AFTER a quality baseline is established.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional, Tuple

import torch


class ShapeInstabilityError(RuntimeError):
    """Raised when a captured static executor is replayed on a new shape."""


class StaticShapeExecutor:
    """CUDA-Graph / static-buffer analogue: capture once, replay for a fixed shape."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self._fn = fn
        self._captured_shape: Optional[Tuple[int, ...]] = None

    def capture(self, example: torch.Tensor) -> None:
        """Record the input shape this executor is specialised for."""
        self._captured_shape = tuple(example.shape)

    @property
    def captured_shape(self) -> Optional[Tuple[int, ...]]:
        return self._captured_shape

    def replay(self, x: torch.Tensor) -> torch.Tensor:
        """Run the captured graph; a differing input shape is a hard error."""
        if self._captured_shape is None:
            raise ShapeInstabilityError("executor not captured")
        if tuple(x.shape) != self._captured_shape:
            raise ShapeInstabilityError(
                f"static executor captured for {self._captured_shape}, "
                f"got {tuple(x.shape)} (use dynamic path for unstable shapes)")
        return self._fn(x)


# ---------------------------------------------------------------------------
# optimization-order gate
# ---------------------------------------------------------------------------
class OptimizationStep(IntEnum):
    PROFILE = 0
    CACHE = 1
    CHUNKED_ATTENTION = 2
    ONNX_EXPORT = 3
    TENSORRT_FP16 = 4
    TENSORRT_INT8 = 5
    DISTILL_QUANTIZE = 6      # planner distill + few-step diffusion
    CUDA_GRAPHS = 7


class OptimizationOrderError(RuntimeError):
    """Raised when an optimization step is applied out of the documented order."""


@dataclass
class OptimizationPlan:
    """Enforces the documented optimization order + the quality-baseline gate."""

    baseline_established: bool = False
    completed: List[OptimizationStep] = field(default_factory=list)

    def establish_baseline(self) -> None:
        self.baseline_established = True

    def can_apply(self, step: OptimizationStep) -> Tuple[bool, str]:
        # profiling + acceptance tests come first (define the baseline).
        if step in (OptimizationStep.TENSORRT_INT8,) and \
                OptimizationStep.TENSORRT_FP16 not in self.completed:
            return False, "INT8 only after FP16/BF16 is validated"
        if step is OptimizationStep.DISTILL_QUANTIZE and not self.baseline_established:
            return False, "distill/quantize only after a quality baseline"
        if step is OptimizationStep.CUDA_GRAPHS and \
                OptimizationStep.CHUNKED_ATTENTION not in self.completed:
            return False, "CUDA graphs only once shapes are stabilised"
        return True, ""

    def apply(self, step: OptimizationStep) -> None:
        ok, reason = self.can_apply(step)
        if not ok:
            raise OptimizationOrderError(reason)
        self.completed.append(step)
