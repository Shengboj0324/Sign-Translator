"""Rendering evaluation with enforced appearance/signing separation (§8).

Metrics: motion-to-photon latency, dropped frames, temporal flicker, mesh/hand
penetration (reuse Doc-04/05), silhouette IoU error, and appearance PSNR/SSIM.

**Innovation — structural separation of concerns.** Appearance metrics live in an
``AppearanceReport`` that is *incapable* of expressing a signing verdict, signing
metrics in a ``SigningReport``, and the code **refuses** to derive a signing verdict
from appearance -- the document's rule ("PSNR/SSIM assess appearance, not correct
signing") made unbreakable rather than merely documented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from ..speech.streaming import percentile
from ..hand_graph.metrics import collision_rate


# ---------------------------------------------------------------------------
# latency / pacing
# ---------------------------------------------------------------------------
def motion_to_photon_p95(latencies: Sequence[float]) -> float:
    """95th-percentile motion-to-photon latency (seconds)."""
    return percentile(list(latencies), 95.0)


def dropped_frame_rate(render_times: Sequence[float], frame_budget: float) -> float:
    """Fraction of frames whose render time exceeded the frame budget (1/fps)."""
    if not render_times:
        return 0.0
    over = sum(1 for t in render_times if t > frame_budget)
    return over / len(render_times)


def temporal_flicker(frames: torch.Tensor) -> torch.Tensor:
    """Mean frame-to-frame change ‖f_{t+1} − f_t‖. ``frames`` (T, ...). 0 = static."""
    if frames.shape[0] < 2:
        return frames.new_zeros(())
    diff = (frames[1:] - frames[:-1]).flatten(1)
    return diff.norm(dim=-1).mean()


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def penetration_rate(joints: torch.Tensor, radii: torch.Tensor) -> float:
    """Mesh/hand self-penetration rate over frames (reuse Doc-04/05 collision)."""
    return collision_rate(joints, radii)


def silhouette_iou_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """1 − IoU of two boolean silhouettes (0 = perfect overlap)."""
    p, g = pred.bool(), gt.bool()
    inter = float((p & g).sum())
    union = float((p | g).sum())
    return 1.0 - (inter / union if union > 0 else 1.0)


# ---------------------------------------------------------------------------
# appearance (NOT signing)
# ---------------------------------------------------------------------------
def psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB). An APPEARANCE metric."""
    mse = float(((pred - gt) ** 2).mean())
    if mse <= 1e-20:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def ssim_global(pred: torch.Tensor, gt: torch.Tensor,
                c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> float:
    """Global SSIM. An APPEARANCE metric (single-window simplification)."""
    x, y = pred.flatten().to(torch.float64), gt.flatten().to(torch.float64)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(unbiased=False), y.var(unbiased=False)
    cov = ((x - mx) * (y - my)).mean()
    return float(((2 * mx * my + c1) * (2 * cov + c2))
                 / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))


# ---------------------------------------------------------------------------
# structural separation of concerns
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppearanceReport:
    """Appearance-only metrics. Deliberately has NO signing field."""

    psnr: float
    ssim: float
    silhouette_iou_error: float

    @property
    def is_signing_verdict(self) -> bool:
        return False                                         # never a signing verdict


@dataclass(frozen=True)
class SigningReport:
    """Signing-correctness metrics. Deliberately has NO appearance field."""

    semantic_accuracy: float
    comprehension: float


def signing_quality_from_appearance(_report: AppearanceReport) -> float:
    """Structural refusal: a signing verdict can NEVER be derived from appearance."""
    raise TypeError(
        "PSNR/SSIM/silhouette assess appearance, not correct signing; a signing "
        "verdict cannot be derived from an AppearanceReport.")


def combined_report(appearance: AppearanceReport, signing: SigningReport
                    ) -> Dict[str, object]:
    """Keep the two report kinds side by side WITHOUT merging them into one score."""
    if not isinstance(appearance, AppearanceReport) or not isinstance(signing, SigningReport):
        raise TypeError("appearance and signing reports must stay typed and separate")
    return {"appearance": appearance, "signing": signing}
