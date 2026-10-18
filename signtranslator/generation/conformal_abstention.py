"""Distribution-free error bounds and abstention for gap reconstruction.

Split conformal prediction turns held-out reconstruction errors into a bound with a
finite-sample guarantee, and the bound decides when to abstain.

**Bound.** For calibration units with nonconformity scores ``s_1..s_n`` and a new
unit with score ``s``, exchangeability gives ``P(s <= s_(k)) >= 1 - alpha`` for
``k = ceil((n + 1)(1 - alpha))``, where ``s_(k)`` is the k-th smallest score; if
``k > n`` the bound is infinite. With distinct scores coverage is also at most
``1 - alpha + 1/(n + 1)``. Scores are normalised by a difficulty ``d`` computed from
visible data only, ``s = e / (floor + d)``, so the bound ``q (floor + d)`` widens
where the gap is harder while the guarantee is unchanged.

**Abstention.** Given a declared tolerance ``tau``, reconstruct only when the bound
is at most ``tau``. Then ``P(accept and e > tau) <= P(e > bound) <= alpha``: the
rate of accepted reconstructions that miss the tolerance is controlled, whatever
the abstention rate turns out to be.

**Units.** One unit is one hidden span of one sign; its error is the largest
geodesic error over the span's scored cells, so the bound holds for every cell of
the span at once. The guarantee needs calibration and test units to be
exchangeable. Calibrating on one performer and testing on another is a check of
that assumption, not an instance of it, and the test coverage is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ..pose.multichannel import MultichannelMotion
from ..pose.rotations import geodesic_distance, rotation_6d_to_matrix


def conformal_quantile(scores, alpha: float) -> float:
    """The exact split-conformal quantile ``s_(ceil((n+1)(1-alpha)))``, or inf."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty finite vector")
    if type(alpha) not in (int, float) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    n = len(scores)
    # Integer arithmetic for the rank, so floating error cannot move it.
    numerator = round((1 - alpha) * 10 ** 12)
    k = -(-(n + 1) * numerator // 10 ** 12)
    return math.inf if k > n else float(np.sort(scores)[k - 1])


@dataclass(frozen=True)
class ConformalBound:
    alpha: float
    quantile: float
    calibration_units: int
    floor: float

    def bound(self, difficulty) -> np.ndarray:
        return self.quantile * (self.floor + np.asarray(difficulty, dtype=np.float64))

    def to_dict(self) -> dict:
        return dict(alpha=self.alpha, quantile=self.quantile, calibration_units=self.calibration_units,
                    floor=self.floor)


def fit_conformal_bound(errors, difficulty, *, alpha: float, floor: float) -> ConformalBound:
    errors = np.asarray(errors, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)
    if errors.shape != difficulty.shape or errors.ndim != 1:
        raise ValueError("errors and difficulty must be matching vectors")
    if np.any(errors < 0) or np.any(difficulty < 0):
        raise ValueError("errors and difficulty must be nonnegative")
    if type(floor) not in (int, float) or not floor > 0:
        raise ValueError("floor must be positive, so every score is defined")
    return ConformalBound(float(alpha), conformal_quantile(errors / (floor + difficulty), alpha),
                          len(errors), float(floor))


def evaluate_abstention(errors, difficulty, bound: ConformalBound, *, tolerance: float) -> dict:
    """Coverage, abstention and the controlled accepted-violation rate on new units."""
    errors = np.asarray(errors, dtype=np.float64)
    limits = bound.bound(difficulty)
    if errors.shape != limits.shape or len(errors) == 0:
        raise ValueError("errors and difficulty must be matching nonempty vectors")
    accept = limits <= tolerance
    return dict(units=len(errors), coverage=float(np.mean(errors <= limits)),
                abstention_rate=float(np.mean(~accept)),
                accepted_violation_rate=float(np.mean(accept & (errors > tolerance))),
                accepted_mean_error=float(errors[accept].mean()) if accept.any() else None,
                guarantee=f"P(accept and error > {tolerance}) <= {bound.alpha} under exchangeability")


def span_error_and_difficulty(state: MultichannelMotion, name: str, hidden: np.ndarray,
                              reconstruction: np.ndarray, covered: np.ndarray, *,
                              allow_inferred: bool = False) -> tuple[float, float] | None:
    """Largest geodesic error (degrees) over a span's scored cells, and its difficulty.

    Difficulty is the largest geodesic angle, over elements, between the visible
    reference rotations immediately before and after the hidden span: the arc the
    interpolation must bridge. It reads no hidden value.
    """
    channel = state.channels[name]
    reference = channel.valid if allow_inferred else channel.valid & channel.observed
    scored = hidden & reference & covered
    if not scored.any():
        return None
    target = rotation_6d_to_matrix(torch.from_numpy(np.ascontiguousarray(channel.values[scored])))
    predicted = rotation_6d_to_matrix(torch.from_numpy(np.ascontiguousarray(reconstruction[scored])))
    error = float(np.degrees(geodesic_distance(target, predicted).numpy()).max())
    angles = []
    visible = reference & ~hidden
    for element in np.flatnonzero(scored.any(axis=0)):
        frames = np.flatnonzero(hidden[:, element])
        before = np.flatnonzero(visible[:frames[0], element])
        after = np.flatnonzero(visible[frames[-1] + 1:, element])
        if not len(before) or not len(after):
            continue
        a = channel.values[before[-1], element]
        b = channel.values[frames[-1] + 1 + after[0], element]
        pair = rotation_6d_to_matrix(torch.from_numpy(np.stack((a, b))))
        angles.append(float(np.degrees(geodesic_distance(pair[:1], pair[1:]).numpy())[0]))
    if not angles:
        return None
    return error, max(angles)


__all__ = ["conformal_quantile", "ConformalBound", "fit_conformal_bound", "evaluate_abstention",
           "span_error_and_difficulty"]
