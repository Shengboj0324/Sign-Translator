"""Masked-reconstruction comparison of canonical-state representations (W4).

Blocker B44 requires that simple baselines stay eligible to win before a learned
tokeniser, autoencoder or generator is preferred. This module supplies the
common protocol: hide declared spans of genuinely observed cells, let each
method reconstruct the full state, and score every method only on the cells
that were observed, hidden and reconstructed by **every** method being compared.

Scored support. For channel ``c`` with validity ``V``, observation ``O``, hidden
mask ``H`` and method coverage ``K_m``, method ``m`` is scored on
``S = V & O & H & (AND over compared methods of K_m)``. Comparing methods on
their own, different supports would confound reconstruction quality with
coverage; each method's own coverage ``|V & O & H & K_m| / |V & O & H|`` is
reported separately so a method cannot win by declining hard cells.

Errors are geometric, in declared units:

* rotations: geodesic angle of ``R^T Rhat`` in degrees;
* translations: Euclidean distance in metres;
* gaze: great-circle angle in degrees;
* facial coefficients and bounded states: absolute difference, unitless.

Interpolation baseline on the true, possibly nonuniform clock. Between the
nearest visible observed neighbours ``a`` and ``b`` with ``tau = (t - t_a) /
(t_b - t_a)``:

* rotations follow the SO(3) geodesic ``R_a exp(tau log(R_a^T R_b))``, exact
  for constant angular velocity about a fixed axis;
* gaze follows the great-circle slerp, exact for constant angular speed;
* translations, coefficients and bounded states interpolate linearly, which
  stays inside ``[0, 1]`` for bounded states because it is convex.

A cell with no visible neighbour on one side is not extrapolated: it is
uncovered. A rotation pair within ``ambiguity_radians`` of ``pi``, or an
antipodal gaze pair, has two equally short geodesics; choosing one would be an
arbitrary guess, so that cell is uncovered too.

Nothing here trains or selects a model. A comparison on fictional fixtures is a
correctness check of the protocol, not evidence about real signing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import numpy as np
import torch

from ..pose.multichannel import CHANNELS, MultichannelMotion
from ..pose.rotations import (
    axis_angle_to_matrix, geodesic_distance, matrix_to_axis_angle,
    matrix_to_rotation_6d, rotation_6d_to_matrix,
)

Reconstruction = dict[str, tuple[np.ndarray, np.ndarray]]  # name -> (values, covered)
Method = Callable[[MultichannelMotion, Mapping[str, np.ndarray]], Reconstruction]


def hidden_spans(state: MultichannelMotion, *, channels: tuple[str, ...],
                 span_seconds: float, spans: int, seed: int) -> dict[str, np.ndarray]:
    """Hide ``spans`` disjoint time spans of ``span_seconds`` on the real clock.

    Spans are placed by a seeded generator over frames, measured in seconds of
    the state's own clock, and apply to every element of each named channel.
    Only frames are chosen here; cells that were never observed are excluded at
    scoring time, so hiding an unavailable cell creates no fake target.
    """
    if not isinstance(state, MultichannelMotion):
        raise ValueError("a typed canonical state is required")
    state.validate()
    if (type(span_seconds) not in (int, float) or not math.isfinite(span_seconds)
            or span_seconds <= 0):
        raise ValueError("span_seconds must be finite and positive")
    if type(spans) is not int or spans < 1 or type(seed) is not int or seed < 0:
        raise ValueError("spans must be positive and seed non-negative integers")
    unknown = set(channels) - set(CHANNELS)
    if unknown or not channels:
        raise ValueError(f"unknown or empty channel selection: {sorted(unknown)}")
    clock = state.timestamps
    rng = np.random.default_rng(seed)
    frames = np.zeros(len(clock), dtype=bool)
    order = rng.permutation(len(clock))
    placed = 0
    for start in order:
        window = (clock >= clock[start]) & (clock < clock[start] + span_seconds)
        # Keep spans disjoint and leave at least one visible frame on each
        # side, so the interpolation baseline is not denied every neighbour.
        if frames[window].any() or window[0] or window[-1]:
            continue
        frames |= window
        placed += 1
        if placed == spans:
            break
    if placed == 0:
        raise ValueError("no interior span fits the declared length on this clock")
    return {name: np.broadcast_to(frames[:, None], state.channels[name].valid.shape).copy()
            for name in channels}


def _visible_neighbours(usable: np.ndarray):
    """Previous and next usable frame index per frame, or -1 when absent."""
    count = len(usable)
    previous = np.full(count, -1)
    following = np.full(count, -1)
    last = -1
    for index in range(count):
        previous[index] = last
        if usable[index]:
            last = index
    last = -1
    for index in range(count - 1, -1, -1):
        following[index] = last
        if usable[index]:
            last = index
    return previous, following


def interpolation_baseline(state: MultichannelMotion, hidden: Mapping[str, np.ndarray], *,
                           ambiguity_radians: float = 1e-6,
                           allow_inferred: bool = False) -> Reconstruction:
    """Geodesic interpolation on the true clock from visible reference neighbours.

    Neighbours must be observed unless ``allow_inferred`` declares that valid
    inferred values (for example a solved skeleton) serve as the reference.
    """
    if type(allow_inferred) is not bool:
        raise ValueError("allow_inferred must be an explicit boolean")
    clock = state.timestamps
    result: Reconstruction = {}
    for name, mask in hidden.items():
        channel = state.channels[name]
        kind = CHANNELS[name][0]
        values = channel.values.astype(np.float64).copy()
        covered = np.zeros(mask.shape, dtype=bool)
        reference = channel.valid if allow_inferred else channel.valid & channel.observed
        visible = reference & ~mask
        for element in range(values.shape[1]):
            previous, following = _visible_neighbours(visible[:, element])
            for frame in np.flatnonzero(mask[:, element]):
                a, b = previous[frame], following[frame]
                if a < 0 or b < 0:
                    continue
                tau = (clock[frame] - clock[a]) / (clock[b] - clock[a])
                left, right = values[a, element], values[b, element]
                if kind == "rotation_6d_columns":
                    first = rotation_6d_to_matrix(torch.from_numpy(left))
                    second = rotation_6d_to_matrix(torch.from_numpy(right))
                    relative = first.T @ second
                    if math.pi - float(geodesic_distance(torch.eye(3, dtype=torch.float64),
                                                         relative)) <= ambiguity_radians:
                        continue
                    step = axis_angle_to_matrix(matrix_to_axis_angle(relative) * tau)
                    values[frame, element] = matrix_to_rotation_6d(first @ step).numpy()
                elif kind == "unit_vector":
                    cosine = float(np.clip(np.dot(left, right), -1., 1.))
                    angle = math.acos(cosine)
                    if math.pi - angle <= ambiguity_radians:
                        continue
                    if angle < 1e-12:
                        values[frame, element] = left
                    else:
                        values[frame, element] = (
                            math.sin((1 - tau) * angle) * left
                            + math.sin(tau * angle) * right) / math.sin(angle)
                else:
                    values[frame, element] = (1 - tau) * left + tau * right
                covered[frame, element] = True
        result[name] = (values, covered)
    return result


def zero_order_hold_baseline(state: MultichannelMotion, hidden: Mapping[str, np.ndarray], *,
                             allow_inferred: bool = False) -> Reconstruction:
    """Repeat the most recent visible reference value; uncovered without one."""
    if type(allow_inferred) is not bool:
        raise ValueError("allow_inferred must be an explicit boolean")
    result: Reconstruction = {}
    for name, mask in hidden.items():
        channel = state.channels[name]
        values = channel.values.astype(np.float64).copy()
        covered = np.zeros(mask.shape, dtype=bool)
        reference = channel.valid if allow_inferred else channel.valid & channel.observed
        visible = reference & ~mask
        for element in range(values.shape[1]):
            previous, _ = _visible_neighbours(visible[:, element])
            for frame in np.flatnonzero(mask[:, element]):
                if previous[frame] >= 0:
                    values[frame, element] = values[previous[frame], element]
                    covered[frame, element] = True
        result[name] = (values, covered)
    return result


def hold_out_rest_baseline(state: MultichannelMotion,
                           hidden: Mapping[str, np.ndarray]) -> Reconstruction:
    """Constant reference: identity rotation, zero translation/coefficient, 0.5 bound.

    Gaze has no reference direction that is not a convention claim, so gaze
    cells are uncovered rather than filled with an assumed forward axis.
    """
    result: Reconstruction = {}
    for name, mask in hidden.items():
        kind = CHANNELS[name][0]
        values = state.channels[name].values.astype(np.float64).copy()
        covered = mask.copy()
        if kind == "rotation_6d_columns":
            values[mask] = [1., 0., 0., 0., 1., 0.]
        elif kind == "bounded_scalar":
            values[mask] = .5
        elif kind == "unit_vector":
            covered[:] = False
        else:
            values[mask] = 0.
        result[name] = (values, covered)
    return result


def _cell_errors(kind: str, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    if kind == "rotation_6d_columns":
        first = rotation_6d_to_matrix(torch.from_numpy(target))
        second = rotation_6d_to_matrix(torch.from_numpy(predicted))
        return geodesic_distance(first, second).numpy() * 180. / math.pi
    if kind == "unit_vector":
        norm = np.linalg.norm(predicted, axis=-1)
        if np.any(norm <= 1e-12):
            raise ValueError("a reconstructed gaze direction has no direction")
        cosine = np.clip((predicted / norm[..., None] * target).sum(-1), -1., 1.)
        return np.degrees(np.arccos(cosine))
    if kind == "translation":
        return np.linalg.norm(predicted - target, axis=-1)
    return np.abs(predicted - target).mean(axis=-1)


UNITS = {"rotation_6d_columns": "degrees", "unit_vector": "degrees",
         "translation": "metres", "coefficient": "unitless", "bounded_scalar": "unitless"}


@dataclass(frozen=True)
class ComparisonReport:
    channels: dict
    methods: tuple[str, ...]

    def to_dict(self) -> dict:
        return dict(methods=list(self.methods), channels=self.channels,
                    note="errors on common support only; coverage is per method")


def compare_reconstructions(state: MultichannelMotion, hidden: Mapping[str, np.ndarray],
                            methods: Mapping[str, Method], *,
                            allow_inferred: bool = False) -> ComparisonReport:
    """Score every method on the cells every method reconstructed.

    The reference is observed cells only, unless ``allow_inferred`` explicitly
    declares that valid inferred values count; the choice is recorded.
    """
    if type(allow_inferred) is not bool:
        raise ValueError("allow_inferred must be an explicit boolean")
    if not isinstance(state, MultichannelMotion):
        raise ValueError("a typed canonical state is required")
    state.validate()
    if len(methods) < 2:
        raise ValueError("a comparison needs at least two methods")
    outputs = {}
    for label, method in methods.items():
        output = method(state, hidden)
        if set(output) != set(hidden):
            raise ValueError(f"{label} must reconstruct exactly the hidden channels")
        outputs[label] = output
    channels = {}
    for name, mask in hidden.items():
        channel = state.channels[name]
        kind = CHANNELS[name][0]
        eligible = (channel.valid if allow_inferred else channel.valid & channel.observed) & mask
        for label, (values, covered) in ((k, v[name]) for k, v in outputs.items()):
            if values.shape != channel.values.shape or covered.shape != mask.shape:
                raise ValueError(f"{label}:{name} reconstruction shape mismatch")
            if not np.isfinite(values[covered & eligible]).all():
                raise ValueError(f"{label}:{name} reconstructed nonfinite values")
            if np.any(covered & ~mask):
                raise ValueError(f"{label}:{name} claims coverage of a visible cell")
        common = eligible.copy()
        for values, covered in (outputs[label][name] for label in outputs):
            common &= covered
        record = dict(unit=UNITS[kind], reference="valid_including_inferred" if allow_inferred
                      else "observed_only", eligible_cells=int(eligible.sum()),
                      common_support_cells=int(common.sum()), methods={})
        for label in outputs:
            values, covered = outputs[label][name]
            own = int((covered & eligible).sum())
            coverage = own / int(eligible.sum()) if eligible.any() else None
            if common.any():
                errors = _cell_errors(kind, values[common], channel.values[common].astype(np.float64))
                record["methods"][label] = dict(
                    coverage=coverage, mean_error=float(errors.mean()),
                    median_error=float(np.median(errors)), max_error=float(errors.max()))
            else:
                record["methods"][label] = dict(coverage=coverage, mean_error=None,
                                                median_error=None, max_error=None)
        channels[name] = record
    return ComparisonReport(channels, tuple(methods))
