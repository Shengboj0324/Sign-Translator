"""Independent cross-modal validation of a solved skeleton against raw markers.

A solved skeleton (for example a glTF animation exported from a marker solve)
and the raw optical markers it was fitted to are two records of one take. Their
agreement can be checked without trusting either file's global frame by using a
quantity that is invariant to rigid motion of the whole body: elbow flexion,
the angle at the elbow between the upper arm and the forearm.

Marker joint centres follow standard practice: each centre is the midpoint of a
lateral/medial (or front/back) marker pair, ``c = (m_1 + m_2) / 2``. A missing
marker coordinate is ``NaN`` and propagates; frames where any centre is missing
are excluded, never filled.

Clocks. The solved skeleton carries seconds; the marker file carries only frame
indices at an unstated rate. The rate ``R`` and integer start offset ``k`` are
searched exhaustively over a declared candidate set: for skeleton times ``t_i``
the marker series is sampled at fractional index ``R (t_i - t_0) + k`` by linear
interpolation, and the pair maximising the Pearson correlation is selected. The
recovered rate is itself evidence and is reported. Correlation measures shape
agreement; the centred mean absolute error measures magnitude agreement after
removing the constant offset that different joint-centre definitions introduce.

The result validates that the two files describe the same take with a coherent
arm chain. It does not validate absolute calibration, other joints, or meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

DEFAULT_RATES = (60., 100., 120., 150., 200., 240., 250.)


def joint_angle_degrees(proximal: np.ndarray, joint: np.ndarray, distal: np.ndarray) -> np.ndarray:
    """Angle at ``joint`` between the two segments, in degrees; NaN where undefined."""
    first, second = proximal - joint, distal - joint
    norms = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(first * second, axis=-1) / norms
    result = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    return np.where(norms > 0, result, np.nan)


def parse_v3d_markers(text: str) -> tuple[list[str], np.ndarray]:
    """Parse a Visual3D marker export: label line, header, then frame rows.

    Returns marker names in column order and an ``(F, M, 3)`` array in the file's
    units; empty cells become NaN. The first line (a performer label) is
    deliberately discarded so no personal label leaves this function.
    """
    lines = [line for line in text.splitlines()]
    if len(lines) < 3:
        raise ValueError("marker file has no data rows")
    header = [cell.strip() for cell in lines[1].split(",")]
    if header[0] != "Frame":
        raise ValueError("marker header must start with Frame")
    names: list[str] = []
    for cell in header[1:]:
        if not cell:
            continue
        name, _, axis = cell.partition("<")
        if axis not in ("T-X>", "T-Y>", "T-Z>"):
            raise ValueError(f"unexpected marker column {cell!r}")
        if axis == "T-X>":
            names.append(name)
    columns = {cell: index for index, cell in enumerate(header)}
    rows = [line.split(",") for line in lines[2:] if line.strip()]
    data = np.full((len(rows), len(names), 3), np.nan)
    for marker, name in enumerate(names):
        for axis, suffix in enumerate(("T-X>", "T-Y>", "T-Z>")):
            column = columns[f"{name}<{suffix}"]
            for row, cells in enumerate(rows):
                if column < len(cells) and cells[column].strip():
                    data[row, marker, axis] = float(cells[column])
    return names, data


def marker_elbow_flexion(names: Sequence[str], data: np.ndarray, side: str) -> np.ndarray:
    """Elbow flexion from midpoint joint centres of a Plug-in Gait-style set."""
    if side not in ("L", "R"):
        raise ValueError("side must be 'L' or 'R'")
    index = {name: i for i, name in enumerate(names)}

    def centre(first, second):
        try:
            return (data[:, index[side + first]] + data[:, index[side + second]]) / 2
        except KeyError as error:
            raise ValueError(f"marker set lacks {error}") from error

    return joint_angle_degrees(centre("FSH", "BSH"), centre("ELB", "IEL"), centre("IWR", "OWR"))


@dataclass(frozen=True)
class AlignmentResult:
    correlation: float
    rate_hz: float
    offset_frames: int
    centred_mae_degrees: float
    samples: int

    def passes(self, *, min_correlation: float, max_centred_mae_degrees: float) -> bool:
        return (self.correlation >= min_correlation
                and self.centred_mae_degrees <= max_centred_mae_degrees)


def align_series(times: np.ndarray, reference: np.ndarray, markers: np.ndarray, *,
                 rates: Sequence[float] = DEFAULT_RATES, min_samples: int = 8) -> AlignmentResult | None:
    """Exhaustive rate/offset search maximising Pearson correlation.

    Returns None when no candidate yields at least ``min_samples`` finite pairs
    with nonzero variance on both sides.
    """
    times = np.asarray(times, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    markers = np.asarray(markers, dtype=np.float64)
    if times.shape != reference.shape or times.ndim != 1 or np.any(np.diff(times) <= 0):
        raise ValueError("times and reference must be matching strictly increasing series")
    elapsed = times - times[0]
    best = None
    positions = np.arange(len(markers), dtype=np.float64)
    for rate in rates:
        span = elapsed[-1] * rate
        for offset in range(0, int(math.floor(len(markers) - 1 - span)) + 1):
            index = elapsed * rate + offset
            sample = np.interp(index, positions, markers)
            # interp ignores NaN placement; recover missingness from neighbours.
            low = np.floor(index).astype(int)
            high = np.minimum(low + 1, len(markers) - 1)
            finite = np.isfinite(markers[low]) & np.isfinite(markers[high]) & np.isfinite(reference)
            if finite.sum() < min_samples:
                continue
            a, b = reference[finite], sample[finite]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            correlation = float(np.corrcoef(a, b)[0, 1])
            if best is None or correlation > best[0]:
                mae = float(np.mean(np.abs((a - a.mean()) - (b - b.mean()))))
                best = (correlation, float(rate), offset, mae, int(finite.sum()))
    if best is None:
        return None
    return AlignmentResult(*best)


@dataclass(frozen=True)
class JointAlignment:
    """One rate and offset for a whole take, with per-series agreement."""

    rate_hz: float
    offset_frames: int
    correlations: tuple[float, ...]
    centred_mae_degrees: tuple[float, ...]
    reference_ranges_degrees: tuple[float, ...]
    samples: tuple[int, ...]

    def passes(self, *, min_correlation: float, max_centred_mae_degrees: float,
               static_range_degrees: float) -> tuple[bool, ...]:
        """Per series: magnitude always; shape only when the series moves.

        Pearson correlation is uninformative as a series' variance approaches
        zero, so a series whose reference range is below ``static_range_degrees``
        is judged by its centred error alone.
        """
        return tuple(mae <= max_centred_mae_degrees
                     and (span < static_range_degrees or r >= min_correlation)
                     for r, mae, span in zip(self.correlations, self.centred_mae_degrees,
                                             self.reference_ranges_degrees))


def align_joint(times: np.ndarray, references: Sequence[np.ndarray],
                markers: Sequence[np.ndarray], *, rates: Sequence[float] = DEFAULT_RATES,
                min_samples: int = 8) -> JointAlignment | None:
    """Search one rate/offset for several series recorded on the same clock.

    The objective is the Pearson correlation of the concatenation of each
    series centred on its own mean, so no series' constant offset influences
    the choice and every series must agree under the same clock.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or np.any(np.diff(times) <= 0) or len(references) != len(markers) or not references:
        raise ValueError("matching nonempty series on a strictly increasing clock are required")
    references = [np.asarray(r, dtype=np.float64) for r in references]
    markers = [np.asarray(m, dtype=np.float64) for m in markers]
    if any(r.shape != times.shape for r in references) or len({len(m) for m in markers}) != 1:
        raise ValueError("references must match the clock and markers must share a length")
    length = len(markers[0])
    positions = np.arange(length, dtype=np.float64)
    elapsed = times - times[0]
    best = None
    for rate in rates:
        span = elapsed[-1] * rate
        for offset in range(0, int(math.floor(length - 1 - span)) + 1):
            index = elapsed * rate + offset
            low = np.floor(index).astype(int)
            high = np.minimum(low + 1, length - 1)
            pieces_a, pieces_b, parts = [], [], []
            for reference, marker in zip(references, markers):
                finite = np.isfinite(marker[low]) & np.isfinite(marker[high]) & np.isfinite(reference)
                if finite.sum() < min_samples:
                    parts = None
                    break
                sample = np.interp(index, positions, marker)
                a, b = reference[finite], sample[finite]
                pieces_a.append(a - a.mean())
                pieces_b.append(b - b.mean())
                parts.append((a, b))
            if parts is None:
                continue
            a, b = np.concatenate(pieces_a), np.concatenate(pieces_b)
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            score = float(np.corrcoef(a, b)[0, 1])
            if best is None or score > best[0]:
                best = (score, float(rate), offset, parts)
    if best is None:
        return None
    _, rate, offset, parts = best
    correlations, maes, ranges, counts = [], [], [], []
    for a, b in parts:
        defined = a.std() > 1e-9 and b.std() > 1e-9
        correlations.append(float(np.corrcoef(a, b)[0, 1]) if defined else float("nan"))
        maes.append(float(np.mean(np.abs((a - a.mean()) - (b - b.mean())))))
        ranges.append(float(np.ptp(a)))
        counts.append(int(a.size))
    return JointAlignment(rate, offset, tuple(correlations), tuple(maes), tuple(ranges), tuple(counts))
