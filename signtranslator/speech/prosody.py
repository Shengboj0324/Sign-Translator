"""Prosody pathway: fundamental frequency, energy, voicing, and pauses.

The source specification retains three pathways from the speech layer -- lexical
posterior, acoustic embeddings, and prosody -- and is explicit that **prosody is
conditioning evidence for discourse and affect, not a deterministic mapping to a
particular facial marker**. This module therefore *measures* prosody and hands
it downstream as evidence; it deliberately does not infer any sign-language
non-manual marker itself. Any such mapping belongs in the planner, where it can
be learned and evaluated rather than hard-coded.

F0 is estimated with YIN (de Cheveigné & Kawahara 2002). The cumulative mean
normalised difference function is what distinguishes YIN from plain
autocorrelation: it suppresses the subharmonic dip at tau = 2T that otherwise
causes octave errors on harmonically rich voices. See
docs/SPEECH_FOUNDATION.md §3.5; the octave-error property is tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import SAMPLE_RATE, HOP_LENGTH


# ---------------------------------------------------------------------------
# Framing helpers
# ---------------------------------------------------------------------------
def frame_signal(x: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """Split ``(T,)`` into ``(n_frames, frame_length)`` with the given hop.

    Non-centred and non-padded: frame ``m`` is ``x[m*hop : m*hop + frame_length]``,
    matching the STFT convention in :mod:`features` so frame indices share a
    common time origin.
    """
    if x.dim() != 1:
        raise ValueError("frame_signal expects a 1-D waveform")
    if x.numel() < frame_length:
        return x.new_zeros((0, frame_length))
    return x.unfold(0, frame_length, hop_length)


def rms_energy(x: torch.Tensor, frame_length: int = 400,
               hop_length: int = HOP_LENGTH) -> torch.Tensor:
    """Per-frame root-mean-square amplitude.

    For a sinusoid of amplitude ``A`` this is ``A/sqrt(2)`` (tested).
    """
    frames = frame_signal(x, frame_length, hop_length)
    if frames.numel() == 0:
        return x.new_zeros((0,))
    return frames.pow(2).mean(dim=1).clamp_min(0).sqrt()


# ---------------------------------------------------------------------------
# YIN fundamental-frequency estimation
# ---------------------------------------------------------------------------
def difference_function(frames: torch.Tensor, window: int,
                        tau_max: int) -> torch.Tensor:
    """YIN squared-difference function ``d(tau)``, shape ``(n_frames, tau_max+1)``.

    ``d(tau) = sum_{n=0}^{W-1} (x[n] - x[n+tau])^2``

    Each frame must supply ``W + tau_max`` samples so every lag uses exactly
    ``W`` terms; otherwise ``d`` would be biased toward large ``tau`` simply
    because fewer terms are summed.
    """
    n_frames = frames.shape[0]
    if frames.shape[1] < window + tau_max:
        raise ValueError("frames too short for the requested window and tau_max")
    ref = frames[:, :window]                                     # (F, W)
    # Shifted views for every lag: (F, tau_max+1, W)
    shifts = frames.unfold(1, window, 1)[:, :tau_max + 1, :]
    diff = ref.unsqueeze(1) - shifts
    return diff.pow(2).sum(dim=-1)


def cumulative_mean_normalized(d: torch.Tensor) -> torch.Tensor:
    """CMND: ``d'(0)=1``, ``d'(tau) = d(tau) / [(1/tau) sum_{j=1..tau} d(j)]``.

    Dividing by the running mean removes the overall downward trend of ``d`` and
    is what makes the *first* sufficiently deep dip (the true period) preferable
    to the deeper dip at twice the period.

    **Degenerate frames.** For a constant frame (digital silence, or DC) every
    ``d(tau)`` is zero and the ratio is ``0/0``. That is *no evidence of
    periodicity*, so the correct value is ``d' = 1`` (maximally aperiodic), not
    ``0``. Clamping the denominator instead would yield ``0``, i.e. below any
    voicing threshold -- reporting silence as a confidently pitched frame.
    """
    n_frames, n_lags = d.shape
    out = torch.ones_like(d)
    if n_lags <= 1:
        return out
    tail = d[:, 1:]
    running = torch.cumsum(tail, dim=1)
    denom = running / torch.arange(1, n_lags, device=d.device,
                                   dtype=d.dtype).unsqueeze(0)
    degenerate = denom <= 1e-12
    ratio = tail / denom.clamp_min(1e-12)
    out[:, 1:] = torch.where(degenerate, torch.ones_like(ratio), ratio)
    return out


def _parabolic_refine(dp: torch.Tensor, tau: int) -> float:
    """Sub-sample period refinement by fitting a parabola through the minimum."""
    if tau <= 0 or tau >= dp.numel() - 1:
        return float(tau)
    y0, y1, y2 = float(dp[tau - 1]), float(dp[tau]), float(dp[tau + 1])
    denom = 2.0 * (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(tau)
    return float(tau) + (y0 - y2) / denom


@dataclass
class F0Result:
    f0: torch.Tensor          # (n_frames,) Hz, 0 where unvoiced
    voiced: torch.Tensor      # (n_frames,) bool
    aperiodicity: torch.Tensor  # (n_frames,) CMND value at the chosen lag


def estimate_f0_yin(x: torch.Tensor, sample_rate: int = SAMPLE_RATE,
                    frame_length: int = 1024, hop_length: int = HOP_LENGTH,
                    f_min: float = 65.0, f_max: float = 400.0,
                    threshold: float = 0.1, min_rms: float = 1e-6) -> F0Result:
    """Estimate F0 per frame with YIN.

    Args:
        f_min/f_max: search range in Hz; sets ``tau_max = sr/f_min`` and
            ``tau_min = sr/f_max``.
        threshold: absolute CMND threshold. The first lag dipping below it is
            taken as the period; a frame whose minimum never dips below is
            declared unvoiced.
        min_rms: frames quieter than this are forced unvoiced. A second line of
            defence (alongside the degenerate-denominator handling in
            :func:`cumulative_mean_normalized`) so that silence can never be
            emitted as pitched -- the system fails closed on absent evidence.
    """
    if f_min <= 0 or f_max <= f_min:
        raise ValueError("require 0 < f_min < f_max")
    tau_min = max(1, int(sample_rate // f_max))
    tau_max = int(sample_rate // f_min)
    if tau_max <= tau_min:
        raise ValueError("f_min/f_max too close for this sample rate")

    total = frame_length + tau_max
    frames = frame_signal(x, total, hop_length)
    if frames.numel() == 0:
        empty = x.new_zeros((0,))
        return F0Result(empty, empty.bool(), empty)

    d = difference_function(frames, frame_length, tau_max)
    dp = cumulative_mean_normalized(d)
    # Per-frame RMS of the reference window, for the energy gate.
    frame_rms = frames[:, :frame_length].pow(2).mean(dim=1).clamp_min(0).sqrt()

    n_frames = dp.shape[0]
    f0 = torch.zeros(n_frames, dtype=x.dtype)
    voiced = torch.zeros(n_frames, dtype=torch.bool)
    aperiodicity = torch.ones(n_frames, dtype=x.dtype)

    for i in range(n_frames):
        row = dp[i]
        if float(frame_rms[i]) < min_rms:
            # Too quiet to carry pitch evidence: fail closed.
            aperiodicity[i] = 1.0
            continue
        search = row[tau_min:tau_max + 1]
        below = (search < threshold).nonzero().flatten()
        if below.numel() > 0:
            # First dip below threshold, then descend to its local minimum --
            # this is the step that avoids locking onto tau = 2T.
            tau = int(below[0]) + tau_min
            while tau + 1 <= tau_max and row[tau + 1] < row[tau]:
                tau += 1
            is_voiced = True
        else:
            tau = int(torch.argmin(search)) + tau_min
            is_voiced = False
        refined = _parabolic_refine(row, tau)
        aperiodicity[i] = row[tau]
        if is_voiced and refined > 0:
            f0[i] = sample_rate / refined
            voiced[i] = True
    return F0Result(f0=f0, voiced=voiced, aperiodicity=aperiodicity)


# ---------------------------------------------------------------------------
# Pauses
# ---------------------------------------------------------------------------
def detect_pauses(energy: torch.Tensor, threshold_db: float = -35.0,
                  min_frames: int = 5) -> List[Tuple[int, int]]:
    """Find silent runs as ``[start, end)`` frame intervals.

    The threshold is relative to the loudest frame, so it is invariant to
    recording gain: ``20*log10(e / max(e)) < threshold_db``. Runs shorter than
    ``min_frames`` are ignored, which is what separates a *pause* (discourse
    evidence) from an inter-phone stop gap.
    """
    if energy.numel() == 0:
        return []
    peak = float(energy.max())
    if peak <= 0:
        return [(0, int(energy.numel()))]
    rel_db = 20.0 * torch.log10((energy / peak).clamp_min(1e-12))
    silent = (rel_db < threshold_db).tolist()

    pauses: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, s in enumerate(silent):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_frames:
                pauses.append((start, i))
            start = None
    if start is not None and len(silent) - start >= min_frames:
        pauses.append((start, len(silent)))
    return pauses


# ---------------------------------------------------------------------------
# Combined extractor
# ---------------------------------------------------------------------------
class ProsodyExtractor(nn.Module):
    """Waveform -> per-frame prosody features.

    Emits a ``(n_frames, 4)`` tensor of
    ``[log-F0 (0 if unvoiced), voiced flag, log-energy, aperiodicity]``.

    log-F0 rather than raw Hz because pitch is perceived multiplicatively (an
    octave is a constant additive step in log space), so a linear layer over
    log-F0 sees musically/linguistically meaningful distances.
    """

    N_FEATURES = 4

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 hop_length: int = HOP_LENGTH, frame_length: int = 1024,
                 energy_frame: int = 400, f_min: float = 65.0,
                 f_max: float = 400.0, threshold: float = 0.1) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.energy_frame = energy_frame
        self.f_min = f_min
        self.f_max = f_max
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        res = estimate_f0_yin(waveform, self.sample_rate, self.frame_length,
                              self.hop_length, self.f_min, self.f_max,
                              self.threshold)
        energy = rms_energy(waveform, self.energy_frame, self.hop_length)
        n = min(res.f0.numel(), energy.numel())
        if n == 0:
            return waveform.new_zeros((0, self.N_FEATURES))

        log_f0 = torch.where(res.voiced[:n],
                             torch.log(res.f0[:n].clamp_min(1e-6)),
                             torch.zeros(n, dtype=waveform.dtype))
        feats = torch.stack([
            log_f0,
            res.voiced[:n].to(waveform.dtype),
            torch.log(energy[:n].clamp_min(1e-10)),
            res.aperiodicity[:n],
        ], dim=1)
        return feats

    def resample_to(self, feats: torch.Tensor, n_target: int) -> torch.Tensor:
        """Linearly resample prosody frames onto another frame grid.

        Prosody needs a longer analysis window than the STFT, so its frame count
        differs from the mel grid; this aligns the two without pretending they
        were computed at the same resolution.
        """
        if feats.numel() == 0 or n_target <= 0:
            return feats.new_zeros((max(n_target, 0), self.N_FEATURES))
        if feats.shape[0] == n_target:
            return feats
        x = feats.t().unsqueeze(0)                       # (1, C, n_frames)
        out = F.interpolate(x, size=n_target, mode="linear", align_corners=True)
        return out.squeeze(0).t().contiguous()
