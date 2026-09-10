"""Continuous sign-language recognition (sign -> gloss) with CTC.

This is the *recognition* direction that makes the system bidirectional. The
ST-GCN encoder produces a per-frame feature sequence; a linear head emits
per-frame class log-probabilities over the gloss vocabulary plus a blank symbol,
and Connectionist Temporal Classification (Graves et al., 2006) aligns the
unsegmented frame sequence to the (shorter) gloss label sequence without frame-
level annotation -- the standard formulation for continuous sign recognition.

Convention: class index ``0`` is the CTC blank; gloss ids occupy ``1..V``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stgcn import STGCNEncoder


def _ctc_target_rows(targets: torch.Tensor,
                     target_lengths: torch.Tensor) -> List[List[int]]:
    integer_dtypes = {torch.int32, torch.int64}
    if targets.dtype not in integer_dtypes or target_lengths.dtype not in integer_dtypes:
        raise TypeError("CTC targets and target_lengths must be int32 or int64")
    if target_lengths.ndim != 1:
        raise ValueError("CTC target_lengths must be one-dimensional")
    lengths = [int(value) for value in target_lengths.detach().cpu().tolist()]
    if any(length < 1 for length in lengths):
        raise ValueError("CTC targets must be non-empty")
    if targets.ndim == 1:
        if sum(lengths) != targets.numel():
            raise ValueError("concatenated CTC targets do not match target_lengths")
        flat = [int(value) for value in targets.detach().cpu().tolist()]
        rows, offset = [], 0
        for length in lengths:
            rows.append(flat[offset:offset + length])
            offset += length
        return rows
    if targets.ndim == 2:
        if targets.shape[0] != len(lengths):
            raise ValueError("padded CTC targets do not match target_lengths batch")
        if any(length > targets.shape[1] for length in lengths):
            raise ValueError("CTC target length exceeds padded target width")
        return [
            [int(value) for value in targets[row, :length].detach().cpu().tolist()]
            for row, length in enumerate(lengths)
        ]
    raise ValueError("CTC targets must be concatenated (1-D) or padded (2-D)")


def assert_ctc_feasible(targets: torch.Tensor, target_lengths: torch.Tensor,
                        input_lengths: torch.Tensor, *, num_classes: int,
                        maximum_input_length: int,
                        expected_batch_size: int) -> None:
    """Reject every impossible CTC alignment before evaluating the objective.

    For target ``y`` of length ``L``, CTC needs one additional frame between each
    adjacent repeated label.  Thus the exact minimum is

    ``L + sum(y[i] == y[i-1] for i in 1..L-1)``.

    Checking only ``T >= L`` is incorrect, and ``zero_infinity=True`` would silently
    turn such invalid examples into zero-loss observations.
    """
    if input_lengths.dtype not in {torch.int32, torch.int64}:
        raise TypeError("CTC input_lengths must be int32 or int64")
    if input_lengths.ndim != 1 or input_lengths.shape != target_lengths.shape:
        raise ValueError("CTC input_lengths and target_lengths must be aligned vectors")
    if input_lengths.numel() != expected_batch_size:
        raise ValueError("CTC length vectors do not match the log-probability batch")
    lengths = [int(value) for value in input_lengths.detach().cpu().tolist()]
    if any(length < 1 or length > maximum_input_length for length in lengths):
        raise ValueError("CTC input length is outside the emitted log-probability range")
    rows = _ctc_target_rows(targets, target_lengths)
    for row_index, (tokens, input_length) in enumerate(zip(rows, lengths)):
        if any(token <= 0 or token >= num_classes for token in tokens):
            raise ValueError(
                f"CTC target row {row_index} contains blank or out-of-range labels")
        minimum = len(tokens) + sum(
            left == right for left, right in zip(tokens, tokens[1:]))
        if input_length < minimum:
            raise ValueError(
                f"CTC target row {row_index} requires at least {minimum} emitted "
                f"frames, got {input_length}")


def ctc_greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> List[List[int]]:
    """Best-path (greedy) CTC decoding.

    Collapses consecutive duplicate labels, then removes blanks.

    Args:
        log_probs: ``(N, T, C)`` per-frame log-probabilities.
        blank: blank class index.

    Returns:
        A list of decoded label sequences (one per batch element).
    """
    best = log_probs.argmax(dim=-1)  # (N, T)
    results: List[List[int]] = []
    for seq in best.tolist():
        out: List[int] = []
        prev = blank
        for s in seq:
            if s != prev and s != blank:
                out.append(s)
            prev = s
        results.append(out)
    return results


class SignRecognizer(nn.Module):
    """ST-GCN encoder + CTC head mapping a pose clip to a gloss sequence."""

    def __init__(self, encoder: STGCNEncoder, num_glosses: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_glosses = num_glosses
        self.num_classes = num_glosses + 1  # +1 for blank (index 0)
        self.classifier = nn.Linear(encoder.out_dim, self.num_classes)
        # Impossible alignments are rejected explicitly before this loss is called.
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=False)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """pose (N, C, T, V) -> log-probs (N, T, num_classes)."""
        feats = self.encoder(pose, return_sequence=True)  # (N, T, D)
        logits = self.classifier(feats)
        return F.log_softmax(logits, dim=-1)

    def loss(self, pose: torch.Tensor, targets: torch.Tensor,
             target_lengths: torch.Tensor,
             input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CTC loss.

        Args:
            pose: ``(N, C, T, V)``.
            targets: concatenated or padded gloss ids in ``1..V`` (no blanks).
            target_lengths: ``(N,)`` true length of each target.
            input_lengths: ``(N,)`` valid frame counts (defaults to full T).
        """
        log_probs = self.forward(pose)          # (N, T, C)
        n, t, _ = log_probs.shape
        if input_lengths is None:
            input_lengths = torch.full((n,), t, dtype=torch.long,
                                       device=log_probs.device)
        return self.loss_from_log_probs(log_probs, targets, target_lengths,
                                        input_lengths)

    def loss_from_log_probs(self, log_probs: torch.Tensor, targets: torch.Tensor,
                            target_lengths: torch.Tensor,
                            input_lengths: torch.Tensor) -> torch.Tensor:
        if log_probs.ndim != 3 or log_probs.shape[2] != self.num_classes:
            raise ValueError("sign CTC log-probabilities have an invalid shape")
        if not torch.isfinite(log_probs).all():
            raise FloatingPointError("sign CTC log-probabilities are non-finite")
        assert_ctc_feasible(
            targets, target_lengths, input_lengths,
            num_classes=self.num_classes,
            maximum_input_length=log_probs.shape[1],
            expected_batch_size=log_probs.shape[0],
        )
        loss = self.ctc(log_probs.permute(1, 0, 2), targets,
                        input_lengths, target_lengths)
        if not torch.isfinite(loss):
            raise FloatingPointError("sign CTC loss is non-finite")
        return loss

    @torch.no_grad()
    def decode(self, pose: torch.Tensor) -> List[List[int]]:
        self.eval()
        return ctc_greedy_decode(self.forward(pose))


def word_error_rate(hypotheses: List[List[int]],
                    references: List[List[int]]) -> float:
    """Mean word (gloss) error rate = Levenshtein distance / reference length."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must align")
    total_dist, total_len = 0, 0
    for hyp, ref in zip(hypotheses, references):
        total_dist += _levenshtein(hyp, ref)
        total_len += max(len(ref), 1)
    return total_dist / max(total_len, 1)


def _levenshtein(a: List[int], b: List[int]) -> int:
    """Classic edit distance (insertions + deletions + substitutions)."""
    m, n = len(a), len(b)
    dp: np.ndarray = np.zeros((m + 1, n + 1), dtype=np.int64)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[m, n])
