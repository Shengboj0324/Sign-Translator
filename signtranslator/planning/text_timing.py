"""Joint source-text label/stop and continuous event-timing baseline.

Generated timed labels are not complete SIR graphs or calibrated ASL decisions.
Annotation-clock origins are explicit inputs, never inferred from target events.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from ..data.governed_corpus import GovernedMotionBatch
from ..data.governed_text import encode_plaintext_transcripts
from .event_timing import AlignedEventIntervals, aligned_timing_loss, event_intervals
from ..inference_context import preserving_eval_mode
from .label_sequence import SIRLabelSequenceLogits, label_sequence_loss, label_sequence_targets
from .label_vocabulary import SIRLabelEntry
from .text_sequence import AutoregressiveTextConfig, AutoregressiveTextSIRLabelModel


@dataclass(frozen=True)
class TemporalTextConfig(AutoregressiveTextConfig):
    timing_scale_seconds: float

    def __post_init__(self):
        super().__post_init__()
        if (type(self.timing_scale_seconds) not in (int, float)
                or not math.isfinite(self.timing_scale_seconds) or self.timing_scale_seconds <= 0):
            raise ValueError('timing_scale_seconds must be finite and positive')


@dataclass(frozen=True)
class TimedLabel:
    label: SIRLabelEntry
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class TemporalSequenceCandidate:
    events: tuple[TimedLabel, ...]
    status: str
    diagnostic_prefix: tuple[SIRLabelEntry, ...]
    vocabulary_sha256: str
    origin_seconds: float


class TemporalTextSIRModel(AutoregressiveTextSIRLabelModel):
    """Shared causal features with a separate continuous onset/duration head.

    Offsets are relative to the source interval origin retained by corpus
    admission, not the earliest target event. No monotone event order is imposed.
    The physical Huber scale is a declared modelling choice, not a measured
    reviewer tolerance or acceptance threshold.
    """

    def __init__(self, vocabulary, *, timing_scale_seconds: float, **kwargs):
        super().__init__(vocabulary, **kwargs)
        self.model_cfg = TemporalTextConfig(**asdict(self.model_cfg),
                                            timing_scale_seconds=timing_scale_seconds)
        self.timing_head = nn.Linear(self.model_cfg.hidden_dim, 2)

    def training_step(self, batch: GovernedMotionBatch, *, weights: dict, with_target_cells=False):
        if not isinstance(batch, GovernedMotionBatch):
            raise ValueError('typed governed training batch required')
        self._check_binding()
        if not isinstance(weights, dict) or set(weights) != {'sir_sequence', 'event_timing'}:
            raise ValueError('declare both sir_sequence and event_timing weights')
        if any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights.values()):
            raise ValueError('temporal objective weights must be finite and positive')
        text = encode_plaintext_transcripts(
            batch.transcript_payloads, batch.annotations,
            declared_encoding=self.model_cfg.declared_encoding, max_bytes=self.model_cfg.max_bytes)
        target = label_sequence_targets(batch.annotations, self.vocabulary)
        if len(batch.annotation_extents) != len(batch.annotations):
            raise ValueError('source clock extent cardinality mismatch')
        origins = torch.tensor([extent[0] for extent in batch.annotation_extents], dtype=torch.float64)
        device = self.byte_embedding.weight.device
        features, valid = self.teacher_features(text.token_ids.to(device), text.lengths,
                                                target.inputs.to(device), target.lengths)
        scores = self.classifier(features)
        scores = torch.where(valid[:, :, None], scores, torch.zeros_like(scores))
        sequence_loss = label_sequence_loss(SIRLabelSequenceLogits(
            scores, target.vocabulary_sha256, target.annotation_sha256), batch.annotations, self.vocabulary,
            with_target_cells=with_target_cells)
        raw = self.timing_head(features[:, :-1, :])
        active = torch.arange(raw.shape[1])[None, :] < (target.lengths - 1)[:, None]
        intervals = event_intervals(raw, origins, active)
        timing_loss = aligned_timing_loss(AlignedEventIntervals(
            intervals, target.vocabulary_sha256, target.annotation_sha256),
            batch.annotations, self.vocabulary, scale_seconds=self.model_cfg.timing_scale_seconds,
            with_target_cells=with_target_cells)
        if with_target_cells:
            sequence_loss, sequence_cells = sequence_loss
            timing_loss, timing_cells = timing_loss
        losses = {'sir_sequence': sequence_loss, 'event_timing': timing_loss,
                'total': weights['sir_sequence'] * sequence_loss + weights['event_timing'] * timing_loss}
        if with_target_cells:
            return losses, {'sir_sequence': sequence_cells, 'event_timing': timing_cells}
        return losses

    @torch.no_grad()
    def generate_temporal(self, token_ids: torch.Tensor, lengths: torch.Tensor, *,
                          origins_seconds: torch.Tensor) -> tuple[TemporalSequenceCandidate, ...]:
        """Generate labels, then time them using their own causal predicted prefix.

        Origins are caller-declared clock coordinates, not source calibration
        evidence. The second pass consumes model-generated labels only; no
        reference event count, label or interval is required. Labels are decoded
        once and are not changed by a timing repair or temporal sorting step.
        """
        if (not isinstance(origins_seconds, torch.Tensor) or origins_seconds.dtype != torch.float64
                or origins_seconds.shape != (token_ids.shape[0],)
                or not bool(torch.isfinite(origins_seconds).all())):
            raise ValueError('explicit finite float64 origins required for temporal generation')
        with preserving_eval_mode(self):
            candidates = super().generate(token_ids, lengths)
            lookup = {entry: i for i, entry in enumerate(self.vocabulary.entries)}
            results = []
            for row, candidate in enumerate(candidates):
                origin = float(origins_seconds[row])
                if candidate.status != 'terminated':
                    results.append(TemporalSequenceCandidate(
                        (), candidate.status, candidate.diagnostic_prefix, candidate.vocabulary_sha256, origin))
                    continue
                prefix = [1] + [lookup[entry] + 2 for entry in candidate.labels]
                inputs = torch.tensor([prefix], dtype=torch.int64, device=token_ids.device)
                features, _ = self.teacher_features(token_ids[row:row+1], lengths[row:row+1],
                                                     inputs, torch.tensor([len(prefix)]))
                raw = self.timing_head(features[:, :-1, :])
                intervals = event_intervals(raw, origins_seconds[row:row+1],
                                            torch.ones(raw.shape[:2], dtype=torch.bool))
                events = tuple(TimedLabel(entry, float(start), float(end))
                               for entry, (start, end) in zip(candidate.labels, intervals[0].tolist()))
                results.append(TemporalSequenceCandidate(
                    events, candidate.status, candidate.diagnostic_prefix, candidate.vocabulary_sha256, origin))
            return tuple(results)
