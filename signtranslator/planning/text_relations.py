"""Partially supervised relation scores attached to generated timed labels.

Scores remain uncalibrated. Missing relation annotations are masked, and this
module supplies neither a threshold-derived SIR graph nor missing referent IDs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from ..data.governed_corpus import GovernedMotionBatch
from ..data.governed_text import encode_plaintext_transcripts
from ..training.objectives import SupportedObjective, SupportedTerm
from ..inference_context import preserving_eval_mode
from .label_sequence import label_sequence_targets
from .relations import SIRRelationLogits, relation_loss_per_example
from .tensors import EDGE_TYPES
from .text_timing import TemporalSequenceCandidate, TemporalTextConfig, TemporalTextSIRModel


@dataclass(frozen=True)
class RelationalTextConfig(TemporalTextConfig):
    relation_supervision: str = 'recorded-positive-structural-negative-v1'
    objective_normalization: str = 'full-population-contribution-v1'

    def __post_init__(self):
        super().__post_init__()
        if self.relation_supervision != 'recorded-positive-structural-negative-v1':
            raise ValueError('unsupported relation supervision semantics')
        if self.objective_normalization != 'full-population-contribution-v1':
            raise ValueError('unsupported joint objective normalization')


@dataclass(frozen=True)
class RelationalSequenceCandidate:
    temporal: TemporalSequenceCandidate
    relation_logits: torch.Tensor | None  # E,E,R; absent for failed label decoding
    relation_valid: torch.Tensor | None   # E,E; self edges excluded
    relation_types: tuple = EDGE_TYPES


class RelationalTextSIRModel(TemporalTextSIRModel):
    """Directed multilabel edge head over causal features and current labels.

    The pair head sees the complete label list during relation prediction. In
    training these are reviewed teacher labels; inference uses generated labels.
    It receives no reference edges, referents, loci or event intervals as inputs.
    """
    governed_objective_schema_version = 1

    def __init__(self, vocabulary, **kwargs):
        super().__init__(vocabulary, **kwargs)
        self.model_cfg = RelationalTextConfig(**asdict(self.model_cfg))
        event_dim = self.model_cfg.hidden_dim + self.model_cfg.embedding_dim
        self.relation_head = nn.Sequential(nn.Linear(2 * event_dim, self.model_cfg.hidden_dim),
                                           nn.Tanh(), nn.Linear(self.model_cfg.hidden_dim, len(EDGE_TYPES)))

    def _relations(self, features, current_label_ids):
        events = torch.cat((features, self.label_embedding(current_label_ids)), dim=-1)
        count = events.shape[1]
        source = events[:, :, None, :].expand(-1, -1, count, -1)
        target = events[:, None, :, :].expand(-1, count, -1, -1)
        return self.relation_head(torch.cat((source, target), dim=-1))

    def training_step(self, batch: GovernedMotionBatch, *, weights: dict) -> SupportedObjective:
        required = {'sir_sequence', 'event_timing', 'sir_relations'}
        if not isinstance(weights, dict) or set(weights) != required:
            raise ValueError('declare sequence, timing and relation objective weights')
        if any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights.values()):
            raise ValueError('all relation-model weights must be finite and positive')
        losses = super().training_step(batch, weights={k: weights[k] for k in ('sir_sequence', 'event_timing')})
        text = encode_plaintext_transcripts(
            batch.transcript_payloads, batch.annotations,
            declared_encoding=self.model_cfg.declared_encoding, max_bytes=self.model_cfg.max_bytes)
        target = label_sequence_targets(batch.annotations, self.vocabulary)
        device = self.byte_embedding.weight.device
        inputs = target.inputs.to(device)
        features, _ = self.teacher_features(text.token_ids.to(device), text.lengths, inputs, target.lengths)
        scores = self._relations(features[:, :-1, :], inputs[:, 1:])
        per_example, supported = relation_loss_per_example(SIRRelationLogits(
            scores, target.annotation_sha256, target.vocabulary_sha256), batch.annotations, self.vocabulary)
        count = len(batch.annotations)
        terms = {name: SupportedTerm(losses[name] * count, count)
                 for name in ('sir_sequence', 'event_timing')}
        terms['sir_relations'] = SupportedTerm(per_example.sum(), int(supported.sum()))
        return SupportedObjective(terms, weights, count)

    @torch.no_grad()
    def generate_relational(self, token_ids: torch.Tensor, lengths: torch.Tensor, *,
                            origins_seconds: torch.Tensor) -> tuple[RelationalSequenceCandidate, ...]:
        """Score generated event pairs without inventing a calibrated threshold."""
        with preserving_eval_mode(self):
            candidates = super().generate_temporal(token_ids, lengths, origins_seconds=origins_seconds)
            lookup = {entry: index for index, entry in enumerate(self.vocabulary.entries)}
            results = []
            for row, candidate in enumerate(candidates):
                if candidate.status != 'terminated':
                    results.append(RelationalSequenceCandidate(candidate, None, None))
                    continue
                prefix = [1] + [lookup[event.label] + 2 for event in candidate.events]
                inputs = torch.tensor([prefix], dtype=torch.int64, device=token_ids.device)
                features, _ = self.teacher_features(token_ids[row:row+1], lengths[row:row+1],
                                                     inputs, torch.tensor([len(prefix)]))
                scores = self._relations(features[:, :-1, :], inputs[:, 1:])[0]
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError('nonfinite generated relation scores')
                valid = ~torch.eye(len(candidate.events), dtype=torch.bool, device=scores.device)
                scores = torch.where(valid[..., None], scores, torch.zeros_like(scores))
                results.append(RelationalSequenceCandidate(candidate, scores, valid))
            return tuple(results)
