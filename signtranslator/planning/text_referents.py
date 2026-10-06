"""Symmetric referent equality scores over generated events, without ID decoding."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from ..data.governed_text import encode_plaintext_transcripts
from ..training.objectives import SupportedObjective, SupportedTerm
from ..inference_context import preserving_eval_mode
from .label_sequence import label_sequence_targets
from .referents import ReferentLogits, referent_loss_per_example
from .text_relations import RelationalSequenceCandidate, RelationalTextConfig, RelationalTextSIRModel


@dataclass(frozen=True)
class ReferentTextConfig(RelationalTextConfig):
    referent_supervision: str = 'known-unordered-equality-v1'

    def __post_init__(self):
        super().__post_init__()
        if self.referent_supervision != 'known-unordered-equality-v1':
            raise ValueError('unsupported referent supervision')


@dataclass(frozen=True)
class ReferentSequenceCandidate:
    relational: RelationalSequenceCandidate
    equality_logits: torch.Tensor | None
    pair_valid: torch.Tensor | None  # symmetric domain excluding self pairs


class ReferentTextSIRModel(RelationalTextSIRModel):
    """Predict same-referent scores, not arbitrary corpus ID numbers.

    Sum/product features impose exact symmetry. Pairwise predictions need not be
    transitive and cannot be used as a partition without explicit decoding.
    """
    def __init__(self, vocabulary, **kwargs):
        super().__init__(vocabulary, **kwargs)
        self.model_cfg = ReferentTextConfig(**asdict(self.model_cfg))
        size = self.model_cfg.hidden_dim + self.model_cfg.embedding_dim
        self.referent_head = nn.Sequential(nn.Linear(2 * size, self.model_cfg.hidden_dim),
                                           nn.Tanh(), nn.Linear(self.model_cfg.hidden_dim, 1))

    def _referents(self, features, current_label_ids):
        events = torch.cat((features, self.label_embedding(current_label_ids)), dim=-1)
        a, b = events[:, :, None, :], events[:, None, :, :]
        return self.referent_head(torch.cat((a + b, a * b), dim=-1)).squeeze(-1)

    def training_step(self, batch, *, weights: dict) -> SupportedObjective:
        required = {'sir_sequence', 'event_timing', 'sir_relations', 'referent_equality'}
        if (not isinstance(weights, dict) or set(weights) != required
                or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights.values())):
            raise ValueError('explicit positive weights for all four reference-model branches required')
        parent = super().training_step(batch, weights={k: weights[k] for k in required if k != 'referent_equality'})
        text = encode_plaintext_transcripts(batch.transcript_payloads, batch.annotations,
                                           declared_encoding=self.model_cfg.declared_encoding,
                                           max_bytes=self.model_cfg.max_bytes)
        target = label_sequence_targets(batch.annotations, self.vocabulary)
        device = self.byte_embedding.weight.device
        inputs = target.inputs.to(device)
        features, _ = self.teacher_features(text.token_ids.to(device), text.lengths, inputs, target.lengths)
        scores = self._referents(features[:, :-1], inputs[:, 1:])
        values, support = referent_loss_per_example(
            ReferentLogits(scores, target.annotation_sha256, target.vocabulary_sha256),
            batch.annotations, self.vocabulary)
        terms = dict(parent.terms)
        terms['referent_equality'] = SupportedTerm(values.sum(), int(support.sum()))
        return SupportedObjective(terms, weights, parent.population_size)

    @torch.no_grad()
    def generate_referents(self, token_ids, lengths, *, origins_seconds):
        with preserving_eval_mode(self):
            candidates = super().generate_relational(token_ids, lengths, origins_seconds=origins_seconds)
            lookup = {entry: index for index, entry in enumerate(self.vocabulary.entries)}
            results = []
            for row, candidate in enumerate(candidates):
                events = candidate.temporal.events
                if candidate.temporal.status != 'terminated':
                    results.append(ReferentSequenceCandidate(candidate, None, None))
                    continue
                prefix = [1] + [lookup[event.label] + 2 for event in events]
                inputs = torch.tensor([prefix], dtype=torch.int64, device=token_ids.device)
                features, _ = self.teacher_features(token_ids[row:row+1], lengths[row:row+1], inputs,
                                                     torch.tensor([len(prefix)]))
                scores = self._referents(features[:, :-1], inputs[:, 1:])[0]
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError('nonfinite generated referent scores')
                valid = ~torch.eye(len(events), dtype=torch.bool, device=scores.device)
                results.append(ReferentSequenceCandidate(candidate, torch.where(valid, scores, torch.zeros_like(scores)), valid))
            return tuple(results)
