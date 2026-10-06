"""Source-conditioned locus logits bound to the reviewed convention alphabet."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from ..data.governed_text import encode_plaintext_transcripts
from ..training.objectives import SupportedObjective, SupportedTerm
from ..inference_context import preserving_eval_mode
from .label_sequence import label_sequence_targets
from .loci import LocusAlphabet, LocusLogits, locus_loss_per_example
from .text_referents import ReferentTextConfig, ReferentTextSIRModel, ReferentSequenceCandidate


@dataclass(frozen=True, kw_only=True)
class LocusTextConfig(ReferentTextConfig):
    locus_count: int
    locus_supervision: str = 'known-convention-alphabet-v1'

    def __post_init__(self):
        super().__post_init__()
        if type(self.locus_count) is not int or not 1 <= self.locus_count <= 4096:
            raise ValueError('bounded positive locus count required')
        if self.locus_supervision != 'known-convention-alphabet-v1':
            raise ValueError('unsupported locus supervision')


@dataclass(frozen=True)
class LocusSequenceCandidate:
    referential: ReferentSequenceCandidate
    locus_logits: torch.Tensor | None
    convention_sha256: str
    locus_identities: tuple[str, ...]


class LocusTextSIRModel(ReferentTextSIRModel):
    def __init__(self, vocabulary, *, locus_alphabet: LocusAlphabet, **kwargs):
        if not isinstance(locus_alphabet, LocusAlphabet) or locus_alphabet.convention != vocabulary.convention:
            raise ValueError('locus alphabet must match the governed vocabulary convention')
        super().__init__(vocabulary, **kwargs)
        self.locus_alphabet = locus_alphabet
        self.model_cfg = LocusTextConfig(**asdict(self.model_cfg), locus_count=len(locus_alphabet.identities))
        size = self.model_cfg.hidden_dim + self.model_cfg.embedding_dim
        self.locus_head = nn.Linear(size, self.model_cfg.locus_count)

    def _check_binding(self):
        super()._check_binding()
        if (self.locus_alphabet.convention != self.vocabulary.convention
                or len(self.locus_alphabet.identities) != self.model_cfg.locus_count):
            raise ValueError('model locus binding changed')

    def _loci(self, features, current_label_ids):
        return self.locus_head(torch.cat((features, self.label_embedding(current_label_ids)), dim=-1))

    def training_step(self, batch, *, weights: dict) -> SupportedObjective:
        required = {'sir_sequence', 'event_timing', 'sir_relations', 'referent_equality', 'locus_assignment'}
        if (not isinstance(weights, dict) or set(weights) != required
                or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights.values())):
            raise ValueError('explicit positive weights for all five locus-model branches required')
        parent = super().training_step(batch, weights={k: weights[k] for k in required if k != 'locus_assignment'})
        text = encode_plaintext_transcripts(batch.transcript_payloads, batch.annotations,
                                           declared_encoding=self.model_cfg.declared_encoding,
                                           max_bytes=self.model_cfg.max_bytes)
        target = label_sequence_targets(batch.annotations, self.vocabulary)
        device = self.byte_embedding.weight.device
        inputs = target.inputs.to(device)
        features, _ = self.teacher_features(text.token_ids.to(device), text.lengths, inputs, target.lengths)
        values, support, cells = locus_loss_per_example(LocusLogits(
            self._loci(features[:, :-1], inputs[:, 1:]), target.annotation_sha256,
            self.model_cfg.convention_sha256, target.vocabulary_sha256),
            batch.annotations, self.vocabulary, self.locus_alphabet, with_target_cells=True)
        terms = dict(parent.terms)
        terms['locus_assignment'] = SupportedTerm(values.sum(), int(support.sum()),
                                                tuple(support.detach().cpu().tolist()), cells)
        return SupportedObjective(terms, weights, parent.population_size)

    @torch.no_grad()
    def generate_loci(self, token_ids, lengths, *, origins_seconds):
        with preserving_eval_mode(self):
            candidates = super().generate_referents(token_ids, lengths, origins_seconds=origins_seconds)
            lookup = {entry: i for i, entry in enumerate(self.vocabulary.entries)}
            results = []
            for row, candidate in enumerate(candidates):
                temporal = candidate.relational.temporal
                scores = None
                if temporal.status == 'terminated':
                    prefix = [1] + [lookup[event.label] + 2 for event in temporal.events]
                    inputs = torch.tensor([prefix], dtype=torch.int64, device=token_ids.device)
                    features, _ = self.teacher_features(token_ids[row:row+1], lengths[row:row+1], inputs,
                                                         torch.tensor([len(prefix)]))
                    scores = self._loci(features[:, :-1], inputs[:, 1:])[0]
                    if not bool(torch.isfinite(scores).all()):
                        raise FloatingPointError('nonfinite generated locus scores')
                results.append(LocusSequenceCandidate(candidate, scores, self.model_cfg.convention_sha256,
                                                       self.locus_alphabet.identities))
            return tuple(results)
