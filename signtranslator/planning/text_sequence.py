"""Text-to-kind/label autoregression without supplied inference event counts.

Outputs are uncalibrated lexical candidates, not temporal SIR or approved ASL.
Neither event serialization order nor a decoder stop implies temporal structure.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..data.governed_corpus import GovernedMotionBatch
from ..data.governed_text import encode_plaintext_transcripts
from ..inference_context import preserving_eval_mode
from .label_sequence import SIRLabelSequenceLogits, label_sequence_loss, label_sequence_targets
from .label_vocabulary import GovernedLabelVocabulary, SIRLabelEntry
from .text_labels import ScaffoldedTextConfig, encode_source_hidden


@dataclass(frozen=True)
class AutoregressiveTextConfig(ScaffoldedTextConfig):
    """Versioned architecture/capacity/vocabulary binding for the sequence model."""


@dataclass(frozen=True)
class LabelSequenceCandidate:
    labels: tuple[SIRLabelEntry, ...]
    status: str                        # terminated, empty_prediction, capacity_exceeded
    diagnostic_prefix: tuple[SIRLabelEntry, ...]
    vocabulary_sha256: str


class AutoregressiveTextSIRLabelModel(nn.Module):
    """Byte encoder and causal label decoder, trained with shifted teacher inputs.

    The objective is ``sir_sequence``. Training includes one observed STOP per
    example. Greedy decoding needs only source IDs and lengths; STOP at the first
    step or failure to stop within capacity returns no usable labels. A terminated
    result still needs timing/graph construction and calibrated linguistic review.
    """
    governed_batch_schema_version = 1

    def __init__(self, vocabulary: GovernedLabelVocabulary, *, declared_encoding: str,
                 max_bytes: int = 4096, max_events: int = 128,
                 embedding_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        if not isinstance(vocabulary, GovernedLabelVocabulary):
            raise ValueError('a governed vocabulary is required')
        self.vocabulary = vocabulary
        self.model_cfg = AutoregressiveTextConfig(
            embedding_dim, hidden_dim, max_bytes, max_events,
            vocabulary.lexicon.sha256, vocabulary.convention.sha256, declared_encoding)
        self.byte_embedding = nn.Embedding(257, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # Input PAD=0 and START=1 are never output classes. STOP is output zero;
        # output class j>0 feeds back as input j+1 on the next step.
        self.label_embedding = nn.Embedding(len(vocabulary.entries) + 2, embedding_dim, padding_idx=0)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, len(vocabulary.entries) + 1)

    def _check_binding(self):
        if (self.vocabulary.lexicon.sha256 != self.model_cfg.lexicon_sha256
                or self.vocabulary.convention.sha256 != self.model_cfg.convention_sha256):
            raise ValueError('model vocabulary differs from constructed class binding')

    def _source(self, token_ids, lengths):
        self._check_binding()
        return encode_source_hidden(token_ids, lengths, max_bytes=self.model_cfg.max_bytes,
                                    embedding=self.byte_embedding, encoder=self.encoder)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor,
                decoder_inputs: torch.Tensor, decoder_lengths: torch.Tensor) -> torch.Tensor:
        """Teacher-forced scores; callers must supply shifted label inputs."""
        output, valid = self.teacher_features(token_ids, lengths, decoder_inputs, decoder_lengths)
        scores = self.classifier(output)
        return torch.where(valid[:, :, None], scores, torch.zeros_like(scores))

    def teacher_features(self, token_ids: torch.Tensor, lengths: torch.Tensor,
                         decoder_inputs: torch.Tensor, decoder_lengths: torch.Tensor):
        """Causal decoder features shared by categorical and temporal heads."""
        hidden = self._source(token_ids, lengths)
        if any(not isinstance(t, torch.Tensor) or t.dtype != torch.int64
               for t in (decoder_inputs, decoder_lengths)):
            raise ValueError('decoder IDs and lengths must be int64 tensors')
        if (decoder_inputs.ndim != 2 or decoder_inputs.shape[0] != token_ids.shape[0]
                or not 2 <= decoder_inputs.shape[1] <= self.model_cfg.max_events + 1
                or decoder_lengths.shape != (token_ids.shape[0],)
                or bool((decoder_lengths < 2).any())
                or bool((decoder_lengths > decoder_inputs.shape[1]).any())):
            raise ValueError('invalid decoder dimensions or event capacity')
        columns = torch.arange(decoder_inputs.shape[1], device=decoder_inputs.device)[None, :]
        valid = columns < decoder_lengths.to(decoder_inputs.device)[:, None]
        lexical = valid & (columns > 0)
        if (bool((decoder_inputs[:, 0] != 1).any())
                or bool(((decoder_inputs[lexical] < 2)
                         | (decoder_inputs[lexical] >= self.label_embedding.num_embeddings)).any())
                or bool((decoder_inputs[~valid] != 0).any())):
            raise ValueError('invalid shifted decoder IDs or padding')
        packed = pack_padded_sequence(self.label_embedding(decoder_inputs),
                                      decoder_lengths.detach().cpu(), batch_first=True,
                                      enforce_sorted=False)
        output, _ = self.decoder(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        total_length=decoder_inputs.shape[1])
        return output, valid

    def training_step(self, batch: GovernedMotionBatch, *, weights: dict) -> dict:
        if not isinstance(batch, GovernedMotionBatch):
            raise ValueError('typed governed training batch required')
        self._check_binding()
        if not isinstance(weights, dict) or set(weights) != {'sir_sequence'}:
            raise ValueError('declare only the sir_sequence objective weight')
        weight = weights['sir_sequence']
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('sir_sequence weight must be finite and positive')
        text = encode_plaintext_transcripts(
            batch.transcript_payloads, batch.annotations,
            declared_encoding=self.model_cfg.declared_encoding, max_bytes=self.model_cfg.max_bytes)
        target = label_sequence_targets(batch.annotations, self.vocabulary)
        device = self.byte_embedding.weight.device
        scores = self(text.token_ids.to(device), text.lengths,
                      target.inputs.to(device), target.lengths)
        prediction = SIRLabelSequenceLogits(scores, target.vocabulary_sha256, target.annotation_sha256)
        loss = label_sequence_loss(prediction, batch.annotations, self.vocabulary)
        return {'sir_sequence': loss, 'total': weight * loss}

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> tuple[LabelSequenceCandidate, ...]:
        """Greedy uncalibrated candidates; never silently accept truncation.

        At most max_events labels can be returned. An extra decoder step is used
        only to observe STOP after an exactly-at-capacity sequence. Failed rows
        retain a diagnostic prefix but have an empty usable ``labels`` tuple.
        """
        with preserving_eval_mode(self):
            hidden = self._source(token_ids, lengths)
            count = token_ids.shape[0]
            inputs = torch.ones((count, 1), device=token_ids.device, dtype=torch.int64)
            prefixes: list[list[SIRLabelEntry]] = [[] for _ in range(count)]
            statuses: list[str | None] = [None] * count
            for step in range(self.model_cfg.max_events + 1):
                output, hidden = self.decoder(self.label_embedding(inputs), hidden)
                scores = self.classifier(output[:, 0, :])
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError('nonfinite autoregressive decoder scores')
                selected = scores.argmax(dim=-1)
                for row, column in enumerate(selected.tolist()):
                    if statuses[row] is not None:
                        continue
                    if column == 0:
                        statuses[row] = 'terminated' if prefixes[row] else 'empty_prediction'
                    elif step == self.model_cfg.max_events:
                        statuses[row] = 'capacity_exceeded'
                    else:
                        prefixes[row].append(self.vocabulary.entries[column - 1])
                if all(status is not None for status in statuses):
                    break
                inputs = (selected + 1)[:, None]
                for row, status in enumerate(statuses):
                    if status is not None:
                        inputs[row, 0] = 0
            return tuple(LabelSequenceCandidate(
                tuple(prefix) if status == 'terminated' else (), str(status),
                tuple(prefix), self.model_cfg.lexicon_sha256)
                for prefix, status in zip(prefixes, statuses))
