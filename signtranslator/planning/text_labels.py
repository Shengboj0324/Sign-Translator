"""Direct text-conditioned categorical baseline on supplied SIR event slots.

This is a scaffolded training component, not a free-generation ASL planner.
Forward inputs contain source byte IDs, lengths and event counts only: target
labels, event kinds, graph edges, intervals and motion cannot enter the encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..config import SerializableConfig
from ..data.governed_corpus import GovernedMotionBatch
from ..data.governed_text import (
    MAX_MODEL_TEXT_BYTES, PLAINTEXT_ENCODING, encode_plaintext_transcripts,
)
from .label_vocabulary import (
    AlignedSIRLabelLogits, GovernedLabelVocabulary, aligned_sir_label_loss,
)


@dataclass(frozen=True)
class ScaffoldedTextConfig(SerializableConfig):
    embedding_dim: int
    hidden_dim: int
    max_bytes: int
    max_events: int
    lexicon_sha256: str
    convention_sha256: str
    declared_encoding: str

    def __post_init__(self):
        for name in ('embedding_dim', 'hidden_dim', 'max_bytes', 'max_events'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_bytes > MAX_MODEL_TEXT_BYTES:
            raise ValueError('max_bytes exceeds supported text capacity')
        if self.declared_encoding != PLAINTEXT_ENCODING:
            raise ValueError('explicit plaintext source encoding required')


def encode_source_hidden(token_ids: torch.Tensor, lengths: torch.Tensor, *,
                         max_bytes: int, embedding: nn.Embedding, encoder: nn.GRU) -> torch.Tensor:
    """Strict right-padded byte input shared by scaffolded and sequence heads."""
    if any(not isinstance(t, torch.Tensor) or t.dtype != torch.int64
           for t in (token_ids, lengths)):
        raise ValueError('text IDs and lengths must be int64 tensors')
    if (token_ids.ndim != 2 or token_ids.shape[0] == 0 or token_ids.shape[1] == 0
            or token_ids.shape[1] > max_bytes or lengths.shape != (token_ids.shape[0],)):
        raise ValueError('invalid source dimensions or capacity')
    if bool((lengths <= 0).any()) or bool((lengths > token_ids.shape[1]).any()):
        raise ValueError('text length exceeds declared capacity')
    valid = torch.arange(token_ids.shape[1], device=token_ids.device)[None, :] < lengths.to(token_ids.device)[:, None]
    if (bool(((token_ids[valid] < 1) | (token_ids[valid] > 256)).any())
            or bool((token_ids[~valid] != 0).any())):
        raise ValueError('invalid byte IDs or nonzero source padding')
    packed = pack_padded_sequence(embedding(token_ids), lengths.detach().cpu(),
                                  batch_first=True, enforce_sorted=False)
    _, hidden = encoder(packed)
    return hidden


class ScaffoldedTextSIRLabelModel(nn.Module):
    """Byte GRU plus learned event-position queries and a joint kind/label head.

    Each class has the exact kind and lexical identity of one governed vocabulary
    entry. Event count/order is supplied; timing, edges, references, motion and
    new events are not predicted. No confidence threshold or accepted inference
    decoder is provided. Scores alone cannot establish linguistic validity.
    """
    governed_batch_schema_version = 1

    def __init__(self, vocabulary: GovernedLabelVocabulary, *, declared_encoding: str,
                 max_bytes: int = 4096, max_events: int = 128,
                 embedding_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        if not isinstance(vocabulary, GovernedLabelVocabulary):
            raise ValueError('a governed vocabulary is required')
        self.vocabulary = vocabulary
        self.model_cfg = ScaffoldedTextConfig(
            embedding_dim, hidden_dim, max_bytes, max_events,
            vocabulary.lexicon.sha256, vocabulary.convention.sha256, declared_encoding)
        self.byte_embedding = nn.Embedding(257, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.event_queries = nn.Embedding(max_events, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, len(vocabulary.entries))

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor,
                event_counts: torch.Tensor) -> torch.Tensor:
        """B,E,C raw scores, zero on padded events; no target content input."""
        if any(not isinstance(t, torch.Tensor) or t.dtype != torch.int64
               for t in (token_ids, lengths, event_counts)):
            raise ValueError('text IDs, lengths and event counts must be int64 tensors')
        if (token_ids.ndim != 2 or token_ids.shape[0] == 0 or token_ids.shape[1] == 0
                or token_ids.shape[1] > self.model_cfg.max_bytes
                or lengths.shape != (token_ids.shape[0],)
                or event_counts.shape != lengths.shape):
            raise ValueError('invalid text/scaffold dimensions or source capacity')
        if (bool((lengths <= 0).any()) or bool((lengths > token_ids.shape[1]).any())
                or bool((event_counts <= 0).any())
                or bool((event_counts > self.model_cfg.max_events).any())):
            raise ValueError('text length or event count exceeds declared capacity')
        hidden = encode_source_hidden(token_ids, lengths, max_bytes=self.model_cfg.max_bytes,
                                      embedding=self.byte_embedding, encoder=self.encoder)
        count = int(event_counts.max())
        queries = self.event_queries(torch.arange(count, device=token_ids.device))
        scores = self.classifier(torch.tanh(hidden[-1, :, None, :] + queries[None, :, :]))
        active = torch.arange(count, device=token_ids.device)[None, :] < event_counts.to(token_ids.device)[:, None]
        return torch.where(active[:, :, None], scores, torch.zeros_like(scores))

    def training_step(self, batch: GovernedMotionBatch, *, weights: dict) -> dict:
        if not isinstance(batch, GovernedMotionBatch):
            raise ValueError('typed governed training batch required')
        if (self.vocabulary.lexicon.sha256 != self.model_cfg.lexicon_sha256
                or self.vocabulary.convention.sha256 != self.model_cfg.convention_sha256):
            raise ValueError('model vocabulary differs from its constructed class binding')
        if not isinstance(weights, dict) or set(weights) != {'sir_labels'}:
            raise ValueError('declare only the sir_labels objective weight for this model')
        weight = weights['sir_labels']
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('sir_labels weight must be finite and positive')
        text = encode_plaintext_transcripts(
            batch.transcript_payloads, batch.annotations,
            declared_encoding=self.model_cfg.declared_encoding,
            max_bytes=self.model_cfg.max_bytes)
        # Only event cardinality is teacher-supplied. Read immutable annotations,
        # not mutable precomputed graph/label tensors, to define the scaffold.
        counts = torch.tensor([len(a.graph().events) for a in batch.annotations], dtype=torch.int64)
        device = self.byte_embedding.weight.device
        scores = self(text.token_ids.to(device), text.lengths, counts)
        prediction = AlignedSIRLabelLogits(scores, self.vocabulary.lexicon.sha256,
                                          text.annotation_sha256)
        loss = aligned_sir_label_loss(prediction, batch.annotations, self.vocabulary)
        return {'sir_labels': loss, 'total': weight * loss}
