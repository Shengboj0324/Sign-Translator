"""Explicit source-text input for governed SIR supervision.

Byte IDs are a fixed transport alphabet, not ASL labels or a learned vocabulary.
The caller must declare that the source bytes are plain UTF-8 text. Structured
transcripts require a separate reviewed extraction and newly bound evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import torch

from ..planning.supervision import GovernedSIRAnnotation


PLAINTEXT_ENCODING = "utf-8-plain-bytes-v1"
MAX_MODEL_TEXT_BYTES = 65_536


@dataclass(frozen=True)
class GovernedTextBatch:
    token_ids: torch.Tensor             # B,L int64; byte b maps to b+1
    valid: torch.Tensor                 # B,L bool; zero is padding only
    lengths: torch.Tensor               # B int64, byte lengths rather than characters
    transcript_sha256: tuple[str, ...]
    annotation_sha256: tuple[str, ...]
    encoding: str = PLAINTEXT_ENCODING


def encode_plaintext_transcripts(
    payloads: Sequence[bytes], annotations: Sequence[GovernedSIRAnnotation], *,
    declared_encoding: str, max_bytes: int,
) -> GovernedTextBatch:
    """Verify bound bytes, decode strictly, and encode without normalization.

    No fallback decoding, Unicode normalization, truncation, token learning or
    target-derived vocabulary. Every active byte is retained, including newline,
    case and multibyte Unicode spelling. IDs 1..256 are fixed; zero pads time.
    A format declaration is a caller contract, not proof of source semantics.
    Returned tensors are on CPU; model code must move them explicitly.
    """
    if declared_encoding != PLAINTEXT_ENCODING:
        raise ValueError("explicit plain UTF-8 byte encoding declaration required")
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_MODEL_TEXT_BYTES:
        raise ValueError("max_bytes must be an integer in [1, 65536]")
    payloads, annotations = tuple(payloads), tuple(annotations)
    if not payloads or len(payloads) != len(annotations):
        raise ValueError("nonempty, equally sized source and annotation batches required")
    hashes, identities = [], []
    for payload, annotation in zip(payloads, annotations):
        if not isinstance(payload, bytes) or not 0 < len(payload) <= max_bytes:
            raise ValueError("source text requires nonempty bounded immutable bytes")
        if not isinstance(annotation, GovernedSIRAnnotation):
            raise ValueError("a governed annotation is required for every source")
        checked = GovernedSIRAnnotation.from_manifest(annotation.to_manifest())
        digest = hashlib.sha256(payload).hexdigest()
        if digest != checked.source.transcript_sha256:
            raise ValueError("source text bytes differ from reviewed transcript identity")
        text = payload.decode('utf-8', errors='strict')
        if not text.strip():
            raise ValueError("source text cannot be blank")
        hashes.append(digest)
        identities.append(checked.content_sha256())
    lengths = torch.tensor([len(payload) for payload in payloads], dtype=torch.int64)
    token_ids = torch.zeros((len(payloads), int(lengths.max())), dtype=torch.int64)
    valid = torch.zeros_like(token_ids, dtype=torch.bool)
    for row, payload in enumerate(payloads):
        token_ids[row, :len(payload)] = torch.tensor([value + 1 for value in payload])
        valid[row, :len(payload)] = True
    return GovernedTextBatch(token_ids, valid, lengths, tuple(hashes), tuple(identities))
