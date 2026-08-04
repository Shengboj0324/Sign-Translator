"""Input and serialization controls for pseudo-gloss candidate generation."""

from __future__ import annotations

import json
import platform
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class InputSecurityPolicy:
    max_bytes: int = 16_384
    max_codepoints: int = 8_192
    max_words: int = 1_024
    max_candidates: int = 64
    max_candidate_tokens: int = 256
    require_nfc: bool = True
    require_nfkc_stable: bool = True
    english_latin_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_bytes", "max_codepoints", "max_words", "max_candidates",
            "max_candidate_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("require_nfc", "require_nfkc_stable", "english_latin_only"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


def runtime_environment() -> dict[str, str]:
    """Exact runtime identity persisted with every model and candidate artifact."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "byteorder": sys.byteorder,
    }


def validate_transcript(text: str, policy: InputSecurityPolicy) -> bytes:
    """Validate an English transcript as inert data and return its exact UTF-8 bytes.

    Prompt-like phrases are not interpreted or heuristically rewritten. Security
    comes from a tool-free local model boundary and schema-constrained output.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("transcript must be a non-empty string")
    if policy.require_nfc and unicodedata.normalize("NFC", text) != text:
        raise ValueError("transcript must be NFC-normalized")
    if policy.require_nfkc_stable and unicodedata.normalize("NFKC", text) != text:
        raise ValueError("transcript contains compatibility or confusable formatting")
    encoded = text.encode("utf-8", errors="strict")
    if len(encoded) > policy.max_bytes or len(text) > policy.max_codepoints:
        raise ValueError("transcript exceeds declared size bounds")
    if len(text.split()) > policy.max_words:
        raise ValueError("transcript exceeds declared word bound")
    for character in text:
        category = unicodedata.category(character)
        if category.startswith("C"):
            raise ValueError("transcript contains a prohibited control or format character")
        if policy.english_latin_only and character.isalpha():
            name = unicodedata.name(character, "")
            if "LATIN" not in name:
                raise ValueError("English transcript contains a non-Latin letter")
            decomposed = unicodedata.normalize("NFKD", character)
            base_letters = [item for item in decomposed if item.isalpha()]
            if not base_letters or any(ord(item) > 0x7F for item in base_letters):
                raise ValueError(
                    "English transcript contains a restricted Latin confusable")
    return encoded


def strict_json_loads(payload: str | bytes, *, max_bytes: int = 1_048_576) -> Any:
    """Parse finite JSON and reject duplicate object keys."""
    if isinstance(payload, str):
        raw = payload.encode("utf-8", errors="strict")
    elif isinstance(payload, bytes):
        raw = payload
        raw.decode("utf-8", errors="strict")
    else:
        raise TypeError("JSON payload must be str or bytes")
    if len(raw) > max_bytes:
        raise ValueError("JSON payload exceeds declared byte bound")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")

    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
