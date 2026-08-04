"""Typed provenance and vocabulary contracts for weak gloss candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:[-+][A-Z0-9]+)*$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
UNKNOWN_TOKEN = "UNKNOWN"


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON used for content-addressed records."""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class LabelType(str, Enum):
    OFFICIAL_HUMAN = "official_human"
    PROJECT_HUMAN = "project_human"
    HUMAN_CORRECTED_PSEUDO = "human_corrected_pseudo"
    UNREVIEWED_PSEUDO = "unreviewed_pseudo"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class GlossLexicon:
    """Versioned, closed candidate vocabulary.

    Token id zero is reserved for the CTC blank. Lexical ids therefore occupy
    ``1..len(tokens)`` in the exact declared order.
    """

    lexicon_id: str
    convention_id: str
    tokens: tuple[str, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.lexicon_id, str) or not self.lexicon_id \
                or not isinstance(self.convention_id, str) or not self.convention_id:
            raise ValueError("lexicon_id and convention_id are required")
        _require_sha256("source_sha256", self.source_sha256)
        if not isinstance(self.tokens, tuple) or not self.tokens \
                or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("lexicon tokens must be non-empty and unique")
        invalid = [token for token in self.tokens if _TOKEN_RE.fullmatch(token) is None]
        if invalid:
            raise ValueError(f"lexicon contains invalid tokens: {invalid[:3]}")
        if UNKNOWN_TOKEN not in self.tokens:
            raise ValueError("lexicon must contain the explicit UNKNOWN token")

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index + 1 for index, token in enumerate(self.tokens)}

    def encode(self, tokens: Sequence[str]) -> tuple[int, ...]:
        if isinstance(tokens, (str, bytes)):
            raise TypeError("tokens must be a sequence of token strings")
        mapping = self.token_to_id
        try:
            encoded = tuple(mapping[token] for token in tokens)
        except (KeyError, TypeError) as error:
            raise ValueError("candidate contains a token outside the closed lexicon") from error
        if not encoded:
            raise ValueError("candidate token sequence must be non-empty")
        return encoded

    def content_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes({
            "convention_id": self.convention_id,
            "lexicon_id": self.lexicon_id,
            "source_sha256": self.source_sha256,
            "tokens": list(self.tokens),
        }))


@dataclass(frozen=True)
class CandidateHypothesis:
    """One text-proposed candidate before video fusion."""

    tokens: tuple[str, ...]
    text_log_probability: float
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple) or not self.tokens \
                or any(not isinstance(token, str) or not token for token in self.tokens):
            raise ValueError("candidate tokens must be non-empty strings")
        if isinstance(self.text_log_probability, bool) \
                or not isinstance(self.text_log_probability, (int, float)) \
                or not math.isfinite(self.text_log_probability) \
                or self.text_log_probability > 0:
            raise ValueError("text_log_probability must be finite and at most zero")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("candidate rank must be positive")


@dataclass(frozen=True)
class CandidateProvenance:
    source_sample_id: str
    source_video_sha256: str
    transcript_sha256: str
    visual_feature_sha256: str
    generator_model_id: str
    model_weight_sha256: str
    tokenizer_sha256: str
    prompt_or_template_sha256: str
    decoding_config_sha256: str
    environment_sha256: str
    code_revision: str
    random_seed: int
    created_at: str

    def __post_init__(self) -> None:
        required_text = (
            self.source_sample_id, self.generator_model_id, self.code_revision,
            self.created_at,
        )
        if any(not isinstance(value, str) or not value for value in required_text):
            raise ValueError("provenance identifiers and timestamp are required")
        if _ID_RE.fullmatch(self.source_sample_id) is None:
            raise ValueError("source_sample_id contains prohibited characters")
        for name in (
            "source_video_sha256", "transcript_sha256", "visual_feature_sha256",
            "model_weight_sha256", "tokenizer_sha256", "prompt_or_template_sha256",
            "decoding_config_sha256",
            "environment_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        try:
            timestamp = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("created_at must be an ISO-8601 timestamp") from error
        if timestamp.tzinfo is None:
            raise ValueError("created_at must include a timezone")


@dataclass(frozen=True)
class WeakGlossCandidateRecord:
    """Append-only weak-label record; never an authentic-gloss field."""

    annotation_id: str
    label_type: LabelType
    review_status: ReviewStatus
    lexicon_id: str
    convention_id: str
    candidate_tokens: tuple[str, ...]
    candidate_log_score: float
    candidate_rank: int
    provenance: CandidateProvenance
    human_annotator_pseudonym: str | None = None
    human_review_protocol: str | None = None
    review_attestation_sha256: str | None = None
    reviewer_qualified_asl: bool = False
    source_video_reviewed: bool = False
    parent_annotation_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (
                self.annotation_id, self.lexicon_id, self.convention_id)):
            raise ValueError("annotation, lexicon, and convention identifiers are required")
        if _ID_RE.fullmatch(self.annotation_id) is None:
            raise ValueError("annotation_id contains prohibited characters")
        if not isinstance(self.candidate_tokens, tuple) or not self.candidate_tokens \
                or any(not isinstance(token, str) or not token
                       for token in self.candidate_tokens):
            raise ValueError("candidate_tokens must be non-empty")
        if isinstance(self.candidate_log_score, bool) \
                or not isinstance(self.candidate_log_score, (int, float)) \
                or not math.isfinite(self.candidate_log_score) \
                or self.candidate_log_score > 0:
            raise ValueError("candidate_log_score must be finite and at most zero")
        if isinstance(self.candidate_rank, bool) \
                or not isinstance(self.candidate_rank, int) or self.candidate_rank < 1:
            raise ValueError("candidate_rank must be positive")
        if not isinstance(self.parent_annotation_ids, tuple) or any(
                not isinstance(item, str) or _ID_RE.fullmatch(item) is None
                for item in self.parent_annotation_ids):
            raise ValueError("parent annotation IDs are invalid")
        if len(set(self.parent_annotation_ids)) != len(self.parent_annotation_ids):
            raise ValueError("parent annotation IDs must be unique")
        if not isinstance(self.limitations, tuple) or any(
                not isinstance(item, str) or not item for item in self.limitations):
            raise ValueError("limitations cannot contain empty values")
        for name in ("human_annotator_pseudonym", "human_review_protocol"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or null")
        if not isinstance(self.reviewer_qualified_asl, bool) \
                or not isinstance(self.source_video_reviewed, bool):
            raise TypeError("review qualification and video-review flags must be boolean")

        if self.label_type not in {
                LabelType.UNREVIEWED_PSEUDO, LabelType.HUMAN_CORRECTED_PSEUDO}:
            raise ValueError("weak candidate record supports only pseudo-derived label types")
        human = self.human_annotator_pseudonym and self.human_review_protocol
        if self.label_type is LabelType.UNREVIEWED_PSEUDO:
            if self.review_status is not ReviewStatus.UNREVIEWED:
                raise ValueError("unreviewed pseudo labels must remain unreviewed")
            if self.human_annotator_pseudonym or self.human_review_protocol \
                    or self.review_attestation_sha256 is not None \
                    or self.reviewer_qualified_asl or self.source_video_reviewed:
                raise ValueError("unreviewed pseudo labels cannot claim human review")
        elif self.review_status is ReviewStatus.APPROVED and not human:
            raise ValueError("approved human labels require reviewer and protocol")
        if self.label_type is LabelType.HUMAN_CORRECTED_PSEUDO:
            if not self.parent_annotation_ids:
                raise ValueError("human-corrected pseudo labels require a machine parent")
            if self.review_status is not ReviewStatus.APPROVED or not human:
                raise ValueError("human-corrected pseudo labels require approved human review")
            if not self.reviewer_qualified_asl or not self.source_video_reviewed \
                    or self.review_attestation_sha256 is None:
                raise ValueError(
                    "human-corrected pseudo labels require qualified video review evidence")
            _require_sha256("review_attestation_sha256", self.review_attestation_sha256)

    def validate_against(self, lexicon: GlossLexicon) -> None:
        if self.lexicon_id != lexicon.lexicon_id or self.convention_id != lexicon.convention_id:
            raise ValueError("candidate lexicon/convention does not match the supplied lexicon")
        lexicon.encode(self.candidate_tokens)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["label_type"] = self.label_type.value
        value["review_status"] = self.review_status.value
        value["candidate_tokens"] = list(self.candidate_tokens)
        value["parent_annotation_ids"] = list(self.parent_annotation_ids)
        value["limitations"] = list(self.limitations)
        return value

    def content_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WeakGlossCandidateRecord":
        required = {
            "annotation_id", "label_type", "review_status", "lexicon_id",
            "convention_id", "candidate_tokens", "candidate_log_score",
            "candidate_rank", "provenance", "human_annotator_pseudonym",
            "human_review_protocol", "parent_annotation_ids", "limitations",
            "review_attestation_sha256", "reviewer_qualified_asl",
            "source_video_reviewed",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"candidate fields must be exactly {sorted(required)}")
        provenance_value = value["provenance"]
        if not isinstance(provenance_value, Mapping):
            raise ValueError("provenance must be an object")
        if not isinstance(value["candidate_tokens"], list) \
                or not isinstance(value["parent_annotation_ids"], list) \
                or not isinstance(value["limitations"], list):
            raise ValueError("candidate token, parent, and limitation fields must be lists")
        if isinstance(value["candidate_rank"], bool) \
                or not isinstance(value["candidate_rank"], int):
            raise ValueError("candidate_rank must be an exact integer")
        if isinstance(value["candidate_log_score"], bool) \
                or not isinstance(value["candidate_log_score"], (int, float)):
            raise ValueError("candidate_log_score must be an exact JSON number")
        provenance_fields = set(CandidateProvenance.__dataclass_fields__)
        if set(provenance_value) != provenance_fields:
            raise ValueError(f"provenance fields must be exactly {sorted(provenance_fields)}")
        return cls(
            annotation_id=value["annotation_id"],
            label_type=LabelType(value["label_type"]),
            review_status=ReviewStatus(value["review_status"]),
            lexicon_id=value["lexicon_id"],
            convention_id=value["convention_id"],
            candidate_tokens=tuple(value["candidate_tokens"]),
            candidate_log_score=value["candidate_log_score"],
            candidate_rank=value["candidate_rank"],
            provenance=CandidateProvenance(**dict(provenance_value)),
            human_annotator_pseudonym=value["human_annotator_pseudonym"],
            human_review_protocol=value["human_review_protocol"],
            review_attestation_sha256=value["review_attestation_sha256"],
            reviewer_qualified_asl=value["reviewer_qualified_asl"],
            source_video_reviewed=value["source_video_reviewed"],
            parent_annotation_ids=tuple(value["parent_annotation_ids"]),
            limitations=tuple(value["limitations"]),
        )


@dataclass(frozen=True)
class HumanGlossAnnotation:
    """Independent human annotation without fabricated machine provenance."""

    annotation_id: str
    source_sample_id: str
    source_video_sha256: str
    label_type: LabelType
    review_status: ReviewStatus
    lexicon_id: str
    convention_id: str
    tokens: tuple[str, ...]
    annotator_pseudonym: str
    review_protocol: str
    review_attestation_sha256: str
    reviewer_qualified_asl: bool
    source_video_reviewed: bool

    def __post_init__(self) -> None:
        if self.label_type not in {LabelType.OFFICIAL_HUMAN, LabelType.PROJECT_HUMAN}:
            raise ValueError("human annotation requires official_human or project_human type")
        if self.review_status is not ReviewStatus.APPROVED:
            raise ValueError("human annotation must be approved")
        for name in (
            "annotation_id", "source_sample_id", "lexicon_id", "convention_id",
            "annotator_pseudonym", "review_protocol",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} is required")
        if _ID_RE.fullmatch(self.annotation_id) is None \
                or _ID_RE.fullmatch(self.source_sample_id) is None:
            raise ValueError("human annotation IDs contain prohibited characters")
        _require_sha256("source_video_sha256", self.source_video_sha256)
        _require_sha256("review_attestation_sha256", self.review_attestation_sha256)
        if self.reviewer_qualified_asl is not True or self.source_video_reviewed is not True:
            raise ValueError("human annotation requires qualified source-video review")
        if not isinstance(self.tokens, tuple) or not self.tokens or any(
                not isinstance(token, str) or not token for token in self.tokens):
            raise ValueError("human annotation tokens are empty")

    def validate_against(self, lexicon: GlossLexicon) -> None:
        if self.lexicon_id != lexicon.lexicon_id or self.convention_id != lexicon.convention_id:
            raise ValueError("human annotation lexicon/convention mismatch")
        lexicon.encode(self.tokens)
