"""Canonical sample schema and dataset map (docs/DATA_ENGINEERING.md §1).

The `Sample` record carries every field the document requires; the governance-
critical fields (license, authorization evidence, consent status, signer-id hash,
split, provenance) are validated fail-closed. `DatasetMap` records each source corpus's best use and material
limitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ..pseudo_gloss.contracts import WeakGlossCandidateRecord


class ConsentState(IntEnum):
    GRANTED = 0
    WITHDRAWN = 1
    # The corpus publisher, rather than this project, collected the participants.
    # This state must never be presented as direct consent held by this project.
    NOT_DIRECTLY_VERIFIED = 2


class AuthorizationBasis(str, Enum):
    """Evidence basis under which this project may process a sample."""

    DIRECT_PARTICIPANT_CONSENT = "direct_participant_consent"
    PUBLISHED_DATASET_LICENSE = "published_dataset_license"


class PersonalityRightsStatus(str, Enum):
    """Status of privacy, publicity, and personality rights outside copyright."""

    VERIFIED = "verified"
    NOT_VERIFIED = "not_verified"


AUTHORIZATION_ACTIONS = frozenset({
    "download", "create_derivatives", "model_training", "commercial_use",
    "redistribution", "identity_use",
})


@dataclass(frozen=True)
class DataAuthorization:
    """Machine-checkable evidence and action scope for data processing.

    Permission fields are recorded claims backed by the immutable evidence file;
    this type does not attempt to infer legal meaning from a license name.
    """

    basis: AuthorizationBasis
    license_identifier: str
    license_url: str
    licensor: str
    evidence_uri: str
    evidence_sha256: str
    permitted_uses: Tuple[str, ...]
    permitted_actions: Tuple[str, ...]
    personality_rights: PersonalityRightsStatus
    attribution_notice: str = ""
    limitations: Tuple[str, ...] = ()

    def to_manifest(self) -> dict:
        return {
            "basis": self.basis.value,
            "license_identifier": self.license_identifier,
            "license_url": self.license_url,
            "licensor": self.licensor,
            "evidence_uri": self.evidence_uri,
            "evidence_sha256": self.evidence_sha256,
            "permitted_uses": list(self.permitted_uses),
            "permitted_actions": list(self.permitted_actions),
            "personality_rights": self.personality_rights.value,
            "attribution_notice": self.attribution_notice,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_manifest(cls, value: object) -> "DataAuthorization":
        if not isinstance(value, dict):
            raise ValueError("authorization must be an object")
        required = {
            "basis", "license_identifier", "license_url", "licensor", "evidence_uri",
            "evidence_sha256", "permitted_uses", "permitted_actions",
            "personality_rights", "attribution_notice", "limitations",
        }
        if set(value) != required:
            raise ValueError(
                f"authorization fields must be exactly {sorted(required)}")
        try:
            basis = AuthorizationBasis(value["basis"])
            rights = PersonalityRightsStatus(value["personality_rights"])
        except (TypeError, ValueError) as error:
            raise ValueError("authorization contains an unknown enum value") from error
        uses = value["permitted_uses"]
        actions = value["permitted_actions"]
        limitations = value["limitations"]
        if not isinstance(uses, list) or not isinstance(actions, list) \
                or not isinstance(limitations, list):
            raise ValueError("authorization uses, actions, and limitations must be lists")
        scalar_names = (
            "license_identifier", "license_url", "licensor", "evidence_uri",
            "evidence_sha256", "attribution_notice",
        )
        if any(not isinstance(value[name], str) for name in scalar_names):
            raise ValueError("authorization scalar fields must be strings")
        if any(not isinstance(item, str) for item in uses + actions + limitations):
            raise ValueError("authorization list fields must contain only strings")
        return cls(
            basis=basis,
            license_identifier=value["license_identifier"],
            license_url=value["license_url"],
            licensor=value["licensor"],
            evidence_uri=value["evidence_uri"],
            evidence_sha256=value["evidence_sha256"],
            permitted_uses=tuple(uses),
            permitted_actions=tuple(actions),
            personality_rights=rights,
            attribution_notice=value["attribution_notice"],
            limitations=tuple(limitations),
        )


def validate_authorization(authorization: DataAuthorization, consent: ConsentState,
                           intended_use: str,
                           requested_actions: Sequence[str] = ()) -> List[str]:
    """Return fail-closed authorization violations without legal inference."""
    violations: List[str] = []
    if not isinstance(authorization, DataAuthorization):
        return ["invalid_authorization_type"]
    if not isinstance(authorization.basis, AuthorizationBasis):
        violations.append("invalid_authorization_basis")
    if not isinstance(authorization.personality_rights, PersonalityRightsStatus):
        violations.append("invalid_personality_rights_status")
    if not isinstance(consent, ConsentState):
        violations.append("invalid_consent_state")
    text_fields = {
        "license_identifier": authorization.license_identifier,
        "license_url": authorization.license_url,
        "licensor": authorization.licensor,
        "evidence_uri": authorization.evidence_uri,
    }
    violations.extend(
        f"missing_{name}" for name, value in text_fields.items()
        if not isinstance(value, str) or not value.strip())
    parsed_license_url = (urlparse(authorization.license_url)
                          if isinstance(authorization.license_url, str) else None)
    if (parsed_license_url is None or parsed_license_url.scheme != "https"
            or not parsed_license_url.netloc):
        violations.append("invalid_license_url")
    digest = (authorization.evidence_sha256.lower()
              if isinstance(authorization.evidence_sha256, str) else "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        violations.append("invalid_authorization_evidence_sha256")
    if (not isinstance(authorization.permitted_uses, tuple)
            or not authorization.permitted_uses
            or any(not isinstance(use, str) or not use.strip()
                   for use in authorization.permitted_uses)
            or len(set(authorization.permitted_uses)) != len(authorization.permitted_uses)):
        violations.append("invalid_permitted_uses")
    elif intended_use not in authorization.permitted_uses:
        violations.append("use_not_permitted")
    if (not isinstance(authorization.permitted_actions, tuple)
            or not authorization.permitted_actions
            or any(not isinstance(action, str) or action not in AUTHORIZATION_ACTIONS
                   for action in authorization.permitted_actions)
            or len(set(authorization.permitted_actions)) != len(authorization.permitted_actions)):
        violations.append("invalid_permitted_actions")
    if (not isinstance(authorization.limitations, tuple)
            or any(not isinstance(item, str) or not item.strip()
                   for item in authorization.limitations)):
        violations.append("invalid_authorization_limitations")
    if isinstance(requested_actions, (str, bytes)) or not isinstance(
            requested_actions, Sequence):
        return [*violations, "invalid_requested_actions"]
    unknown_requested = any(
        not isinstance(action, str) or action not in AUTHORIZATION_ACTIONS
        for action in requested_actions)
    if unknown_requested:
        violations.append("unknown_requested_action")
    for action in requested_actions:
        if isinstance(action, str) and action not in authorization.permitted_actions:
            violations.append(f"action_not_permitted:{action}")
    if authorization.basis is AuthorizationBasis.DIRECT_PARTICIPANT_CONSENT:
        if consent is not ConsentState.GRANTED:
            violations.append("direct_consent_not_granted")
    elif authorization.basis is AuthorizationBasis.PUBLISHED_DATASET_LICENSE:
        if consent is not ConsentState.NOT_DIRECTLY_VERIFIED:
            violations.append("secondary_license_consent_state_mismatch")
        if (not isinstance(authorization.attribution_notice, str)
                or not authorization.attribution_notice.strip()):
            violations.append("missing_attribution_notice")
        if (authorization.personality_rights is PersonalityRightsStatus.NOT_VERIFIED
                and not authorization.limitations):
            violations.append("unrecorded_personality_rights_limitation")
    if ("identity_use" in requested_actions
            and authorization.personality_rights is not PersonalityRightsStatus.VERIFIED):
        violations.append("personality_rights_not_verified")
    return violations


VALID_SPLITS = frozenset({"train", "val", "test"})


@dataclass
class Sample:
    """A canonical dataset sample (all fields from the document's schema)."""

    sample_id: str
    source_id: str
    signer_id_hash: str                       # HASHED identity, never raw
    target_language: str                      # e.g. "ASL"
    license: str                              # e.g. "CC-BY-NC-4.0"
    consent: ConsentState
    intended_use: str
    smplx_version: str
    provenance: str                           # Merkle provenance root (§2)
    split: str                                # train / val / test
    dialect: Optional[str] = None
    video_uri: Optional[str] = None
    audio_uri: Optional[str] = None
    calibration: Optional[Dict] = None
    transcript_lattice: Optional[List] = None
    semantic_plan: Optional[object] = None
    annotation_tiers: Dict[str, List] = field(default_factory=dict)
    confidence_2d: Optional[float] = None
    confidence_3d: Optional[float] = None
    frame_transform: Optional[Dict] = None
    time_transform: Optional[Dict] = None
    retention_date: Optional[float] = None    # governance (§7)
    authorization: Optional[DataAuthorization] = None
    # Machine-generated annotation hypotheses remain parallel to authentic
    # annotation tiers. They are validated below but never promoted implicitly.
    weak_gloss_candidates: Tuple["WeakGlossCandidateRecord", ...] = ()
    # DELIBERATELY absent: any sensitive-trait field (§7 non-inference guard).

    @property
    def group_key(self) -> tuple:
        """The leakage grouping key: (signer, source recording)."""
        return (self.signer_id_hash, self.source_id)


def validate_sample(s: Sample) -> List[str]:
    """Return violated schema rules (empty == valid). Governance fields required."""
    v: List[str] = []
    if not s.sample_id:
        v.append("missing_sample_id")
    if not s.signer_id_hash:
        v.append("missing_signer_id_hash")            # never anonymous-untracked
    if not s.license:
        v.append("missing_license")
    if not s.intended_use:
        v.append("missing_intended_use")
    if not s.provenance:
        v.append("missing_provenance")
    if not s.target_language:
        v.append("missing_target_language")
    if s.authorization is None:
        v.append("missing_authorization")
    elif not isinstance(s.authorization, DataAuthorization):
        v.append("invalid_authorization_type")
    else:
        if s.authorization.license_identifier != s.license:
            v.append("authorization_license_mismatch")
        v.extend(validate_authorization(
            s.authorization, s.consent, s.intended_use, requested_actions=("download",)))
    if s.split not in VALID_SPLITS:
        v.append("invalid_split")
    for name, c in (("confidence_2d", s.confidence_2d), ("confidence_3d", s.confidence_3d)):
        if c is not None and not (0.0 <= c <= 1.0):
            v.append(f"{name}_out_of_range")
    if not isinstance(s.weak_gloss_candidates, tuple):
        v.append("weak_gloss_candidates_must_be_tuple")
    elif s.weak_gloss_candidates:
        from ..pseudo_gloss.contracts import WeakGlossCandidateRecord

        annotation_ids = []
        for candidate in s.weak_gloss_candidates:
            if not isinstance(candidate, WeakGlossCandidateRecord):
                v.append("invalid_weak_gloss_candidate_type")
                continue
            annotation_ids.append(candidate.annotation_id)
            if candidate.provenance.source_sample_id != s.sample_id:
                v.append("weak_gloss_candidate_sample_mismatch")
        if len(annotation_ids) != len(set(annotation_ids)):
            v.append("duplicate_weak_gloss_candidate_id")
    return v


# ---------------------------------------------------------------------------
# dataset map
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetResource:
    name: str
    best_use: str
    limitation: str
    redistributable: bool


DATASET_MAP: Dict[str, DatasetResource] = {
    "How2Sign": DatasetResource(
        "How2Sign", "80+h continuous multimodal ASL; speech/text/depth; multiview",
        "interpreted instructional domain; alignment and signer limitations", False),
    "WLASL": DatasetResource(
        "WLASL", "isolated recognition and visual pretraining",
        "isolated lexical signs are not continuous production", False),
    "MS-ASL": DatasetResource(
        "MS-ASL", "isolated recognition and visual pretraining",
        "isolated lexical signs are not continuous production", False),
    "ASLLVD": DatasetResource(
        "ASLLVD", "lexical/phonological study",
        "not a broad continuous speech-to-sign corpus", False),
    "PHOENIX14T": DatasetResource(
        "PHOENIX14T", "established continuous SLT benchmark",
        "German Sign Language/weather domain, not ASL", False),
    "SignAvatars": DatasetResource(
        "SignAvatars", "large fitted holistic 3D motion and multiple prompts",
        "automatically reconstructed 3D contains model bias/error", False),
    "ASL3DWord": DatasetResource(
        "ASL3DWord", "expressive isolated 3D word generation",
        "isolated-word scope", False),
}


def dataset_map_is_complete() -> bool:
    """Every registered resource has a best-use and a stated limitation."""
    return all(r.best_use and r.limitation for r in DATASET_MAP.values())
