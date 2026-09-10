"""Fail-closed evidence gate for pre-Phase-2 data and human support.

A requirement bundle passes only when one source co-observes every required
capability, has been locally verified, and has recorded permission for the
declared use.  Capabilities from different sources are never combined into a
fictional training example.  Rights states are evidence records, not legal advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import re
from typing import Iterable
from urllib.parse import urlparse


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


class EvidenceLevel(IntEnum):
    NONE = 0
    PUBLISHER_CLAIM = 1
    LOCAL_VERIFIED = 2
    QUALIFIED_HUMAN = 3


class RightsStatus(str, Enum):
    PROHIBITED = "prohibited"
    UNRESOLVED = "unresolved"
    PERMISSION_REQUIRED = "permission_required"
    PERMITTED = "permitted"


class AccessStatus(str, Enum):
    LOCAL_VERIFIED = "local_verified"
    PUBLIC_DOWNLOAD = "public_download"
    ACCOUNT_REQUIRED = "account_required"
    REQUEST_REQUIRED = "request_required"
    UNRELEASED = "unreleased"


class IntendedUse(str, Enum):
    RESEARCH_TRAINING = "research_training"
    COMMERCIAL_TRAINING = "commercial_training"
    COMMERCIAL_DEPLOYMENT = "commercial_deployment"


@dataclass(frozen=True)
class CapabilityEvidence:
    capability: str
    level: EvidenceLevel

    def __post_init__(self) -> None:
        if not isinstance(self.capability, str) or not _IDENTIFIER.fullmatch(
                self.capability):
            raise ValueError("capability must be a lowercase ASCII identifier")
        if not isinstance(self.level, EvidenceLevel) or self.level is EvidenceLevel.NONE:
            raise ValueError("a claimed capability requires positive evidence")


@dataclass(frozen=True)
class SourceCandidate:
    source_id: str
    title: str
    official_url: str
    access: AccessStatus
    capabilities: tuple[CapabilityEvidence, ...]
    research_rights: RightsStatus
    commercial_training_rights: RightsStatus
    commercial_deployment_rights: RightsStatus
    limitation: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not _IDENTIFIER.fullmatch(
                self.source_id):
            raise ValueError("source_id must be a lowercase ASCII identifier")
        parsed = urlparse(self.official_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("official_url must be an absolute HTTPS URL")
        if not isinstance(self.title, str) or not isinstance(self.limitation, str) \
                or not self.title.strip() or not self.limitation.strip():
            raise ValueError("source title and limitation are required")
        enum_values = (
            (self.access, AccessStatus),
            (self.research_rights, RightsStatus),
            (self.commercial_training_rights, RightsStatus),
            (self.commercial_deployment_rights, RightsStatus),
        )
        if any(not isinstance(value, enum_type) for value, enum_type in enum_values):
            raise ValueError("source access and rights fields must be typed enums")
        if not isinstance(self.capabilities, tuple) or not self.capabilities:
            raise ValueError("source capabilities must be a non-empty tuple")
        capability_names = [item.capability for item in self.capabilities]
        if len(capability_names) != len(set(capability_names)):
            raise ValueError("source capabilities must be unique")
        if not all(isinstance(item, CapabilityEvidence) for item in self.capabilities):
            raise ValueError("source capabilities must carry typed evidence")

    def rights_for(self, intended_use: IntendedUse) -> RightsStatus:
        if intended_use is IntendedUse.RESEARCH_TRAINING:
            return self.research_rights
        if intended_use is IntendedUse.COMMERCIAL_TRAINING:
            return self.commercial_training_rights
        if intended_use is IntendedUse.COMMERCIAL_DEPLOYMENT:
            return self.commercial_deployment_rights
        raise ValueError("unsupported intended use")


@dataclass(frozen=True)
class RequirementBundle:
    requirement_id: str
    required_capabilities: frozenset[str]
    minimum_evidence: EvidenceLevel
    intended_use: IntendedUse

    def __post_init__(self) -> None:
        if not isinstance(self.requirement_id, str) or not _IDENTIFIER.fullmatch(
                self.requirement_id):
            raise ValueError("requirement_id must be a lowercase ASCII identifier")
        if (not isinstance(self.required_capabilities, frozenset)
                or not self.required_capabilities):
            raise ValueError("a requirement bundle cannot be empty")
        if any(not isinstance(item, str) or not _IDENTIFIER.fullmatch(item)
               for item in self.required_capabilities):
            raise ValueError("required capabilities must be lowercase ASCII identifiers")
        if (not isinstance(self.minimum_evidence, EvidenceLevel)
                or self.minimum_evidence is EvidenceLevel.NONE):
            raise ValueError("minimum_evidence must be positive")
        if not isinstance(self.intended_use, IntendedUse):
            raise ValueError("intended_use must be typed")


@dataclass(frozen=True)
class BundleDecision:
    requirement_id: str
    passed: bool
    satisfying_sources: tuple[str, ...]
    source_failures: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class PortfolioDecision:
    approved: bool
    bundle_decisions: tuple[BundleDecision, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "approved": self.approved,
            "composition_rule": (
                "each bundle must be satisfied by one locally verified and authorized source"
            ),
            "cross_source_capability_stitching_allowed": False,
            "bundle_decisions": [
                {
                    "requirement_id": decision.requirement_id,
                    "passed": decision.passed,
                    "satisfying_sources": list(decision.satisfying_sources),
                    "source_failures": {
                        source_id: list(failures)
                        for source_id, failures in decision.source_failures
                    },
                }
                for decision in self.bundle_decisions
            ],
        }


def _source_failures(source: SourceCandidate,
                     requirement: RequirementBundle) -> tuple[str, ...]:
    failures: list[str] = []
    evidence = {item.capability: item.level for item in source.capabilities}
    missing = sorted(requirement.required_capabilities - evidence.keys())
    failures.extend(f"missing_capability:{item}" for item in missing)
    insufficient = sorted(
        capability for capability in requirement.required_capabilities & evidence.keys()
        if evidence[capability] < requirement.minimum_evidence
    )
    failures.extend(f"insufficient_evidence:{item}" for item in insufficient)
    if source.access is not AccessStatus.LOCAL_VERIFIED:
        failures.append(f"not_locally_verified:{source.access.value}")
    rights = source.rights_for(requirement.intended_use)
    if rights is not RightsStatus.PERMITTED:
        failures.append(f"rights_not_permitted:{rights.value}")
    return tuple(failures)


def assess_source_portfolio(
    sources: Iterable[SourceCandidate],
    requirements: Iterable[RequirementBundle],
) -> PortfolioDecision:
    source_list = tuple(sources)
    requirement_list = tuple(requirements)
    if not source_list or not requirement_list:
        raise ValueError("sources and requirements must both be non-empty")
    source_ids = [source.source_id for source in source_list]
    requirement_ids = [requirement.requirement_id for requirement in requirement_list]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source identifiers must be unique")
    if len(requirement_ids) != len(set(requirement_ids)):
        raise ValueError("requirement identifiers must be unique")

    decisions: list[BundleDecision] = []
    for requirement in requirement_list:
        failures = tuple(
            (source.source_id, _source_failures(source, requirement))
            for source in sorted(source_list, key=lambda item: item.source_id)
        )
        satisfying = tuple(source_id for source_id, reasons in failures if not reasons)
        decisions.append(BundleDecision(
            requirement_id=requirement.requirement_id,
            passed=bool(satisfying),
            satisfying_sources=satisfying,
            source_failures=failures,
        ))
    return PortfolioDecision(
        approved=all(decision.passed for decision in decisions),
        bundle_decisions=tuple(decisions),
    )


def evidence(*capabilities: str,
             level: EvidenceLevel = EvidenceLevel.PUBLISHER_CLAIM,
             ) -> tuple[CapabilityEvidence, ...]:
    """Build a sorted, duplicate-rejecting capability-evidence tuple."""
    if len(capabilities) != len(set(capabilities)):
        raise ValueError("capabilities must be unique")
    return tuple(CapabilityEvidence(item, level) for item in sorted(capabilities))


PRE_PHASE_2_REQUIREMENTS = (
    RequirementBundle(
        "continuous_coobserved_3d_research",
        frozenset({
            "continuous_asl", "body_3d", "left_hand_3d", "right_hand_3d",
            "face_expression", "head_pose", "eye_gaze", "eyelid_state",
            "stable_signer_id", "text_alignment",
        }),
        EvidenceLevel.LOCAL_VERIFIED,
        IntendedUse.RESEARCH_TRAINING,
    ),
    RequirementBundle(
        "continuous_linguistic_reference",
        frozenset({
            "continuous_asl", "authentic_gloss", "manual_timing",
            "handshape_annotation", "nonmanual_annotation", "head_movement_annotation",
            "eye_gaze_annotation", "qualified_asl_annotation",
        }),
        EvidenceLevel.LOCAL_VERIFIED,
        IntendedUse.RESEARCH_TRAINING,
    ),
    RequirementBundle(
        "commercial_training_authorization",
        frozenset({
            "continuous_asl", "body_3d", "left_hand_3d", "right_hand_3d",
            "face_expression", "head_pose", "eye_gaze", "eyelid_state",
            "stable_signer_id", "text_alignment",
        }),
        EvidenceLevel.LOCAL_VERIFIED,
        IntendedUse.COMMERCIAL_TRAINING,
    ),
    RequirementBundle(
        "commercial_render_rig",
        frozenset({"body_rig", "hand_rig", "face_rig", "eye_rig"}),
        EvidenceLevel.LOCAL_VERIFIED,
        IntendedUse.COMMERCIAL_DEPLOYMENT,
    ),
    RequirementBundle(
        "qualified_asl_governance",
        frozenset({"qualified_asl_review", "deaf_co_design", "accessible_consent"}),
        EvidenceLevel.QUALIFIED_HUMAN,
        IntendedUse.COMMERCIAL_DEPLOYMENT,
    ),
)


CURRENT_SOURCE_CANDIDATES = (
    SourceCandidate(
        "how2sign_local", "How2Sign local training release",
        "https://how2sign.github.io/", AccessStatus.LOCAL_VERIFIED,
        evidence("continuous_asl", "rgb_front", "two_d_pose", "text_alignment",
                 "stable_signer_id", level=EvidenceLevel.LOCAL_VERIFIED),
        RightsStatus.PERMITTED, RightsStatus.PROHIBITED, RightsStatus.PROHIBITED,
        "Local release is 2D frontal data under CC BY-NC 4.0; it is not a commercial 3D corpus.",
    ),
    SourceCandidate(
        "signavatars", "SignAvatars",
        "https://github.com/ZhengdiYu/SignAvatars", AccessStatus.REQUEST_REQUIRED,
        evidence("continuous_asl", "body_3d", "left_hand_3d", "right_hand_3d",
                 "face_expression", "head_pose", "stable_signer_id", "text_alignment"),
        RightsStatus.PERMISSION_REQUIRED, RightsStatus.PERMISSION_REQUIRED,
        RightsStatus.PERMISSION_REQUIRED,
        "Publisher annotations omit explicit eye-gaze evidence and access is restricted.",
    ),
    SourceCandidate(
        "asllrp", "ASLLRP SignStream corpora",
        "https://www.bu.edu/asllrp/", AccessStatus.ACCOUNT_REQUIRED,
        evidence("continuous_asl", "authentic_gloss", "manual_timing",
                 "handshape_annotation", "nonmanual_annotation",
                 "head_movement_annotation", "eye_gaze_annotation",
                 "qualified_asl_annotation", "camera_calibration"),
        RightsStatus.PERMISSION_REQUIRED, RightsStatus.PERMISSION_REQUIRED,
        RightsStatus.PERMISSION_REQUIRED,
        "Linguistically rich reference data; terms require permission and do not supply 3D motion.",
    ),
    SourceCandidate(
        "asl_citizen", "ASL Citizen",
        "https://www.microsoft.com/en-us/research/project/asl-citizen/",
        AccessStatus.REQUEST_REQUIRED,
        evidence("isolated_asl", "qualified_asl_annotation", "participant_consent"),
        RightsStatus.PERMISSION_REQUIRED, RightsStatus.PERMISSION_REQUIRED,
        RightsStatus.PERMISSION_REQUIRED,
        "Isolated signs are useful auxiliary evidence but cannot replace "
        "continuous co-observation.",
    ),
    SourceCandidate(
        "nvidia_asl", "NVIDIA Trustworthy AI ASL dataset",
        "https://www.nvidia.com/en-us/gated-resources/"
        "trustworthy-ai-american-sign-language/dataset/",
        AccessStatus.ACCOUNT_REQUIRED,
        evidence("isolated_asl", "two_d_pose", "hand_landmarks", "face_mesh"),
        RightsStatus.PERMISSION_REQUIRED, RightsStatus.UNRESOLVED,
        RightsStatus.UNRESOLVED,
        "Gated isolated-sign data; commercial-training and deployment rights are unresolved.",
    ),
    SourceCandidate(
        "smplx", "SMPL-X body model",
        "https://smpl-x.is.tue.mpg.de/modellicense.html", AccessStatus.ACCOUNT_REQUIRED,
        evidence("body_rig", "hand_rig", "face_rig", "eye_rig"),
        RightsStatus.PERMISSION_REQUIRED, RightsStatus.PERMISSION_REQUIRED,
        RightsStatus.PERMISSION_REQUIRED,
        "Research registration does not itself establish commercial model rights.",
    ),
    SourceCandidate(
        "makehuman", "MakeHuman core assets",
        "https://static.makehumancommunity.org/about/license.html",
        AccessStatus.PUBLIC_DOWNLOAD,
        evidence("body_rig", "hand_rig", "face_rig", "eye_rig"),
        RightsStatus.PERMITTED, RightsStatus.PERMITTED, RightsStatus.PERMITTED,
        "Asset licensing is promising, but the exact rig has not been locally qualified.",
    ),
    SourceCandidate(
        "gallaudet_mll", "Gallaudet Motion Light Lab",
        "https://gallaudet.edu/visual-language-visual-learning/ml2/",
        AccessStatus.REQUEST_REQUIRED,
        evidence("qualified_asl_review", "deaf_co_design", "motion_capture_partner"),
        RightsStatus.PERMISSION_REQUIRED, RightsStatus.PERMISSION_REQUIRED,
        RightsStatus.PERMISSION_REQUIRED,
        "Potential expert partnership only; no agreement or accessible-consent "
        "protocol exists yet.",
    ),
)


CURRENT_PRE_PHASE_2_DECISION = assess_source_portfolio(
    CURRENT_SOURCE_CANDIDATES, PRE_PHASE_2_REQUIREMENTS)


__all__ = [
    "EvidenceLevel", "RightsStatus", "AccessStatus", "IntendedUse",
    "CapabilityEvidence", "SourceCandidate", "RequirementBundle",
    "BundleDecision", "PortfolioDecision", "assess_source_portfolio", "evidence",
    "PRE_PHASE_2_REQUIREMENTS", "CURRENT_SOURCE_CANDIDATES",
    "CURRENT_PRE_PHASE_2_DECISION",
]
