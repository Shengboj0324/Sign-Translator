"""Canonical sample schema and dataset map (docs/DATA_ENGINEERING.md §1).

The `Sample` record carries every field the document requires; the governance-
critical fields (license, consent, signer-id hash, split, provenance) are never
optional. `DatasetMap` records each source corpus's best use and material
limitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


class ConsentState(IntEnum):
    GRANTED = 0
    WITHDRAWN = 1


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
    if s.split not in VALID_SPLITS:
        v.append("invalid_split")
    for name, c in (("confidence_2d", s.confidence_2d), ("confidence_3d", s.confidence_3d)):
        if c is not None and not (0.0 <= c <= 1.0):
            v.append(f"{name}_out_of_range")
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
