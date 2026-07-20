"""Datasheet + preprocessing manifest (Doc-10 §8; Gebru et al. 2018).

A `Datasheet` carries the Gebru et al. sections; a `PreprocessingManifest` records
the exact ordered steps and the §2 Merkle provenance root, so the dataset is
documented and its construction reproducible. Deaf annotators are credited in the
maintenance section (governance requirement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .provenance import ProvenanceChain, ProvenanceStep

# The seven Gebru et al. datasheet sections.
DATASHEET_SECTIONS = (
    "motivation", "composition", "collection", "preprocessing",
    "uses", "distribution", "maintenance",
)


@dataclass
class Datasheet:
    motivation: str = ""
    composition: str = ""
    collection: str = ""
    preprocessing: str = ""
    uses: str = ""
    distribution: str = ""
    maintenance: str = ""
    deaf_annotator_credits: tuple = ()   # governance: credit Deaf annotators

    def missing_sections(self) -> List[str]:
        return [s for s in DATASHEET_SECTIONS if not getattr(self, s)]

    def is_complete(self) -> bool:
        # Complete iff every section is filled AND Deaf annotators are credited.
        return not self.missing_sections() and bool(self.deaf_annotator_credits)


@dataclass
class PreprocessingManifest:
    """The ordered preprocessing steps + the provenance root that certifies them."""

    steps: List[ProvenanceStep] = field(default_factory=list)
    provenance_root: str = "0" * 64

    @classmethod
    def from_chain(cls, chain: ProvenanceChain) -> "PreprocessingManifest":
        return cls(steps=list(chain.steps), provenance_root=chain.root)

    def verify(self) -> bool:
        """The recorded steps must reproduce the stored provenance root."""
        return ProvenanceChain.recompute_root(self.steps) == self.provenance_root

    def step_names(self) -> Sequence[str]:
        return [s.name for s in self.steps]
