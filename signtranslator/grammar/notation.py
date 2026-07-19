"""Phonological notation (HamNoSys-style) and variation-preserving provenance.

A sign has sub-lexical phonological structure: **handshape, location, movement,
orientation** (the classic parameters). HamNoSys and SignWriting are established
notations for this; SiGML is an XML serialization of HamNoSys used to drive
avatars. This module models a compact, validated version of that structure with
a SiGML-like serialization.

Two requirements from the document and the position paper are treated as
first-class, not afterthoughts:

* **Never silently normalise variation into "incorrect" signing.** A concept can
  have several *dialect* / *register* variants; all are valid and are kept
  distinct. A ``VariationSet`` holds them without collapsing to a canonical form,
  and equality/normalisation deliberately does **not** merge them.

* **Automatically generated glosses are noisy labels with provenance.** A
  ``GlossLabel`` records where it came from and how much to trust it, so a
  downstream consumer can weigh a machine gloss against a signer-validated one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Phonological parameters (compact inventories)
# ---------------------------------------------------------------------------
class Handshape(Enum):
    FLAT = "flat"; FIST = "fist"; INDEX = "index"; SPREAD = "spread"
    CUP = "cup"; PINCH = "pinch"; THUMB = "thumb"; CLAW = "claw"


class Location(Enum):
    NEUTRAL = "neutral"; HEAD = "head"; CHEST = "chest"; SHOULDER = "shoulder"
    CHIN = "chin"; FOREHEAD = "forehead"; CHEEK = "cheek"


class Movement(Enum):
    NONE = "none"; STRAIGHT = "straight"; ARC = "arc"; CIRCLE = "circle"
    WIGGLE = "wiggle"; TAP = "tap"


class Orientation(Enum):
    PALM_UP = "palm_up"; PALM_DOWN = "palm_down"; PALM_IN = "palm_in"
    PALM_OUT = "palm_out"; PALM_LEFT = "palm_left"; PALM_RIGHT = "palm_right"


@dataclass(frozen=True)
class SignPhonology:
    """The sub-lexical form of one sign."""

    handshape: Handshape
    location: Location
    movement: Movement
    orientation: Orientation
    two_handed: bool = False

    def validate(self) -> List[str]:
        """Type-level validity (enums guarantee inventory membership)."""
        problems: List[str] = []
        for name, cls in (("handshape", Handshape), ("location", Location),
                          ("movement", Movement), ("orientation", Orientation)):
            if not isinstance(getattr(self, name), cls):
                problems.append(f"{name}_invalid")
        return problems

    def to_sigml(self) -> str:
        """A SiGML-like serialization (compact, deterministic, parseable)."""
        hands = "2" if self.two_handed else "1"
        return (f"<hns hands='{hands}'>"
                f"<hamshape v='{self.handshape.value}'/>"
                f"<hamloc v='{self.location.value}'/>"
                f"<hammove v='{self.movement.value}'/>"
                f"<hamori v='{self.orientation.value}'/>"
                f"</hns>")

    @staticmethod
    def from_sigml(text: str) -> "SignPhonology":
        import re
        def grab(tag):
            m = re.search(rf"<{tag} v='([^']+)'/>", text)
            if not m:
                raise ValueError(f"missing {tag} in SiGML")
            return m.group(1)
        two = "hands='2'" in text
        return SignPhonology(
            handshape=Handshape(grab("hamshape")),
            location=Location(grab("hamloc")),
            movement=Movement(grab("hammove")),
            orientation=Orientation(grab("hamori")),
            two_handed=two)


# ---------------------------------------------------------------------------
# Dialect / register variation -- never silently normalised
# ---------------------------------------------------------------------------
class Register(Enum):
    NEUTRAL = "neutral"; FORMAL = "formal"; INFORMAL = "informal"


@dataclass(frozen=True)
class SignVariant:
    """One realisation of a concept, tagged with dialect and register."""

    phonology: SignPhonology
    dialect: str = "general"
    register: Register = Register.NEUTRAL


@dataclass
class VariationSet:
    """All known variants of one concept, kept DISTINCT.

    There is no ``canonicalize`` that picks a "correct" form -- doing so would
    erase legitimate dialect/register variation, which the document forbids. The
    set only *selects* by tag; it never merges.
    """

    concept: int
    variants: List[SignVariant] = field(default_factory=list)

    def add(self, variant: SignVariant) -> None:
        self.variants.append(variant)

    def by_dialect(self, dialect: str) -> List[SignVariant]:
        return [v for v in self.variants if v.dialect == dialect]

    def by_register(self, register: Register) -> List[SignVariant]:
        return [v for v in self.variants if v.register is register]

    def dialects(self) -> List[str]:
        seen: List[str] = []
        for v in self.variants:
            if v.dialect not in seen:
                seen.append(v.dialect)
        return seen

    def is_variant_of_same_concept(self, other: "VariationSet") -> bool:
        return self.concept == other.concept

    def __len__(self) -> int:
        return len(self.variants)


# ---------------------------------------------------------------------------
# Noisy-gloss provenance
# ---------------------------------------------------------------------------
class GlossSource(Enum):
    SIGNER_VALIDATED = "signer_validated"   # gold: a fluent signer confirmed it
    ELAN_TIER = "elan_tier"                 # human annotation tier
    AUTO = "auto"                           # machine-generated: noisy


@dataclass(frozen=True)
class GlossLabel:
    """A gloss token with provenance, so noisy labels can be down-weighted."""

    gloss: int
    source: GlossSource
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @property
    def is_noisy(self) -> bool:
        return self.source is GlossSource.AUTO

    @property
    def is_gold(self) -> bool:
        return self.source is GlossSource.SIGNER_VALIDATED

    def training_weight(self) -> float:
        """Down-weight noisy labels; gold labels keep full weight.

        A simple, defensible scheme: gold=1, ELAN tier=confidence, auto=half its
        confidence -- so an unvalidated machine gloss never counts as much as a
        signer-confirmed one at equal nominal confidence.
        """
        if self.source is GlossSource.SIGNER_VALIDATED:
            return 1.0
        if self.source is GlossSource.ELAN_TIER:
            return self.confidence
        return 0.5 * self.confidence
