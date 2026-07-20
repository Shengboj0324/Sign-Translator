"""Linguistically-grounded hard negatives + shortcut falsification (Doc-11 §4).

Hard negatives flip exactly one licensed grammatical feature (negation, question
type, entity, aspect, number, role-shift) via the Doc-03 oracle, so each is a
MINIMAL linguistic contrast — not a random clip. The shortcut-falsification
embeddings prove the document's claim: a signer/length shortcut solves the
random-negative task but fails on hard negatives.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from ..grammar.grammar_tests import (
    ControllableASLBuilder, GrammarFeatures, QuestionType, Aspect, minimal_pair,
    changed_sir_fields,
)

# document's contrast dimensions -> (GrammarFeatures field, value chooser)
HARD_NEGATIVE_DIMENSIONS = (
    "negated", "question", "entity", "aspect", "number", "role_shift",
)


def _flip_value(base: GrammarFeatures, dimension: str):
    """The new (feature_name, value) that realises a one-feature contrast."""
    if dimension == "negated":
        return "negated", (not base.negated)
    if dimension == "question":
        nv = QuestionType.YESNO if base.question == QuestionType.NONE else QuestionType.NONE
        return "question", nv
    if dimension == "entity":
        cur = base.object if base.object is not None else 0
        return "object", cur + 1                      # a different referent id
    if dimension == "aspect":
        nv = Aspect.CONTINUATIVE if base.aspect == Aspect.NONE else Aspect.NONE
        return "aspect", nv
    if dimension == "number":
        return "plural_subject", (not base.plural_subject)
    if dimension == "role_shift":
        return "role_shift", (not base.role_shift)
    raise ValueError(f"unknown hard-negative dimension {dimension!r}")


def hard_negative(base: GrammarFeatures, dimension: str) -> GrammarFeatures:
    """Return a minimal linguistic contrast of ``base`` along ``dimension``."""
    field, value = _flip_value(base, dimension)
    return replace(base, **{field: value})


def contrast_changed_fields(builder: ControllableASLBuilder,
                            base: GrammarFeatures, dimension: str) -> set:
    """The SIR fields that flipping ``dimension`` changes."""
    neg = hard_negative(base, dimension)
    return changed_sir_fields(builder.build(base), builder.build(neg))


def is_minimal_linguistic_contrast(builder: ControllableASLBuilder,
                                   base: GrammarFeatures, dimension: str) -> bool:
    """True iff the flip is a GENUINE single-feature contrast (SIRs differ).

    This is the hard-negative mining guarantee: exactly one grammatical feature is
    flipped (by construction of `hard_negative`) and the realised SIR actually
    differs (non-vacuous). It does NOT require the change be a subset of the
    pre-declared licensed set — e.g. an entity swap legitimately also reorders
    naming signs; that stronger property is `is_licensed_contrast`.
    """
    return bool(contrast_changed_fields(builder, base, dimension))


def is_licensed_contrast(builder: ControllableASLBuilder,
                         base: GrammarFeatures, dimension: str) -> bool:
    """Stronger: the flip changes ONLY the feature's licensed SIR fields."""
    field, value = _flip_value(base, dimension)
    res = minimal_pair(builder, base, field, value)
    return bool(res.changed) and res.changed.issubset(res.licensed)


# ---------------------------------------------------------------------------
# shortcut-falsification embeddings
# ---------------------------------------------------------------------------
def signer_shortcut_embedding(signers: Sequence[int], num_signers: int
                              ) -> torch.Tensor:
    """A representation that encodes ONLY signer identity (a shortcut)."""
    idx = torch.as_tensor(list(signers), dtype=torch.long)
    return torch.nn.functional.one_hot(idx, num_signers).float()


def length_shortcut_embedding(lengths: Sequence[int], bin_width: int = 10,
                              num_bins: int = 16) -> torch.Tensor:
    """A representation that encodes ONLY clip length, as a binned one-hot.

    A binned one-hot (not a raw scalar) is used deliberately: a scalar magnitude
    is destroyed by the L2 normalisation inside the InfoNCE, whereas a one-hot
    survives it — so length can genuinely act as a shortcut, and length-matched
    hard negatives collapse to the same code.
    """
    idx = torch.tensor([min(int(l) // bin_width, num_bins - 1) for l in lengths],
                       dtype=torch.long)
    return torch.nn.functional.one_hot(idx, num_bins).float()


def content_embedding(labels: Sequence[int], num_labels: int) -> torch.Tensor:
    """A representation that encodes the linguistic label (genuine content)."""
    idx = torch.as_tensor(list(labels), dtype=torch.long)
    return torch.nn.functional.one_hot(idx, num_labels).float()
