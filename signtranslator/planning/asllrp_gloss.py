"""ASLLRP gloss tokens -> planner targets: base gloss, locus slots and non-manual markers (plan item 2).

The planner's target is the dominant-hand **surface** tier of an ASLLRP utterance (``build_asllrp_english_gloss.py``):
the signs in order, with their agreement and index loci and the grammatical markers that span them. Each surface token
is parsed into

- ``base``: the lexical gloss without annotation decoration - trailing hold underscores (``AGREE__``), the hand prefix
  (``(1h)``, ``(2h)``), locus indices and a classifier's free-text description (``ICL:tight-C"eating`` keeps
  ``ICL:tight-C``; descriptions between backslashes are removed). Name signs (``ns-``), fingerspelling (``fs-``),
  lexicalised fingerspelling (``#``), repetition (``+``) and gestures (``5"wave``) are lexical and kept;
- ``source`` / ``target``: locus letters of an agreement verb (``i:MOVE:j``), of an index or possessive
  (``IX-3p:i``: target ``i``) or of a located sign (``MOVE:i``). Letters are renumbered per utterance in order of first
  appearance (``1`` is the first referent placed in space), the order the motion generator's locus slots use.

Markers (``topic/focus``, ``wh question``, ``y/n question``, ``rhq question``, ``negative``, ``cond/when``,
``relative cl``) become a multi-hot vector per surface token.

``GlossVocabulary`` keeps whole base glosses that occur at least ``min_count`` times in the training targets and
spells any other gloss as UTF-8 byte tokens closed by an end-of-gloss token, so every target is representable exactly.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
import hashlib
import json
import re

MARKERS = ("topic/focus", "wh question", "y/n question", "rhq question", "negative", "cond/when", "relative cl")
MAX_LOCI = 4                                   # locus slots 1..4 (0 = none), as the generator's locus conditioning
PAD, BOS, EOS, GLOSS_END = 0, 1, 2, 259        # 3..258 are the 256 byte tokens
BYTE0, FIRST_GLOSS = 3, 260

_HAND = re.compile(r"^\((?:\d)?h\)")
_AGREE = re.compile(r"^([a-z]):(.+?):([a-z])$")
_SOURCE = re.compile(r"^([a-z]):(.+)$")
_TARGET = re.compile(r"^(.+?):([a-z])$")


@dataclass(frozen=True)
class SurfaceSign:
    base: str
    source: str | None = None                  # locus letter as annotated (renumbered by ``canonical_loci``)
    target: str | None = None


def parse_surface_token(token: str) -> SurfaceSign:
    """One dominant-hand surface token -> (base gloss, source locus, target locus)."""
    t = token.strip()
    t = re.sub(r"\\[^\\]*\\", "", t)           # classifier / gesture descriptions between backslashes
    t = re.sub(r"_+$", "", t)                  # hold duration
    t = _HAND.sub("", t)
    source = target = None
    m = _AGREE.match(t)
    if m:
        source, t, target = m.group(1), m.group(2), m.group(3)
    else:
        m = _SOURCE.match(t)
        if m and not re.match(r"^[A-Z]", m.group(1)):
            source, t = m.group(1), m.group(2)
        m = _TARGET.match(t)
        if m:
            t, target = m.group(1), m.group(2)
    t = re.sub(r"_+$", "", t)
    if '"' in t and re.match(r"^[A-Z]*CL:", t):  # classifier: keep the class and handshape, drop the description
        t = t.split('"', 1)[0]
    t = t.rstrip('"')
    return SurfaceSign(base=t or token.strip(), source=source, target=target)


def canonical_loci(signs: list[SurfaceSign]) -> list[tuple[int, int]]:
    """(source, target) slot per sign, letters renumbered by first appearance; 0 = none, capped at ``MAX_LOCI``."""
    order: dict[str, int] = {}

    def slot(letter):
        if letter is None:
            return 0
        if letter not in order:
            order[letter] = len(order) + 1
        return min(order[letter], MAX_LOCI)
    return [(slot(s.source), slot(s.target)) for s in signs]


def marker_matrix(n: int, markers: list[dict]) -> list[list[int]]:
    """Multi-hot (n, len(MARKERS)) of the markers spanning each surface token."""
    out = [[0] * len(MARKERS) for _ in range(n)]
    for m in markers:
        if m.get("tier") not in MARKERS or m.get("token_start") is None:
            continue
        k = MARKERS.index(m["tier"])
        for i in range(m["token_start"], min(m["token_end"], n)):
            out[i][k] = 1
    return out


@dataclass(frozen=True)
class PlanTarget:
    """A planner target: one entry per surface sign."""
    bases: tuple[str, ...]
    loci: tuple[tuple[int, int], ...]
    markers: tuple[tuple[int, ...], ...]


def plan_target(pair: dict) -> PlanTarget:
    signs = [parse_surface_token(t) for t in pair["surface"]]
    return PlanTarget(bases=tuple(s.base for s in signs), loci=tuple(canonical_loci(signs)),
                      markers=tuple(tuple(r) for r in marker_matrix(len(signs), pair.get("markers") or [])))


class GlossVocabulary:
    def __init__(self, glosses: list[str]):
        self.glosses = list(glosses)
        self.index = {g: FIRST_GLOSS + i for i, g in enumerate(self.glosses)}

    @classmethod
    def build(cls, targets, *, min_count: int = 2, max_glosses: int = 4000) -> "GlossVocabulary":
        counts = collections.Counter(b for t in targets for b in t.bases)
        ranked = sorted((g for g, c in counts.items() if c >= min_count), key=lambda g: (-counts[g], g))
        return cls(ranked[:max_glosses])

    def __len__(self) -> int:
        return FIRST_GLOSS + len(self.glosses)

    def pieces(self, base: str) -> list[int]:
        """Token ids of one gloss: its own id, or its UTF-8 bytes then GLOSS_END."""
        if base in self.index:
            return [self.index[base]]
        return [BYTE0 + b for b in base.encode("utf-8")] + [GLOSS_END]

    def decode(self, ids) -> list[str]:
        out, spelled = [], bytearray()
        for i in ids:
            i = int(i)
            if i in (PAD, BOS):
                continue
            if i == EOS:
                break
            if BYTE0 <= i < BYTE0 + 256:
                spelled.append(i - BYTE0)
            elif i == GLOSS_END:
                out.append(spelled.decode("utf-8", errors="replace")); spelled = bytearray()
            elif FIRST_GLOSS <= i < len(self):
                out.append(self.glosses[i - FIRST_GLOSS])
        if spelled:
            out.append(spelled.decode("utf-8", errors="replace"))
        return out

    def to_dict(self) -> dict:
        return dict(glosses=self.glosses)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.glosses).encode()).hexdigest()


def is_unit_end(token_id: int) -> bool:
    """A position that closes one surface sign (a whole-gloss token or GLOSS_END): attributes are predicted there."""
    return token_id == GLOSS_END or token_id >= FIRST_GLOSS


def encode_target(vocab: GlossVocabulary, target: PlanTarget, *, max_tokens: int = 128):
    """(token ids with BOS/EOS, per-token source slot, target slot, marker multi-hot, unit-end mask).

    Attributes sit on the token that closes each sign; other positions carry zeros and a False mask. A target longer
    than ``max_tokens`` is truncated at a sign boundary.
    """
    ids, src, dst, mk, end = [BOS], [0], [0], [[0] * len(MARKERS)], [False]
    for base, (s, d), m in zip(target.bases, target.loci, target.markers):
        pieces = vocab.pieces(base)
        if len(ids) + len(pieces) + 1 > max_tokens:
            break
        for j, p in enumerate(pieces):
            last = j == len(pieces) - 1
            ids.append(p); src.append(s if last else 0); dst.append(d if last else 0)
            mk.append(list(m) if last else [0] * len(MARKERS)); end.append(last)
    ids.append(EOS); src.append(0); dst.append(0); mk.append([0] * len(MARKERS)); end.append(False)
    return ids, src, dst, mk, end


__all__ = ["MARKERS", "MAX_LOCI", "PAD", "BOS", "EOS", "GLOSS_END", "BYTE0", "FIRST_GLOSS", "SurfaceSign",
           "parse_surface_token", "canonical_loci", "marker_matrix", "PlanTarget", "plan_target", "GlossVocabulary",
           "is_unit_end", "encode_target"]
