"""Versioned sign lexicon, retrieval grounding, and hallucination detection.

The specification requires grounding the decoder with retrieval ``D(x)`` from a
**versioned** sign lexicon/corpus, and reporting *hallucinated lexical entries*
and *invalid spatial references*.

Versioning is not decoration. A sign plan is only reproducible and auditable if
the lexicon it was grounded against is pinned: two runs that disagree must be
attributable either to the model or to a lexicon change, never ambiguously to
both. Each :class:`SignLexicon` therefore carries a semantic ``version`` string
and a **content hash** computed from its entries, so provenance can record
exactly which lexicon produced a plan and a silent edit is detectable.

Retrieval here is deliberately simple (embedding cosine similarity over a small
in-memory store): the retrieval *interface and its grounding role* are what
matter for this layer, not a production ANN index.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class LexEntry:
    """One lexicon entry: a lexeme id, a gloss label, and an embedding."""

    lexeme: int
    gloss: str
    embedding: Tuple[float, ...]

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor(self.embedding, dtype=torch.float32)


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved candidate with its score and its lexicon provenance."""

    lexeme: int
    gloss: str
    score: float
    lexicon_version: str
    lexicon_hash: str


class SignLexicon:
    """A versioned, content-hashed lexicon supporting retrieval and membership."""

    def __init__(self, entries: Sequence[LexEntry], version: str = "0.0.0",
                 embedding_dim: Optional[int] = None) -> None:
        if not entries:
            raise ValueError("lexicon must have at least one entry")
        dims = {len(e.embedding) for e in entries}
        if len(dims) != 1:
            raise ValueError("all entries must share an embedding dimension")
        self.embedding_dim = embedding_dim or next(iter(dims))
        if self.embedding_dim != next(iter(dims)):
            raise ValueError("embedding_dim mismatch with entries")

        lexemes = [e.lexeme for e in entries]
        if len(lexemes) != len(set(lexemes)):
            raise ValueError("duplicate lexeme ids in lexicon")

        self.version = version
        self._by_lexeme: Dict[int, LexEntry] = {e.lexeme: e for e in entries}
        self._entries: List[LexEntry] = list(entries)
        # Row-normalised embedding matrix for cosine retrieval.
        mat = torch.stack([e.as_tensor() for e in entries], dim=0)
        self._matrix = torch.nn.functional.normalize(mat, dim=-1)
        self._lexeme_order = lexemes
        self.content_hash = self._compute_hash()

    # -- versioning ---------------------------------------------------------
    def _compute_hash(self) -> str:
        """SHA-256 over the sorted, serialised entries -- edit-detecting."""
        payload = json.dumps(
            [[e.lexeme, e.gloss, [round(x, 6) for x in e.embedding]]
             for e in sorted(self._entries, key=lambda e: e.lexeme)],
            sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def fingerprint(self) -> str:
        return f"{self.version}@{self.content_hash}"

    def __len__(self) -> int:
        return len(self._entries)

    # -- membership ---------------------------------------------------------
    def contains(self, lexeme: int) -> bool:
        return lexeme in self._by_lexeme

    def gloss_of(self, lexeme: int) -> Optional[str]:
        entry = self._by_lexeme.get(lexeme)
        return entry.gloss if entry else None

    # -- retrieval D(x) -----------------------------------------------------
    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> List[RetrievalResult]:
        """Return the ``top_k`` entries by cosine similarity to ``query``.

        Results carry the lexicon fingerprint so downstream provenance can pin
        exactly which lexicon version produced them.
        """
        if query.dim() != 1 or query.shape[0] != self.embedding_dim:
            raise ValueError(f"query must be a ({self.embedding_dim},) vector")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        q = torch.nn.functional.normalize(query.float(), dim=-1)
        scores = self._matrix @ q                       # cosine, in [-1, 1]
        k = min(top_k, len(self._entries))
        top = torch.topk(scores, k)
        out = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            entry = self._entries[idx]
            out.append(RetrievalResult(
                lexeme=entry.lexeme, gloss=entry.gloss, score=float(score),
                lexicon_version=self.version, lexicon_hash=self.content_hash))
        return out

    def provenance_stamp(self) -> Dict[str, str]:
        return {"lexicon_version": self.version, "lexicon_hash": self.content_hash}


# ---------------------------------------------------------------------------
# Hallucination / invalid-reference reporting
# ---------------------------------------------------------------------------
@dataclass
class GroundingReport:
    """What a plan asked for that the lexicon / signing space cannot support."""

    hallucinated_units: List[int] = field(default_factory=list)     # unit indices
    invalid_spatial_refs: List[int] = field(default_factory=list)   # referent ids
    num_units: int = 0
    num_referents: int = 0

    @property
    def hallucination_rate(self) -> float:
        return len(self.hallucinated_units) / self.num_units if self.num_units else 0.0

    @property
    def invalid_reference_rate(self) -> float:
        return (len(self.invalid_spatial_refs) / self.num_referents
                if self.num_referents else 0.0)

    @property
    def is_grounded(self) -> bool:
        return not self.hallucinated_units and not self.invalid_spatial_refs

    def summary(self) -> str:
        return (f"hallucinated {len(self.hallucinated_units)}/{self.num_units} units, "
                f"invalid {len(self.invalid_spatial_refs)}/{self.num_referents} refs")


def ground_plan(plan, lexicon: SignLexicon, num_loci: int) -> GroundingReport:
    """Check a plan against the lexicon and the finite signing-space loci.

    * A **hallucinated unit** is a manual unit whose lexeme is not in the
      lexicon and which is not fingerspelled (fingerspelling is the licensed
      escape hatch for out-of-lexicon items).
    * An **invalid spatial reference** is a referent placed at a locus outside
      the fixed locus alphabet, or an arg/topic/focus referent with no locus at
      all (it cannot be pointed at in signing space).
    """
    report = GroundingReport(num_units=len(plan.manual_units),
                             num_referents=len(plan.referents))
    fs = set(plan.fingerspelling)
    for i, lex in enumerate(plan.manual_units):
        if not lexicon.contains(lex) and i not in fs:
            report.hallucinated_units.append(i)

    for ref in plan.referents:
        locus = plan.loci.get(ref)
        if locus is None or not 0 <= locus < num_loci:
            report.invalid_spatial_refs.append(ref)
    # An arg/topic/focus referent must also be placeable.
    used = {r for _, r in plan.frame.args}
    if plan.topic is not None:
        used.add(plan.topic)
    if plan.focus is not None:
        used.add(plan.focus)
    for ref in used:
        locus = plan.loci.get(ref)
        if (locus is None or not 0 <= locus < num_loci) and ref not in report.invalid_spatial_refs:
            report.invalid_spatial_refs.append(ref)
    return report
