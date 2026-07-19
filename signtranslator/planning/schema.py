"""Typed sign-plan schema, serialization, and structural validation.

The plan is the *typed interface* the specification demands in place of
free-form gloss. It is linearized to a token sequence over a fixed vocabulary
with a fixed slot order, so that (a) an autoregressive decoder can emit it, (b) a
finite-state automaton can constrain that emission to well-formed skeletons
(Stage 2b), and (c) serialize/deserialize is an exact round-trip.

Serialization grammar (fixed slot order; ``*`` = zero or more, ``?`` = optional):

    BOP  PRED <pred>  ARGS (<role> <ref>)*  REFS (<ref>)*  TAM <tam>
    TOPIC <ref>?  FOCUS <ref>?  LOCI (<ref> <locus>)*
    UNITS (<lex>)*  CLS (<lex>)*  NMS (<marker> <i> <j>)*
    FS (<i>)*  CONF <conf>  EOP

Every slot marker is always present (even for empty variable-length slots and for
the optional TOPIC/FOCUS referents), so the language is regular: the set of legal
next tokens depends only on a finite "slot state", never on unbounded history.
TOPIC/FOCUS carry an *optional* single referent (the marker is emitted with a
following REF iff the field is set), so information-structure marking round-trips
exactly. Cross-slot *consistency* constraints (a referent used must be declared, a
locus must be unique, ...) are NOT part of the grammar; they are enforced by
:func:`validate_plan`. The grammar guarantees a well-formed *skeleton*; the
validator guarantees a well-formed *plan*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Token "kinds" -- disjoint namespaces whose union is the vocabulary.
STRUCTURAL = ("BOP", "PRED", "ARGS", "REFS", "TAM", "TOPIC", "FOCUS", "LOCI",
              "UNITS", "CLS", "NMS", "FS", "CONF", "EOP")
VALUE_KINDS = ("ROLE", "REF", "PRED_V", "TAM_V", "LOCUS", "LEX", "NM", "IDX", "CONF_V")


@dataclass(frozen=True)
class PlanVocabulary:
    """Sizes of the typed inventories, and the resulting token id layout.

    Token ids are assigned in fixed, contiguous blocks so that a model's output
    vocabulary and the automaton's alphabet are the same integers.
    """

    num_predicates: int = 8
    num_roles: int = 6
    num_referents: int = 6
    num_tam: int = 6
    num_loci: int = 7
    num_lexemes: int = 32
    num_nonmanual: int = 5
    max_units: int = 12          # bounds IDX tokens (indices into manual_units)
    num_conf_buckets: int = 11   # quantised confidence levels -> {0/10 .. 10/10}
    max_args: int = 8            # cap on semantic-frame arguments
    max_nonmanual: int = 8       # cap on non-manual scope spans

    def __post_init__(self) -> None:
        for name in ("num_predicates", "num_roles", "num_referents", "num_tam",
                     "num_loci", "num_lexemes", "num_nonmanual", "max_units"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.num_conf_buckets < 2:
            raise ValueError("num_conf_buckets must be >= 2")
        if self.max_args < 1 or self.max_nonmanual < 1:
            raise ValueError("max_args and max_nonmanual must be >= 1")
        # Precompute the id layout once. This is a frozen dataclass, so the
        # tables are immutable and safe to cache; recomputing them per call made
        # automaton enumeration and model decoding quadratically slow.
        sizes = [
            *[(k, 1) for k in STRUCTURAL],
            ("ROLE", self.num_roles), ("REF", self.num_referents),
            ("PRED_V", self.num_predicates), ("TAM_V", self.num_tam),
            ("LOCUS", self.num_loci), ("LEX", self.num_lexemes),
            ("NM", self.num_nonmanual), ("IDX", self.max_units),
            ("CONF_V", self.num_conf_buckets),
        ]
        offsets, cursor, decode_table = {}, 0, []
        for kind, size in sizes:
            offsets[kind] = cursor
            for v in range(size):
                decode_table.append((kind, v))
            cursor += size
        object.__setattr__(self, "_size_map", dict(sizes))
        object.__setattr__(self, "_offset_map", offsets)
        object.__setattr__(self, "_decode_table", tuple(decode_table))
        object.__setattr__(self, "_total_size", cursor)

    @property
    def size(self) -> int:
        return self._total_size

    def token(self, kind: str, value: int = 0) -> int:
        """Global token id for ``kind`` (structural) or ``kind[value]`` (value)."""
        if kind in STRUCTURAL:
            if value != 0:
                raise ValueError(f"structural token {kind} takes no value")
            return self._offset_map[kind]
        if kind not in VALUE_KINDS:
            raise ValueError(f"unknown token kind {kind}")
        size = self._size_map[kind]
        if not 0 <= value < size:
            raise ValueError(f"{kind} value {value} out of range [0, {size})")
        return self._offset_map[kind] + value

    def decode(self, token: int) -> Tuple[str, int]:
        """Inverse of :meth:`token`: global id -> (kind, value)."""
        if not 0 <= token < self._total_size:
            raise ValueError(f"token id {token} out of range [0, {self._total_size})")
        return self._decode_table[token]

    def value_kind_size(self, kind: str) -> int:
        """Number of distinct values for a value kind."""
        return self._size_map[kind]

    def confidence_of_bucket(self, bucket: int) -> float:
        return bucket / (self.num_conf_buckets - 1)

    def bucket_of_confidence(self, confidence: float) -> int:
        c = min(1.0, max(0.0, float(confidence)))
        return int(round(c * (self.num_conf_buckets - 1)))


DEFAULT_VOCAB = PlanVocabulary()


# ---------------------------------------------------------------------------
# Plan dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SemanticFrame:
    """A predicate and its (role, referent) arguments."""

    predicate: int
    args: List[Tuple[int, int]] = field(default_factory=list)  # (role, referent)


@dataclass(frozen=True)
class NonmanualSpan:
    """A non-manual marker scoping over ``manual_units[start:end+1]``."""

    marker: int
    start: int
    end: int


@dataclass
class SignPlan:
    """The typed sign plan (see module docstring for the field semantics)."""

    frame: SemanticFrame
    referents: List[int] = field(default_factory=list)
    tam: int = 0
    topic: Optional[int] = None
    focus: Optional[int] = None
    loci: Dict[int, int] = field(default_factory=dict)          # referent -> locus
    manual_units: List[int] = field(default_factory=list)       # lexeme ids
    classifiers: List[int] = field(default_factory=list)        # lexeme ids
    nonmanual: List[NonmanualSpan] = field(default_factory=list)
    fingerspelling: List[int] = field(default_factory=list)     # indices into units
    conf_bucket: int = 0
    # provenance is system-attached bookkeeping, NOT part of the token grammar
    # the decoder generates; it is kept here for the typed interface.
    provenance: Dict[str, str] = field(default_factory=dict)

    def confidence(self, vocab: PlanVocabulary = DEFAULT_VOCAB) -> float:
        return vocab.confidence_of_bucket(self.conf_bucket)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def serialize_plan(plan: SignPlan, vocab: PlanVocabulary = DEFAULT_VOCAB) -> List[int]:
    """Linearize a plan to token ids following the fixed-order grammar.

    Serialization covers every decoder-generatable field (all of the typed plan
    except ``provenance``, which is system-attached bookkeeping). In particular
    ``topic``/``focus`` (optional referents) and ``classifiers`` DO round-trip.
    ``deserialize_plan`` is its exact inverse.
    """
    t = vocab.token
    out: List[int] = [t("BOP")]

    out += [t("PRED"), t("PRED_V", plan.frame.predicate)]
    out.append(t("ARGS"))
    for role, ref in plan.frame.args:
        out += [t("ROLE", role), t("REF", ref)]
    out.append(t("REFS"))
    for ref in plan.referents:
        out.append(t("REF", ref))
    out += [t("TAM"), t("TAM_V", plan.tam)]
    # TOPIC / FOCUS: marker always present, followed by a single REF iff set.
    out.append(t("TOPIC"))
    if plan.topic is not None:
        out.append(t("REF", plan.topic))
    out.append(t("FOCUS"))
    if plan.focus is not None:
        out.append(t("REF", plan.focus))
    out.append(t("LOCI"))
    for ref, locus in plan.loci.items():
        out += [t("REF", ref), t("LOCUS", locus)]
    out.append(t("UNITS"))
    for lex in plan.manual_units:
        out.append(t("LEX", lex))
    out.append(t("CLS"))
    for lex in plan.classifiers:
        out.append(t("LEX", lex))
    out.append(t("NMS"))
    for span in plan.nonmanual:
        out += [t("NM", span.marker), t("IDX", span.start), t("IDX", span.end)]
    out.append(t("FS"))
    for idx in plan.fingerspelling:
        out.append(t("IDX", idx))
    out += [t("CONF"), t("CONF_V", plan.conf_bucket)]
    out.append(t("EOP"))
    return out


class DeserializationError(ValueError):
    """Raised when a token stream does not conform to the grammar skeleton."""


def deserialize_plan(tokens: Sequence[int],
                     vocab: PlanVocabulary = DEFAULT_VOCAB) -> SignPlan:
    """Exact inverse of :func:`serialize_plan`.

    Raises :class:`DeserializationError` if the stream is not a well-formed
    skeleton (wrong marker order, wrong value kind, truncated pair, ...). Note
    this checks the *skeleton*, not cross-slot consistency -- use
    :func:`validate_plan` for that.
    """
    kinds = [vocab.decode(int(tok)) for tok in tokens]
    pos = 0

    def expect_struct(name: str) -> None:
        nonlocal pos
        if pos >= len(kinds) or kinds[pos] != (name, 0):
            got = kinds[pos] if pos < len(kinds) else "END"
            raise DeserializationError(f"expected {name}, got {got} at {pos}")
        pos += 1

    def take_value(kind: str) -> int:
        nonlocal pos
        if pos >= len(kinds) or kinds[pos][0] != kind:
            got = kinds[pos] if pos < len(kinds) else "END"
            raise DeserializationError(f"expected {kind} value, got {got} at {pos}")
        val = kinds[pos][1]
        pos += 1
        return val

    def peek_kind() -> Optional[str]:
        return kinds[pos][0] if pos < len(kinds) else None

    expect_struct("BOP")
    expect_struct("PRED")
    predicate = take_value("PRED_V")
    frame = SemanticFrame(predicate=predicate)

    expect_struct("ARGS")
    while peek_kind() == "ROLE":
        role = take_value("ROLE")
        ref = take_value("REF")            # a truncated pair raises here
        frame.args.append((role, ref))

    plan = SignPlan(frame=frame)
    expect_struct("REFS")
    while peek_kind() == "REF":
        plan.referents.append(take_value("REF"))

    expect_struct("TAM")
    plan.tam = take_value("TAM_V")

    # TOPIC / FOCUS carry an OPTIONAL single referent (zero-or-one REF).
    expect_struct("TOPIC")
    plan.topic = take_value("REF") if peek_kind() == "REF" else None
    expect_struct("FOCUS")
    plan.focus = take_value("REF") if peek_kind() == "REF" else None

    expect_struct("LOCI")
    while peek_kind() == "REF":
        ref = take_value("REF")
        locus = take_value("LOCUS")
        plan.loci[ref] = locus

    expect_struct("UNITS")
    while peek_kind() == "LEX":
        plan.manual_units.append(take_value("LEX"))

    expect_struct("CLS")
    while peek_kind() == "LEX":
        plan.classifiers.append(take_value("LEX"))

    expect_struct("NMS")
    while peek_kind() == "NM":
        marker = take_value("NM")
        start = take_value("IDX")
        end = take_value("IDX")
        plan.nonmanual.append(NonmanualSpan(marker, start, end))

    expect_struct("FS")
    while peek_kind() == "IDX":
        plan.fingerspelling.append(take_value("IDX"))

    expect_struct("CONF")
    plan.conf_bucket = take_value("CONF_V")
    expect_struct("EOP")
    if pos != len(kinds):
        raise DeserializationError(f"trailing tokens after EOP at {pos}")
    return plan


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------
def validate_plan(plan: SignPlan, vocab: PlanVocabulary = DEFAULT_VOCAB,
                  lexicon: Optional["object"] = None) -> List[str]:
    """Return the list of violated structural rules (empty == valid).

    Rules mirror docs/SEMANTIC_PLANNER.md §2.1. ``lexicon`` (Stage 2c) enables
    the hallucination rule; without it that rule is skipped.
    """
    violations: List[str] = []
    declared = set(plan.referents)
    n_units = len(plan.manual_units)

    # 1. args reference only declared referents
    for role, ref in plan.frame.args:
        if not 0 <= role < vocab.num_roles:
            violations.append("role_out_of_range")
        if ref not in declared:
            violations.append("arg_referent_undeclared")

    # 2. topic/focus are declared referents
    if plan.topic is not None and plan.topic not in declared:
        violations.append("topic_undeclared")
    if plan.focus is not None and plan.focus not in declared:
        violations.append("focus_undeclared")

    # ranges of scalar fields
    if not 0 <= plan.frame.predicate < vocab.num_predicates:
        violations.append("predicate_out_of_range")
    if not 0 <= plan.tam < vocab.num_tam:
        violations.append("tam_out_of_range")
    for ref in plan.referents:
        if not 0 <= ref < vocab.num_referents:
            violations.append("referent_out_of_range")

    # 3. every declared referent has a locus in the locus alphabet
    for ref in declared:
        if ref not in plan.loci:
            violations.append("referent_without_locus")
    for ref, locus in plan.loci.items():
        if ref not in declared:
            violations.append("locus_for_undeclared_referent")
        if not 0 <= locus < vocab.num_loci:
            violations.append("locus_out_of_range")

    # 4. distinct referents occupy distinct loci
    assigned = list(plan.loci.values())
    if len(assigned) != len(set(assigned)):
        violations.append("locus_collision")

    # 5. non-manual scope spans are within the manual-unit range and ordered
    for span in plan.nonmanual:
        if not 0 <= span.marker < vocab.num_nonmanual:
            violations.append("nonmanual_marker_out_of_range")
        if not (0 <= span.start <= span.end < max(n_units, 1)) or n_units == 0:
            violations.append("nonmanual_scope_out_of_bounds")

    # 6. fingerspelling indices point at real units
    for idx in plan.fingerspelling:
        if not 0 <= idx < n_units:
            violations.append("fingerspell_index_out_of_bounds")

    # 7. confidence bucket in range
    if not 0 <= plan.conf_bucket < vocab.num_conf_buckets:
        violations.append("confidence_out_of_range")

    # lexeme range
    for lex in plan.manual_units:
        if not 0 <= lex < vocab.num_lexemes:
            violations.append("lexeme_out_of_range")

    # 8. hallucination rule: each unit is a lexicon entry or fingerspelled
    if lexicon is not None:
        fs = set(plan.fingerspelling)
        for i, lex in enumerate(plan.manual_units):
            in_lex = lexicon.contains(lex) if hasattr(lexicon, "contains") else (lex in lexicon)
            if not in_lex and i not in fs:
                violations.append("hallucinated_lexical_entry")

    # de-duplicate while preserving order, so a rule reports once
    seen, unique = set(), []
    for v in violations:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique
