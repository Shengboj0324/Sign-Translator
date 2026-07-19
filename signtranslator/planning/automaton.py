"""Schema automaton: a DFA over the plan serialization grammar.

The grammar of ``schema.py`` has a fixed slot order and finite, history-depth-
independent value alphabets, so its language is **regular** and recognised by a
deterministic finite automaton. This automaton drives constrained decoding
(``constrained.py``): at each step it supplies the set of legal next tokens, and
a decoder masked to that set can only ever produce a well-formed skeleton.

Three properties are proved in the tests, not assumed:

* **Soundness** -- every string the DFA accepts deserializes without error.
* **Liveness** -- every reachable non-accepting state has a non-empty allowed
  set, so decoding never dead-ends before ``EOP``.
* **Determinism** -- from any state, a token's *kind* selects at most one edge,
  so the DFA is a genuine (partial) function. Asserted at construction time.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from .schema import PlanVocabulary, DEFAULT_VOCAB, STRUCTURAL

STRUCTURAL_KINDS = frozenset(STRUCTURAL)


class S(Enum):
    """Automaton states, one per position in the fixed slot order."""

    START = auto()            # expect BOP
    AFTER_BOP = auto()        # expect PRED marker
    EXPECT_PRED_V = auto()    # expect predicate value
    AFTER_PRED_V = auto()     # expect ARGS marker
    ARGS_ROLE_OR_END = auto()  # expect ROLE (new arg) or REFS marker
    ARGS_REF = auto()         # expect REF (second half of an arg pair)
    REFS_REF_OR_END = auto()  # expect REF or TAM marker
    EXPECT_TAM_V = auto()     # expect TAM value
    AFTER_TAM_V = auto()      # expect LOCI marker
    LOCI_REF_OR_END = auto()  # expect REF or UNITS marker
    LOCI_LOCUS = auto()       # expect LOCUS (second half of a locus pair)
    UNITS_LEX_OR_END = auto()  # expect LEX or NMS marker
    NMS_NM_OR_END = auto()    # expect NM or FS marker
    NMS_I = auto()            # expect IDX (span start)
    NMS_J = auto()            # expect IDX (span end)
    FS_IDX_OR_END = auto()    # expect IDX or CONF marker
    EXPECT_CONF_V = auto()    # expect CONF value
    AFTER_CONF_V = auto()     # expect EOP
    ACCEPT = auto()           # terminal


# Each edge is (token-kind, next-state). A "kind" is either a structural marker
# name ("BOP", "ARGS", ...) or a value kind ("ROLE", "REF", ...). All values of
# a value kind share the edge. Within a state, kinds are distinct -> determinism.
_TRANSITIONS: Dict[S, List[Tuple[str, S]]] = {
    S.START:            [("BOP", S.AFTER_BOP)],
    S.AFTER_BOP:        [("PRED", S.EXPECT_PRED_V)],
    S.EXPECT_PRED_V:    [("PRED_V", S.AFTER_PRED_V)],
    S.AFTER_PRED_V:     [("ARGS", S.ARGS_ROLE_OR_END)],
    S.ARGS_ROLE_OR_END: [("ROLE", S.ARGS_REF), ("REFS", S.REFS_REF_OR_END)],
    S.ARGS_REF:         [("REF", S.ARGS_ROLE_OR_END)],
    S.REFS_REF_OR_END:  [("REF", S.REFS_REF_OR_END), ("TAM", S.EXPECT_TAM_V)],
    S.EXPECT_TAM_V:     [("TAM_V", S.AFTER_TAM_V)],
    S.AFTER_TAM_V:      [("LOCI", S.LOCI_REF_OR_END)],
    S.LOCI_REF_OR_END:  [("REF", S.LOCI_LOCUS), ("UNITS", S.UNITS_LEX_OR_END)],
    S.LOCI_LOCUS:       [("LOCUS", S.LOCI_REF_OR_END)],
    S.UNITS_LEX_OR_END: [("LEX", S.UNITS_LEX_OR_END), ("NMS", S.NMS_NM_OR_END)],
    S.NMS_NM_OR_END:    [("NM", S.NMS_I), ("FS", S.FS_IDX_OR_END)],
    S.NMS_I:            [("IDX", S.NMS_J)],
    S.NMS_J:            [("IDX", S.NMS_NM_OR_END)],
    S.FS_IDX_OR_END:    [("IDX", S.FS_IDX_OR_END), ("CONF", S.EXPECT_CONF_V)],
    S.EXPECT_CONF_V:    [("CONF_V", S.AFTER_CONF_V)],
    S.AFTER_CONF_V:     [("EOP", S.ACCEPT)],
    S.ACCEPT:           [],
}


# The "decision" states are where a variable-length slot either repeats (adds
# another element) or advances (emits its closing marker). Each maps to the
# token *kind* that starts a new element and the vocab attribute that bounds how
# many elements the slot may hold. Bounding these makes the language FINITE, so
# constrained decoding is guaranteed to terminate -- an unbounded DFA lets a
# poorly-conditioned model loop forever in a repeatable slot.
_REPEAT_DECISION: Dict[S, Tuple[str, str]] = {
    S.ARGS_ROLE_OR_END: ("ROLE", "max_args"),
    S.REFS_REF_OR_END:  ("REF", "num_referents"),
    S.LOCI_REF_OR_END:  ("REF", "num_referents"),
    S.UNITS_LEX_OR_END: ("LEX", "max_units"),
    S.NMS_NM_OR_END:    ("NM", "max_nonmanual"),
    S.FS_IDX_OR_END:    ("IDX", "max_units"),
}


class SchemaAutomaton:
    """Deterministic recogniser of the plan serialization language.

    Two layers:

    * the **base DFA** (``allowed_tokens`` / ``step`` / ``accepts``) recognises
      the full, unbounded serialization language; its soundness/liveness/
      determinism are the properties proved in the tests;
    * a **bounded runtime** (``bounded_allowed`` / ``bounded_step``) additionally
      caps each variable-length slot, yielding a finite sub-language. Every
      bounded run is also a base run (the caps only *remove* the repeat option),
      so soundness transfers, and the advance marker is always retained, so
      bounded liveness holds too. This layer is what guarantees decoding halts.
    """

    def __init__(self, vocab: PlanVocabulary = DEFAULT_VOCAB) -> None:
        self.vocab = vocab
        self.transitions = _TRANSITIONS
        self._check_determinism()
        # Precompute per-state allowed sets and per-state (kind -> next) maps.
        # These are constant for a given vocab and are hit on every decode step.
        self._allowed: Dict[S, Set[int]] = {}
        self._edge_by_kind: Dict[S, Dict[str, S]] = {}
        for state, edges in self.transitions.items():
            self._edge_by_kind[state] = {kind: nxt for kind, nxt in edges}
            self._allowed[state] = self._compute_allowed(state)

    def _check_determinism(self) -> None:
        for state, edges in self.transitions.items():
            kinds = [kind for kind, _ in edges]
            if len(kinds) != len(set(kinds)):
                raise AssertionError(f"non-deterministic edges from {state}: {kinds}")

    # -- basic DFA interface ------------------------------------------------
    @property
    def initial_state(self) -> S:
        return S.START

    def is_accepting(self, state: S) -> bool:
        return state is S.ACCEPT

    def _compute_allowed(self, state: S) -> Set[int]:
        out: Set[int] = set()
        for kind, _ in self.transitions[state]:
            if kind in STRUCTURAL_KINDS:
                out.add(self.vocab.token(kind))
            else:
                for v in range(self.vocab.value_kind_size(kind)):
                    out.add(self.vocab.token(kind, v))
        return out

    def step(self, state: S, token: int) -> Optional[S]:
        """Advance on a token id; ``None`` if the token is not allowed here."""
        kind, _ = self.vocab.decode(int(token))
        return self._edge_by_kind[state].get(kind)

    def allowed_tokens(self, state: S) -> Set[int]:
        """The set of legal next token ids from ``state`` (empty at ACCEPT)."""
        return self._allowed[state]

    # -- recognition --------------------------------------------------------
    # -- bounded runtime (guarantees termination) --------------------------
    def cap_of(self, state: S) -> Optional[int]:
        """The repeat cap at a decision state, else ``None``."""
        dec = _REPEAT_DECISION.get(state)
        return getattr(self.vocab, dec[1]) if dec else None

    def bounded_allowed(self, state: S, counts: Dict[S, int]) -> Set[int]:
        """Allowed tokens with the slot-repeat cap applied.

        At a decision state whose repeat count has reached its cap, the tokens
        that would start another element are removed, leaving only the advance
        marker -- so the set is never empty for a non-accepting state.
        """
        allowed = self._allowed[state]
        dec = _REPEAT_DECISION.get(state)
        if dec is None:
            return allowed
        kind, attr = dec
        if counts.get(state, 0) >= getattr(self.vocab, attr):
            return {t for t in allowed if self.vocab.decode(t)[0] != kind}
        return allowed

    def bounded_step(self, state: S, token: int,
                     counts: Dict[S, int]) -> Tuple[Optional[S], Dict[S, int]]:
        """Advance the bounded runtime, updating slot counts."""
        kind, _ = self.vocab.decode(int(token))
        new_counts = dict(counts)
        dec = _REPEAT_DECISION.get(state)
        if dec is not None and kind == dec[0]:
            new_counts[state] = new_counts.get(state, 0) + 1
        return self.step(state, token), new_counts

    def max_generated_length(self) -> int:
        """An upper bound on any bounded run's length -> a safe decode cap."""
        # fixed markers/values + capped repeats (pairs/triples counted).
        fixed = 14
        return (fixed + self.vocab.max_args * 2 + self.vocab.num_referents
                + self.vocab.num_referents * 2 + self.vocab.max_units
                + self.vocab.max_nonmanual * 3 + self.vocab.max_units)

    def accepts(self, tokens) -> bool:
        """Whether the DFA accepts the full token sequence."""
        state = self.initial_state
        for tok in tokens:
            nxt = self.step(state, int(tok))
            if nxt is None:
                return False
            state = nxt
        return self.is_accepting(state)

    # -- analysis (used by the proofs) --------------------------------------
    def reachable_states(self) -> Set[S]:
        seen = {self.initial_state}
        frontier = [self.initial_state]
        while frontier:
            state = frontier.pop()
            for _, nxt in self.transitions[state]:
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen

    def enumerate_accepted(self, max_length: int):
        """Yield every accepted token sequence of length <= ``max_length``.

        Breadth-first over (state, prefix). Exponential in general, so only for
        the tiny vocabularies used to prove soundness exhaustively.
        """
        stack = [(self.initial_state, [])]
        while stack:
            state, prefix = stack.pop()
            if self.is_accepting(state):
                yield list(prefix)
            if len(prefix) >= max_length:
                continue
            for tok in sorted(self.allowed_tokens(state)):
                nxt = self.step(state, tok)
                if nxt is not None:
                    stack.append((nxt, prefix + [tok]))
