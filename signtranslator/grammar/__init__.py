"""Sign-language grammar layer: the Structured Intermediate Representation (SIR).

Implements `03_sign_language_grammar_translation.md` (see docs/GRAMMAR_SIR.md).

We model one named language (ASL). The SIR is a temporal graph whose nodes are
manual / non-manual events carrying time intervals and whose typed edges encode
precedence, overlap, scope, co-reference, and spatial locus. A gloss sequence is
one observed *projection* of this graph, never the whole of it.
"""

from .sir import (
    EventKind, EdgeType, SIREvent, SIREdge, SIRGraph,
    validate_sir, gloss_projection, is_topological_order,
)

__all__ = [
    "EventKind", "EdgeType", "SIREvent", "SIREdge", "SIRGraph",
    "validate_sir", "gloss_projection", "is_topological_order",
]
