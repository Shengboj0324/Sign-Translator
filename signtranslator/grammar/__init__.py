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
from .temporal import (
    AllenRelation, classify_relation, intervals_intersect,
    validity_loss, precedence_loss, meets_loss, contains_loss, during_loss,
    overlap_loss, equals_loss, edge_temporal_loss, sir_temporal_loss,
)
from .graph_transformer import (
    relation_biased_attention, RelationBiasedAttention, GraphTransformerLayer,
    SIRDecoder,
)
from .nonmanual import (
    MarkerSpan, multilabel_scope_bce, spans_from_activations,
    scope_containment_loss, NonmanualScopeHead,
)
from .notation import (
    Handshape, Location, Movement, Orientation, SignPhonology,
    Register, SignVariant, VariationSet, GlossSource, GlossLabel,
)
from .signbleu import (
    sign_bleu, SignBLEUResult, modified_precision, within_channel_ngrams,
    blended_ngrams, cohens_kappa, fleiss_kappa, GrammaticalityRating,
    agreement_on_grammaticality,
)
from .grammar_tests import (
    Aspect, QuestionType, GrammarFeatures, ControllableASLBuilder,
    LICENSED, MinimalPairResult, minimal_pair, changed_sir_fields,
    build_discourse, locus_of_referent, realise_with_fingerspelling,
    NM_NEG, NM_WH, NM_YN, NM_TOPIC, NM_COND, NM_ROLESHIFT,
)
from .integration import plan_to_sir, plan_manual_units

__all__ = [
    "EventKind", "EdgeType", "SIREvent", "SIREdge", "SIRGraph",
    "validate_sir", "gloss_projection", "is_topological_order",
    "AllenRelation", "classify_relation", "intervals_intersect",
    "validity_loss", "precedence_loss", "meets_loss", "contains_loss",
    "during_loss", "overlap_loss", "equals_loss", "edge_temporal_loss",
    "sir_temporal_loss",
    "relation_biased_attention", "RelationBiasedAttention",
    "GraphTransformerLayer", "SIRDecoder",
    "MarkerSpan", "multilabel_scope_bce", "spans_from_activations",
    "scope_containment_loss", "NonmanualScopeHead",
    "Handshape", "Location", "Movement", "Orientation", "SignPhonology",
    "Register", "SignVariant", "VariationSet", "GlossSource", "GlossLabel",
    "sign_bleu", "SignBLEUResult", "modified_precision", "within_channel_ngrams",
    "blended_ngrams", "cohens_kappa", "fleiss_kappa", "GrammaticalityRating",
    "agreement_on_grammaticality",
    "Aspect", "QuestionType", "GrammarFeatures", "ControllableASLBuilder",
    "LICENSED", "MinimalPairResult", "minimal_pair", "changed_sir_fields",
    "build_discourse", "locus_of_referent", "realise_with_fingerspelling",
    "NM_NEG", "NM_WH", "NM_YN", "NM_TOPIC", "NM_COND", "NM_ROLESHIFT",
    "plan_to_sir", "plan_manual_units",
]
