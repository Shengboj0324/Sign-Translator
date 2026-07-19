"""LLM semantic reasoning layer: the constrained sign-plan planner.

Implements `02_llm_semantic_reasoning_layer.md` (see docs/SEMANTIC_PLANNER.md).

The planner consumes evidence (transcript hypotheses, acoustic states, discourse
memory, lexicon retrieval) and emits a *typed sign plan* with explicit
uncertainty -- never free-form gloss text. Stage 2a: the typed schema, its
deterministic serialization, and the structural validator.
"""

from .schema import (
    PlanVocabulary, SignPlan, SemanticFrame, NonmanualSpan,
    serialize_plan, deserialize_plan, validate_plan,
    DEFAULT_VOCAB,
)
from .automaton import SchemaAutomaton, S
from .constrained import (
    ConstrainedDecoder, masked_log_softmax, masked_distribution, allowed_mask,
)
from .lexicon import (
    LexEntry, RetrievalResult, SignLexicon, GroundingReport, ground_plan,
)
from .planner import SemanticPlanner, pad_plan_batch
from .consistency import (
    SemanticFeatures, ControllablePlanBuilder, counterfactual_diff,
    CounterfactualResult, changed_fields, check_semantic_consistency,
    ConsistencyReport, NM_NEG, NM_WH, NM_YN, NM_TOPIC, NM_COND,
)
from .preference import (
    SequencePreferenceDPO, DPOStats, sequence_log_prob,
)
from .factorized import (
    EvidenceEncoder, ContentHead, HeavyDecoder, factorized_train, joint_train,
    representation_probe_accuracy, DominanceReport, run_dominance_experiment,
    TrainingRegimeResult,
)

__all__ = [
    "PlanVocabulary", "SignPlan", "SemanticFrame", "NonmanualSpan",
    "serialize_plan", "deserialize_plan", "validate_plan",
    "DEFAULT_VOCAB",
    "SchemaAutomaton", "S",
    "ConstrainedDecoder", "masked_log_softmax", "masked_distribution",
    "allowed_mask",
    "LexEntry", "RetrievalResult", "SignLexicon", "GroundingReport", "ground_plan",
    "SemanticPlanner", "pad_plan_batch",
    "SemanticFeatures", "ControllablePlanBuilder", "counterfactual_diff",
    "CounterfactualResult", "changed_fields", "check_semantic_consistency",
    "ConsistencyReport", "NM_NEG", "NM_WH", "NM_YN", "NM_TOPIC", "NM_COND",
    "SequencePreferenceDPO", "DPOStats", "sequence_log_prob",
    "EvidenceEncoder", "ContentHead", "HeavyDecoder", "factorized_train",
    "joint_train", "representation_probe_accuracy", "DominanceReport",
    "run_dominance_experiment", "TrainingRegimeResult",
]
