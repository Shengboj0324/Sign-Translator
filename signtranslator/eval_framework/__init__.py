"""Evaluation framework (Document 12).

Implements `12_evaluation_framework.md` (see docs/EVALUATION_FRAMEWORK.md): a chain of
falsifiable contracts across seven caveat-bound metric layers, a rigorous statistical
protocol (pre-registration + test firewall, paired/permutation/bootstrap statistics),
reproducible text metrics (SacreBLEU + BERTScore), blinded comprehension scoring, and
baselines/stratification/model card. Aggregates the audited leaf metrics from Docs
01-11; the governing principle is that a single metric cannot establish adequacy.
"""

from .contracts import Direction, Contract, EvaluationChain
from .metric_stack import (
    Layer, REQUIRED_CAVEATS, MetricResult, metric, to_contract, MetricStackReport,
)
from .statistics import (
    paired_differences, paired_t_statistic, paired_permutation_pvalue,
    sign_test_pvalue, bootstrap_ci, SeedSummary, aggregate_seeds,
    significant_and_meaningful,
)
from .protocol import (
    ProtocolError, PreRegistration, EvaluationFirewall, signer_held_out_split,
)
from .text_metrics import (
    tokenize, corpus_bleu, BLEUSignature, idf_weights, bert_score,
)
from .comprehension import (
    proposition_prf, comprehension_f1, mean_comprehension_f1,
    proposition_agreement, SystemScores, preference_comprehension_dissociate,
    BlindedTrial,
)
from .reporting import (
    BaselineType, REQUIRED_BASELINES, has_required_baselines,
    exceeds_human_upper_reference, worst_slice, aggregate_hides_worst_slice,
    MODEL_CARD_SECTIONS, ModelCard,
)

__all__ = [
    "Direction", "Contract", "EvaluationChain",
    "Layer", "REQUIRED_CAVEATS", "MetricResult", "metric", "to_contract",
    "MetricStackReport",
    "paired_differences", "paired_t_statistic", "paired_permutation_pvalue",
    "sign_test_pvalue", "bootstrap_ci", "SeedSummary", "aggregate_seeds",
    "significant_and_meaningful",
    "ProtocolError", "PreRegistration", "EvaluationFirewall", "signer_held_out_split",
    "tokenize", "corpus_bleu", "BLEUSignature", "idf_weights", "bert_score",
    "proposition_prf", "comprehension_f1", "mean_comprehension_f1",
    "proposition_agreement", "SystemScores", "preference_comprehension_dissociate",
    "BlindedTrial",
    "BaselineType", "REQUIRED_BASELINES", "has_required_baselines",
    "exceeds_human_upper_reference", "worst_slice", "aggregate_hides_worst_slice",
    "MODEL_CARD_SECTIONS", "ModelCard",
]
