"""Caveat-bound metric stack (Doc-12 §2).

The seven metric layers, each with the mandatory caveat from the document's table.
A MetricResult must carry EXACTLY its layer's required caveat; constructing one
without it (or with the wrong caveat) raises. So a metric can never be reported
stripped of its caveat — "transcript accuracy = sign adequacy" is unstatable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from .contracts import Contract, Direction


class Layer(Enum):
    SPEECH = "speech"
    PLAN = "plan"
    MANUAL = "manual"
    NONMANUAL = "non-manual"
    DISTRIBUTION = "distribution"
    RENDERING = "rendering"
    HUMAN = "human"


#: the required caveat for each layer (verbatim intent of the document's table).
REQUIRED_CAVEATS: Dict[Layer, str] = {
    Layer.SPEECH: "transcript accuracy does not equal sign adequacy",
    Layer.PLAN: "gloss agreement is annotation-dependent",
    Layer.MANUAL: "one valid production may differ from reference",
    Layer.NONMANUAL: "landmark accuracy is not grammatical accuracy",
    Layer.DISTRIBUTION: "embedding choice can bias results",
    Layer.RENDERING: "appearance quality is not comprehension",
    Layer.HUMAN: "use fluent target-language signers",
}


@dataclass(frozen=True)
class MetricResult:
    layer: Layer
    name: str
    value: float
    caveat: str

    def __post_init__(self):
        required = REQUIRED_CAVEATS[self.layer]
        if self.caveat != required:
            raise ValueError(
                f"{self.layer.value}/{self.name} must carry its required caveat "
                f"{required!r}, got {self.caveat!r}")


def metric(layer: Layer, name: str, value: float) -> MetricResult:
    """Build a metric result with its layer's required caveat auto-attached."""
    return MetricResult(layer, name, float(value), REQUIRED_CAVEATS[layer])


def to_contract(result: MetricResult, threshold: float,
                direction: Direction) -> Contract:
    """Bridge a caveat-bound metric to a falsifiable contract (caveat flows in)."""
    return Contract(result.name, result.layer.value, result.value, threshold,
                    direction, result.caveat)


@dataclass
class MetricStackReport:
    results: List[MetricResult]

    def by_layer(self) -> Dict[Layer, List[MetricResult]]:
        out: Dict[Layer, List[MetricResult]] = {}
        for r in self.results:
            out.setdefault(r.layer, []).append(r)
        return out

    def caveats(self) -> Dict[Layer, str]:
        """Every reported layer maps to its (mandatory, present) caveat."""
        return {layer: REQUIRED_CAVEATS[layer] for layer in self.by_layer()}

    def covers_all_layers(self) -> bool:
        return set(self.by_layer()) == set(Layer)
