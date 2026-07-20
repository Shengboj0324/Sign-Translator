"""Adversarial tests for the caveat-bound metric stack (Doc-12, stage 12b)."""

import pytest

from signtranslator.eval_framework.metric_stack import (
    Layer, REQUIRED_CAVEATS, MetricResult, metric, to_contract, MetricStackReport,
)
from signtranslator.eval_framework.contracts import Direction


def test_all_seven_layers_have_caveats():
    assert set(REQUIRED_CAVEATS) == set(Layer)
    assert all(v.strip() for v in REQUIRED_CAVEATS.values())


def test_metric_factory_attaches_required_caveat():
    m = metric(Layer.SPEECH, "wer", 0.1)
    assert m.caveat == REQUIRED_CAVEATS[Layer.SPEECH]


def test_wrong_caveat_is_rejected():
    with pytest.raises(ValueError):
        MetricResult(Layer.SPEECH, "wer", 0.1, "appearance quality is not comprehension")
    with pytest.raises(ValueError):
        MetricResult(Layer.MANUAL, "geo", 0.05, "")     # empty caveat


def test_caveat_flows_into_contract():
    m = metric(Layer.RENDERING, "flicker", 0.01)
    c = to_contract(m, threshold=0.05, direction=Direction.LE)
    assert c.caveat == REQUIRED_CAVEATS[Layer.RENDERING]
    assert c.layer == "rendering" and c.passed


def test_report_groups_by_layer_and_lists_caveats():
    report = MetricStackReport([
        metric(Layer.SPEECH, "wer", 0.1),
        metric(Layer.SPEECH, "cer", 0.05),
        metric(Layer.MANUAL, "geo", 0.03),
    ])
    grouped = report.by_layer()
    assert len(grouped[Layer.SPEECH]) == 2 and len(grouped[Layer.MANUAL]) == 1
    assert report.caveats()[Layer.SPEECH] == REQUIRED_CAVEATS[Layer.SPEECH]
    assert not report.covers_all_layers()


def test_full_stack_covers_all_layers():
    report = MetricStackReport([metric(l, "m", 0.5) for l in Layer])
    assert report.covers_all_layers()
