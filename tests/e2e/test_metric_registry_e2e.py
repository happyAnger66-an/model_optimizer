"""Metric registry E2E regression tests."""

from __future__ import annotations

import pytest


@pytest.mark.e2e
def test_pi05_metrics_registered_for_all_stages():
    from model_optimizer.evaluate.metrics import create_metric, list_metrics
    from model_optimizer.evaluate.metrics.pi05 import Pi05Metric

    expected = {
        ("pi05", "vit"),
        ("pi05", "embed_prefix"),
        ("pi05", "llm"),
        ("pi05", "expert"),
        ("pi05", "denoise"),
    }

    assert expected.issubset(set(list_metrics()))
    metric = create_metric("pi05", "llm", [{"past_keys": 1}])
    assert isinstance(metric, Pi05Metric)
    assert metric.get_result() == [{"past_keys": 1}]


@pytest.mark.e2e
def test_metric_registry_supports_non_pi05_architecture():
    from model_optimizer.evaluate.metrics import Metric, create_metric, register_metric

    class FakeMetric(Metric):
        def print(self):
            return None

        def compare(self, other):
            return {
                "self": self.get_result(),
                "other": other.get_result(),
            }

    register_metric("world_model_e2e", "decoder", FakeMetric)

    metric = create_metric("world_model_e2e", "decoder", {"loss": 1.0})
    other = create_metric("world_model_e2e", "decoder", {"loss": 2.0})

    assert isinstance(metric, FakeMetric)
    assert metric.compare(other) == {
        "self": {"loss": 1.0},
        "other": {"loss": 2.0},
    }
