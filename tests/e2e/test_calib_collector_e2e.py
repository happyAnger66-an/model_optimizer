"""Calibration collector E2E regression tests."""

from __future__ import annotations

import pytest


@pytest.mark.e2e
def test_pi05_calib_collectors_registered():
    from model_optimizer.calibrate.collectors import get_calib_collector, list_calib_collectors

    expected = {
        ("pi05", "vit"),
        ("pi05", "embed_prefix"),
        ("pi05", "llm"),
        ("pi05", "expert"),
        ("pi05", "denoise"),
    }

    assert expected.issubset(set(list_calib_collectors()))
    assert get_calib_collector("pi05", "llm").stage == "llm"


@pytest.mark.e2e
def test_pi05_calib_collector_maps_stage_to_component(monkeypatch):
    import model_optimizer.calibrate.pi05_calib_load as pi05_calib_load
    from model_optimizer.calibrate.collectors import get_calib_collector

    calls = []

    def fake_open(calib_data, *, component):
        calls.append((calib_data, component))
        return [{"component": component}]

    monkeypatch.setattr(pi05_calib_load, "open_pi05_calib_for_quantize", fake_open)

    assert get_calib_collector("pi05", "vit").load("/calib") == [{"component": "pi05_vit"}]
    assert get_calib_collector("pi05", "embed_prefix").load("/calib") == [
        {"component": "pi05_embed_prefix"}
    ]
    assert get_calib_collector("pi05", "llm").load("/calib") == [{"component": "pi05_llm"}]
    assert get_calib_collector("pi05", "expert").load("/calib") == [
        {"component": "pi05_expert"}
    ]
    assert get_calib_collector("pi05", "denoise").load("/calib") == [
        {"component": "pi05_denoise"}
    ]

    assert calls == [
        ("/calib", "pi05_vit"),
        ("/calib", "pi05_embed_prefix"),
        ("/calib", "pi05_llm"),
        ("/calib", "pi05_expert"),
        ("/calib", "pi05_denoise"),
    ]


@pytest.mark.e2e
def test_calib_collector_registry_supports_non_pi05_architecture():
    from model_optimizer.calibrate.collectors import (
        CalibCollector,
        get_calib_collector,
        register_calib_collector,
    )

    class FakeWorldModelCollector(CalibCollector):
        def load(self, calib_data: str):
            return {"path": calib_data, "stage": self.stage}

    register_calib_collector(
        FakeWorldModelCollector(
            architecture="world_model_e2e",
            stage="encoder",
        )
    )

    collector = get_calib_collector("world_model_e2e", "encoder")
    assert collector.load("/tmp/calib") == {
        "path": "/tmp/calib",
        "stage": "encoder",
    }
