"""Feature validation E2E regression tests."""

from __future__ import annotations

import json
import sys
import types

import pytest


@pytest.mark.e2e
def test_validate_feature_config_rejects_unknown_feature():
    from model_optimizer.config.feature_config import FeatureConfig
    from model_optimizer.models.features import validate_feature_config

    cfg = FeatureConfig.from_dict(
        {
            "features": {
                "typo_feature_name": {"enabled": True},
            },
        }
    )

    with pytest.raises(ValueError, match="unknown feature"):
        validate_feature_config(cfg, model_name="pi05_libero/llm")


@pytest.mark.e2e
def test_validate_feature_config_rejects_enabled_unsupported_feature():
    from model_optimizer.config.feature_config import FeatureConfig
    from model_optimizer.models.features import register_feature, validate_feature_config

    register_feature(
        "llm_only_feature_e2e",
        default_enabled=False,
        apply_fn=lambda target, params, ctx: None,
        supported_models=("pi05_libero/llm",),
    )
    cfg = FeatureConfig.from_dict(
        {
            "features": {
                "llm_only_feature_e2e": {"enabled": True},
            },
        }
    )

    with pytest.raises(ValueError, match="not supported"):
        validate_feature_config(cfg, model_name="pi05_libero/vit")

    disabled = FeatureConfig.from_dict(
        {
            "features": {
                "llm_only_feature_e2e": {"enabled": False},
            },
        }
    )
    validate_feature_config(disabled, model_name="pi05_libero/vit")


@pytest.mark.e2e
def test_export_cli_validates_feature_config_before_model_construction(tmp_path, monkeypatch):
    from model_optimizer.convert.convert_formt import convert_model

    feature_path = tmp_path / "bad_feature.json"
    feature_path.write_text(json.dumps({"features": {"missing_feature": True}}))
    constructed = []

    class FakeModel:
        @classmethod
        def construct_from_name_path(cls, *args, **kwargs):
            constructed.append((args, kwargs))
            return cls()

    fake_registry = types.ModuleType("model_optimizer.models.registry")
    fake_registry.get_model_cls = lambda _name: FakeModel
    monkeypatch.setitem(sys.modules, "model_optimizer.models.registry", fake_registry)

    with pytest.raises(ValueError, match="unknown feature"):
        convert_model(
            [
                "workflow",
                "--model_name",
                "pi05_libero/llm",
                "--model_path",
                "/models/pi05",
                "--export_dir",
                str(tmp_path / "export"),
                "--feature_config",
                str(feature_path),
            ]
        )

    assert constructed == []
