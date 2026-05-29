# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""模型特性开关配置（JSON 驱动）。

CLI ``export`` / ``quantize`` 可通过 ``--feature_config x.json`` 动态启停
``fmha_d256_attention`` / ``fused_mlp`` 等模型级特性，并为每个特性携带私有参数。

JSON 格式::

    {
      "version": 1,
      "features": {
        "fmha_d256_attention": { "enabled": true,  "params": { "use_fp16": true } },
        "fused_mlp":           { "enabled": false },
        "kv_cache_fp8":        true
      },
      "export":   { "dynamo": false },
      "quantize": { "dynamic_quant": "fp16" }
    }

- ``features.<name>`` 可为对象 ``{"enabled": bool, "params": {...}}`` 或布尔简写。
- 省略某特性 → 用注册表默认值（见 :mod:`model_optimizer.models.features`）。

特性是否启用的优先级（高 → 低）::

    JSON 显式 enabled  >  环境变量 MODEL_OPT_FEATURE_<NAME>  >  注册表 default_enabled

不传 ``--feature_config`` 时返回空配置，所有特性走默认，行为与历史一致。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_ENV_PREFIX = "MODEL_OPT_FEATURE_"
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def _env_override(name: str) -> bool | None:
    """读取 ``MODEL_OPT_FEATURE_<NAME>`` 环境变量；未设置返回 ``None``。"""
    raw = os.environ.get(_ENV_PREFIX + name.upper())
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    logger.warning(
        "feature env %s%s=%r not recognized, treating as enabled",
        _ENV_PREFIX, name.upper(), raw,
    )
    return True


@dataclass
class FeatureSpec:
    """单个特性的配置项。

    Attributes:
        enabled: ``None`` 表示 JSON 未显式指定（交由 env / 默认决定）。
        params:  透传给特性 ``apply_fn`` 的私有参数字典。
    """

    enabled: bool | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "FeatureSpec":
        if isinstance(raw, bool):
            return cls(enabled=raw, params={})
        if isinstance(raw, dict):
            enabled = raw.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                raise ValueError(f"feature 'enabled' must be bool, got {enabled!r}")
            params = raw.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"feature 'params' must be an object, got {params!r}")
            return cls(enabled=enabled, params=dict(params))
        raise ValueError(
            f"feature spec must be bool or object, got {type(raw).__name__}: {raw!r}"
        )


@dataclass
class FeatureConfig:
    """解析后的特性配置。

    通常由 :meth:`load` 从 JSON 文件构建；不传文件时用 :meth:`empty`。
    """

    features: dict[str, FeatureSpec] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)
    quantize: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    # ── 构造 ────────────────────────────────────────────────────────────────
    @classmethod
    def empty(cls) -> "FeatureConfig":
        return cls()

    @classmethod
    def load(cls, path: str | None) -> "FeatureConfig":
        """从 JSON 文件加载；``path`` 为 ``None``/空 时返回空配置。"""
        if not path:
            return cls.empty()
        if not os.path.exists(path):
            raise FileNotFoundError(f"feature_config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureConfig":
        if not isinstance(data, dict):
            raise ValueError(f"feature_config root must be an object, got {type(data).__name__}")
        raw_features = data.get("features", {})
        if not isinstance(raw_features, dict):
            raise ValueError("feature_config 'features' must be an object")
        features = {name: FeatureSpec.from_raw(spec) for name, spec in raw_features.items()}
        export = data.get("export", {}) or {}
        quantize = data.get("quantize", {}) or {}
        if not isinstance(export, dict) or not isinstance(quantize, dict):
            raise ValueError("feature_config 'export'/'quantize' must be objects")
        return cls(features=features, export=dict(export), quantize=dict(quantize), raw=data)

    # ── 查询 ────────────────────────────────────────────────────────────────
    def is_enabled(self, name: str, default: bool) -> bool:
        """按优先级 JSON > env > default 判定特性是否启用。"""
        spec = self.features.get(name)
        if spec is not None and spec.enabled is not None:
            return spec.enabled
        env = _env_override(name)
        if env is not None:
            return env
        return default

    def params(self, name: str) -> dict[str, Any]:
        spec = self.features.get(name)
        return dict(spec.params) if spec is not None else {}

    def __repr__(self) -> str:  # pragma: no cover - 仅日志
        items = {n: (s.enabled, s.params) for n, s in self.features.items()}
        return f"FeatureConfig(features={items}, export={self.export}, quantize={self.quantize})"


__all__ = ["FeatureConfig", "FeatureSpec"]
