# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""模型特性注册表（可扩展的"结构改写"开关）。

每个特性是一次对模型（或其子模块）的 **就地结构改写**，例如：

- ``fmha_d256_attention``：把各层 ``self_attn`` 包成 CuTe DSL FMHA D=256 插件路径。
- ``fused_mlp``：把各层 ``GemmaMLP`` 的 gate/up 合并为单次 GEMM。

注册一个新特性只需：

    register_feature(
        "my_feature",
        default_enabled=False,
        apply_fn=lambda target, params, ctx: ...,
        supported_models=("pi05_libero/llm_with_cutedsl",),
        description="...",
    )

随后 JSON 里写 ``{"features": {"my_feature": {"enabled": true, "params": {...}}}}``
即可启用，**无需改 CLI 或配置 schema**。

是否启用由 :meth:`FeatureConfig.is_enabled` 按 ``JSON > env > default`` 决定。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from model_optimizer.config.feature_config import FeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureContext:
    """传给特性 ``apply_fn`` 的上下文（按需扩展字段）。

    Attributes:
        model_name: 注册名（如 ``pi05_libero/llm_with_cutedsl``），用于 ``supported_models`` 过滤。
        dtype:      构造期目标 dtype。
        extra:      模型自定义透传数据。
    """

    model_name: str
    dtype: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


# apply_fn 签名：(target_module, params, ctx) -> None
ApplyFn = Callable[[Any, dict[str, Any], FeatureContext], Any]


@dataclass
class Feature:
    name: str
    default_enabled: bool
    apply_fn: ApplyFn
    description: str = ""
    supported_models: tuple[str, ...] | None = None  # None = 适用所有模型
    conflicts: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def supports(self, model_name: str) -> bool:
        return self.supported_models is None or model_name in self.supported_models


FEATURE_REGISTRY: dict[str, Feature] = {}


def register_feature(
    name: str,
    *,
    default_enabled: bool,
    apply_fn: ApplyFn,
    description: str = "",
    supported_models: tuple[str, ...] | None = None,
    conflicts: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    override: bool = True,
) -> Feature:
    """注册（或覆盖）一个特性。模块多次 import 时默认覆盖，保证幂等。"""
    if name in FEATURE_REGISTRY and not override:
        return FEATURE_REGISTRY[name]
    feat = Feature(
        name=name,
        default_enabled=default_enabled,
        apply_fn=apply_fn,
        description=description,
        supported_models=supported_models,
        conflicts=tuple(conflicts),
        requires=tuple(requires),
    )
    FEATURE_REGISTRY[name] = feat
    return feat


def get_feature(name: str) -> Feature | None:
    return FEATURE_REGISTRY.get(name)


def validate_feature_config(feature_config: FeatureConfig | None, *, model_name: str) -> None:
    """Validate explicit feature settings before mutating model modules."""

    fc = feature_config or FeatureConfig.empty()
    unknown = [name for name in fc.features if name not in FEATURE_REGISTRY]
    if unknown:
        raise ValueError(
            f"feature_config references unknown feature(s): {unknown}; "
            f"known={sorted(FEATURE_REGISTRY)}"
        )

    unsupported = []
    for name, spec in fc.features.items():
        if spec.enabled is not True:
            continue
        feat = FEATURE_REGISTRY[name]
        if not feat.supports(model_name):
            unsupported.append(name)
    if unsupported:
        raise ValueError(
            f"feature(s) {unsupported} are not supported by model {model_name!r}"
        )

    enabled = {
        name
        for name, feat in FEATURE_REGISTRY.items()
        if feat.supports(model_name) and fc.is_enabled(name, feat.default_enabled)
    }
    conflicts: list[tuple[str, str]] = []
    missing_requires: list[tuple[str, str]] = []
    for name in sorted(enabled):
        feat = FEATURE_REGISTRY[name]
        for conflict in feat.conflicts:
            if conflict in enabled:
                conflicts.append((name, conflict))
        for requirement in feat.requires:
            if requirement not in enabled:
                missing_requires.append((name, requirement))
    if conflicts:
        raise ValueError(f"feature conflict(s) for model {model_name!r}: {conflicts}")
    if missing_requires:
        raise ValueError(
            f"feature requirement(s) not met for model {model_name!r}: {missing_requires}"
        )


def apply_features(
    target: Any,
    feature_config: FeatureConfig | None,
    ctx: FeatureContext,
) -> list[str]:
    """遍历注册表，对 ``ctx.model_name`` 支持且被启用的特性逐个应用到 ``target``。

    Returns:
        实际应用的特性名列表（按注册顺序）。
    """
    fc = feature_config or FeatureConfig.empty()
    validate_feature_config(fc, model_name=ctx.model_name)

    applied: list[str] = []
    for name, feat in FEATURE_REGISTRY.items():
        if not feat.supports(ctx.model_name):
            # JSON 显式为不支持的模型开启了该特性 → 告警，便于排错。
            if fc.is_enabled(name, False) and name in fc.features:
                logger.warning(
                    "feature %r not supported by model %r; skipped.",
                    name, ctx.model_name,
                )
            continue
        if not fc.is_enabled(name, feat.default_enabled):
            logger.info("feature %r disabled, skipping.", name)
            continue
        params = fc.params(name)
        logger.info("apply feature %r params=%s", name, params)
        feat.apply_fn(target, params, ctx)
        applied.append(name)
    return applied


__all__ = [
    "Feature",
    "FeatureContext",
    "FEATURE_REGISTRY",
    "register_feature",
    "get_feature",
    "validate_feature_config",
    "apply_features",
]
