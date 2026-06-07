"""Metric registry."""

from __future__ import annotations

from .metric import Metric

_METRICS: dict[tuple[str, str], type[Metric]] = {}


def register_metric(architecture: str, stage: str, metric_cls: type[Metric]) -> type[Metric]:
    _METRICS[(architecture, stage)] = metric_cls
    return metric_cls


def get_metric_cls(architecture: str, stage: str) -> type[Metric]:
    key = (architecture, stage)
    try:
        return _METRICS[key]
    except KeyError as exc:
        available = ", ".join(f"{arch}/{name}" for arch, name in sorted(_METRICS))
        raise ValueError(f"Metric {architecture}/{stage} not found. Available: {available}") from exc


def create_metric(architecture: str, stage: str, result) -> Metric:
    return get_metric_cls(architecture, stage)(result)


def list_metrics() -> tuple[tuple[str, str], ...]:
    return tuple(sorted(_METRICS))
