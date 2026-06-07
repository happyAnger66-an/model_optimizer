"""Calibration collector registry."""

from __future__ import annotations

from .base import CalibCollector

_COLLECTORS: dict[tuple[str, str], CalibCollector] = {}


def register_calib_collector(collector: CalibCollector) -> CalibCollector:
    key = (collector.architecture, collector.stage)
    _COLLECTORS[key] = collector
    return collector


def get_calib_collector(architecture: str, stage: str) -> CalibCollector:
    key = (architecture, stage)
    try:
        return _COLLECTORS[key]
    except KeyError as exc:
        available = ", ".join(f"{arch}/{name}" for arch, name in sorted(_COLLECTORS))
        raise ValueError(
            f"Calib collector {architecture}/{stage} not found. Available: {available}"
        ) from exc


def list_calib_collectors() -> tuple[tuple[str, str], ...]:
    return tuple(sorted(_COLLECTORS))
