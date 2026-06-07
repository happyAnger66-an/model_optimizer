"""Pi05 metric registrations."""

from __future__ import annotations

from .pi05 import Pi05Metric
from .registry import register_metric

_PI05_STAGES = ("vit", "embed_prefix", "llm", "expert", "denoise")


def register() -> None:
    for stage in _PI05_STAGES:
        register_metric("pi05", stage, Pi05Metric)


register()
