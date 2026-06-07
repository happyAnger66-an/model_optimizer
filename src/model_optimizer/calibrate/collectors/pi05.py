"""Pi05 calibration collectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import CalibCollector
from .registry import register_calib_collector

_PI05_COMPONENTS = {
    "vit": "pi05_vit",
    "embed_prefix": "pi05_embed_prefix",
    "llm": "pi05_llm",
    "expert": "pi05_expert",
    "denoise": "pi05_denoise",
}


@dataclass(frozen=True)
class Pi05CalibCollector(CalibCollector):
    component: str = ""

    def load(self, calib_data: str) -> Any:
        from model_optimizer.calibrate.pi05_calib_load import open_pi05_calib_for_quantize

        return open_pi05_calib_for_quantize(calib_data, component=self.component)


def register() -> tuple[Pi05CalibCollector, ...]:
    collectors = tuple(
        Pi05CalibCollector(
            architecture="pi05",
            stage=stage,
            component=component,
            description=f"Pi05 {stage} calibration data loader.",
        )
        for stage, component in _PI05_COMPONENTS.items()
    )
    for collector in collectors:
        register_calib_collector(collector)
    return collectors


register()
