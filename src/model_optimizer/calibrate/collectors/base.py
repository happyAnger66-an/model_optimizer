"""Calibration collector interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CalibCollector:
    """Load calibration data for one architecture stage."""

    architecture: str
    stage: str
    description: str = ""

    def load(self, calib_data: str) -> Any:
        raise NotImplementedError
