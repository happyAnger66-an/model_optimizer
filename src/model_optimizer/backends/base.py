"""Backend installation interface.

Backend installers mutate an already-created policy by mounting runtime
implementations such as TensorRT, ONNX Runtime, or native CUDA graph paths.
They are intentionally separate from policy adapters: adapters create policies,
installers attach execution backends to those policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[[str, str], None]


class BackendInstaller(ABC):
    """Install a backend for a model architecture."""

    architecture: str
    backend: str

    @abstractmethod
    def install(self, policy: Any, config: Any, on_progress: ProgressCallback) -> None:
        """Mount this backend onto ``policy`` using ``config``."""
