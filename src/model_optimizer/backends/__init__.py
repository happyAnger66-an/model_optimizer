"""Backend installers."""

from .base import BackendInstaller, ProgressCallback
from .pi05 import (
    Pi05NativeBackendInstaller,
    Pi05OnnxRTBackendInstaller,
    Pi05TensorRTBackendInstaller,
)
from .registry import (
    get_backend_installer,
    list_backend_installers,
    register_backend_installer,
)

__all__ = [
    "BackendInstaller",
    "ProgressCallback",
    "Pi05NativeBackendInstaller",
    "Pi05OnnxRTBackendInstaller",
    "Pi05TensorRTBackendInstaller",
    "get_backend_installer",
    "list_backend_installers",
    "register_backend_installer",
]
