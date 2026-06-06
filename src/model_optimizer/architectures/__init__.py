"""Model architecture specifications."""

from .base import ArchitectureSpec, StageSpec
from .pi05 import PI05_ARCHITECTURE_NAME, PI05_ARCHITECTURE_SPEC
from .registry import (
    get_architecture_spec,
    list_architecture_specs,
    register_architecture_spec,
)

__all__ = [
    "ArchitectureSpec",
    "StageSpec",
    "PI05_ARCHITECTURE_NAME",
    "PI05_ARCHITECTURE_SPEC",
    "get_architecture_spec",
    "list_architecture_specs",
    "register_architecture_spec",
]
