"""Architecture specification registry."""

from __future__ import annotations

from .base import ArchitectureSpec

_ARCHITECTURE_SPECS: dict[str, ArchitectureSpec] = {}


def register_architecture_spec(spec: ArchitectureSpec, *, override: bool = True) -> ArchitectureSpec:
    """Register an architecture spec by name."""

    if spec.name in _ARCHITECTURE_SPECS and not override:
        return _ARCHITECTURE_SPECS[spec.name]
    _ARCHITECTURE_SPECS[spec.name] = spec
    return spec


def get_architecture_spec(name: str) -> ArchitectureSpec:
    """Return an architecture spec by name."""

    try:
        return _ARCHITECTURE_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"Architecture {name!r} not found; available={sorted(_ARCHITECTURE_SPECS)}"
        ) from exc


def list_architecture_specs() -> tuple[str, ...]:
    """Return registered architecture names."""

    return tuple(sorted(_ARCHITECTURE_SPECS))
