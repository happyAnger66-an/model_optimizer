"""Policy adapter registry."""

from __future__ import annotations

from .base import PolicyAdapter

_POLICY_ADAPTERS: dict[str, PolicyAdapter] = {}


def register_policy_adapter(adapter: PolicyAdapter, *, override: bool = True) -> PolicyAdapter:
    """Register a policy adapter for an architecture."""

    if adapter.architecture in _POLICY_ADAPTERS and not override:
        return _POLICY_ADAPTERS[adapter.architecture]
    _POLICY_ADAPTERS[adapter.architecture] = adapter
    return adapter


def get_policy_adapter(architecture: str) -> PolicyAdapter:
    """Return a policy adapter for an architecture."""

    try:
        return _POLICY_ADAPTERS[architecture]
    except KeyError as exc:
        raise ValueError(
            f"Policy adapter for architecture {architecture!r} not found; "
            f"available={sorted(_POLICY_ADAPTERS)}"
        ) from exc


def list_policy_adapters() -> tuple[str, ...]:
    """Return registered policy adapter architecture names."""

    return tuple(sorted(_POLICY_ADAPTERS))
