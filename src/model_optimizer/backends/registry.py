"""Backend installer registry."""

from __future__ import annotations

from .base import BackendInstaller

_BACKEND_INSTALLERS: dict[tuple[str, str], BackendInstaller] = {}


def register_backend_installer(
    installer: BackendInstaller,
    *,
    override: bool = True,
) -> BackendInstaller:
    """Register a backend installer by ``(architecture, backend)``."""

    key = (installer.architecture, installer.backend)
    if key in _BACKEND_INSTALLERS and not override:
        return _BACKEND_INSTALLERS[key]
    _BACKEND_INSTALLERS[key] = installer
    return installer


def get_backend_installer(architecture: str, backend: str) -> BackendInstaller:
    """Return a backend installer for ``architecture`` and ``backend``."""

    key = (architecture, backend)
    try:
        return _BACKEND_INSTALLERS[key]
    except KeyError as exc:
        available = sorted(f"{arch}:{be}" for arch, be in _BACKEND_INSTALLERS)
        raise ValueError(
            f"Backend installer for architecture={architecture!r}, backend={backend!r} "
            f"not found; available={available}"
        ) from exc


def list_backend_installers() -> tuple[tuple[str, str], ...]:
    """Return registered ``(architecture, backend)`` installer keys."""

    return tuple(sorted(_BACKEND_INSTALLERS))
