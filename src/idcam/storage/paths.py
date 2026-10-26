"""Writable runtime directory helpers."""

from __future__ import annotations

import os
from pathlib import Path

from idcam.exceptions import ConfigurationError

INSTALL_HINT = "On Raspberry Pi, install with: sudo ./scripts/install.sh"


def ensure_writable_directory(path: Path, description: str) -> Path:
    """Create ``path`` if needed and require that the current user can write it.

    Production directories are created by the installer and systemd, not by
    granting world-writable permissions. This helper fails with a clear
    ConfigurationError instead of a raw PermissionError.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise ConfigurationError(
            f"Cannot create {description} directory {path}: permission denied. {INSTALL_HINT}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot create {description} directory {path}: {exc}. {INSTALL_HINT}"
        ) from exc

    if not os.access(path, os.W_OK | os.X_OK):
        raise ConfigurationError(
            f"{description.capitalize()} directory {path} is not writable. {INSTALL_HINT}"
        )
    return path
