"""Configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from idcam.config.schema import EXAMPLE_SERVER_HOSTS, AppConfig, CameraConfig
from idcam.exceptions import ConfigurationError


DEFAULT_CONFIG_PATHS = (
    Path("/etc/idcam/config.yaml"),
    Path("config.yaml"),
)


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate configuration from YAML file."""
    config_path = path or _resolve_config_path()
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except PermissionError as exc:
        raise ConfigurationError(
            f"Permission denied reading {config_path}. "
            "Run diagnostics as the idcam user (sudo -u idcam ...) or set IDCAM_CONFIG "
            "to a readable config path."
        ) from exc
    except OSError as exc:
        raise ConfigurationError(f"Cannot read {config_path}: {exc}") from exc

    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration must be a mapping, got {type(raw)}")

    _apply_env_overrides(raw)

    try:
        config = AppConfig.model_validate(raw)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Configuration validation failed: {exc}") from exc

    _enforce_production_endpoint(config)
    return config


def load_camera_config_for_diagnostic(
    path: Path | None = None,
) -> tuple[CameraConfig, dict[str, str]]:
    """
  Load camera settings for hardware diagnostics.

  Returns (camera_config, warnings). On failure uses CameraConfig defaults and
  records a clear warning instead of silently ignoring permission errors.
    """
    warnings: dict[str, str] = {}
    try:
        return load_config(path).camera, warnings
    except ConfigurationError as exc:
        warnings["config_error"] = str(exc)
        warnings["config_hint"] = (
            "Stop idcam service, run as: sudo -u idcam /opt/idcam/venv/bin/python scripts/... "
            "or export IDCAM_CONFIG=/path/to/readable/config.yaml"
        )
        camera = CameraConfig()
        _apply_camera_env_overrides(camera)
        return camera, warnings


def _apply_camera_env_overrides(camera: CameraConfig) -> None:
    """Allow diagnostics without YAML when env vars are set."""
    pw = os.environ.get("IDCAM_PREVIEW_WIDTH")
    ph = os.environ.get("IDCAM_PREVIEW_HEIGHT")
    cw = os.environ.get("IDCAM_CAPTURE_WIDTH")
    ch = os.environ.get("IDCAM_CAPTURE_HEIGHT")
    if pw:
        camera.preview_width = int(pw)
    if ph:
        camera.preview_height = int(ph)
    if cw:
        camera.capture_width = int(cw)
    if ch:
        camera.capture_height = int(ch)


def _resolve_config_path() -> Path:
    env_path = os.environ.get("IDCAM_CONFIG")
    if env_path:
        return Path(env_path)
    for candidate in DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return candidate
    return DEFAULT_CONFIG_PATHS[0]


def _apply_env_overrides(raw: dict) -> None:
    token = os.environ.get("IDCAM_AUTH_TOKEN")
    if token:
        auth = raw.setdefault("auth", {})
        auth["type"] = "bearer"
        auth["token"] = token

    device_id = os.environ.get("IDCAM_DEVICE_ID")
    if device_id:
        raw["device_id"] = device_id

    server_url = os.environ.get("IDCAM_SERVER_URL")
    if server_url:
        server = raw.setdefault("server", {})
        server["url"] = server_url


def _production_mode() -> bool:
    return os.environ.get("IDCAM_ENV", "").strip().lower() in {"production", "prod"}


def _enforce_production_endpoint(config: AppConfig) -> None:
    """Refuse placeholder example hosts when running as the production service."""
    if not _production_mode():
        return
    host = (config.server.url.host or "").lower().rstrip(".")
    if host in EXAMPLE_SERVER_HOSTS or host.endswith(".example.com"):
        raise ConfigurationError(
            f"Refusing to start: server.url ({config.server.url}) is a placeholder. "
            "Set the real API URL in /etc/idcam/config.yaml or IDCAM_SERVER_URL."
        )
