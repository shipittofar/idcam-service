"""Runtime directory helpers must work for a non-root service user."""

from __future__ import annotations

from pathlib import Path

import pytest

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.app.state_machine import AppState
from idcam.config.schema import AppConfig, LoggingConfig, ServerConfig, UIConfig
from idcam.exceptions import ConfigurationError
from idcam.logging.setup import setup_logging
from idcam.storage.paths import ensure_writable_directory
from idcam.storage.temp_files import TempFileStore
from idcam.ui.status import StatusPublisher


def test_ensure_writable_directory_non_root(tmp_path):
    target = tmp_path / "var" / "lib" / "idcam" / "tmp"
    ensure_writable_directory(target, "temp")
    probe = target / "probe.txt"
    probe.write_text("ok", encoding="utf-8")
    assert probe.read_text(encoding="utf-8") == "ok"


def test_ensure_writable_directory_permission_error(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked"
    original = Path.mkdir

    def boom(self, *args, **kwargs):
        if self == blocked:
            raise PermissionError("denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", boom)
    with pytest.raises(ConfigurationError, match="permission denied"):
        ensure_writable_directory(blocked, "log")


def test_setup_logging_uses_configured_writable_dir(tmp_path):
    log_dir = tmp_path / "var" / "log" / "idcam"
    setup_logging(LoggingConfig(log_dir=log_dir, json_format=False))
    assert (log_dir / "idcam.log").exists()
    (log_dir / "idcam.log").write_text("", encoding="utf-8")  # still writable


def test_status_publisher_without_root(tmp_path):
    status = tmp_path / "run" / "idcam" / "status.json"
    pub = StatusPublisher(status)
    assert status.parent.is_dir()
    pub.update(AppState.IDLE)
    assert status.exists()
    assert "IDLE" in status.read_text(encoding="utf-8")


def test_temp_file_store_without_root(tmp_path):
    store = TempFileStore(tmp_path / "var" / "lib" / "idcam" / "tmp")
    path = store.create_temp_path()
    path.write_bytes(b"data")
    assert path.exists()


def test_orchestrator_instantiates_without_root(tmp_path):
    config = AppConfig(
        device_id="svc-user",
        temp_dir=tmp_path / "var" / "lib" / "idcam" / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        logging=LoggingConfig(log_dir=tmp_path / "var" / "log" / "idcam"),
        ui=UIConfig(status_file=tmp_path / "run" / "idcam" / "status.json"),
    )
    orch = ApplicationOrchestrator(config, use_mock_camera=True)
    assert orch.temp_store.base_dir.exists()
    assert config.ui.status_file.parent.exists()
    (orch.temp_store.base_dir / "w.txt").write_text("ok", encoding="utf-8")
