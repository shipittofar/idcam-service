"""User and admin status publishing."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from idcam.app.state_machine import AppState
from idcam.storage.paths import ensure_writable_directory
from idcam.ui.messages import FAILURE_MESSAGES, USER_MESSAGES


@dataclass
class StatusSnapshot:
    state: str
    message: str
    capture_id: str | None = None
    detail_message: str | None = None
    camera: dict[str, Any] = field(default_factory=dict)
    network: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    application: dict[str, Any] = field(default_factory=dict)
    last_success_at: float | None = None
    last_error: str | None = None
    version: str = "1.0.0"
    uptime_seconds: float = 0.0
    updated_at: float = field(default_factory=time.time)

    # Legacy flat fields for backward compatibility
    @property
    def camera_healthy(self) -> bool:
        return bool(self.camera.get("healthy", False))

    @property
    def network_healthy(self) -> bool:
        return bool(self.network.get("healthy", False))

    @property
    def server_healthy(self) -> bool:
        return bool(self.server.get("healthy", False))

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "message": self.message,
            "capture_id": self.capture_id,
            "detail_message": self.detail_message,
            "camera": self.camera,
            "network": self.network,
            "server": self.server,
            "storage": self.storage,
            "application": self.application,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "updated_at": self.updated_at,
            # Legacy compatibility
            "camera_healthy": self.camera_healthy,
            "network_healthy": self.network_healthy,
            "server_healthy": self.server_healthy,
        }


class StatusPublisher:
    """Publishes status to a JSON file for UI consumption."""

    def __init__(self, status_file: Path, version: str = "1.0.0") -> None:
        self.status_file = status_file
        self.version = version
        self._snapshot = StatusSnapshot(state=AppState.BOOT.value, message=USER_MESSAGES[AppState.BOOT])
        ensure_writable_directory(self.status_file.parent, "runtime status")

    @property
    def snapshot(self) -> StatusSnapshot:
        return self._snapshot

    def update(
        self,
        state: AppState,
        *,
        capture_id: str | None = None,
        detail: str | None = None,
        camera: dict[str, Any] | None = None,
        network: dict[str, Any] | None = None,
        server: dict[str, Any] | None = None,
        storage: dict[str, Any] | None = None,
        application: dict[str, Any] | None = None,
        last_success_at: float | None = None,
        last_error: str | None = None,
        failure_key: str | None = None,
        uptime_seconds: float | None = None,
    ) -> None:
        message = USER_MESSAGES.get(state, state.value)
        if failure_key and failure_key in FAILURE_MESSAGES:
            detail = FAILURE_MESSAGES[failure_key]

        snap = StatusSnapshot(
            state=state.value,
            message=message,
            capture_id=capture_id,
            detail_message=detail,
            camera=camera if camera is not None else self._snapshot.camera,
            network=network if network is not None else self._snapshot.network,
            server=server if server is not None else self._snapshot.server,
            storage=storage if storage is not None else self._snapshot.storage,
            application=application if application is not None else self._snapshot.application,
            last_success_at=last_success_at if last_success_at is not None else self._snapshot.last_success_at,
            last_error=last_error,
            version=self.version,
            uptime_seconds=uptime_seconds if uptime_seconds is not None else self._snapshot.uptime_seconds,
        )
        self._snapshot = snap
        self._write()

    def _write(self) -> None:
        tmp = self.status_file.with_suffix(".tmp")
        payload = json.dumps(self._snapshot.to_dict(), indent=2)
        tmp.write_text(payload, encoding="utf-8")
        try:
            with tmp.open("rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass
        tmp.replace(self.status_file)

    def get_admin_info(self) -> dict:
        return self._snapshot.to_dict()
