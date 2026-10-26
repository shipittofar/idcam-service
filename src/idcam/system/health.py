"""System health monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class HealthStatus:
    camera_healthy: bool = False
    network_healthy: bool = False
    server_healthy: bool = False
    last_success_at: float | None = None
    last_error: str | None = None
    last_error_at: float | None = None
    recovery_count: int = 0
    total_captures: int = 0
    total_failures: int = 0
    uptime_start: float = field(default_factory=time.time)

    def record_success(self) -> None:
        self.last_success_at = time.time()
        self.total_captures += 1

    def record_failure(self, error: str) -> None:
        self.last_error = error
        self.last_error_at = time.time()
        self.total_failures += 1

    def record_recovery(self) -> None:
        self.recovery_count += 1

    def uptime_sec(self) -> float:
        return time.time() - self.uptime_start

    def to_dict(self) -> dict:
        return {
            "camera_healthy": self.camera_healthy,
            "network_healthy": self.network_healthy,
            "server_healthy": self.server_healthy,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
            "recovery_count": self.recovery_count,
            "total_captures": self.total_captures,
            "total_failures": self.total_failures,
            "uptime_sec": self.uptime_sec(),
        }
