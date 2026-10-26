"""Component health status model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time


class HealthLevel(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class ComponentStatus:
    level: HealthLevel = HealthLevel.UNAVAILABLE
    message: str | None = None
    details: dict = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.level == HealthLevel.HEALTHY

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "level": self.level.value,
            "message": self.message,
            **self.details,
        }


@dataclass
class SystemHealth:
    """Aggregated system health."""

    camera: ComponentStatus = field(default_factory=ComponentStatus)
    network: ComponentStatus = field(default_factory=ComponentStatus)
    server: ComponentStatus = field(default_factory=ComponentStatus)
    storage: ComponentStatus = field(default_factory=ComponentStatus)
    application: ComponentStatus = field(default_factory=ComponentStatus)
    last_success_at: float | None = None
    last_error: str | None = None
    last_error_at: float | None = None
    recovery_count: int = 0
    total_captures: int = 0
    total_failures: int = 0
    uptime_start: float = field(default_factory=time.time)
    autofocus_available: bool = False
    autofocus_degraded: bool = False

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

    @property
    def overall_ready(self) -> bool:
        return (
            self.camera.healthy
            and self.storage.healthy
            and self.application.healthy
        )

    def to_dict(self) -> dict:
        return {
            "camera": self.camera.to_dict(),
            "network": self.network.to_dict(),
            "server": self.server.to_dict(),
            "storage": self.storage.to_dict(),
            "application": self.application.to_dict(),
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
            "recovery_count": self.recovery_count,
            "total_captures": self.total_captures,
            "total_failures": self.total_failures,
            "uptime_seconds": round(self.uptime_sec(), 1),
            "autofocus_available": self.autofocus_available,
            "autofocus_degraded": self.autofocus_degraded,
        }
