"""Disk space health monitoring."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from idcam.config.schema import StorageConfig
from idcam.logging.setup import get_logger

logger = get_logger("storage.disk")


class StorageHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class StorageStatus:
    health: StorageHealth
    free_mb: float
    min_free_mb: float
    message: str | None = None

    @property
    def healthy(self) -> bool:
        return self.health == StorageHealth.HEALTHY

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "health": self.health.value,
            "free_mb": round(self.free_mb, 1),
            "min_free_mb": self.min_free_mb,
            "message": self.message,
        }


class DiskMonitor:
    """Monitors free disk space for the application temp directory."""

    def __init__(self, config: StorageConfig, path: Path) -> None:
        self.config = config
        self.path = path

    def _free_bytes(self) -> int:
        if hasattr(os, "statvfs"):
            stat = os.statvfs(self.path)
            return stat.f_bavail * stat.f_frsize
        usage = shutil.disk_usage(self.path)
        return usage.free

    def check(self) -> StorageStatus:
        try:
            free_bytes = self._free_bytes()
            free_mb = free_bytes / (1024 * 1024)
        except OSError as exc:
            logger.error("Cannot read disk space for %s: %s", self.path, exc)
            return StorageStatus(
                health=StorageHealth.UNAVAILABLE,
                free_mb=0.0,
                min_free_mb=self.config.min_free_mb,
                message="Cannot read disk space",
            )

        if free_mb < self.config.min_free_mb:
            logger.warning(
                "Storage below threshold: %.1f MB free (min %d MB)",
                free_mb,
                self.config.min_free_mb,
            )
            return StorageStatus(
                health=StorageHealth.DEGRADED,
                free_mb=free_mb,
                min_free_mb=self.config.min_free_mb,
                message="Storage full — Please contact support",
            )

        return StorageStatus(
            health=StorageHealth.HEALTHY,
            free_mb=free_mb,
            min_free_mb=self.config.min_free_mb,
        )

    def can_capture(self, required_mb: float | None = None) -> bool:
        status = self.check()
        if not status.healthy:
            return False
        if required_mb is not None and status.free_mb < required_mb:
            return False
        return True
