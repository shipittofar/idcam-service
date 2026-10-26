"""Structured logging setup."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from idcam.config.schema import LoggingConfig
from idcam.storage.paths import ensure_writable_directory


class JsonFormatter(logging.Formatter):
    """JSON log formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "capture_id") and record.capture_id:
            payload["capture_id"] = record.capture_id
        if hasattr(record, "state") and record.state:
            payload["state"] = record.state
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key in ("extra_fields",):
            if hasattr(record, key):
                payload.update(getattr(record, key))
        return json.dumps(payload, ensure_ascii=False)


class CaptureContextFilter(logging.Filter):
    """Inject capture_id and state into log records."""

    def __init__(self) -> None:
        super().__init__()
        self.capture_id: str | None = None
        self.state: str | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        record.capture_id = self.capture_id
        record.state = self.state
        return True


_context_filter = CaptureContextFilter()


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Configure application logging with rotation."""
    ensure_writable_directory(config.log_dir, "log")
    log_file = config.log_dir / "idcam.log"

    root = logging.getLogger("idcam")
    root.handlers.clear()
    root.setLevel(getattr(logging, config.level))
    root.propagate = False

    if config.json_format:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] capture=%(capture_id)s state=%(state)s %(message)s"
        )

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_context_filter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_context_filter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"idcam.{name}")


def set_log_context(*, capture_id: str | None = None, state: str | None = None) -> None:
    if capture_id is not None:
        _context_filter.capture_id = capture_id
    if state is not None:
        _context_filter.state = state
