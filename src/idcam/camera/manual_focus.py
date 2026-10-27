"""Persisted manual focus value (``camera.focus_mode: manual``).

The value chosen and confirmed in the debug panel is written here and read back at every start, so
the same lens position is used for every later capture until it is changed again.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

from idcam.logging.setup import get_logger

logger = get_logger("camera.manual_focus")

LENS_MIN = 0.0
LENS_MAX = 32.0  # advertised LensPosition range of the Arducam 64MP Hawkeye (dioptres)


def valid_lens(value: object) -> float | None:
    """The value as a float when it is a finite number in the lens range, else None."""
    try:
        lens = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lens) or not LENS_MIN <= lens <= LENS_MAX:
        return None
    return lens


class ManualFocusStore:
    """Small JSON file with the confirmed manual lens position."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> float | None:
        """The saved lens position, or None if nothing valid was saved (never raises)."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.warning("event=manual_focus_unreadable path=%s error=%s", self.path, exc)
            return None
        lens = valid_lens(data.get("lens_position")) if isinstance(data, dict) else None
        if lens is None:
            logger.warning("event=manual_focus_invalid path=%s", self.path)
        return lens

    def save(self, lens: float) -> None:
        """Atomically write the confirmed lens position (raises OSError if the file cannot be written)."""
        value = valid_lens(lens)
        if value is None:
            raise ValueError(f"lens position out of range {LENS_MIN}..{LENS_MAX}: {lens!r}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "lens_position": round(value, 3),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        fd, tmp = tempfile.mkstemp(prefix=".manual_focus.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.info("event=manual_focus_saved lens=%.3f path=%s", value, self.path)
