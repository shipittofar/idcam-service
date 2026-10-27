"""WS2812B (NeoPixel) status LED strip for Raspberry Pi 5.

Optional and off by default. The strip is a status aid only: if the LED
libraries are missing or the hardware fails, the indicator logs a warning
and turns itself off. It never raises into the capture pipeline.
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from idcam.app.state_machine import AppState
from idcam.logging.setup import get_logger

if TYPE_CHECKING:
    from idcam.config.schema import LedConfig

logger = get_logger("led")

Color = tuple[int, int, int]

OFF: Color = (0, 0, 0)
WHITE: Color = (255, 255, 255)
YELLOW: Color = (255, 255, 0)
BLUE: Color = (0, 0, 255)
GREEN: Color = (0, 255, 0)
RED: Color = (255, 0, 0)

FRAME_INTERVAL_SEC = 0.05
# Seconds to wait before restarting a worker that died, by consecutive failure (the last value
# repeats). At boot the strip's driver or Blinka's dependency check can fail for a while.
WORKER_BACKOFF_SEC = (5.0, 10.0, 20.0, 40.0, 60.0)
# A worker that stays up this long counts as healthy again (failure counter resets).
WORKER_STABLE_SEC = 10.0


@dataclass(frozen=True)
class LedPattern:
    color: Color
    mode: str = "solid"  # solid | blink | pulse
    period_sec: float = 1.0

    def color_at(self, t: float) -> Color:
        if self.mode == "blink":
            return self.color if (t % self.period_sec) < self.period_sec / 2 else OFF
        if self.mode == "pulse":
            level = 0.15 + 0.85 * (0.5 - 0.5 * math.cos(2 * math.pi * t / self.period_sec))
            return tuple(int(c * level) for c in self.color)  # type: ignore[return-value]
        return self.color


# The strip is dark while starting, while waiting for a document and while a document is being
# detected, stabilised, focused and captured: it only speaks once there is something for the
# user to do or to wait for (processing, done, remove, failure, recovering, error). A status can
# still be switched on with led.colors, e.g. {"ready": [255, 255, 255]}.
STARTING = DARK = LedPattern(OFF)
READY = DOCUMENT = DARK
PROCESSING = LedPattern(BLUE, "blink", 0.6)
DONE = LedPattern(GREEN)
# Uploaded and waiting for the document to be taken away: green blinking.
REMOVAL_WAIT = LedPattern(GREEN, "blink", 0.8)
RECOVERING = LedPattern(YELLOW, "blink", 1.0)
FAULT = LedPattern(RED, "blink", 1.0)
FAILURE_FLASH = LedPattern(RED)
# Failed repeatedly and the document is still there: take it away, then place it again.
REMOVE_HINT = LedPattern(WHITE, "blink", 0.8)

STATE_PATTERNS: dict[AppState, LedPattern] = {
    AppState.BOOT: STARTING,
    AppState.INITIALIZING: STARTING,
    AppState.CAMERA_INITIALIZING: STARTING,
    AppState.CAMERA_READY: STARTING,
    AppState.IDLE: READY,
    AppState.DOCUMENT_DETECTED: DOCUMENT,
    AppState.WAITING_FOR_STABILITY: DOCUMENT,
    AppState.FOCUSING: DOCUMENT,
    AppState.FOCUS_VERIFICATION: DOCUMENT,
    AppState.CAPTURING: DOCUMENT,
    AppState.IMAGE_VALIDATION: PROCESSING,
    AppState.UPLOADING: PROCESSING,
    AppState.WAITING_FOR_SERVER: PROCESSING,
    AppState.PROCESSING_RESPONSE: PROCESSING,
    AppState.SUCCESS: DONE,
    AppState.WAITING_FOR_REMOVAL: REMOVAL_WAIT,
    AppState.RECOVERY: RECOVERING,
    AppState.ERROR: FAULT,
    AppState.SHUTDOWN: DARK,
}


_STARTUP_STATES = (AppState.BOOT, AppState.INITIALIZING, AppState.CAMERA_INITIALIZING, AppState.CAMERA_READY)
_WORKING_STATES = (
    AppState.DOCUMENT_DETECTED,
    AppState.WAITING_FOR_STABILITY,
    AppState.FOCUSING,
    AppState.FOCUS_VERIFICATION,
    AppState.CAPTURING,
)
_PROCESSING_STATES = (
    AppState.IMAGE_VALIDATION,
    AppState.UPLOADING,
    AppState.WAITING_FOR_SERVER,
    AppState.PROCESSING_RESPONSE,
)


def _recolor(pattern: LedPattern, color: Color | None) -> LedPattern:
    if color is None or tuple(color) == pattern.color:
        return pattern
    return LedPattern(tuple(color), pattern.mode, pattern.period_sec)  # type: ignore[arg-type]


def build_patterns(config: LedConfig) -> tuple[dict[AppState, LedPattern], dict[str, LedPattern]]:
    """State -> pattern table and the named extras (remove hint, failure flash) for a config.

    With the default config this is exactly ``STATE_PATTERNS``.
    """
    colors = {name: tuple(rgb) for name, rgb in config.colors.items()}
    table = dict(STATE_PATTERNS)
    groups = {
        "starting": (_STARTUP_STATES, STARTING),
        "ready": ((AppState.IDLE,), READY),
        "working": (_WORKING_STATES, DOCUMENT),
        "processing": (_PROCESSING_STATES, PROCESSING),
        "success": ((AppState.SUCCESS,), DONE),
        # same colour as "success", blinking, while the user is asked to take the document away
        "success_wait": ((AppState.WAITING_FOR_REMOVAL,), REMOVAL_WAIT),
        "recovering": ((AppState.RECOVERY,), RECOVERING),
        "error": ((AppState.ERROR,), FAULT),
    }
    for name, (states, base) in groups.items():
        pattern = _recolor(base, colors.get("success" if name == "success_wait" else name))
        for state in states:
            table[state] = pattern
    extras = {
        "remove": _recolor(REMOVE_HINT, colors.get("remove")),
        "failure": _recolor(FAILURE_FLASH, colors.get("failure")),
    }
    return table, extras


class LedBackend(Protocol):
    def show(self, color: Color) -> None: ...

    def close(self) -> None: ...


class Pi5NeopixelBackend:
    """Drives the strip through the Pi 5 RP1 PIO (same stack as ws2812_test.py)."""

    def __init__(self, config: LedConfig) -> None:
        import adafruit_pixelbuf
        import board
        from adafruit_raspberry_pi5_neopixel_write import neopixel_write

        pin = getattr(board, f"D{config.gpio_pin}")

        class _Pixelbuf(adafruit_pixelbuf.PixelBuf):
            def _transmit(self, buf):
                neopixel_write(pin, buf)

        self._pixels = _Pixelbuf(
            size=config.count,
            auto_write=False,
            byteorder=config.byteorder,
            brightness=config.brightness,
        )

    def show(self, color: Color) -> None:
        self._pixels.fill(color)
        self._pixels.show()

    def close(self) -> None:
        self.show(OFF)


class SubprocessLedBackend:
    """Drives the strip through ``idcam.ui.led_worker`` and restarts it when it dies.

    ``show`` and ``poll`` never raise: a strip that cannot be opened only means no LEDs. Colours
    travel as ``R G B`` lines over the worker's stdin; after a restart the last colour is sent
    again, so the strip comes up in the right state without waiting for the next state change.
    """

    def __init__(self, config: LedConfig, command: list[str] | None = None) -> None:
        self._command = command or [
            sys.executable,
            "-m",
            "idcam.ui.led_worker",
            "--pin",
            str(config.gpio_pin),
            "--count",
            str(config.count),
            "--brightness",
            str(config.brightness),
            "--byteorder",
            config.byteorder,
        ]
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._err: tempfile._TemporaryFileWrapper | None = None  # worker stderr (a file never fills up like a pipe)
        self._last: Color = OFF
        self._failures = 0
        self._next_spawn = 0.0
        self._spawned_at = 0.0

    def show(self, color: Color) -> None:
        with self._lock:
            self._last = color
            if not self._ensure_worker():  # a freshly started worker already got this colour
                self._send(color)

    def poll(self) -> None:
        """Called by the indicator thread: restarts a dead worker even when the colour is steady."""
        with self._lock:
            self._ensure_worker()

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is None:
                return
            try:
                self._write(proc, OFF)
                proc.stdin.close()  # type: ignore[union-attr]
                proc.wait(timeout=2.0)
            except Exception:
                proc.kill()
            finally:
                self._drop_err()

    # -- internals (lock held) ---------------------------------------------------------------

    def _ensure_worker(self) -> bool:
        """Make sure a worker is running; True when one was just started (and sent ``_last``)."""
        proc = self._proc
        now = time.monotonic()
        if proc is not None and proc.poll() is None:
            if self._failures and now - self._spawned_at >= WORKER_STABLE_SEC:
                self._failures = 0
            return False
        if proc is not None:
            self._note_exit(proc, now)
            self._proc = None
        if now < self._next_spawn:
            return False
        try:
            self._err = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._err,
                text=True,
                encoding="ascii",
            )
        except OSError as exc:
            self._drop_err()
            self._failures += 1
            self._next_spawn = now + self._backoff()
            logger.warning("event=led_worker_spawn_failed error=%s", exc)
            return False
        self._spawned_at = now
        self._write(self._proc, self._last)  # colour the strip as soon as it opens
        return True

    def _backoff(self) -> float:
        return WORKER_BACKOFF_SEC[min(max(self._failures, 1), len(WORKER_BACKOFF_SEC)) - 1]

    def _drop_err(self) -> None:
        err, self._err = self._err, None
        if err is not None:
            try:
                err.close()
            except Exception:
                pass

    def _note_exit(self, proc: subprocess.Popen, now: float) -> None:
        error: list[str] = []
        try:
            if self._err is not None:
                self._err.seek(0)
                error = self._err.read().strip().splitlines()[-1:]
        except Exception:
            error = []
        finally:
            self._drop_err()
        self._failures += 1
        self._next_spawn = now + self._backoff()
        # Log the first failure and then only every 5th, so a missing strip does not flood the log.
        if self._failures == 1 or self._failures % 5 == 0:
            logger.warning(
                "event=led_unavailable exit_code=%s attempt=%d retry_in_sec=%.0f error=%s",
                proc.returncode,
                self._failures,
                self._backoff(),
                error[0] if error else "",
            )

    def _send(self, color: Color) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self._write(proc, color)

    @staticmethod
    def _write(proc: subprocess.Popen, color: Color) -> None:
        try:
            proc.stdin.write(f"{color[0]} {color[1]} {color[2]}\n")  # type: ignore[union-attr]
            proc.stdin.flush()  # type: ignore[union-attr]
        except (BrokenPipeError, OSError, ValueError):
            pass  # worker is gone; the next poll()/show() restarts it


class LedIndicator:
    """Shows the application state on a WS2812B strip from a background thread."""

    def __init__(self, config: LedConfig, backend: LedBackend | None = None) -> None:
        self.config = config
        self._backend = backend
        self._lock = threading.Lock()
        self._patterns, self._extras = build_patterns(config)
        self._pattern = self._patterns[AppState.BOOT]
        self._pattern_since = time.monotonic()
        self._hint: LedPattern | None = None
        self._hint_since = 0.0
        self._flash: LedPattern | None = None
        self._flash_until = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_color: Color | None = None

    @property
    def active(self) -> bool:
        return self._backend is not None and self._thread is not None and self._thread.is_alive()

    @property
    def pattern(self) -> LedPattern:
        with self._lock:
            return self._pattern

    def start(self) -> None:
        if not self.config.enabled or self._thread is not None:
            return
        if self._backend is None:
            self._backend = SubprocessLedBackend(self.config)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="idcam-led", daemon=True)
        self._thread.start()
        logger.info(
            "event=led_started gpio=%d count=%d brightness=%.2f",
            self.config.gpio_pin,
            self.config.count,
            self.config.brightness,
        )

    def set_state(self, state: AppState) -> None:
        pattern = self._patterns.get(state, FAULT)
        with self._lock:
            self._hint = None  # every transition ends a hint; the caller re-raises it if still needed
            if state == AppState.SHUTDOWN:
                self._flash = None
            if pattern is not self._pattern:
                self._pattern = pattern
                self._pattern_since = time.monotonic()

    def set_remove_hint(self, active: bool) -> None:
        """Blink the "remove the document" colour on top of the current (IDLE) pattern."""
        with self._lock:
            if active and self._hint is None:
                self._hint = self._extras["remove"]
                self._hint_since = time.monotonic()
            elif not active:
                self._hint = None

    def flash_failure(self, seconds: float | None = None) -> None:
        """Show solid red for a moment, then return to the current state's pattern."""
        duration = self.config.failure_flash_sec if seconds is None else seconds
        with self._lock:
            self._flash = self._extras["failure"]
            self._flash_until = time.monotonic() + duration

    def current_color(self, now: float | None = None) -> Color:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._flash is not None and now < self._flash_until:
                return self._flash.color
            self._flash = None
            if self._hint is not None:
                return self._hint.color_at(now - self._hint_since)
            return self._pattern.color_at(now - self._pattern_since)

    def stop(self) -> None:
        self.set_state(AppState.SHUTDOWN)
        thread = self._thread
        if thread is not None:
            self._stop.set()
            thread.join(timeout=2.0)
            self._thread = None
        backend = self._backend
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                logger.warning("event=led_off_failed error=%s", exc)

    def _run(self) -> None:
        while not self._stop.is_set():
            poll = getattr(self._backend, "poll", None)
            if callable(poll):
                poll()
            color = self.current_color()
            if color != self._last_color:
                try:
                    self._backend.show(color)  # type: ignore[union-attr]
                except Exception as exc:
                    logger.warning("event=led_write_failed error=%s; LED indicator disabled", exc)
                    self._backend = None
                    return
                self._last_color = color
            self._stop.wait(FRAME_INTERVAL_SEC)
