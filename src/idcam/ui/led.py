"""WS2812B (NeoPixel) status LED strip for Raspberry Pi 5.

Optional and off by default. The strip is a status aid only: if the LED
libraries are missing or the hardware fails, the indicator logs a warning
and turns itself off. It never raises into the capture pipeline.
"""

from __future__ import annotations

import math
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
ORANGE: Color = (255, 80, 0)
CYAN: Color = (0, 200, 255)
BLUE: Color = (0, 0, 255)
GREEN: Color = (0, 255, 0)
RED: Color = (255, 0, 0)

FRAME_INTERVAL_SEC = 0.05
# At boot the service can start before /dev/pio0 (RP1 PIO) or its permissions exist, and the
# first attempt then failed for good, leaving the strip dark until the next restart.
CONNECT_RETRY_SEC = 5.0
CONNECT_MAX_ATTEMPTS = 120  # ~10 minutes


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


STARTING = LedPattern(BLUE, "pulse", 2.0)
READY = LedPattern(WHITE)
# Detection through capture reuses the ready pattern itself, so the strip
# stays the same steady white from IDLE through autofocus and the still
# capture and the light on the document never changes mid-capture.
DOCUMENT = READY
PROCESSING = LedPattern(BLUE, "blink", 0.6)
DONE = LedPattern(GREEN)
RECOVERING = LedPattern(YELLOW, "blink", 1.0)
FAULT = LedPattern(RED, "blink", 1.0)
DARK = LedPattern(OFF)
FAILURE_FLASH = LedPattern(RED)
# Failed repeatedly and the document is still there: take it away, then place it again.
REMOVE_HINT = LedPattern(ORANGE, "blink", 0.8)
# "working" colour when illuminate_during_capture is off.
WORKING_COLORED = LedPattern(CYAN, "pulse", 1.5)

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
    AppState.WAITING_FOR_REMOVAL: DONE,
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
_SUCCESS_STATES = (AppState.SUCCESS, AppState.WAITING_FOR_REMOVAL)


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
    working = READY if config.illuminate_during_capture else WORKING_COLORED
    groups = {
        "starting": (_STARTUP_STATES, STARTING),
        "ready": ((AppState.IDLE,), READY),
        "working": (_WORKING_STATES, working),
        "processing": (_PROCESSING_STATES, PROCESSING),
        "success": (_SUCCESS_STATES, DONE),
        "recovering": ((AppState.RECOVERY,), RECOVERING),
        "error": ((AppState.ERROR,), FAULT),
    }
    for name, (states, base) in groups.items():
        pattern = _recolor(base, colors.get(name))
        for state in states:
            table[state] = pattern
    # "working" follows "ready" when it is meant to be the same light on the document.
    if config.illuminate_during_capture and "working" not in colors:
        for state in _WORKING_STATES:
            table[state] = table[AppState.IDLE]
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
        self._stop.clear()
        if self._backend is None:
            try:
                self._backend = Pi5NeopixelBackend(self.config)
            except Exception as exc:
                logger.warning(
                    "event=led_unavailable error=%s retrying_every_sec=%.0f gpio=%d",
                    exc,
                    CONNECT_RETRY_SEC,
                    self.config.gpio_pin,
                )
                self._thread = threading.Thread(target=self._connect_then_run, name="idcam-led", daemon=True)
                self._thread.start()
                return
        self._thread = threading.Thread(target=self._run, name="idcam-led", daemon=True)
        self._thread.start()
        self._log_started()

    def _log_started(self, attempt: int | None = None) -> None:
        logger.info(
            "event=led_started gpio=%d count=%d brightness=%.2f%s",
            self.config.gpio_pin,
            self.config.count,
            self.config.brightness,
            f" after_attempts={attempt}" if attempt else "",
        )

    def _connect_then_run(self) -> None:
        """Keep trying to open the strip (hardware not ready yet at boot), then drive it."""
        for attempt in range(2, CONNECT_MAX_ATTEMPTS + 2):
            if self._stop.wait(CONNECT_RETRY_SEC):
                return
            try:
                self._backend = Pi5NeopixelBackend(self.config)
            except Exception as exc:
                if attempt % 12 == 0:
                    logger.warning("event=led_still_unavailable attempts=%d error=%s", attempt, exc)
                continue
            self._log_started(attempt)
            self._run()
            return
        logger.warning("event=led_gave_up attempts=%d", CONNECT_MAX_ATTEMPTS + 1)

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
