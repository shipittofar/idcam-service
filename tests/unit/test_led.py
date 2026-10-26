"""WS2812B status LED indicator."""

import time
from unittest.mock import MagicMock

import pytest

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.app.state_machine import AppState
from idcam.app.transaction import ActiveCaptureTransaction
from idcam.config.schema import AppConfig, LedConfig, ServerConfig, UIConfig
from idcam.system.component_health import ComponentStatus, HealthLevel
from idcam.ui import led as led_mod
from idcam.ui.led import (
    BLUE,
    GREEN,
    OFF,
    RED,
    STATE_PATTERNS,
    WHITE,
    YELLOW,
    LedIndicator,
    LedPattern,
)

STARTUP_STATES = [AppState.BOOT, AppState.INITIALIZING, AppState.CAMERA_INITIALIZING, AppState.CAMERA_READY]
CAPTURE_STATES = [
    AppState.DOCUMENT_DETECTED,
    AppState.WAITING_FOR_STABILITY,
    AppState.FOCUSING,
    AppState.FOCUS_VERIFICATION,
    AppState.CAPTURING,
]
PROCESSING_STATES = [
    AppState.IMAGE_VALIDATION,
    AppState.UPLOADING,
    AppState.WAITING_FOR_SERVER,
    AppState.PROCESSING_RESPONSE,
]
DONE_STATES = [AppState.SUCCESS, AppState.WAITING_FOR_REMOVAL]


class FakeBackend:
    def __init__(self, fail_on_show: bool = False) -> None:
        self.shown: list[tuple[int, int, int]] = []
        self.closed = False
        self.fail_on_show = fail_on_show

    def show(self, color):
        if self.fail_on_show:
            raise OSError("pio unavailable")
        self.shown.append(color)

    def close(self):
        self.closed = True


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _indicator(**kwargs) -> LedIndicator:
    return LedIndicator(LedConfig(enabled=True, **kwargs), backend=FakeBackend())


def _sample(pattern: LedPattern, seconds: float = 4.0, step: float = 0.05) -> set:
    return {pattern.color_at(i * step) for i in range(int(seconds / step))}


# --- state-to-pattern mapping ------------------------------------------------


def test_every_state_has_a_pattern():
    assert set(STATE_PATTERNS) == set(AppState)


@pytest.mark.parametrize("state", STARTUP_STATES)
def test_startup_is_blue_pulsing(state):
    pattern = STATE_PATTERNS[state]
    assert pattern.color == BLUE
    assert pattern.mode == "pulse"
    colors = _sample(pattern)
    assert all(c[0] == 0 and c[1] == 0 and c[2] > 0 for c in colors)
    assert len(colors) > 2


def test_ready_is_solid_white():
    pattern = STATE_PATTERNS[AppState.IDLE]
    assert pattern.mode == "solid"
    assert _sample(pattern) == {WHITE}


@pytest.mark.parametrize("state", CAPTURE_STATES)
def test_capture_states_are_solid_white(state):
    pattern = STATE_PATTERNS[state]
    assert pattern.mode == "solid"
    assert _sample(pattern) == {WHITE}
    assert YELLOW not in _sample(pattern)


@pytest.mark.parametrize("state", PROCESSING_STATES)
def test_processing_and_upload_are_blue_blinking(state):
    pattern = STATE_PATTERNS[state]
    assert pattern.mode == "blink"
    assert _sample(pattern) == {BLUE, OFF}


@pytest.mark.parametrize("state", DONE_STATES)
def test_success_is_solid_green(state):
    pattern = STATE_PATTERNS[state]
    assert pattern.mode == "solid"
    assert _sample(pattern) == {GREEN}


def test_recovery_is_yellow_blinking():
    pattern = STATE_PATTERNS[AppState.RECOVERY]
    assert pattern.mode == "blink"
    assert _sample(pattern) == {YELLOW, OFF}


def test_error_is_red_blinking():
    pattern = STATE_PATTERNS[AppState.ERROR]
    assert pattern.mode == "blink"
    assert _sample(pattern) == {RED, OFF}


def test_shutdown_is_off():
    assert _sample(STATE_PATTERNS[AppState.SHUTDOWN]) == {OFF}


def test_ready_to_capture_keeps_white_without_restarting_pattern():
    indicator = _indicator()
    indicator.set_state(AppState.IDLE)
    for state in CAPTURE_STATES:
        indicator.set_state(state)
        assert indicator.current_color() == WHITE


# --- failure flash -----------------------------------------------------------


def test_failure_is_red_for_two_seconds_then_current_state():
    indicator = _indicator()
    assert indicator.config.failure_flash_sec == 2.0
    indicator.set_state(AppState.IDLE)
    now = time.monotonic()
    indicator.flash_failure()
    assert indicator.current_color(now + 0.1) == RED
    assert indicator.current_color(now + 1.9) == RED
    assert indicator.current_color(now + 2.1) == WHITE


def test_state_change_during_flash_is_shown_after_flash():
    indicator = _indicator()
    indicator.set_state(AppState.IDLE)
    now = time.monotonic()
    indicator.flash_failure()
    indicator.set_state(AppState.DOCUMENT_DETECTED)
    indicator.set_state(AppState.UPLOADING)
    indicator.set_state(AppState.WAITING_FOR_REMOVAL)
    assert indicator.current_color(now + 1.0) == RED
    assert indicator.current_color(now + 2.5) == GREEN


def test_flash_does_not_linger_after_expiry():
    indicator = _indicator(failure_flash_sec=0.2)
    indicator.set_state(AppState.IDLE)
    now = time.monotonic()
    indicator.flash_failure()
    assert indicator.current_color(now + 0.5) == WHITE
    indicator.set_state(AppState.ERROR)
    # A later state uses its own pattern; the expired flash never comes back.
    assert indicator.current_color(now + 0.6) in {RED, OFF}
    indicator.set_state(AppState.IDLE)
    assert indicator.current_color(now + 0.7) == WHITE


def test_shutdown_cancels_flash():
    indicator = _indicator()
    indicator.set_state(AppState.IDLE)
    indicator.flash_failure()
    indicator.set_state(AppState.SHUTDOWN)
    assert indicator.current_color() == OFF


# --- driver thread -----------------------------------------------------------


def test_state_changes_reach_the_strip_and_stop_turns_it_off():
    backend = FakeBackend()
    indicator = LedIndicator(LedConfig(enabled=True), backend=backend)
    indicator.start()
    try:
        indicator.set_state(AppState.IDLE)
        assert _wait_for(lambda: backend.shown and backend.shown[-1] == WHITE)
        indicator.set_state(AppState.WAITING_FOR_REMOVAL)
        assert _wait_for(lambda: backend.shown[-1] == GREEN)
    finally:
        indicator.stop()
    assert backend.closed
    assert not indicator.active


def test_rapid_transitions_settle_on_latest_state():
    backend = FakeBackend()
    indicator = LedIndicator(LedConfig(enabled=True), backend=backend)
    indicator.start()
    try:
        for state in [AppState.IDLE, *CAPTURE_STATES, *PROCESSING_STATES, AppState.SUCCESS]:
            indicator.set_state(state)
        indicator.set_state(AppState.WAITING_FOR_REMOVAL)
        assert _wait_for(lambda: backend.shown and backend.shown[-1] == GREEN)
        time.sleep(0.15)
        assert backend.shown[-1] == GREEN
    finally:
        indicator.stop()


def test_flash_then_restore_on_strip():
    backend = FakeBackend()
    indicator = LedIndicator(LedConfig(enabled=True, failure_flash_sec=0.2), backend=backend)
    indicator.start()
    try:
        indicator.set_state(AppState.IDLE)
        assert _wait_for(lambda: backend.shown and backend.shown[-1] == WHITE)
        indicator.flash_failure()
        assert _wait_for(lambda: backend.shown[-1] == RED)
        assert _wait_for(lambda: backend.shown[-1] == WHITE)
    finally:
        indicator.stop()


def test_calls_do_not_block_caller():
    class SlowBackend(FakeBackend):
        def show(self, color):
            time.sleep(0.3)
            super().show(color)

    indicator = LedIndicator(LedConfig(enabled=True), backend=SlowBackend())
    indicator.start()
    try:
        started = time.monotonic()
        for state in [AppState.IDLE, AppState.FOCUSING, AppState.UPLOADING, AppState.SUCCESS]:
            indicator.set_state(state)
        indicator.flash_failure()
        assert time.monotonic() - started < 0.05
    finally:
        indicator.stop()


# --- hardware unavailable ----------------------------------------------------


def test_disabled_by_default_never_touches_hardware(monkeypatch):
    def boom(_config):
        raise AssertionError("backend must not be created when disabled")

    monkeypatch.setattr(led_mod, "Pi5NeopixelBackend", boom)
    indicator = LedIndicator(LedConfig())
    indicator.start()
    indicator.set_state(AppState.IDLE)
    indicator.flash_failure()
    indicator.stop()
    assert not indicator.active


def test_missing_libraries_disable_indicator_without_raising(monkeypatch):
    def missing(_config):
        raise ModuleNotFoundError("No module named 'board'")

    monkeypatch.setattr(led_mod, "Pi5NeopixelBackend", missing)
    indicator = LedIndicator(LedConfig(enabled=True))
    indicator.start()
    assert not indicator.active
    indicator.set_state(AppState.ERROR)
    indicator.flash_failure()
    indicator.stop()


def test_write_failure_disables_indicator():
    backend = FakeBackend(fail_on_show=True)
    indicator = LedIndicator(LedConfig(enabled=True), backend=backend)
    indicator.start()
    assert _wait_for(lambda: not indicator.active)
    indicator.set_state(AppState.IDLE)
    indicator.stop()


# --- orchestrator wiring -----------------------------------------------------


@pytest.fixture
def orch(tmp_path):
    config = AppConfig(
        device_id="led",
        temp_dir=tmp_path / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        ui=UIConfig(
            status_file=tmp_path / "run" / "status.json",
            admin_socket=tmp_path / "run" / "admin.sock",
        ),
    )
    o = ApplicationOrchestrator(config, use_mock_camera=True)
    o.health.camera = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
    o.health.storage = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
    o.disk.can_capture = MagicMock(return_value=True)
    return o


def test_orchestrator_transitions_drive_led(orch):
    orch.sm.state = AppState.CAMERA_READY
    orch._transition(AppState.IDLE)
    assert orch.led.pattern is STATE_PATTERNS[AppState.IDLE]
    orch._transition(AppState.DOCUMENT_DETECTED)
    orch._transition(AppState.FOCUSING)
    orch._transition(AppState.CAPTURING)
    assert orch.led.current_color() == WHITE
    orch._transition(AppState.IMAGE_VALIDATION)
    assert orch.led.pattern is STATE_PATTERNS[AppState.UPLOADING]


@pytest.mark.asyncio
async def test_orchestrator_failure_flashes_red_then_restores(orch):
    orch.sm.state = AppState.FOCUSING
    orch._resume_state = AppState.IDLE
    orch._transaction = ActiveCaptureTransaction(capture_id="cap-fail")
    now = time.monotonic()
    await orch._fail_transaction_and_wait("focus_failed", "soft")
    assert orch.sm.state == AppState.IDLE
    assert orch.led.current_color(now + 0.5) == RED
    assert orch.led.current_color(now + 2.5) == WHITE


class TestGuidanceColours:
    def test_default_config_uses_the_standard_table(self):
        table, extras = led_mod.build_patterns(LedConfig())
        assert all(table[state] == pattern for state, pattern in STATE_PATTERNS.items())
        assert extras["remove"].color == led_mod.ORANGE

    def test_every_phase_has_a_distinct_meaning(self):
        table, extras = led_mod.build_patterns(LedConfig())
        assert table[AppState.IDLE].color == WHITE  # place a document
        assert table[AppState.FOCUSING].color == WHITE  # hold still, lit for the camera
        assert table[AppState.UPLOADING].color == BLUE  # captured: wait
        assert table[AppState.SUCCESS].color == GREEN  # done: remove the document
        assert extras["remove"].color not in (WHITE, BLUE, GREEN, RED, YELLOW)  # failed: remove, retry

    def test_colours_can_be_overridden_per_status_without_touching_the_blink_style(self):
        cfg = LedConfig(colors={"success": (0, 128, 255), "processing": (255, 0, 255)})
        table, _ = led_mod.build_patterns(cfg)
        assert table[AppState.SUCCESS].color == (0, 128, 255) and table[AppState.SUCCESS].mode == "solid"
        assert table[AppState.UPLOADING].color == (255, 0, 255) and table[AppState.UPLOADING].mode == "blink"

    def test_unknown_status_or_bad_channel_is_rejected(self):
        with pytest.raises(ValueError, match="unknown status"):
            LedConfig(colors={"banana": (1, 2, 3)})
        with pytest.raises(ValueError, match="0-255"):
            LedConfig(colors={"success": (0, 300, 0)})

    def test_working_phase_can_use_its_own_colour(self):
        table, _ = led_mod.build_patterns(LedConfig(illuminate_during_capture=False))
        assert table[AppState.FOCUSING].color == led_mod.CYAN
        assert table[AppState.IDLE].color == WHITE

    def test_gpio_pin_is_configurable_and_validated(self):
        assert LedConfig().gpio_pin == 18
        assert LedConfig(gpio_pin=12).gpio_pin == 12
        with pytest.raises(ValueError):
            LedConfig(gpio_pin=40)

    def test_remove_hint_blinks_orange_and_any_transition_clears_it(self):
        led = LedIndicator(LedConfig(enabled=True), backend=FakeBackend())
        led.set_state(AppState.IDLE)
        led.set_remove_hint(True)
        on = led.current_color(now=led._hint_since + 0.1)
        off = led.current_color(now=led._hint_since + 0.6)
        assert on == led_mod.ORANGE and off == OFF
        led.set_state(AppState.DOCUMENT_DETECTED)
        assert led.current_color(now=time.monotonic()) == WHITE

    def test_failure_flash_wins_over_the_hint(self):
        led = LedIndicator(LedConfig(enabled=True), backend=FakeBackend())
        led.set_state(AppState.IDLE)
        led.set_remove_hint(True)
        led.flash_failure(5.0)
        assert led.current_color() == RED
