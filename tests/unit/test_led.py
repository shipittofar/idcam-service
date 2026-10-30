"""WS2812B status LED indicator."""

import os
import sys
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
def test_startup_is_off(state):
    assert _sample(STATE_PATTERNS[state]) == {OFF}


def test_ready_is_steady_white_and_detection_turns_it_off():
    assert _sample(STATE_PATTERNS[AppState.IDLE]) == {WHITE}
    assert _sample(STATE_PATTERNS[AppState.DOCUMENT_DETECTED]) == {OFF}


@pytest.mark.parametrize("state", CAPTURE_STATES)
def test_detection_focus_and_capture_are_off(state):
    pattern = STATE_PATTERNS[state]
    assert pattern.mode == "solid"
    assert _sample(pattern) == {OFF}


@pytest.mark.parametrize("state", PROCESSING_STATES)
def test_processing_and_upload_are_blue_blinking(state):
    pattern = STATE_PATTERNS[state]
    assert pattern.mode == "blink"
    assert _sample(pattern) == {BLUE, OFF}


def test_success_is_solid_green():
    pattern = STATE_PATTERNS[AppState.SUCCESS]
    assert pattern.mode == "solid"
    assert _sample(pattern) == {GREEN}


def test_waiting_for_removal_is_green_blinking():
    pattern = STATE_PATTERNS[AppState.WAITING_FOR_REMOVAL]
    assert pattern.mode == "blink"
    assert _sample(pattern) == {GREEN, OFF}


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


def test_ready_to_capture_stays_dark_without_restarting_pattern():
    indicator = _indicator()
    indicator.set_state(AppState.IDLE)
    for state in CAPTURE_STATES:
        indicator.set_state(state)
        assert indicator.current_color() == OFF


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
        assert _wait_for(lambda: backend.shown and backend.shown[-1] == WHITE)  # ready: strip white
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


def test_missing_libraries_never_raise_and_stay_dark(monkeypatch):
    """The strip lives in a worker process; failing to open it must not touch the service."""
    failing = [sys.executable, "-c", "import sys; print('ModuleNotFoundError: board', file=sys.stderr); sys.exit(3)"]
    monkeypatch.setattr(led_mod, "WORKER_BACKOFF_SEC", (0.05,))
    indicator = LedIndicator(LedConfig(enabled=True), backend=led_mod.SubprocessLedBackend(LedConfig(), command=failing))
    indicator.start()
    indicator.set_state(AppState.ERROR)
    indicator.flash_failure()
    time.sleep(0.3)
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
    assert orch.led.current_color() == OFF
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
        assert extras["remove"].color == WHITE and extras["remove"].mode == "blink"

    def test_every_phase_has_a_distinct_meaning(self):
        table, extras = led_mod.build_patterns(LedConfig())
        assert table[AppState.IDLE].color == WHITE  # no document: white
        assert table[AppState.FOCUSING].color == OFF  # document detected / focus / capture: dark
        assert table[AppState.UPLOADING].color == BLUE  # captured: wait
        assert table[AppState.WAITING_FOR_REMOVAL].color == GREEN  # done: remove the document
        assert table[AppState.WAITING_FOR_REMOVAL].mode == "blink"
        assert extras["remove"].mode == "blink"  # failed: blinking white, unlike the steady ready white

    def test_colours_can_be_overridden_per_status_without_touching_the_blink_style(self):
        cfg = LedConfig(colors={"success": (0, 128, 255), "processing": (255, 0, 255)})
        table, _ = led_mod.build_patterns(cfg)
        assert table[AppState.SUCCESS].color == (0, 128, 255) and table[AppState.SUCCESS].mode == "solid"
        assert table[AppState.WAITING_FOR_REMOVAL].color == (0, 128, 255)
        assert table[AppState.WAITING_FOR_REMOVAL].mode == "blink"
        assert table[AppState.UPLOADING].color == (255, 0, 255) and table[AppState.UPLOADING].mode == "blink"

    def test_unknown_status_or_bad_channel_is_rejected(self):
        with pytest.raises(ValueError, match="unknown status"):
            LedConfig(colors={"banana": (1, 2, 3)})
        with pytest.raises(ValueError, match="0-255"):
            LedConfig(colors={"success": (0, 300, 0)})

    def test_dark_statuses_can_be_switched_on_with_colors(self):
        cfg = LedConfig(colors={"ready": (255, 255, 255), "working": (0, 200, 255), "starting": (0, 0, 255)})
        table, _ = led_mod.build_patterns(cfg)
        assert table[AppState.IDLE].color == (255, 255, 255)
        assert table[AppState.FOCUSING].color == (0, 200, 255)
        assert table[AppState.CAPTURING].color == (0, 200, 255)
        assert table[AppState.BOOT].color == (0, 0, 255)
        assert table[AppState.UPLOADING].color == BLUE  # untouched

    def test_old_illuminate_setting_in_an_existing_config_is_ignored(self):
        cfg = LedConfig.model_validate({"enabled": True, "illuminate_during_capture": True})
        table, _ = led_mod.build_patterns(cfg)
        assert table[AppState.FOCUSING].color == OFF

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
        assert on == WHITE and off == OFF
        led.set_state(AppState.DOCUMENT_DETECTED)
        assert led.current_color(now=time.monotonic()) == OFF

    def test_failure_flash_wins_over_the_hint(self):
        led = LedIndicator(LedConfig(enabled=True), backend=FakeBackend())
        led.set_state(AppState.IDLE)
        led.set_remove_hint(True)
        led.flash_failure(5.0)
        assert led.current_color() == RED


# --- worker process supervision ----------------------------------------------------------

ECHO_WORKER = (
    "import sys, os; out = open(sys.argv[1], 'a', buffering=1)\n"
    "for line in sys.stdin: out.write(line)\n"
)


def _worker_backend(tmp_path, script=ECHO_WORKER):
    log = tmp_path / "colours.txt"
    return led_mod.SubprocessLedBackend(LedConfig(), command=[sys.executable, "-c", script, str(log)]), log


def test_worker_receives_each_colour_as_rgb_line(tmp_path):
    backend, log = _worker_backend(tmp_path)
    try:
        backend.show((255, 255, 255))
        backend.show((0, 255, 0))
        assert _wait_for(lambda: log.exists() and log.read_text().split("\n")[:2] == ["255 255 255", "0 255 0"], 3.0)
    finally:
        backend.close()


def test_dead_worker_is_restarted_and_gets_the_current_colour_again(tmp_path, monkeypatch):
    monkeypatch.setattr(led_mod, "WORKER_BACKOFF_SEC", (0.05,))
    # first run exits right away (hardware not ready at boot); later runs stay up and log colours
    script = (
        "import sys, os\n"
        "marker = sys.argv[1] + '.ran'\n"
        "if not os.path.exists(marker):\n"
        "    open(marker, 'w').close(); sys.exit(3)\n"
        "out = open(sys.argv[1], 'a', buffering=1)\n"
        "for line in sys.stdin: out.write(line)\n"
    )
    backend, log = _worker_backend(tmp_path, script)
    try:
        backend.show((255, 255, 255))  # worker spawns and dies at once
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and not (log.exists() and "255 255 255" in log.read_text()):
            backend.poll()  # the indicator thread does this every frame
            time.sleep(0.05)
        assert log.exists() and "255 255 255" in log.read_text()  # steady colour was re-sent
    finally:
        backend.close()


def test_indicator_thread_restarts_a_late_strip_without_a_colour_change(tmp_path, monkeypatch):
    monkeypatch.setattr(led_mod, "WORKER_BACKOFF_SEC", (0.05,))
    script = (
        "import sys, os\n"
        "marker = sys.argv[1] + '.ran'\n"
        "if not os.path.exists(marker):\n"
        "    open(marker, 'w').close(); sys.exit(3)\n"
        "out = open(sys.argv[1], 'a', buffering=1)\n"
        "for line in sys.stdin: out.write(line)\n"
    )
    backend, log = _worker_backend(tmp_path, script)
    indicator = LedIndicator(LedConfig(enabled=True), backend=backend)
    indicator.start()
    indicator.set_state(AppState.SUCCESS)  # solid green, then nothing changes for the rest of the test
    try:
        assert _wait_for(lambda: log.exists() and "0 255 0" in log.read_text(), 5.0)
    finally:
        indicator.stop()


def test_worker_exit_error_is_logged_once_and_backoff_grows(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(led_mod, "WORKER_BACKOFF_SEC", (0.01, 0.02))
    failing = [sys.executable, "-c", "import sys; print('RuntimeError: pio unavailable', file=sys.stderr); sys.exit(3)"]
    backend = led_mod.SubprocessLedBackend(LedConfig(), command=failing)
    try:
        with caplog.at_level("WARNING"):
            for _ in range(60):
                backend.poll()
                time.sleep(0.02)
        messages = [r.getMessage() for r in caplog.records if "led_unavailable" in r.getMessage()]
        assert messages and "pio unavailable" in messages[0]
        assert "attempt=1 " in messages[0]
        assert len(messages) < 2 or "attempt=5 " in messages[1]  # logged 1st, 5th, 10th ... not every retry
    finally:
        backend.close()


def test_close_turns_the_strip_off_and_stops_the_worker(tmp_path):
    backend, log = _worker_backend(tmp_path)
    backend.show((255, 255, 255))
    assert _wait_for(lambda: log.exists() and "255 255 255" in log.read_text(), 3.0)
    proc = backend._proc
    backend.close()
    assert proc.poll() is not None
    assert "0 0 0" in log.read_text()


def test_worker_module_protocol_ignores_garbage_and_clamps(monkeypatch, tmp_path):
    from idcam.ui import led_worker

    shown = []

    class Strip:
        def __init__(self, _config):
            pass

        def show(self, color):
            shown.append(color)

        def close(self):
            shown.append("closed")

    import idcam.ui.led as led_mod2

    monkeypatch.setattr(led_mod2, "Pi5NeopixelBackend", Strip)
    pipe = tmp_path / "in.txt"
    pipe.write_text("10 20 30\nnot a colour\n999 -5 7\n1 2\n", encoding="ascii")
    monkeypatch.setattr(led_worker, "_take_command_pipe", lambda: open(pipe, encoding="ascii"))

    assert led_worker.main(["--pin", "18", "--count", "30", "--brightness", "0.2"]) == 0
    assert shown == [(10, 20, 30), (255, 0, 7), "closed"]


def test_worker_reports_init_failure_with_exit_code_3(monkeypatch, tmp_path, capsys):
    from idcam.ui import led_worker

    import idcam.ui.led as led_mod2

    def boom(_config):
        raise SystemExit("Blinka found installed platform dependencies with unsupported versions")

    monkeypatch.setattr(led_mod2, "Pi5NeopixelBackend", boom)
    monkeypatch.setattr(led_worker, "_take_command_pipe", lambda: open(os.devnull, encoding="ascii"))

    assert led_worker.main(["--pin", "18", "--count", "30", "--brightness", "0.2"]) == led_worker.EXIT_INIT_FAILED
    assert "unsupported versions" in capsys.readouterr().err
