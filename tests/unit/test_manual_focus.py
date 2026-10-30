"""Manual focus mode: autofocus off, lens value chosen and saved in the debug panel."""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.app.state_machine import AppState
from idcam.app.transaction import ActiveCaptureTransaction
from idcam.camera.autofocus import AutofocusController, AutofocusResult
from idcam.camera.manager import CameraManager
from idcam.camera.manual_focus import ManualFocusStore, valid_lens
from idcam.config.schema import AppConfig, CameraConfig, DebugConfig, ServerConfig, UIConfig
from idcam.exceptions import AutofocusError, CameraError
from idcam.preview.live_viewer import LiveDebugViewer
from idcam.system.component_health import ComponentStatus, HealthLevel

# --------------------------------------------------------------------------- store


class TestManualFocusStore:
    def test_save_then_load_round_trip(self, tmp_path):
        store = ManualFocusStore(tmp_path / "state" / "manual_focus.json")
        assert store.load() is None  # nothing saved yet
        store.save(8.4)
        assert store.load() == 8.4
        data = json.loads(store.path.read_text(encoding="utf-8"))
        assert data["lens_position"] == 8.4 and "saved_at" in data

    def test_a_new_save_replaces_the_old_value_and_leaves_no_temp_files(self, tmp_path):
        store = ManualFocusStore(tmp_path / "manual_focus.json")
        store.save(8.4)
        store.save(9.15)
        assert store.load() == 9.15
        assert [p.name for p in tmp_path.iterdir()] == ["manual_focus.json"]

    @pytest.mark.parametrize("content", ["not json", "[]", '{"lens_position": "abc"}', '{"lens_position": 99}', '{"lens_position": -1}', "{}"])
    def test_unreadable_or_out_of_range_file_is_ignored_not_fatal(self, tmp_path, content):
        path = tmp_path / "manual_focus.json"
        path.write_text(content, encoding="utf-8")
        assert ManualFocusStore(path).load() is None

    @pytest.mark.parametrize("bad", [-0.1, 32.5, float("nan"), float("inf"), "x", None])
    def test_invalid_values_are_never_saved(self, tmp_path, bad):
        store = ManualFocusStore(tmp_path / "manual_focus.json")
        with pytest.raises(ValueError):
            store.save(bad)  # type: ignore[arg-type]
        assert not store.path.exists()

    def test_unwritable_location_raises_oserror(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        with pytest.raises(OSError):
            ManualFocusStore(blocker / "manual_focus.json").save(8.0)

    @pytest.mark.parametrize("value,expected", [(8, 8.0), ("9.5", 9.5), (0, 0.0), (32, 32.0), (33, None), (-1, None), (float("nan"), None), ("x", None), (None, None)])
    def test_valid_lens(self, value, expected):
        assert valid_lens(value) == expected


# --------------------------------------------------------------------------- config


class TestConfig:
    def test_auto_is_the_default_and_manual_is_a_switch(self):
        assert CameraConfig().focus_mode == "auto"
        assert CameraConfig(focus_mode="manual").focus_mode == "manual"
        with pytest.raises(ValueError):
            CameraConfig(focus_mode="semi")  # type: ignore[arg-type]

    def test_manual_lens_range_is_validated(self):
        assert CameraConfig().manual_lens_position == 8.0
        with pytest.raises(ValueError):
            CameraConfig(manual_lens_position=40.0)


# --------------------------------------------------------------------------- camera manager


class FakePicam:
    camera_controls = {
        "AfMode": (0, 2, 0),
        "AfSpeed": (0, 1, 0),
        "AfTrigger": (0, 1, 0),
        "LensPosition": (0.0, 32.0, 1.0),
    }

    def __init__(self) -> None:
        self.controls: list[dict] = []

    def set_controls(self, controls: dict) -> None:
        self.controls.append(dict(controls))


def _real_manager(config: CameraConfig) -> tuple[CameraManager, FakePicam]:
    manager = CameraManager(config, use_mock=True)
    picam = FakePicam()
    manager._use_mock = False
    manager._picam2 = picam
    manager._started = True
    return manager, picam


class TestManagerManualMode:
    def test_preview_start_parks_the_lens_instead_of_starting_autofocus(self):
        manager, picam = _real_manager(CameraConfig(focus_mode="manual", manual_lens_position=9.2))
        manager._enable_preview_autofocus(wait=False)

        assert picam.controls[-1]["LensPosition"] == 9.2
        assert picam.controls[-1]["AfMode"] == manager._af_mode_manual()
        assert all("AfTrigger" not in c for c in picam.controls)  # autofocus is never triggered
        assert all(c.get("AfMode") != manager._af_mode_continuous() for c in picam.controls)
        assert manager._held_lens == 9.2  # the still capture uses it

    def test_auto_mode_still_starts_continuous_autofocus(self):
        manager, picam = _real_manager(CameraConfig())
        manager._enable_preview_autofocus(wait=False)
        assert picam.controls[-1]["AfMode"] == manager._af_mode_continuous()

    def test_set_manual_lens_moves_the_lens_right_away(self):
        manager, picam = _real_manager(CameraConfig(focus_mode="manual"))
        assert manager.set_manual_lens(10.5) == 10.5
        assert picam.controls[-1] == {"AfMode": manager._af_mode_manual(), "LensPosition": 10.5}
        assert manager.get_manual_lens() == 10.5 and manager._held_lens == 10.5

    def test_out_of_range_is_rejected_without_touching_the_lens(self):
        manager, picam = _real_manager(CameraConfig(focus_mode="manual"))
        with pytest.raises(CameraError):
            manager.set_manual_lens(40.0)
        assert picam.controls == []

    def test_still_capture_restore_keeps_the_manual_lens(self):
        manager, picam = _real_manager(CameraConfig(focus_mode="manual", manual_lens_position=8.8))
        manager._enable_preview_autofocus(wait=False)  # what capture_still_to_file does after the still
        assert picam.controls[-1]["LensPosition"] == 8.8

    def test_auto_mode_has_no_manual_lens(self):
        manager = CameraManager(CameraConfig(), use_mock=True)
        assert not manager.is_manual_focus and manager.get_manual_lens() is None


# --------------------------------------------------------------------------- autofocus controller


def _frame() -> np.ndarray:
    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    frame[20:100, 20:30] = 0
    frame[20:30, 20:140] = 0
    return frame


class ManualCamera:
    def __init__(self, lens: float | None = 8.8) -> None:
        self.lens = lens
        self.set_calls: list[float] = []
        self.cycle_calls = 0
        self.search_calls = 0

    def get_controls(self) -> dict:
        return {"AfMode": (0, 2, 0), "AfTrigger": (0, 1, 0), "LensPosition": (0.0, 32.0, 1.0)}

    def capture_metadata(self) -> dict:
        return {"LensPosition": self.lens}

    def capture_preview_array(self) -> np.ndarray:
        return _frame()

    def capture_fresh_preview_array(self) -> np.ndarray:
        return _frame()

    def apply_document_focus_window(self, preview_roi, preview_size=None) -> dict:
        size = preview_size or (160, 120)
        return {"preview_roi": preview_roi, "preview_size": size, "scaler_crop": (0, 0, *size), "sensor_roi": preview_roi}

    def get_manual_lens(self):
        return self.lens

    def set_manual_lens(self, value: float) -> float:
        self.set_calls.append(value)
        return value

    # autofocus entry points that must stay unused in manual mode
    def autofocus_cycle(self) -> bool:
        self.cycle_calls += 1
        return True

    def set_lens_position(self, value: float) -> None:
        self.search_calls += 1


class TestAutofocusControllerManual:
    @pytest.mark.asyncio
    async def test_no_autofocus_and_no_lens_search_only_the_manual_value(self):
        camera = ManualCamera(8.8)
        ctl = AutofocusController(CameraConfig(focus_mode="manual", af_refine_settle_sec=0.05))
        ctl.probe_capabilities(camera)

        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)

        assert result.success and result.method == "manual"
        assert result.lens_after == 8.8
        assert camera.set_calls == [8.8]
        assert camera.cycle_calls == 0 and camera.search_calls == 0

    @pytest.mark.asyncio
    async def test_a_soft_manual_focus_is_reported_but_never_blocks_the_capture(self, caplog):
        camera = ManualCamera(8.8)
        ctl = AutofocusController(CameraConfig(focus_mode="manual", af_refine_settle_sec=0.05))
        ctl.probe_capabilities(camera)

        with caplog.at_level("WARNING"):
            result = await ctl.run_autofocus(camera, sharpness_threshold=1e12)

        assert result.success
        assert "manual_focus_soft" in caplog.text

    @pytest.mark.asyncio
    async def test_manual_mode_without_a_lens_value_fails_clearly(self):
        camera = ManualCamera(None)
        ctl = AutofocusController(CameraConfig(focus_mode="manual", af_refine_settle_sec=0.05))
        ctl.probe_capabilities(camera)
        with pytest.raises(AutofocusError, match="manual"):
            await ctl.run_autofocus(camera, sharpness_threshold=1.0)

    @pytest.mark.asyncio
    async def test_remembered_lens_is_ignored_in_manual_mode(self):
        camera = ManualCamera(9.0)
        ctl = AutofocusController(CameraConfig(focus_mode="manual", af_refine_settle_sec=0.05))
        ctl.probe_capabilities(camera)
        ctl.remember_lens(5.0)
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.lens_after == 9.0 and camera.search_calls == 0


# --------------------------------------------------------------------------- orchestrator


def _config(tmp_path, **camera) -> AppConfig:
    return AppConfig(
        device_id="manual-focus",
        temp_dir=tmp_path / "state" / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        camera=CameraConfig(af_refine_settle_sec=0.05, **camera),
        ui=UIConfig(status_file=tmp_path / "run" / "status.json", admin_socket=tmp_path / "run" / "admin.sock"),
    )


def _orch(tmp_path, **camera) -> ApplicationOrchestrator:
    orch = ApplicationOrchestrator(_config(tmp_path, **camera), use_mock_camera=True)
    orch.health.camera = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
    return orch


class TestOrchestratorManualFocus:
    def test_saved_value_is_used_at_startup_else_the_config_default(self, tmp_path):
        orch = _orch(tmp_path, focus_mode="manual", manual_lens_position=7.5)
        orch._load_manual_focus()
        assert orch.camera.get_manual_lens() == 7.5  # nothing saved yet: config value

        orch.manual_focus_store.save(9.1)
        again = _orch(tmp_path, focus_mode="manual", manual_lens_position=7.5)
        again._load_manual_focus()
        assert again.camera.get_manual_lens() == 9.1  # the confirmed value wins

    def test_default_save_location_is_next_to_the_temp_dir(self, tmp_path):
        orch = _orch(tmp_path, focus_mode="manual")
        assert orch.manual_focus_store.path == tmp_path / "state" / "manual_focus.json"
        custom = _orch(tmp_path, focus_mode="manual", manual_focus_file=tmp_path / "elsewhere.json")
        assert custom.manual_focus_store.path == tmp_path / "elsewhere.json"

    def test_apply_then_confirm_and_save(self, tmp_path):
        orch = _orch(tmp_path, focus_mode="manual")
        orch.sm.state = AppState.IDLE

        applied = orch.set_manual_focus(9.3, save=False)
        assert applied["ok"] and applied["status"] == "applied"
        assert orch.camera.get_manual_lens() == 9.3
        assert orch.manual_focus_store.load() is None  # preview only, not saved

        saved = orch.set_manual_focus(9.3, save=True)
        assert saved["ok"] and saved["status"] == "saved" and saved["saved_lens"] == 9.3
        assert orch.manual_focus_store.load() == 9.3
        assert orch.focus_status()["saved_lens"] == 9.3

    def test_rejected_in_auto_mode_for_bad_values_and_while_a_capture_runs(self, tmp_path):
        auto = _orch(tmp_path)  # focus_mode auto
        assert auto.set_manual_focus(8.0)["status"] == "disabled"
        assert auto.focus_status() == {"mode": "auto"}

        orch = _orch(tmp_path, focus_mode="manual")
        orch.sm.state = AppState.IDLE
        assert orch.set_manual_focus("abc")["status"] == "invalid"
        assert orch.set_manual_focus(99)["status"] == "invalid"
        assert orch.set_manual_focus(float("nan"))["status"] == "invalid"

        orch.sm.state = AppState.FOCUSING
        busy = orch.set_manual_focus(8.0)
        assert busy["status"] == "busy" and not busy["ok"]
        assert orch.focus_status()["busy"] is True

    def test_save_failure_is_reported_and_the_lens_still_moved(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        orch = _orch(tmp_path, focus_mode="manual", manual_focus_file=blocker / "m.json")
        orch.sm.state = AppState.IDLE
        result = orch.set_manual_focus(8.4, save=True)
        assert not result["ok"] and result["status"] == "save_failed"
        assert orch.camera.get_manual_lens() == 8.4

    def test_status_describes_the_manual_state(self, tmp_path):
        orch = _orch(tmp_path, focus_mode="manual", manual_lens_position=8.0)
        orch.sm.state = AppState.IDLE
        status = orch.focus_status()
        assert status["mode"] == "manual"
        assert status["lens_position"] == 8.0
        assert status["saved_lens"] is None
        assert status["range"] == [0.0, 32.0]
        assert status["busy"] is False

    @pytest.mark.asyncio
    async def test_focusing_state_uses_the_manual_lens_and_runs_no_autofocus(self, tmp_path):
        orch = _orch(tmp_path, focus_mode="manual")
        orch.camera.initialize()
        orch.camera.set_manual_lens(8.8)
        orch.camera.autofocus_cycle = MagicMock()
        orch.camera.trigger_autofocus = MagicMock()
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-manual")
        frame = _document()
        orch._safe_preview = AsyncMock(return_value=frame)
        orch.detector.lock_document(frame, allow_fallback=True)
        orch.detector.is_scene_empty = MagicMock(return_value=False)

        await orch._focusing_tick()

        assert orch.sm.state == AppState.CAPTURING
        assert orch._last_focus_lens == 8.8
        orch.camera.autofocus_cycle.assert_not_called()
        orch.camera.trigger_autofocus.assert_not_called()


class TestFocusingStateIsSkipped:
    @pytest.mark.asyncio
    async def test_stable_document_goes_straight_to_capturing(self, tmp_path):
        orch = _orch(tmp_path, focus_mode="manual")
        orch.camera.initialize()
        orch.camera.set_manual_lens(8.8)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-skip")
        frame = _document()
        orch._safe_preview = AsyncMock(return_value=frame)
        orch.detector.lock_document(frame, allow_fallback=True)
        orch.detector.is_scene_empty = MagicMock(return_value=False)
        orch.autofocus.run_autofocus = AsyncMock()
        seen = []
        for _ in range(20):
            await orch._stability_tick()
            seen.append(orch.sm.state)
            if orch.sm.state == AppState.CAPTURING:
                break
        assert orch.sm.state == AppState.CAPTURING
        assert AppState.FOCUSING not in seen
        assert orch._last_focus_lens == 8.8
        orch.autofocus.run_autofocus.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_capture_request_also_skips_focusing(self, tmp_path):
        orch = _orch(tmp_path, focus_mode="manual")
        orch.camera.initialize()
        orch.camera.set_manual_lens(8.8)
        orch.sm.state = AppState.DOCUMENT_DETECTED
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-man", trigger="manual")
        orch._safe_preview = AsyncMock(return_value=_document())
        await orch._document_detected_tick()
        assert orch.sm.state == AppState.CAPTURING


class TestFindBestFocus:
    @pytest.mark.asyncio
    async def test_search_applies_the_best_lens_without_saving_it(self, tmp_path):
        orch = _orch(tmp_path, focus_mode="manual")
        orch.camera.initialize()
        orch.camera.set_manual_lens(8.0)
        orch.sm.state = AppState.IDLE
        orch._event_loop = asyncio.get_running_loop()
        orch._safe_preview = AsyncMock(return_value=_document())
        seen = {}

        async def fake_af(camera, **kwargs):
            seen.update(kwargs)
            seen["busy_during"] = orch._finding_focus
            return AutofocusResult(True, 120.0, 1, 0, "test", 10.0, lens_after=9.4)

        orch.autofocus.run_autofocus = fake_af
        result = await asyncio.to_thread(orch.find_manual_focus)

        assert result["ok"] and result["status"] == "found" and result["lens_position"] == 9.4
        assert seen["force_auto"] is True and seen["busy_during"] is True
        assert orch.camera.get_manual_lens() == 9.4
        assert orch.manual_focus_store.load() is None  # only "Confirm & save" persists
        assert orch._finding_focus is False

    @pytest.mark.asyncio
    async def test_failed_search_restores_the_previous_lens(self, tmp_path):
        orch = _orch(tmp_path, focus_mode="manual")
        orch.camera.initialize()
        orch.camera.set_manual_lens(8.0)
        orch.sm.state = AppState.IDLE
        orch._event_loop = asyncio.get_running_loop()
        orch._safe_preview = AsyncMock(return_value=_document())
        orch.autofocus.run_autofocus = AsyncMock(side_effect=AutofocusError("no peak"))
        result = await asyncio.to_thread(orch.find_manual_focus)
        assert not result["ok"] and "no peak" in result["message"]
        assert orch.camera.get_manual_lens() == 8.0 and orch._finding_focus is False

    def test_rejected_in_auto_mode_and_while_a_capture_runs(self, tmp_path):
        assert _orch(tmp_path).find_manual_focus()["status"] == "disabled"
        orch = _orch(tmp_path, focus_mode="manual")
        orch._event_loop = None
        assert orch.find_manual_focus()["status"] == "error"
        orch._event_loop = MagicMock()
        orch._event_loop.is_running.return_value = True
        orch.sm.state = AppState.CAPTURING
        assert orch.find_manual_focus()["status"] == "busy"

    @pytest.mark.asyncio
    async def test_idle_tick_does_nothing_during_the_search(self, tmp_path):
        orch = _orch(tmp_path, focus_mode="manual")
        orch.sm.state = AppState.IDLE
        orch._finding_focus = True
        orch._safe_preview = AsyncMock(return_value=_document())
        await orch._idle_tick()
        orch._safe_preview.assert_not_called()
        assert orch.focus_status()["busy"] is True

    def test_panel_auto_request_goes_to_the_auto_handler(self, viewer):
        calls = []
        viewer.set_focus_handlers(
            lambda: dict(MANUAL_STATUS),
            lambda lens, save: {"ok": True, "status": "applied"},
            lambda: calls.append(1) or {"ok": True, "status": "found", "message": "ok", "lens_position": 9.4},
        )
        code, payload = _post(viewer, {"auto": True})
        assert code == 200 and payload["status"] == "found" and payload["lens_position"] == 9.4
        assert calls == [1]

    def test_panel_auto_request_without_handler_is_unavailable(self, viewer):
        viewer.set_focus_handlers(lambda: dict(MANUAL_STATUS), lambda lens, save: {"ok": True, "status": "applied"})
        code, payload = _post(viewer, {"auto": True})
        assert code == 503 and payload["status"] == "unavailable"

    def test_page_has_the_find_button(self, viewer):
        with urllib.request.urlopen(_url(viewer, "/preview/live"), timeout=3) as resp:
            assert "Find best focus" in resp.read().decode()


class TestUploadCopyOverlap:
    @staticmethod
    def _jpeg(orch, size=(1600, 1200)):
        from PIL import Image

        path = orch.temp_store.create_temp_path()
        Image.new("RGB", size, (180, 180, 180)).save(path, format="JPEG", quality=90)
        return path

    @pytest.mark.asyncio
    async def test_copy_is_built_during_validation_and_used_for_the_upload(self, tmp_path):
        orch = _orch(tmp_path)
        original = self._jpeg(orch)
        orch._capture_path = original
        orch._start_upload_copy(original, (200, 150, 800, 600), (800, 600), None)
        assert orch._upload_crop_roi is not None and orch._upload_copy_task is not None
        await orch._use_upload_copy()
        assert orch._capture_path != original and orch._capture_path.exists()
        assert not original.exists()  # the full-size capture is deleted once the copy replaces it
        assert orch._upload_copy_task is None

    @pytest.mark.asyncio
    async def test_a_rejected_image_leaves_no_copy_behind(self, tmp_path):
        orch = _orch(tmp_path)
        original = self._jpeg(orch)
        orch._start_upload_copy(original, (200, 150, 800, 600), (800, 600), None)
        task = orch._upload_copy_task
        await orch._discard_upload_copy()
        prepared = await task
        assert prepared is None or not prepared.path.exists()
        assert orch._upload_copy_task is None

    @pytest.mark.asyncio
    async def test_unreadable_image_falls_back_to_the_original(self, tmp_path):
        orch = _orch(tmp_path)
        broken = orch.temp_store.create_temp_path()
        broken.write_bytes(b"not a jpeg")
        orch._capture_path = broken
        orch._start_upload_copy(broken, (0, 0, 100, 100), (800, 600), None)
        await orch._use_upload_copy()
        assert orch._capture_path == broken


class TestRemovalCounting:
    @pytest.mark.asyncio
    async def test_one_noisy_frame_does_not_restart_the_removal_count(self, tmp_path):
        orch = _orch(tmp_path)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-rm")
        orch._transaction.mark_completed()
        orch._safe_preview = AsyncMock(return_value=_document())
        need = orch.config.detection.removal_frames_required
        flags = iter([True] * (need - 2) + [False] + [True] * 3)
        orch.detector.is_scene_empty = MagicMock(side_effect=lambda frame: next(flags))
        for _ in range(need - 2):
            await orch._removal_tick()
        assert orch._removal_count == need - 2
        await orch._removal_tick()  # a noisy "page still there" frame
        assert orch._removal_count == need - 3  # decays by one instead of resetting to zero
        for _ in range(3):  # 7 -> 10
            await orch._removal_tick()
        assert orch.sm.state == AppState.IDLE

    @pytest.mark.asyncio
    async def test_a_page_that_stays_never_counts_down_to_removed(self, tmp_path):
        orch = _orch(tmp_path)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-stay")
        orch._transaction.mark_completed()
        orch._safe_preview = AsyncMock(return_value=_document())
        orch.detector.is_scene_empty = MagicMock(return_value=False)
        for _ in range(30):
            await orch._removal_tick()
        assert orch.sm.state == AppState.WAITING_FOR_REMOVAL and orch._removal_count == 0


def _document() -> np.ndarray:
    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    frame[20:100, 20:140] = 40
    frame[20:28, 20:140] = 0
    frame[92:100, 20:140] = 0
    return frame


# --------------------------------------------------------------------------- panel (HTTP)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    if port < 1024:
        pytest.skip("no ephemeral port >= 1024")
    return port


@pytest.fixture
def viewer():
    port = _free_port()
    v = LiveDebugViewer(DebugConfig(enabled=True, host="127.0.0.1", port=port, fps=5.0))
    v.start()
    try:
        if not v.is_running:
            pytest.skip("debug HTTP server failed to bind")
        yield v
    finally:
        v.stop()


def _url(v, path):
    return f"http://127.0.0.1:{v.config.port}{path}"


def _post(v, body, content_type="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(_url(v, "/preview/focus"), data=data, method="POST", headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


MANUAL_STATUS = {"mode": "manual", "lens_position": 8.0, "saved_lens": None, "range": [0.0, 32.0], "busy": False}


class TestPanelEndpoints:
    def test_get_focus_defaults_to_auto_without_a_handler(self, viewer):
        with urllib.request.urlopen(_url(viewer, "/preview/focus"), timeout=3) as resp:
            assert json.loads(resp.read().decode()) == {"mode": "auto"}

    def test_status_and_focus_report_manual_mode_with_live_sharpness(self, viewer):
        viewer.set_focus_handlers(lambda: dict(MANUAL_STATUS), lambda lens, save: {"ok": True, "status": "applied"})
        viewer.publish(
            _document(), state="IDLE", document_detected=True, motion=0.0, edge_density=0.1,
            motion_threshold=1.0, edge_density_threshold=0.05, stable_count=0, stable_required=3,
            stable=False, camera_healthy=True, autofocus_available=True,
        )
        for path in ("/preview/focus", "/preview/status"):
            with urllib.request.urlopen(_url(viewer, path), timeout=3) as resp:
                payload = json.loads(resp.read().decode())
            focus = payload if path.endswith("focus") else payload["focus"]
            assert focus["mode"] == "manual" and focus["range"] == [0.0, 32.0]
            assert "sharpness" in focus

    def test_page_has_the_manual_focus_card(self, viewer):
        with urllib.request.urlopen(_url(viewer, "/preview/live"), timeout=3) as resp:
            html = resp.read().decode()
        for needle in ("Manual focus", "lens_slider", "Confirm &amp; save", "/preview/focus"):
            assert needle in html

    def test_post_applies_and_saves_through_the_handler(self, viewer):
        calls = []

        def set_fn(lens, save):
            calls.append((lens, save))
            return {"ok": True, "status": "saved" if save else "applied", "lens_position": lens, "token": "SECRET"}

        viewer.set_focus_handlers(lambda: dict(MANUAL_STATUS), set_fn)
        code, payload = _post(viewer, {"lens_position": 8.4})
        assert code == 200 and payload["status"] == "applied"
        code, payload = _post(viewer, {"lens_position": 8.4, "save": True})
        assert code == 200 and payload["status"] == "saved" and payload["lens_position"] == 8.4
        assert calls == [(8.4, False), (8.4, True)]
        assert "SECRET" not in json.dumps(payload)  # only known fields are echoed

    def test_error_statuses_map_to_http_codes(self, viewer):
        for status, code in (("busy", 409), ("disabled", 409), ("invalid", 400)):
            viewer.set_focus_handlers(lambda: dict(MANUAL_STATUS), lambda lens, save, s=status: {"ok": False, "status": s, "message": s})
            got, payload = _post(viewer, {"lens_position": 8.0})
            assert got == code and payload["ok"] is False

    def test_post_requires_json_content_type_so_other_sites_cannot_move_the_lens(self, viewer):
        called = []
        viewer.set_focus_handlers(lambda: dict(MANUAL_STATUS), lambda lens, save: called.append(lens) or {"ok": True, "status": "applied"})
        code, _ = _post(viewer, b'{"lens_position": 8}', content_type="text/plain")
        assert code == 415 and called == []

    def test_invalid_body_and_missing_handler(self, viewer):
        code, _ = _post(viewer, b"{not json")
        assert code == 400
        code, payload = _post(viewer, {"lens_position": 8})
        assert code == 503 and payload["status"] == "unavailable"

    def test_handler_exception_is_a_500_without_details(self, viewer):
        def boom(lens, save):
            raise RuntimeError("camera exploded with secret path /etc/idcam")

        viewer.set_focus_handlers(lambda: dict(MANUAL_STATUS), boom)
        code, payload = _post(viewer, {"lens_position": 8})
        assert code == 500 and "secret" not in json.dumps(payload)

    def test_orchestrator_wires_the_panel_when_debug_is_enabled(self, tmp_path):
        config = _config(tmp_path, focus_mode="manual")
        config = config.model_copy(update={"debug": DebugConfig(enabled=True, host="127.0.0.1", port=_free_port())})
        orch = ApplicationOrchestrator(config, use_mock_camera=True)
        assert orch._debug_viewer is not None
        assert orch._debug_viewer.focus_set_handler == orch.set_manual_focus
        assert orch._debug_viewer.focus_payload()["mode"] == "manual"
