"""End-to-end capture state-machine behaviour (the Pi bugs unit tests missed)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.app.state_machine import AppState
from idcam.app.transaction import ActiveCaptureTransaction
from idcam.camera.autofocus import AutofocusResult
from idcam.config.schema import AppConfig, CameraConfig, DetectionConfig, QualityConfig, ServerConfig, UIConfig
from idcam.network.protocol import ServerResponse, UploadOutcome
from idcam.system.component_health import ComponentStatus, HealthLevel


@pytest.fixture
def app_config(tmp_path):
    return AppConfig(
        device_id="lifecycle",
        temp_dir=tmp_path / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        quality=QualityConfig(
            min_sharpness=10.0,
            min_file_bytes=100,
            min_width=64,
            min_height=64,
            require_expected_dimensions=False,
        ),
        detection=DetectionConfig(
            motion_threshold=5.0,
            edge_density_threshold=0.01,
            stable_frames_required=2,
            removal_frames_required=2,
            detect_frames_required=1,
        ),
        camera=CameraConfig(preview_fps=30.0, af_timeout_sec=1.0, af_max_retries=1),
        ui=UIConfig(
            status_file=tmp_path / "run" / "status.json",
            admin_socket=tmp_path / "run" / "admin.sock",
        ),
    )


def _orch(app_config):
    orch = ApplicationOrchestrator(app_config, use_mock_camera=True)
    orch.health.camera = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
    orch.disk.can_capture = MagicMock(return_value=True)
    return orch


def _empty_frame():
    return np.full((120, 160, 3), 200, dtype=np.uint8)


def _document_frame():
    frame = _empty_frame()
    frame[20:100, 20:140] = 40
    frame[20:28, 20:140] = 0
    frame[92:100, 20:140] = 0
    frame[20:100, 20:28] = 0
    frame[20:100, 132:140] = 0
    return frame


class TestStartupGoesIdle:
    @pytest.mark.asyncio
    async def test_startup_idle_not_waiting_for_removal(self, app_config):
        orch = _orch(app_config)
        with patch.object(orch.api, "start", AsyncMock()), patch.object(
            orch.api, "check_connectivity", AsyncMock(return_value=True)
        ):
            await orch._startup()
        assert orch.sm.state == AppState.IDLE
        assert orch.sm.state != AppState.WAITING_FOR_REMOVAL
        assert not orch.has_active_transaction
        available, _phase, _msg = orch._manual_status()
        assert available is True


class TestAutoCapturePath:
    @pytest.mark.asyncio
    async def test_idle_document_starts_detected(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch.detector.update_baseline(_empty_frame())
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await orch._idle_tick()
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._transaction is not None
        assert orch._transaction.trigger == "auto"

    @pytest.mark.asyncio
    async def test_detected_to_stability(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.DOCUMENT_DETECTED
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await orch._document_detected_tick()
        assert orch.sm.state == AppState.WAITING_FOR_STABILITY

    @pytest.mark.asyncio
    async def test_stability_to_focusing(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch.detector.update_baseline(_empty_frame())
        frame = _document_frame()
        orch._safe_preview = AsyncMock(return_value=frame)
        for _ in range(6):
            await orch._stability_tick()
            if orch.sm.state == AppState.FOCUSING:
                break
        assert orch.sm.state == AppState.FOCUSING

    @pytest.mark.asyncio
    async def test_focusing_to_capturing(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch._check_motion_abort = AsyncMock(return_value=True)
        orch.autofocus.run_autofocus = AsyncMock(
            return_value=AutofocusResult(True, 90.0, 1, None, "hardware_cycle", 10.0)
        )
        orch.autofocus.verify_focus = AsyncMock(
            return_value=AutofocusResult(True, 90.0, 0, None, "verification", 10.0)
        )
        await orch._focusing_tick()
        assert orch.sm.state == AppState.CAPTURING

    @pytest.mark.asyncio
    async def test_capturing_to_upload(self, app_config, tmp_path):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-1")
        orch._check_motion_abort = AsyncMock(return_value=True)

        async def fake_capture(_camera, path):
            path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 500)
            return path

        orch.capture_service.capture = fake_capture
        orch.validator.validate = MagicMock(
            return_value=MagicMock(file_bytes=500, width=100, height=80, sha256="ab", sharpness=90.0)
        )
        await orch._capturing_tick()
        assert orch.sm.state == AppState.IMAGE_VALIDATION
        await orch._validation_tick()
        assert orch.sm.state == AppState.UPLOADING

    @pytest.mark.asyncio
    async def test_success_to_waiting_for_removal(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.SUCCESS
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        with patch("idcam.app.orchestrator.asyncio.sleep", AsyncMock()):
            await orch._success_tick()
        assert orch.sm.state == AppState.WAITING_FOR_REMOVAL


class TestIdleDoesNotStayWhenReady:
    @pytest.mark.asyncio
    async def test_awaiting_clear_scene_retries_when_document_still_present(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._awaiting_clear_scene = True
        orch.detector.update_baseline(_empty_frame())
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await orch._idle_tick()
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._awaiting_clear_scene is False
        assert orch._transaction is not None
        assert orch._transaction.trigger == "auto"

    @pytest.mark.asyncio
    async def test_repeated_failures_hold_until_document_removed(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._awaiting_clear_scene = True
        orch._consecutive_failures = orch.MAX_AUTO_RETRIES_WITH_DOCUMENT
        orch.detector.update_baseline(_empty_frame())
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._awaiting_clear_scene is True
        assert orch._transaction is None


class TestAutoDetectStableFocusCapture:
    @pytest.mark.asyncio
    async def test_detect_stable_focus_capture(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch.detector.update_baseline(_empty_frame())
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch.autofocus.run_autofocus = AsyncMock(
            return_value=AutofocusResult(True, 35.0, 1, None, "hardware_trigger", 28.0, 10.0, True)
        )
        orch.autofocus.verify_focus = AsyncMock(
            return_value=AutofocusResult(True, 35.0, 0, None, "verification", 28.0)
        )

        async def fake_capture(_camera, path):
            path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 500)
            return path

        orch.capture_service.capture = fake_capture
        await orch._idle_tick()
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        await orch._document_detected_tick()
        assert orch.sm.state == AppState.WAITING_FOR_STABILITY
        for _ in range(8):
            await orch._stability_tick()
            if orch.sm.state == AppState.FOCUSING:
                break
        assert orch.sm.state == AppState.FOCUSING
        await orch._focusing_tick()
        assert orch.sm.state == AppState.CAPTURING
        await orch._capturing_tick()
        assert orch.sm.state == AppState.IMAGE_VALIDATION


class TestRemovalHold:
    @pytest.mark.asyncio
    async def test_document_still_present_stays_in_removal(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        orch.detector.update_baseline(_empty_frame())
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await orch._removal_tick()
        await orch._removal_tick()
        assert orch.sm.state == AppState.WAITING_FOR_REMOVAL
        available, _phase, _msg = orch._manual_status()
        assert available is True

    @pytest.mark.asyncio
    async def test_document_removed_returns_idle(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        empty = _empty_frame()
        orch.detector.update_baseline(empty)
        orch._safe_preview = AsyncMock(return_value=empty)
        await orch._removal_tick()
        await orch._removal_tick()
        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None


class TestManualCapturePolicy:
    def test_manual_from_idle(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        available, _phase, _msg = orch._manual_status()
        assert available is True

    def test_manual_from_waiting_for_removal(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-ok")
        orch._transaction.mark_completed()
        available, _phase, _msg = orch._manual_status()
        assert available is True
        result = orch.request_manual_capture(timeout=0.05)
        assert result["status"] == "timeout" or result.get("ok") is False
        assert orch._pending_command in {"manual_capture", None}

    def test_manual_busy_while_focusing(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-busy")
        available, _phase, _msg = orch._manual_status()
        assert available is False
        result = orch.request_manual_capture(timeout=0.1)
        assert result["status"] == "busy"
        assert result["state"] == "FOCUSING"

    def test_manual_busy_while_uploading(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.UPLOADING
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-up")
        available, _phase, _msg = orch._manual_status()
        assert available is False
        result = orch.request_manual_capture(timeout=0.1)
        assert result["status"] == "busy"

    @pytest.mark.asyncio
    async def test_removal_tick_starts_manual_pipeline(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-old")
        orch._transaction.mark_completed()
        orch._pending_command = "manual_capture"
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        await orch._removal_tick()
        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._transaction.trigger == "manual"
        assert orch._resume_state == AppState.WAITING_FOR_REMOVAL

    @pytest.mark.asyncio
    async def test_failure_from_idle_capture_returns_idle_not_removal(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._resume_state = AppState.IDLE
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-fail")
        await orch._fail_transaction_and_wait("focus_failed", "soft")
        assert orch.sm.state == AppState.IDLE
        assert orch._awaiting_clear_scene is True
        available, _phase, _msg = orch._manual_status()
        assert available is True


class TestWaitsForRealDocument:
    @pytest.mark.asyncio
    async def test_empty_scene_after_detection_returns_to_idle_without_failure(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._begin_transaction(trigger="auto")
        orch._safe_preview = AsyncMock(return_value=_empty_frame())
        orch.stability.update = MagicMock(return_value=False)

        for _ in range(orch.DOCUMENT_LOSS_FRAMES):
            await orch._stability_tick()

        assert orch.sm.state == AppState.IDLE
        assert orch._transaction is None
        assert orch._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_document_in_view_keeps_pipeline_running(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._begin_transaction(trigger="auto")
        orch.detector.update_baseline(_empty_frame())
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch.stability.update = MagicMock(return_value=False)

        for _ in range(orch.DOCUMENT_LOSS_FRAMES + 1):
            await orch._stability_tick()

        assert orch.sm.state != AppState.IDLE
        assert orch._transaction is not None

    @pytest.mark.asyncio
    async def test_auto_capture_without_geometry_does_not_lock_a_fake_region(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.DOCUMENT_DETECTED
        orch._begin_transaction(trigger="auto")
        orch._safe_preview = AsyncMock(return_value=_empty_frame())

        await orch._document_detected_tick()

        assert orch.sm.state == AppState.IDLE
        assert not orch.detector.lock.is_locked
        assert orch._transaction is None


class TestMotionBeforeShutter:
    @staticmethod
    def _capturing(app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.CAPTURING
        orch._begin_transaction(trigger="auto")
        orch.detector.lock_document(_document_frame(), allow_fallback=True)
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch.detector.is_scene_empty = MagicMock(return_value=False)
        orch.detector.structural_motion_in_lock = MagicMock(return_value=100.0)
        return orch

    @pytest.mark.asyncio
    async def test_apparent_motion_restabilizes_instead_of_failing(self, app_config):
        orch = self._capturing(app_config)

        proceed = await orch._check_motion_abort()

        assert proceed is False
        assert orch.sm.state == AppState.WAITING_FOR_STABILITY
        assert orch._consecutive_failures == 0
        assert orch._transaction is not None
        assert orch._motion_retries == 1

    @pytest.mark.asyncio
    async def test_persistent_motion_eventually_fails_the_capture(self, app_config):
        orch = self._capturing(app_config)
        orch._motion_retries = orch.MAX_MOTION_RESTABILIZE

        proceed = await orch._check_motion_abort()

        assert proceed is False
        assert orch.sm.state == AppState.IDLE
        assert orch._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_still_page_goes_on_to_capture(self, app_config):
        orch = self._capturing(app_config)
        orch.detector.structural_motion_in_lock = MagicMock(return_value=1.0)

        assert await orch._check_motion_abort() is True
        assert orch.sm.state == AppState.CAPTURING


class TestLedGuidanceInFlow:
    @pytest.mark.asyncio
    async def test_repeated_failures_with_document_present_show_remove_hint_until_cleared(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.IDLE
        orch._awaiting_clear_scene = True
        orch._consecutive_failures = orch.MAX_AUTO_RETRIES_WITH_DOCUMENT
        orch.detector.update_baseline(_empty_frame())
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        assert orch.led._hint is None

        await orch._idle_tick()
        assert orch.led._hint is not None  # tell the user to take the document away

        orch._safe_preview = AsyncMock(return_value=_empty_frame())
        for _ in range(12):  # the detector releases its latch after a few empty frames
            await orch._idle_tick()
        assert orch.led._hint is None  # removed: back to the ready colour


class TestFocusAndTimingIntegration:
    @pytest.mark.asyncio
    async def test_page_moving_during_focus_waits_for_stability_without_a_failure(self, app_config):
        from idcam.exceptions import FocusMotionError

        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._begin_transaction(trigger="auto")
        orch.detector.lock_document(_document_frame(), allow_fallback=True)
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch.detector.is_scene_empty = MagicMock(return_value=False)
        orch.autofocus.run_autofocus = AsyncMock(side_effect=FocusMotionError("moving"))

        await orch._focusing_tick()

        assert orch.sm.state == AppState.WAITING_FOR_STABILITY
        assert orch._consecutive_failures == 0
        assert orch._transaction is not None

    @pytest.mark.asyncio
    async def test_persistent_motion_during_focus_eventually_fails_and_drops_the_lens_memory(self, app_config):
        from idcam.exceptions import FocusMotionError

        orch = _orch(app_config)
        orch.autofocus.remember_lens(8.4)
        orch.sm.state = AppState.FOCUSING
        orch._begin_transaction(trigger="auto")
        orch._motion_retries = orch.MAX_MOTION_RESTABILIZE
        orch.detector.lock_document(_document_frame(), allow_fallback=True)
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch.detector.is_scene_empty = MagicMock(return_value=False)
        orch.autofocus.run_autofocus = AsyncMock(side_effect=FocusMotionError("moving"))

        await orch._focusing_tick()

        assert orch.sm.state == AppState.IDLE
        assert orch._consecutive_failures == 1
        assert orch.autofocus.remembered_lens is None

    @pytest.mark.asyncio
    async def test_focus_failure_drops_the_remembered_lens(self, app_config):
        from idcam.exceptions import AutofocusError

        orch = _orch(app_config)
        orch.autofocus.remember_lens(8.4)
        orch.sm.state = AppState.FOCUSING
        orch._begin_transaction(trigger="auto")
        orch.detector.lock_document(_document_frame(), allow_fallback=True)
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch.detector.is_scene_empty = MagicMock(return_value=False)
        orch.autofocus.run_autofocus = AsyncMock(side_effect=AutofocusError("no peak"))

        await orch._focusing_tick()

        assert orch.autofocus.remembered_lens is None

    @pytest.mark.asyncio
    async def test_validated_capture_remembers_its_lens_and_a_rejected_one_forgets_it(self, app_config, tmp_path):
        from idcam.exceptions import ImageValidationError

        orch = _orch(app_config)
        orch.sm.state = AppState.IMAGE_VALIDATION
        orch._begin_transaction(trigger="auto")
        orch._capture_path = tmp_path / "c.jpg"
        orch._capture_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 500)
        orch._last_focus_lens = 8.42
        orch.validator.validate = MagicMock(
            return_value=MagicMock(file_bytes=500, width=100, height=80, sha256="ab", sharpness=90.0)
        )

        await orch._validation_tick()

        assert orch.sm.state == AppState.UPLOADING
        assert orch.autofocus.remembered_lens == 8.42
        assert orch._txn_validated_ms is not None

        orch2 = _orch(app_config)
        orch2.autofocus.remember_lens(8.42)
        orch2.sm.state = AppState.IMAGE_VALIDATION
        orch2._begin_transaction(trigger="auto")
        orch2._capture_path = tmp_path / "d.jpg"
        orch2._capture_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 500)
        orch2.validator.validate = MagicMock(side_effect=ImageValidationError("blurry"))

        await orch2._validation_tick()

        assert orch2.autofocus.remembered_lens is None

    def test_timing_summary_reports_each_stage_from_a_monotonic_clock(self, app_config, caplog):
        orch = _orch(app_config)
        orch._txn_marks = [
            ("DETECTED", 100.0),
            ("DOCUMENT_DETECTED", 100.0),
            ("WAITING_FOR_STABILITY", 100.05),
            ("FOCUSING", 101.75),
            ("CAPTURING", 108.2),
            ("IMAGE_VALIDATION", 110.6),
            ("UPLOADING", 111.2),
        ]
        orch._txn_detect_latency_ms = 700
        orch._txn_validated_ms = 11300

        with caplog.at_level("INFO"), patch("idcam.app.orchestrator.time.monotonic", return_value=112.0):
            orch._log_timing_summary("success")

        line = next(r.getMessage() for r in caplog.records if "event=timing_summary" in r.getMessage())
        for expected in (
            "outcome=success",
            "detect_ms=700",
            "stability_ms=1700",
            "focus_ms=6450",
            "capture_ms=2400",
            "validation_ms=600",
            "upload_ms=800",
            "total_ms=12000",
            "end_to_end_ms=12700",
            "detect_to_validated_ms=11300",
        ):
            assert expected in line, expected
        assert orch._txn_marks == []  # logged once per transaction

    def test_timing_summary_without_a_transaction_logs_nothing(self, app_config, caplog):
        orch = _orch(app_config)
        with caplog.at_level("INFO"):
            orch._log_timing_summary("success")
        assert "timing_summary" not in caplog.text

    def test_detector_measures_how_long_a_document_took_to_be_detected(self):
        from idcam.config.schema import DetectionConfig
        from idcam.vision.document_detector import DocumentDetector

        det = DocumentDetector(DetectionConfig(min_geometry_confidence=0.25, detect_frames_required=2))
        assert det.detection_latency_ms == 0
        det.detect(_document_frame())
        assert det._first_seen_at is not None
        assert det.detection_latency_ms >= 0
        det.detect(_empty_frame())
        assert det.detection_latency_ms == 0

    def test_loop_stats_are_logged_once_a_minute_and_reset(self, app_config, caplog):
        orch = _orch(app_config)
        orch._loop_window_start = time.monotonic() - 61.0
        orch._loop_ticks = 300
        orch._loop_tick_ms = 300 * 90.0
        orch._loop_tick_max_ms = 400.0

        with caplog.at_level("INFO"):
            orch._log_loop_stats(0.2)

        line = next(r.getMessage() for r in caplog.records if "event=loop_stats" in r.getMessage())
        assert "avg_tick_ms=90" in line and "max_tick_ms=400" in line and "target_fps=5.0" in line
        assert orch._loop_ticks == 0

        caplog.clear()
        with caplog.at_level("INFO"):
            orch._log_loop_stats(0.2)  # inside the new window: silent
        assert "loop_stats" not in caplog.text


class TestFocusMemoryWiring:
    @pytest.mark.asyncio
    async def test_a_successful_focus_is_remembered_immediately(self, app_config):
        orch = _orch(app_config)
        orch.sm.state = AppState.FOCUSING
        orch._begin_transaction(trigger="auto")
        orch.detector.lock_document(_document_frame(), allow_fallback=True)
        orch._safe_preview = AsyncMock(return_value=_document_frame())
        orch.detector.is_scene_empty = MagicMock(return_value=False)
        orch.autofocus.run_autofocus = AsyncMock(
            return_value=AutofocusResult(True, 90.0, 1, None, "hardware_cycle+refine", 10.0, lens_after=8.37)
        )

        await orch._focusing_tick()

        assert orch.sm.state == AppState.CAPTURING
        assert orch.autofocus.remembered_lens == 8.37

    def test_pre_capture_motion_check_uses_the_relaxed_stability_threshold(self, app_config):
        orch = _orch(app_config)
        base = app_config.detection.motion_threshold
        assert orch.stability.effective_motion_threshold == pytest.approx(base / 0.36)
