"""Application orchestrator — single authority for state and transactions."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import threading
import time
from pathlib import Path

from PIL import Image

from idcam import __version__
from idcam.app.state_machine import AppState, StateMachine
from idcam.app.transaction import ActiveCaptureTransaction, TransactionPhase
from idcam.camera.autofocus import AutofocusController
from idcam.camera.capture import CaptureService
from idcam.camera.manager import CameraManager
from idcam.camera.lens_search import find_sharpest_lens
from idcam.camera.manual_focus import ManualFocusStore, valid_lens
from idcam.camera.quality import ImageValidator
from idcam.camera.upload_image import prepare_upload_image
from idcam.config.schema import AppConfig
from idcam.exceptions import (
    AutofocusError,
    CameraError,
    CameraUnavailableError,
    CaptureError,
    FocusMotionError,
    ImageValidationError,
    NetworkError,
    ProtocolError,
    RecoveryExhaustedError,
    ServerRejectedError,
    ServerError,
    ShutdownRequested,
)
from idcam.logging.setup import get_logger, set_log_context
from idcam.network.api_client import ApiClient
from idcam.network.protocol import UploadOutcome, UploadRequest
from idcam.preview.live_viewer import LiveDebugViewer
from idcam.storage.disk_monitor import DiskMonitor, StorageHealth
from idcam.storage.paths import ensure_writable_directory
from idcam.storage.temp_files import TempFileStore
from idcam.system.component_health import ComponentStatus, HealthLevel, SystemHealth
from idcam.system.watchdog import Watchdog
from idcam.ui.led import LedIndicator
from idcam.ui.status import StatusPublisher
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.sharpness import scale_preview_roi_to_image
from idcam.vision.stability_detector import StabilityDetector

logger = get_logger("orchestrator")

# States that run a single long-lived action per state entry (no re-entry per loop tick).
ONE_SHOT_STATES = frozenset({
    AppState.DOCUMENT_DETECTED,
    AppState.FOCUSING,
    AppState.FOCUS_VERIFICATION,
    AppState.CAPTURING,
    AppState.IMAGE_VALIDATION,
    AppState.UPLOADING,
    AppState.WAITING_FOR_SERVER,
    AppState.PROCESSING_RESPONSE,
    AppState.SUCCESS,
})


class ApplicationOrchestrator:
    """
    Main application loop with explicit state machine.

    Exactly one capture transaction at a time. No queue.
    """

    # Failed attempts retried automatically while the same document stays in view.
    MAX_AUTO_RETRIES_WITH_DOCUMENT = 2
    # Consecutive empty-scene frames before a locked document counts as removed.
    DOCUMENT_LOSS_FRAMES = 3
    # Times the page may seem to move right before the shutter before the capture is given up.
    MAX_MOTION_RESTABILIZE = 3

    def __init__(self, config: AppConfig, *, use_mock_camera: bool = False) -> None:
        self.config = config
        self.sm = StateMachine()
        self.sm.set_timeouts(config.state_timeouts)
        self.health = SystemHealth()
        self.status = StatusPublisher(config.ui.status_file, version=__version__)
        self.camera = CameraManager(config.camera, use_mock=use_mock_camera)
        self.autofocus = AutofocusController(config.camera, config.quality)
        # Manual focus (camera.focus_mode: manual): autofocus off, lens value set in the debug panel.
        self._manual_focus = config.camera.focus_mode == "manual"
        self._finding_focus = False  # the panel's "Find best focus" search is running
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self.manual_focus_store = ManualFocusStore(
            config.camera.manual_focus_file or (config.temp_dir.parent / "manual_focus.json")
        )
        self.temp_store = TempFileStore(config.temp_dir)
        self.capture_service = CaptureService(config.camera, self.temp_store)
        self.validator = ImageValidator(config.quality, config.camera)
        self.detector = DocumentDetector(config.detection)
        self.stability = StabilityDetector(config.detection)
        self.disk = DiskMonitor(config.storage, config.temp_dir)
        self.api = ApiClient(config)
        self.watchdog = Watchdog(config.watchdog, config.state_timeouts)
        self.led = LedIndicator(config.led)
        self._shutdown_event = asyncio.Event()
        self._transaction: ActiveCaptureTransaction | None = None
        self._recovery_attempts = 0
        self._next_recovery_at: float = 0.0
        self._recovery_in_progress = False
        self._removal_count = 0
        self._removal_ticks = 0
        self._state_tick_running = False
        self._state_action_claimed: AppState | None = None
        self._error_backoff_claimed = False
        self._capture_path: Path | None = None
        self._pending_response = None
        self._debug_viewer: LiveDebugViewer | None = None
        self._idle_reject_count = 0
        self._cmd_lock = threading.Lock()
        self._pending_command: str | None = None
        self._command_result: dict | None = None
        self._command_event = threading.Event()
        self._manual_phase = "idle"
        self._manual_message = ""
        self._resume_state = AppState.IDLE
        self._awaiting_clear_scene = False
        self._clear_scene_count = 0
        self._awaiting_removal_log_count = 0
        self._consecutive_failures = 0
        self._presence_misses = 0
        self._motion_retries = 0
        self._last_focus_lens: float | None = None
        # (state name, monotonic time) marks of the active transaction, for the timing summary.
        self._txn_marks: list[tuple[str, float]] = []
        self._txn_detect_latency_ms = 0
        self._txn_validated_ms: int | None = None
        self._loop_window_start = time.monotonic()
        self._loop_ticks = 0
        self._loop_tick_ms = 0.0
        self._loop_tick_max_ms = 0.0
        self._upload_crop_roi: tuple[int, int, int, int] | None = None
        # Cropped/resized upload copy, built while the full image is being validated.
        self._upload_copy_task: asyncio.Future | None = None
        # What the document of the current/last capture looked like, to notice that the next person
        # put a different one in its place without the scene ever being empty.
        self._doc_signature = None
        self._removal_disturbance = 0  # frames in which the picture changed while waiting for the next document
        self._background: set[asyncio.Task] = set()
        self._last_scene_empty = False
        self._last_transition_reason = ""
        self._capture_started = False
        self._upload_started = False
        self._upload_result = ""
        self._manual_capture_requested = False
        self._last_preview_size = (
            int(config.camera.preview_width),
            int(config.camera.preview_height),
        )
        if config.debug.enabled:
            self._debug_viewer = LiveDebugViewer(config.debug)
            self._debug_viewer.set_capture_handler(self.request_manual_capture)
            self._debug_viewer.set_focus_handlers(self.focus_status, self.set_manual_focus, self.find_manual_focus)

    @property
    def has_active_transaction(self) -> bool:
        """True while autofocus/capture/upload is running. Not WAITING_FOR_REMOVAL."""
        return self.sm.in_transaction()

    async def run(self) -> int:
        self._install_signal_handlers()
        try:
            await self._startup()
            await self._main_loop()
        except ShutdownRequested:
            logger.info("Shutdown requested")
        except RecoveryExhaustedError:
            return 1
        except Exception as exc:
            logger.exception("Fatal error: %s", exc)
            self._transition(AppState.ERROR, reason=str(exc))
            return 1
        finally:
            try:
                await self._shutdown()
            finally:
                # Never leave the strip lit if shutdown itself fails.
                self.led.stop()
        return 0

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        self._event_loop = loop
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._request_shutdown)
            except NotImplementedError:
                signal.signal(sig, lambda *_: self._request_shutdown())

    def _request_shutdown(self) -> None:
        logger.info("Signal received, requesting shutdown")
        self._shutdown_event.set()

    async def _startup(self) -> None:
        self.led.start()
        self._transition(AppState.INITIALIZING)
        if self._debug_viewer is not None:
            self._debug_viewer.start()
            self._publish_debug()
        ensure_writable_directory(self.config.temp_dir, "temp")
        removed = self.temp_store.cleanup_stale(self.config.storage.stale_temp_max_age_sec)
        if removed:
            logger.info("Startup: removed %d stale temp files", removed)

        storage_status = self.disk.check()
        self._update_storage_health(storage_status)

        await self.api.start()
        network_ok = await self.api.check_connectivity()
        self.health.network = ComponentStatus(
            level=HealthLevel.HEALTHY if network_ok else HealthLevel.DEGRADED,
            message=None if network_ok else "Network unavailable",
            details={"healthy": network_ok},
        )
        self.health.server = ComponentStatus(
            level=HealthLevel.HEALTHY if network_ok else HealthLevel.DEGRADED,
            details={"configured": True, "reachable": network_ok},
        )

        if self._manual_focus:
            self._load_manual_focus()
        self._transition(AppState.CAMERA_INITIALIZING)
        try:
            await asyncio.to_thread(self.camera.initialize)
        except CameraUnavailableError as exc:
            logger.error("event=camera_init_failed %s", exc)
            self.health.camera = ComponentStatus(
                level=HealthLevel.UNAVAILABLE,
                message=str(exc),
                details={"healthy": False},
            )
            self._transition(AppState.RECOVERY, reason="camera_init_failed")
            return

        caps = self.autofocus.probe_capabilities(self.camera)
        self.health.autofocus_available = caps.hardware_af_available
        self.health.autofocus_degraded = caps.degraded
        self.health.camera = ComponentStatus(
            level=HealthLevel.HEALTHY,
            details={
                "healthy": True,
                "autofocus_available": caps.hardware_af_available,
                "autofocus_degraded": caps.degraded,
            },
        )
        self.health.application = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})

        self._transition(AppState.CAMERA_READY)
        if storage_status.health != StorageHealth.HEALTHY:
            self._transition(AppState.ERROR, reason="storage_degraded")
            return
        self._transition(AppState.IDLE)
        logger.info("Application ready")

    async def _main_loop(self) -> None:
        frame_interval = 1.0 / self.config.camera.preview_fps

        while not self._shutdown_event.is_set():
            started = time.monotonic()
            state_before = self.sm.state
            self._refresh_health()
            alert = self.watchdog.check(
                self.sm,
                capture_id=self._transaction.capture_id if self._transaction else None,
            )
            if alert is not None:
                await self._handle_watchdog(alert)
            elif not await self._handle_state_timeout():
                await self._dispatch_state_tick()

            tick_ms = (time.monotonic() - started) * 1000.0
            self._loop_ticks += 1
            self._loop_tick_ms += tick_ms
            self._loop_tick_max_ms = max(self._loop_tick_max_ms, tick_ms)
            self._log_loop_stats(frame_interval)

            # Fixed-rate loop: the old unconditional sleep(frame_interval) came on top of the
            # tick's own work (~0.2 s + 0.5 s detection per step). Sleep only the remainder,
            # and not at all right after a state change: the next state fetches its own frame.
            if self.sm.state != state_before:
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(max(0.0, frame_interval - (time.monotonic() - started)))

    async def _handle_state_timeout(self) -> bool:
        if not self.sm.is_timed_out():
            return False
        state = self.sm.state
        logger.warning(
            "event=state_timeout state=%s elapsed=%.1fs capture_id=%s",
            state.value,
            self.sm.elapsed(),
            self._transaction.capture_id if self._transaction else None,
        )
        if state == AppState.DOCUMENT_DETECTED:
            self._abort_transaction("document_detected_timeout")
            self._clear_transaction()
            self._transition(AppState.IDLE, reason="document_detected_timeout")
            return True
        if state == AppState.WAITING_FOR_STABILITY:
            await self._fail_transaction_and_wait("stability_timeout", "Document not stable in time")
            return True
        if state in {AppState.FOCUSING, AppState.FOCUS_VERIFICATION}:
            await self._fail_transaction_and_wait("focus_failed", f"Focus timed out in {state.value}")
            return True
        if state in {AppState.UPLOADING, AppState.WAITING_FOR_SERVER, AppState.PROCESSING_RESPONSE}:
            await self._fail_transaction_and_wait("upload_failed", "Server did not respond in time")
            return True
        if state == AppState.SUCCESS:
            self._transition(AppState.WAITING_FOR_REMOVAL)
            return True
        if state == AppState.CAPTURING:
            await self._fail_transaction_and_wait("capture_failed", "Capture timed out")
            return True
        if state == AppState.IMAGE_VALIDATION:
            await self._fail_transaction_and_wait("image_invalid", "Image validation timed out")
            return True
        return False

    def _claim_state_action(self) -> bool:
        """Return True if this tick may run the state's action (one-shot per entry)."""
        if self.sm.state not in ONE_SHOT_STATES:
            return True
        if self._state_action_claimed == self.sm.state:
            return False
        self._state_action_claimed = self.sm.state
        return True

    async def _dispatch_state_tick(self) -> None:
        if self._state_tick_running:
            return
        self._state_tick_running = True
        try:
            handlers = {
                AppState.IDLE: self._idle_tick,
                AppState.DOCUMENT_DETECTED: self._document_detected_tick,
                AppState.WAITING_FOR_STABILITY: self._stability_tick,
                AppState.FOCUSING: self._focusing_tick,
                AppState.FOCUS_VERIFICATION: self._focus_verification_tick,
                AppState.CAPTURING: self._capturing_tick,
                AppState.IMAGE_VALIDATION: self._validation_tick,
                AppState.UPLOADING: self._uploading_tick,
                AppState.WAITING_FOR_SERVER: self._waiting_server_tick,
                AppState.PROCESSING_RESPONSE: self._processing_response_tick,
                AppState.SUCCESS: self._success_tick,
                AppState.WAITING_FOR_REMOVAL: self._removal_tick,
                AppState.RECOVERY: self._recovery_tick,
                AppState.ERROR: self._error_tick,
            }
            handler = handlers.get(self.sm.state)
            if handler:
                await handler()
        finally:
            self._state_tick_running = False

    def _note_preview_frame(self, frame) -> None:
        if frame is not None and getattr(frame, "ndim", 0) >= 2:
            h, w = int(frame.shape[0]), int(frame.shape[1])
            self._last_preview_size = (w, h)

    async def _safe_preview(self, *, fresh: bool = False):
        """Preview frame; ``fresh`` = first frame exposed after now (not one queued earlier)."""
        capture = self.camera.capture_preview_array
        if fresh and callable(getattr(self.camera, "capture_fresh_preview_array", None)):
            capture = self.camera.capture_fresh_preview_array
        try:
            return await asyncio.to_thread(capture)
        except CameraError as exc:
            logger.error("event=camera_preview_failed %s", exc)
            self._publish_debug(camera_error=str(exc))
            await self._enter_recovery(str(exc))
            return None

    async def _check_motion_abort(self) -> bool:
        """Return False if locked document moved significantly (not AE noise)."""
        frame = await self._safe_preview(fresh=True)
        if frame is None:
            return False
        if not self.detector.lock.is_locked:
            return True
        if await self._abort_if_document_gone(frame):
            return False
        motion = self.detector.structural_motion_in_lock(frame)
        threshold = self.stability.effective_motion_threshold * 2.5
        self._publish_debug(frame)
        if motion > threshold:
            logger.warning(
                "event=document_moved_during_capture state=%s motion=%.2f threshold=%.2f",
                self.sm.state.value,
                motion,
                threshold,
            )
            self._motion_retries += 1
            if self._motion_retries <= self.MAX_MOTION_RESTABILIZE:
                # Not a failure: wait for the page to settle again, then refocus and retry.
                self.stability.reset()
                self.stability.set_roi(self.detector.lock.roi)
                self._transition(AppState.WAITING_FOR_STABILITY, reason="moved_before_capture")
                return False
            await self._fail_transaction_and_wait("stability_timeout", "Document moved during capture")
            return False
        return True

    async def _idle_tick(self) -> None:
        if self.has_active_transaction or self._finding_focus:
            return
        if not self.disk.can_capture(self.config.storage.capture_reserve_mb):
            self._update_storage_health(self.disk.check())
            return
        frame = await self._safe_preview()
        if frame is None:
            return
        if self._take_manual_request():
            logger.info("event=pipeline_stage stage=manual_capture_start state=IDLE")
            self.stability.reset()
            self._resume_state = AppState.IDLE
            self._awaiting_clear_scene = False
            self._manual_capture_requested = True
            self._begin_transaction(trigger="manual")
            self._set_manual_phase("focusing", "Manual capture started")
            self._publish_debug(frame)
            self._transition(AppState.DOCUMENT_DETECTED, reason="manual_capture")
            return
        detected = await asyncio.to_thread(self.detector.detect, frame)
        self._publish_debug(frame)
        if self._awaiting_clear_scene and not detected:
            self._last_scene_empty = True
            self._clear_scene_count += 1
            if self._clear_scene_count >= self.config.detection.removal_frames_required:
                self._awaiting_clear_scene = False
                self._clear_scene_count = 0
                self._consecutive_failures = 0
                self.led.set_remove_hint(False)
                logger.info("event=auto_capture_armed reason=scene_cleared_after_failure")
            return
        if self._awaiting_clear_scene and self.detector.calm_streak == 0:
            self._removal_disturbance += 1
        if (
            self._awaiting_clear_scene
            and detected
            and self.detector.document_replaced(frame, self._doc_signature, disturbed=self._removal_disturbance >= 2)
        ):
            logger.info("event=document_replaced state=IDLE failures=%d", self._consecutive_failures)
            self._awaiting_clear_scene = False
            self._clear_scene_count = 0
            self._consecutive_failures = 0
            self.led.set_remove_hint(False)
        if self._awaiting_clear_scene and detected:
            if self._consecutive_failures >= self.MAX_AUTO_RETRIES_WITH_DOCUMENT:
                # Retrying on the same paper looped forever without ever asking the user to
                # remove it. After a couple of failed attempts, hold until the scene is clear;
                # the failure_key in status carries "Please remove the document and try again".
                self._clear_scene_count = 0
                self._last_scene_empty = False
                self._awaiting_removal_log_count += 1
                self.led.set_remove_hint(True)
                if self._awaiting_removal_log_count == 1 or self._awaiting_removal_log_count % 100 == 0:
                    logger.warning(
                        "event=waiting_for_removal reason=repeated_failures_document_present failures=%d",
                        self._consecutive_failures,
                    )
                return
            logger.info("event=auto_capture_retry reason=document_still_present_after_failure")
            self._awaiting_clear_scene = False
            self._clear_scene_count = 0
        if detected:
            logger.info(
                "event=document_detected confidence=%.2f area=%.3f roi=%s led=%s motion=%.2f edge_density=%.4f "
                "latency_ms=%d",
                self.detector.debug_info().geometry_confidence,
                getattr(self.detector._last_geometry, "area_fraction", 0.0),
                self.detector.debug_info().document_roi,
                "on" if self.led.config.enabled else "off",
                self.detector.last_motion,
                self.detector.last_edge_density,
                self.detector.detection_latency_ms,
            )
            logger.info("event=pipeline_stage stage=document_detected")
            self.stability.reset()
            self._begin_transaction(trigger="auto")
            self._transition(AppState.DOCUMENT_DETECTED, reason="document_detected")
        else:
            self._idle_reject_count += 1
            if self._idle_reject_count == 1 or self._idle_reject_count % 50 == 0:
                info = self.detector.debug_info()
                logger.info(
                    "event=idle_no_document frames=%d motion=%.2f frame_motion=%.2f "
                    "edge_density=%.4f reason=%s roi=%dx%d frame=%dx%d gray_std=%.1f best_bbox=%s",
                    self._idle_reject_count,
                    info.motion,
                    info.frame_motion,
                    info.edge_density,
                    info.reject_reason,
                    info.roi_width,
                    info.roi_height,
                    info.frame_width,
                    info.frame_height,
                    info.gray_std,
                    info.document_roi,
                )

    def _begin_transaction(self, *, trigger: str = "auto") -> ActiveCaptureTransaction:
        if self._transaction is not None and not self._transaction.is_terminal:
            raise RuntimeError("Cannot begin transaction: one already active")
        if self.sm.state == AppState.WAITING_FOR_REMOVAL:
            self._resume_state = AppState.WAITING_FOR_REMOVAL
        else:
            self._resume_state = AppState.IDLE
        with self._cmd_lock:
            if self._pending_command == "manual_capture":
                self._pending_command = "manual_capture_running"
                trigger = "manual"
                self._manual_phase = "starting"
                self._manual_message = "Manual capture started"
        txn = ActiveCaptureTransaction(capture_id=ActiveCaptureTransaction.generate_id(), trigger=trigger)
        txn.mark_document_detected()
        self._transaction = txn
        self._capture_path = None
        self._idle_reject_count = 0
        self._capture_started = False
        self._upload_started = False
        self._upload_result = ""
        self._presence_misses = 0
        self._motion_retries = 0
        self._last_focus_lens = None
        self._txn_marks = [("DETECTED", time.monotonic())]
        self._txn_detect_latency_ms = self.detector.detection_latency_ms
        self._txn_validated_ms = None
        self._upload_crop_roi = None
        self._upload_copy_task = None
        set_log_context(capture_id=txn.capture_id)
        logger.info("event=transaction_started capture_id=%s trigger=%s", txn.capture_id, trigger)
        return txn

    # -- manual focus (HTTP-thread entry points used by the debug panel) --------------------------

    def _load_manual_focus(self) -> None:
        """Use the confirmed lens position from the last session, else camera.manual_lens_position."""
        saved = self.manual_focus_store.load()
        lens = saved if saved is not None else float(self.config.camera.manual_lens_position)
        self.camera.set_manual_lens(lens)
        logger.info(
            "event=manual_focus_active lens=%.3f source=%s file=%s autofocus=disabled",
            lens,
            "saved" if saved is not None else "config",
            self.manual_focus_store.path,
        )
        if not self.config.debug.enabled:
            logger.warning(
                "event=manual_focus_no_panel hint=set debug.enabled: true to change the focus from the panel"
            )

    def _focus_busy(self) -> bool:
        txn_active = self._transaction is not None and not self._transaction.is_terminal
        return (
            self._finding_focus
            or self.sm.state not in {AppState.IDLE, AppState.WAITING_FOR_REMOVAL}
            or self.has_active_transaction
            or txn_active
        )

    def focus_status(self) -> dict:
        """Focus mode and values for the panel (safe to call from any thread)."""
        if not self._manual_focus:
            return {"mode": "auto"}
        lo, hi = self.camera.lens_position_range()
        return {
            "mode": "manual",
            "lens_position": self.camera.get_manual_lens(),
            "saved_lens": self.manual_focus_store.load(),
            "range": [lo, hi],
            "busy": self._focus_busy(),
        }

    def find_manual_focus(self) -> dict:
        """Panel button: search the sharpest lens position on the document in view and apply it.

        Not saved: the operator checks the result in the preview and confirms with "Confirm & save".
        Blocks the calling (HTTP) thread until the search is done.
        """
        if not self._manual_focus:
            return {"ok": False, "status": "disabled", "message": "Manual focus is off (camera.focus_mode is auto)"}
        loop = self._event_loop
        if loop is None or not loop.is_running():
            return {"ok": False, "status": "error", "message": "The service is not running yet"}
        if self._focus_busy():
            return {"ok": False, "status": "busy", "message": "A capture is running; try again when it is done"}
        self._finding_focus = True
        try:
            future = asyncio.run_coroutine_threadsafe(self._find_manual_focus(), loop)
            return future.result(timeout=150.0)
        except Exception as exc:
            logger.warning("event=manual_focus_search_failed error=%s", exc)
            return {"ok": False, "status": "error", "message": "Focus search failed; see the log"}
        finally:
            self._finding_focus = False

    async def _find_manual_focus(self) -> dict:
        previous = self.camera.get_manual_lens()
        frame = await self._safe_preview()
        if frame is None:
            return {"ok": False, "status": "error", "message": "No camera frame"}
        roi = self.detector.focus_preview_roi(frame)
        lo, hi = self.camera.lens_position_range()
        logger.info("event=manual_focus_search_started roi=%s range=%.1f..%.1f", roi, lo, hi)
        try:
            result = await asyncio.to_thread(find_sharpest_lens, self.camera, roi, lo, hi)
        except AutofocusError as exc:
            logger.warning("event=manual_focus_search_failed error=%s", exc)
            if previous is not None:
                self.camera.set_manual_lens(previous)
            return {"ok": False, "status": "error", "message": f"No sharp position found: {exc}"}
        lens = self.camera.set_manual_lens(result.lens)
        logger.info(
            "event=manual_focus_found lens=%.3f sharpness=%.1f peak_ratio=%.1f samples=%d",
            lens,
            result.sharpness,
            result.peak_ratio,
            result.samples,
        )
        return {
            "ok": True,
            "status": "found",
            "message": "Best position found; check the preview, then press Confirm & save",
            "lens_position": lens,
        }

    def set_manual_focus(self, lens: object, save: bool = False) -> dict:
        """Move the lens to ``lens`` (live preview); with ``save`` also keep it for all later captures."""
        if not self._manual_focus:
            return {"ok": False, "status": "disabled", "message": "Manual focus is off (camera.focus_mode is auto)"}
        value = valid_lens(lens)
        if value is None:
            return {"ok": False, "status": "invalid", "message": "Lens position must be a number from 0 to 32"}
        if self._focus_busy():
            return {"ok": False, "status": "busy", "message": "A capture is running; change the focus when it is done"}
        try:
            value = self.camera.set_manual_lens(value)
        except CameraError as exc:
            return {"ok": False, "status": "error", "message": str(exc)}
        if not save:
            return {"ok": True, "status": "applied", "lens_position": value}
        try:
            self.manual_focus_store.save(value)
        except (OSError, ValueError) as exc:
            logger.error("event=manual_focus_save_failed error=%s", exc)
            return {"ok": False, "status": "save_failed", "lens_position": value, "message": f"Could not save: {exc}"}
        return {"ok": True, "status": "saved", "lens_position": value, "saved_lens": value}

    def request_manual_capture(self, timeout: float = 180.0) -> dict:
        """HTTP-thread entry: queue a capture through the existing state machine."""
        state = self.sm.state
        txn_active = self._transaction is not None and not self._transaction.is_terminal
        allowed = {AppState.IDLE, AppState.WAITING_FOR_REMOVAL}
        if state not in allowed or self.has_active_transaction or txn_active:
            logger.info("event=manual_capture_rejected status=busy state=%s", state.value)
            return {
                "ok": False,
                "status": "busy",
                "state": state.value,
                "message": f"Busy ({state.value})",
            }
        if not self.health.camera.healthy:
            return {
                "ok": False,
                "status": "busy",
                "state": state.value,
                "message": "Camera is not healthy",
            }
        with self._cmd_lock:
            if self._pending_command is not None:
                return {
                    "ok": False,
                    "status": "busy",
                    "state": state.value,
                    "message": "Capture already in progress",
                }
            self._pending_command = "manual_capture"
            self._command_result = None
            self._command_event.clear()
            self._manual_phase = "queued"
            self._manual_message = "Waiting for capture pipeline"
            self._manual_capture_requested = True
        logger.info("event=pipeline_stage stage=manual_capture_queued")
        if not self._command_event.wait(timeout):
            with self._cmd_lock:
                if self._pending_command == "manual_capture":
                    self._pending_command = None
                self._manual_phase = "timeout"
                self._manual_message = "Timed out waiting for capture"
            logger.warning("event=manual_capture_timeout")
            return {"ok": False, "status": "timeout", "message": "Timed out waiting for capture"}
        with self._cmd_lock:
            result = dict(self._command_result or {"ok": False, "status": "error", "message": "No result"})
        return result

    def _take_manual_request(self) -> bool:
        with self._cmd_lock:
            if self._pending_command == "manual_capture":
                self._pending_command = "manual_capture_running"
                self._manual_phase = "starting"
                return True
            return False

    def _set_manual_phase(self, phase: str, message: str = "") -> None:
        with self._cmd_lock:
            if self._pending_command in {"manual_capture", "manual_capture_running"}:
                self._manual_phase = phase
                if message:
                    self._manual_message = message

    def _finish_manual_request(self, result: dict) -> None:
        with self._cmd_lock:
            if self._pending_command not in {"manual_capture", "manual_capture_running"}:
                return
            self._command_result = result
            self._pending_command = None
            self._manual_phase = "success" if result.get("ok") else "failed"
            self._manual_message = str(result.get("message") or "")
            self._command_event.set()

    def _manual_status(self) -> tuple[bool, str, str]:
        with self._cmd_lock:
            txn_active = self._transaction is not None and not self._transaction.is_terminal
            available = (
                self.health.camera.healthy
                and self._pending_command is None
                and not self.has_active_transaction
                and not txn_active
                and self.sm.state in {AppState.IDLE, AppState.WAITING_FOR_REMOVAL}
            )
            return available, self._manual_phase, self._manual_message

    async def _document_detected_tick(self) -> None:
        if not self._claim_state_action():
            return
        frame = await self._safe_preview()
        manual = bool(self._transaction and self._transaction.trigger == "manual")
        if frame is not None:
            locked = self.detector.lock_document(frame, allow_fallback=manual)
            if not locked:
                # Nothing real to lock: go back to waiting for a document instead of
                # stabilising/focusing/capturing an empty scene.
                self._abort_transaction("no_document_geometry")
                self._clear_transaction()
                self._transition(AppState.IDLE, reason="document_not_confirmed")
                return
            self.stability.set_roi(self.detector.lock.roi)
            self._doc_signature = self.detector.document_signature(frame)
            self._publish_debug(frame)
            logger.info("event=pipeline_stage stage=document_locked roi=%s", self.detector.lock.roi)
        if manual:
            if self._manual_focus:
                await self._capture_with_manual_focus("manual_capture_skip_stability")
                return
            self._set_manual_phase("focusing", "Focusing...")
            self._transition(AppState.FOCUSING, reason="manual_capture_skip_stability")
            return
        self._transition(AppState.WAITING_FOR_STABILITY, reason="auto_wait_stability")

    async def _abort_if_document_gone(self, frame) -> bool:
        """Back to IDLE (no failure counted) when the locked document has left the scene.

        The pipeline must only continue while a document is actually in view: stability,
        focus and capture all treat an empty table as perfectly "stable".
        """
        if self._transaction is None or self._transaction.trigger == "manual":
            return False
        empty = await asyncio.to_thread(self.detector.is_scene_empty, frame)
        self._presence_misses = self._presence_misses + 1 if empty else 0
        if self._presence_misses < self.DOCUMENT_LOSS_FRAMES:
            return False
        logger.warning("event=document_lost state=%s frames=%d", self.sm.state.value, self._presence_misses)
        self._presence_misses = 0
        self._log_timing_summary("document_removed")
        self._abort_transaction("document_removed")
        self._clear_transaction()
        self._transition(AppState.IDLE, reason="document_removed_before_capture")
        return True

    async def _stability_tick(self) -> None:
        frame = await self._safe_preview()
        if frame is None:
            return
        if await self._abort_if_document_gone(frame):
            return
        # A page held in a hand is judged by how far its corners move, not by the picture difference.
        corners = await asyncio.to_thread(self.detector.current_corners, frame) if self.stability.hand_present else None
        if self.stability.update(frame, corners=corners):
            self._publish_debug(frame)
            logger.info(
                "event=pipeline_stage stage=stability_confirmed frames=%d motion=%.2f",
                self.stability.stable_count,
                self.stability.debug_info().motion,
            )
            assert self._transaction is not None
            self._transaction.phase = TransactionPhase.STABILIZING
            if await asyncio.to_thread(self.detector.refresh_lock, frame):
                # the page was swapped (or moved a lot) while waiting: lock the one that is there now
                logger.info("event=lock_refreshed roi=%s", self.detector.lock.roi)
                self.stability.set_roi(self.detector.lock.roi)
            self._doc_signature = await asyncio.to_thread(self.detector.document_signature, frame)
            if self._manual_focus:
                await self._capture_with_manual_focus("document_stable")
                return
            self._set_manual_phase("focusing", "Focusing...")
            self._transition(AppState.FOCUSING, reason="document_stable")
        else:
            self._publish_debug(frame)
            self.sm.touch_progress()

    async def _capture_with_manual_focus(self, reason: str) -> None:
        """Manual focus mode: there is nothing to focus, so FOCUSING is never entered.

        The lens already sits on the confirmed value (the preview keeps it there); go straight to
        CAPTURING after re-measuring the locked outline on a settled frame.
        """
        assert self._transaction is not None
        self._last_focus_lens = self.camera.get_manual_lens()
        logger.info("event=pipeline_stage stage=focus_skipped mode=manual lens=%s", self._last_focus_lens)
        settled = await self._safe_preview(fresh=True)
        if settled is not None:
            await asyncio.to_thread(self.detector.rebaseline_motion, settled)
        self._transition(AppState.CAPTURING, reason=reason)

    async def _focusing_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        if self._shutdown_event.is_set():
            return
        # Lens motion during AF must not be treated as the document moving.
        frame = await self._safe_preview()
        if frame is None:
            return
        if await self._abort_if_document_gone(frame):
            return
        preview_roi = self.detector.focus_preview_roi(frame)
        self._publish_debug(frame)
        self._transaction.mark_focus_started()
        self._transaction.focus_attempts += 1
        self._set_manual_phase("focusing", "Focusing...")
        try:
            result = await self.autofocus.run_autofocus(
                self.camera,
                roi_fraction=self.config.detection.roi_center_fraction,
                preview_roi=preview_roi,
            )
            self._transaction.mark_focus_completed(result.sharpness, result.min_required_sharpness)
            self._last_focus_lens = result.lens_after
            # The last successful focus is the starting point for the next document; it is dropped
            # again if the image taken with it is rejected, or if a later focus fails.
            self.autofocus.remember_lens(result.lens_after)
            self.sm.touch_progress()
            logger.info(
                "event=pipeline_stage stage=focus_verified method=%s sharpness=%.1f ratio=%.2f",
                result.method,
                result.sharpness,
                result.improvement_ratio,
            )
            settled = await self._safe_preview(fresh=True)
            if settled is not None:
                await asyncio.to_thread(self.detector.rebaseline_motion, settled)
            self._transition(AppState.CAPTURING, reason="focus_verified")
        except FocusMotionError as exc:
            # The page or camera moved while focusing: not a failure, wait for it to settle again.
            logger.warning("event=focus_motion %s", exc)
            self._motion_retries += 1
            if self._motion_retries <= self.MAX_MOTION_RESTABILIZE:
                self.stability.reset()
                self.stability.set_roi(self.detector.lock.roi)
                self._transition(AppState.WAITING_FOR_STABILITY, reason="moved_during_focus")
                return
            self.autofocus.forget_lens("focus_motion")
            await self._fail_transaction_and_wait("focus_failed", str(exc))
        except AutofocusError as exc:
            logger.error("event=autofocus_failed %s", exc)
            self.autofocus.forget_lens("focus_failed")
            await self._fail_transaction_and_wait("focus_failed", str(exc))

    async def _focus_verification_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        self._transaction.focus_verify_attempts += 1
        preview_roi = self.detector.focus_preview_roi()
        try:
            result = await self.autofocus.verify_focus(
                self.camera,
                roi_fraction=self.config.detection.roi_center_fraction,
                preview_roi=preview_roi,
            )
            self._transaction.mark_focus_completed(result.sharpness, result.min_required_sharpness)
            self.sm.touch_progress()
            self._transition(AppState.CAPTURING, reason="focus_verified")
        except AutofocusError:
            if self._transaction.focus_verify_attempts < self.config.camera.af_verify_max_retries:
                self._transition(AppState.FOCUSING, reason="refocus")
            else:
                await self._fail_transaction_and_wait("focus_failed", "Focus verification failed")

    async def _capturing_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        if not await self._check_motion_abort():
            return
        if not self.disk.can_capture(self.config.storage.capture_reserve_mb):
            await self._fail_transaction_and_wait("storage_full", "Insufficient disk space")
            return
        # The still takes a few seconds; connect to the server meanwhile.
        self._background.add(task := asyncio.create_task(self.api.prewarm()))
        task.add_done_callback(self._background.discard)
        output_path = self.temp_store.create_temp_path()
        self._transaction.mark_capture_started(str(output_path))
        self._capture_started = True
        self._set_manual_phase("capturing", "Capturing...")
        logger.info("event=pipeline_stage stage=capturing path=%s", output_path)
        try:
            self._capture_path = await self.capture_service.capture(self.camera, output_path)
            self.sm.touch_progress()
            self._transition(AppState.IMAGE_VALIDATION)
        except CaptureError as exc:
            logger.error("event=capture_failed %s", exc)
            self.temp_store.cleanup(output_path)
            msg = str(exc)
            if any(
                token in msg.lower()
                for token in ("camera not started", "failed to restore preview", "camera unavailable")
            ):
                await self._enter_recovery(msg)
            else:
                await self._fail_transaction_and_wait("capture_failed", msg)

    async def _validation_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        assert self._capture_path is not None
        self._set_manual_phase("validating", "Validating image...")
        logger.info("event=pipeline_stage stage=image_validation path=%s", self._capture_path)
        try:
            focus_roi = self.detector.focus_preview_roi()
            doc_roi = self.detector.document_preview_roi()
            preview_size = self._last_preview_size
            scaler_crop = self.camera.current_scaler_crop()
            image_crop = getattr(self.camera, "last_still_scaler_crop", None)
            self._start_upload_copy(self._capture_path, doc_roi, preview_size, scaler_crop, image_crop)
            result = await asyncio.to_thread(
                self.validator.validate,
                self._capture_path,
                document_roi=focus_roi,
                preview_size=preview_size,
                scaler_crop=scaler_crop,
                image_crop=image_crop,
                reference_sharpness=self._transaction.sharpness_score or None,
            )
            self._transaction.mark_validated(
                size=result.file_bytes,
                width=result.width,
                height=result.height,
                sha256=result.sha256,
                sharpness=result.sharpness,
            )
            logger.info(
                "event=image_validated sha256=%s sharpness=%.1f doc_sharpness=%.1f bytes=%d",
                result.sha256[:16],
                result.sharpness,
                result.document_sharpness,
                result.file_bytes,
            )
            self.sm.touch_progress()
            self.autofocus.remember_lens(self._last_focus_lens)  # a validated lens position
            if self._txn_marks:
                self._txn_validated_ms = int(
                    (time.monotonic() - self._txn_marks[0][1]) * 1000 + self._txn_detect_latency_ms
                )
            self._transition(AppState.UPLOADING)
        except ImageValidationError as exc:
            logger.error("event=image_validation_failed %s", exc)
            await self._discard_upload_copy()
            self.autofocus.forget_lens("image_invalid")
            self._keep_rejected_for_debug(self._capture_path)
            self.temp_store.cleanup(self._capture_path)
            await self._fail_transaction_and_wait("image_invalid", str(exc))

    def _keep_rejected_for_debug(self, path: Path) -> None:
        """Opt-in (IDCAM_KEEP_REJECTED=1): keep the last 5 rejected images to inspect focus.

        Off by default: captured images may be identity documents and must not linger on disk.
        """
        if os.environ.get("IDCAM_KEEP_REJECTED", "").lower() not in ("1", "true"):
            return
        try:
            dest_dir = self.config.temp_dir.parent / "rejected"
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_dir / f"rejected-{int(time.time())}.jpg")
            for old in sorted(dest_dir.glob("rejected-*.jpg"))[:-5]:
                old.unlink(missing_ok=True)
            logger.warning("event=rejected_image_kept dir=%s", dest_dir)
        except OSError:
            logger.warning("event=rejected_image_keep_failed", exc_info=True)

    def _start_upload_copy(self, path: Path, doc_roi, preview_size, scaler_crop, image_crop=None) -> None:
        """Start building the upload copy in a worker thread while the image is validated.

        The crop box only needs the image size, which the JPEG header gives at once, so the two
        jobs (both read-only on the capture) overlap instead of running one after the other.
        """
        cfg = self.config.upload_image
        self._upload_copy_task = None
        self._upload_crop_roi = None
        try:
            with Image.open(path) as img:
                size = img.size
            self._upload_crop_roi = scale_preview_roi_to_image(
                doc_roi, preview_size, size, scaler_crop=scaler_crop, image_crop=image_crop
            )
        except Exception:
            logger.warning("event=upload_crop_roi_failed", exc_info=True)
        if not cfg.crop_to_document and not cfg.max_dimension:
            return
        roi = self._upload_crop_roi
        self._upload_copy_task = asyncio.ensure_future(asyncio.to_thread(self._make_upload_copy, path, roi))

    async def _discard_upload_copy(self) -> None:
        task, self._upload_copy_task = self._upload_copy_task, None
        if task is None:
            return
        try:
            prepared = await task
        except Exception:
            return
        if prepared is not None:
            self.temp_store.cleanup(prepared.path)

    def _make_upload_copy(self, original: Path, crop_roi):
        """Cropped, size-limited copy of ``original`` (None on any problem: the original is sent)."""
        dst = self.temp_store.create_temp_path()
        started = time.monotonic()
        try:
            prepared = prepare_upload_image(original, dst, crop_roi, self.config.upload_image)
        except Exception:
            logger.warning("event=upload_image_prepare_failed using=original", exc_info=True)
            self.temp_store.cleanup(dst)
            return None
        logger.info(
            "event=upload_image_prepared source=%dx%d output=%dx%d cropped=%s bytes_before=%d bytes_after=%d "
            "duration_ms=%d",
            prepared.source_width,
            prepared.source_height,
            prepared.width,
            prepared.height,
            prepared.cropped,
            original.stat().st_size,
            prepared.file_bytes,
            int((time.monotonic() - started) * 1000),
        )
        return prepared

    async def _use_upload_copy(self) -> None:
        """Send the prepared copy instead of the full-resolution capture (built meanwhile)."""
        assert self._capture_path is not None
        task, self._upload_copy_task = self._upload_copy_task, None
        if task is None:
            return
        prepared = await task
        if prepared is None:
            return
        self.temp_store.cleanup(self._capture_path)
        self._capture_path = prepared.path

    async def _uploading_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        assert self._capture_path is not None
        self._transaction.mark_upload_started()
        self._upload_started = True
        self._upload_result = "started"
        self._set_manual_phase("uploading", "Uploading...")
        logger.info(
            "event=pipeline_stage stage=uploading capture_id=%s path=%s bytes=%s sha256=%s",
            self._transaction.capture_id,
            self._capture_path,
            self._transaction.image_size,
            (self._transaction.image_sha256 or "")[:16],
        )
        await self._use_upload_copy()
        request = UploadRequest(
            capture_id=self._transaction.capture_id,
            device_id=self.config.device_id,
            image_path=str(self._capture_path),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        try:
            self._pending_response = await self.api.upload_document(request)
            self._transaction.mark_upload_completed(self._pending_response.http_status)
            self._upload_result = f"http_{self._pending_response.http_status}"
            self.sm.touch_progress()
            self._transition(AppState.WAITING_FOR_SERVER, reason="upload_sent")
        except (ServerRejectedError, ServerError, ProtocolError) as exc:
            self.temp_store.cleanup(self._capture_path)
            key = "server_rejected" if isinstance(exc, ServerRejectedError) else "upload_failed"
            self._upload_result = key
            await self._fail_transaction_and_wait(key, str(exc))
        except NetworkError as exc:
            self.temp_store.cleanup(self._capture_path)
            self._upload_result = "upload_failed"
            await self._fail_transaction_and_wait("upload_failed", str(exc))

    async def _waiting_server_tick(self) -> None:
        if not self._claim_state_action():
            return
        self._transition(AppState.PROCESSING_RESPONSE)

    async def _processing_response_tick(self) -> None:
        if not self._claim_state_action():
            return
        assert self._transaction is not None
        response = self._pending_response
        logger.info(
            "event=pipeline_stage stage=server_response success=%s outcome=%s http_status=%s capture_id=%s",
            bool(response and response.success),
            response.outcome.value if response else "none",
            response.http_status if response else None,
            self._transaction.capture_id,
        )
        if response and response.success and response.outcome == UploadOutcome.ACCEPTED:
            self.health.record_success()
            self.health.server = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True, "reachable": True})
            self._transaction.mark_completed()
            self.temp_store.cleanup(self._capture_path)
            logger.info("event=pipeline_stage stage=success capture_id=%s", self._transaction.capture_id)
            self._log_timing_summary("success")
            self._finish_manual_request(
                {
                    "ok": True,
                    "status": "success",
                    "state": AppState.SUCCESS.value,
                    "capture_id": self._transaction.capture_id,
                    "message": "Upload accepted",
                }
            )
            self._upload_result = "accepted"
            self._transition(AppState.SUCCESS, reason="upload_accepted")
        else:
            msg = response.message if response else "Upload failed"
            self.health.record_failure(msg)
            self.temp_store.cleanup(self._capture_path)
            failure_key = "upload_failed"
            if response and response.outcome == UploadOutcome.REJECTED:
                failure_key = "server_rejected"
            self._upload_result = failure_key
            await self._fail_transaction_and_wait(failure_key, msg or "Upload failed")

    async def _success_tick(self) -> None:
        if not self._claim_state_action():
            return
        await asyncio.sleep(0.8)  # long enough to see the green; the next person must not wait
        self._consecutive_failures = 0
        self._removal_count = 0
        self._removal_ticks = 0
        self._removal_disturbance = 0
        self._last_scene_empty = False
        self._transition(AppState.WAITING_FOR_REMOVAL, reason="capture_success")

    async def _removal_tick(self) -> None:
        frame = await self._safe_preview()
        if frame is None:
            return
        if self._take_manual_request():
            logger.info("event=pipeline_stage stage=manual_capture_start state=WAITING_FOR_REMOVAL")
            self.stability.reset()
            self._begin_transaction(trigger="manual")
            self._manual_capture_requested = True
            self._set_manual_phase("focusing", "Manual capture started")
            self._publish_debug(frame)
            self._transition(AppState.DOCUMENT_DETECTED, reason="manual_capture")
            return
        empty = self.detector.is_scene_empty(frame)
        self._last_scene_empty = empty
        if self.detector.calm_streak == 0:
            self._removal_disturbance += 1
        if not empty and self.detector.document_replaced(
            frame, self._doc_signature, disturbed=self._removal_disturbance >= 2
        ):
            logger.info(
                "event=document_replaced state=WAITING_FOR_REMOVAL waited_ms=%d count=%d disturbed=%d",
                int(self.sm.elapsed() * 1000),
                self._removal_count,
                self._removal_disturbance,
            )
            self._clear_transaction()
            self._awaiting_clear_scene = False
            self._consecutive_failures = 0
            self.led.set_remove_hint(False)
            self.stability.reset()
            self._begin_transaction(trigger="auto")
            self._publish_debug(frame)
            self._transition(AppState.DOCUMENT_DETECTED, reason="document_replaced")
            return
        if empty:
            self._removal_count += 1
        else:
            # One noisy frame (a hand passing, a flicker near the threshold) must not restart the
            # whole count; a page that is really still there keeps it at zero.
            self._removal_count = max(0, self._removal_count - 1)
            self.sm.touch_progress()
        self._removal_ticks += 1
        if self._removal_ticks % 25 == 1:
            logger.info(
                "event=waiting_for_removal_progress empty=%s count=%d/%d confidence=%.2f",
                empty,
                self._removal_count,
                self.config.detection.removal_frames_required,
                self.detector.debug_info().geometry_confidence,
            )
        self._publish_debug(frame)

        if self._removal_count >= self.config.detection.removal_frames_required:
            logger.info(
                "event=document_removed reason=scene_empty consecutive_frames=%d motion=%.2f edge_density=%.4f "
                "waited_ms=%d",
                self._removal_count,
                self.detector.last_motion,
                self.detector.last_edge_density,
                int(self.sm.elapsed() * 1000),
            )
            self._clear_transaction()
            self.detector.reset_baseline()
            self._awaiting_clear_scene = False
            self._transition(AppState.IDLE, reason="document_removed")

    async def _recovery_tick(self) -> None:
        if self._recovery_in_progress:
            return
        if time.monotonic() < self._next_recovery_at:
            return
        if not self._claim_state_action():
            return

        self._recovery_in_progress = True
        max_attempts = self.config.camera.recovery_max_attempts
        try:
            if self._recovery_attempts >= max_attempts:
                logger.error("Recovery exhausted after %d attempts", max_attempts)
                self.health.record_failure("recovery_exhausted")
                self.health.camera = ComponentStatus(level=HealthLevel.UNAVAILABLE, message="Camera recovery exhausted")
                self._transition(AppState.ERROR)
                raise RecoveryExhaustedError("Recovery attempts exhausted")

            self._recovery_attempts += 1
            logger.info("event=recovery_started attempt=%d/%d", self._recovery_attempts, max_attempts)
            self.health.record_recovery()
            self._abort_transaction("recovery")

            await asyncio.sleep(self.config.camera.recovery_delay_sec)
            await asyncio.to_thread(self.camera.restart)
            self.health.camera = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
            self._recovery_attempts = 0
            self._next_recovery_at = 0.0
            if self.disk.check().healthy:
                self._transition(AppState.IDLE)
            else:
                self._transition(AppState.ERROR, reason="storage_still_degraded")
        except CameraError as exc:
            logger.error("event=recovery_failed %s", exc)
            self.health.camera = ComponentStatus(level=HealthLevel.UNAVAILABLE, message=str(exc))
            self._next_recovery_at = time.monotonic() + self.config.camera.recovery_delay_sec
            self._state_action_claimed = None
        finally:
            self._recovery_in_progress = False

    async def _error_tick(self) -> None:
        storage = self.disk.check()
        self._update_storage_health(storage)
        if not storage.healthy or self._shutdown_event.is_set():
            return
        if self._error_backoff_claimed:
            return
        self._error_backoff_claimed = True
        try:
            await asyncio.sleep(10.0)
            if not self._shutdown_event.is_set():
                self._transition(AppState.RECOVERY, reason="auto_recovery_from_error")
        finally:
            self._error_backoff_claimed = False

    async def _handle_watchdog(self, alert) -> None:
        state = alert.state
        if state in {
            AppState.IMAGE_VALIDATION,
            AppState.UPLOADING,
            AppState.WAITING_FOR_SERVER,
            AppState.PROCESSING_RESPONSE,
        }:
            await self._fail_transaction_and_wait("timeout", f"watchdog:{alert.reason}")
            return
        if state == AppState.CAPTURING:
            await self._enter_recovery(f"watchdog:{alert.reason}")
            return
        await self._enter_recovery(f"watchdog:{alert.reason}")

    async def _enter_recovery(self, reason: str) -> None:
        logger.warning(
            "event=enter_recovery reason=%s capture_id=%s",
            reason,
            getattr(self._transaction, "capture_id", None),
        )
        self._cleanup_transaction_files()
        if self._transaction and not self._transaction.is_terminal:
            self._transaction.mark_failed(reason)
        self._finish_manual_request(
            {
                "ok": False,
                "status": "failed",
                "state": "RECOVERY",
                "capture_id": getattr(self._transaction, "capture_id", None),
                "message": reason,
            }
        )
        self._next_recovery_at = 0.0
        self._state_action_claimed = None
        self._transition(AppState.RECOVERY, reason=reason)

    # state name -> field of the timing summary (sums when a state repeats, e.g. re-stabilizing)
    _SUMMARY_FIELDS = {
        "DOCUMENT_DETECTED": "lock_ms",
        "WAITING_FOR_STABILITY": "stability_ms",
        "FOCUSING": "focus_ms",
        "CAPTURING": "capture_ms",
        "IMAGE_VALIDATION": "validation_ms",
        "UPLOADING": "upload_ms",  # includes preparing the smaller upload copy
        "WAITING_FOR_SERVER": "response_ms",
        "PROCESSING_RESPONSE": "response_ms",
    }

    def _log_timing_summary(self, outcome: str) -> None:
        """One line per transaction with the time spent in each stage (monotonic clock)."""
        marks = self._txn_marks
        if not marks:
            return
        now = time.monotonic()
        fields: dict[str, int] = {}
        for (name, start), nxt in zip(marks, [m[1] for m in marks[1:]] + [now]):
            field = self._SUMMARY_FIELDS.get(name)
            if field:
                fields[field] = fields.get(field, 0) + round((nxt - start) * 1000)
        total_ms = round((now - marks[0][1]) * 1000)
        logger.info(
            "event=timing_summary outcome=%s detect_ms=%d %s total_ms=%d end_to_end_ms=%d "
            "detect_to_validated_ms=%s",
            outcome,
            self._txn_detect_latency_ms,
            " ".join(f"{key}={value}" for key, value in fields.items()),
            total_ms,
            total_ms + self._txn_detect_latency_ms,
            self._txn_validated_ms,
        )
        self._txn_marks = []

    def _log_loop_stats(self, target_interval: float) -> None:
        """Every 60 s: how fast the main loop really runs (CPU / frame delivery check)."""
        now = time.monotonic()
        window = now - self._loop_window_start
        if window < 60.0:
            return
        if self._loop_ticks:
            logger.info(
                "event=loop_stats window_s=%.0f ticks=%d achieved_fps=%.2f target_fps=%.1f avg_tick_ms=%.0f "
                "max_tick_ms=%.0f state=%s",
                window,
                self._loop_ticks,
                self._loop_ticks / window,
                1.0 / target_interval if target_interval > 0 else 0.0,
                self._loop_tick_ms / self._loop_ticks,
                self._loop_tick_max_ms,
                self.sm.state.value,
            )
        self._loop_window_start = now
        self._loop_ticks = 0
        self._loop_tick_ms = 0.0
        self._loop_tick_max_ms = 0.0

    def _abort_transaction(self, reason: str) -> None:
        if self._transaction and not self._transaction.is_terminal:
            self._transaction.mark_failed(reason)
        self._finish_manual_request(
            {
                "ok": False,
                "status": "failed",
                "state": self.sm.state.value,
                "capture_id": self._transaction.capture_id if self._transaction else None,
                "message": reason,
            }
        )
        self._cleanup_transaction_files()

    async def _fail_transaction_and_wait(self, failure_key: str, detail: str) -> None:
        self._consecutive_failures += 1
        self._log_timing_summary(f"failed:{failure_key}")
        if self._transaction:
            self._transaction.mark_failed(detail)
        self.health.record_failure(detail)
        self._cleanup_transaction_files()
        self.led.flash_failure()
        dest = (
            AppState.WAITING_FOR_REMOVAL
            if self._resume_state == AppState.WAITING_FOR_REMOVAL
            else AppState.IDLE
        )
        self.status.update(
            dest,
            capture_id=self._transaction.capture_id if self._transaction else None,
            failure_key=failure_key,
            last_error=detail,
            **self._health_status_kwargs(),
        )
        logger.info(
            "event=pipeline_stage stage=failed reason=%s detail=%s capture_id=%s next_state=%s",
            failure_key,
            detail,
            self._transaction.capture_id if self._transaction else None,
            dest.value,
        )
        self._finish_manual_request(
            {
                "ok": False,
                "status": "failed",
                "state": dest.value,
                "capture_id": self._transaction.capture_id if self._transaction else None,
                "message": f"{failure_key}: {detail}",
            }
        )
        self.stability.reset()
        self._removal_disturbance = 0
        if dest == AppState.IDLE:
            self._awaiting_clear_scene = True
            self._clear_scene_count = 0
            self._awaiting_removal_log_count = 0
            self._clear_transaction()
            self._transition(AppState.IDLE, reason=f"{failure_key}")
        else:
            self._transition(AppState.WAITING_FOR_REMOVAL, reason=f"{failure_key}")

    def _cleanup_transaction_files(self) -> None:
        if self._capture_path:
            self.temp_store.cleanup(self._capture_path)
            self._capture_path = None

    def _clear_transaction(self) -> None:
        capture_id = self._transaction.capture_id if self._transaction else None
        logger.info("event=transaction_cleared capture_id=%s", capture_id)
        self._transaction = None
        self._capture_path = None
        self._pending_response = None
        self.detector.release_lock()
        set_log_context(capture_id=None)

    def _refresh_health(self) -> None:
        storage = self.disk.check()
        self._update_storage_health(storage)

    def _update_storage_health(self, storage_status) -> None:
        level = HealthLevel.HEALTHY
        if storage_status.health.value == "degraded":
            level = HealthLevel.DEGRADED
        elif storage_status.health.value == "unavailable":
            level = HealthLevel.UNAVAILABLE
        self.health.storage = ComponentStatus(
            level=level,
            message=storage_status.message,
            details=storage_status.to_dict(),
        )

    def _health_status_kwargs(self) -> dict:
        return {
            "camera": self.health.camera.to_dict(),
            "network": self.health.network.to_dict(),
            "server": self.health.server.to_dict(),
            "storage": self.health.storage.to_dict(),
            "application": self.health.application.to_dict(),
            "uptime_seconds": self.health.uptime_sec(),
        }

    def _transition(self, target: AppState, *, reason: str = "") -> None:
        if self._shutdown_event.is_set() and target not in {AppState.SHUTDOWN, AppState.RECOVERY}:
            target = AppState.SHUTDOWN
        prev = self.sm.state
        self._last_transition_reason = reason or f"{prev.value}->{target.value}"
        self.sm.transition(target, reason=reason)
        if self._txn_marks:
            self._txn_marks.append((target.value, time.monotonic()))
        self._state_action_claimed = None
        if target != AppState.ERROR:
            self._error_backoff_claimed = False
        set_log_context(state=target.value)
        doc = self.detector.debug_info()
        stab = self.stability.debug_info()
        logger.info(
            "event=state_transition from=%s to=%s reason=%s capture_id=%s "
            "document_detected=%s motion=%.2f frame_motion=%.2f edge_density=%.4f "
            "stable=%s removal_detected=%s removal_count=%d",
            prev.value,
            target.value,
            reason,
            self._transaction.capture_id if self._transaction else None,
            doc.detected,
            doc.motion,
            doc.frame_motion,
            doc.edge_density,
            stab.stable,
            self._last_scene_empty,
            self._removal_count,
        )
        ready = target == AppState.IDLE and self.health.overall_ready
        message_state = target
        if target == AppState.IDLE and not ready:
            message_state = AppState.ERROR if not self.health.storage.healthy else target

        self.led.set_state(message_state)
        self.status.update(
            message_state,
            capture_id=self._transaction.capture_id if self._transaction else None,
            last_success_at=self.health.last_success_at,
            last_error=self.health.last_error,
            **self._health_status_kwargs(),
        )
        self._publish_debug()

    def _publish_debug(self, frame=None, *, camera_error: str | None = None) -> None:
        """Push a debug snapshot. Never mutates ``frame``; the viewer copies it."""
        viewer = self._debug_viewer
        if viewer is None:
            return
        self._note_preview_frame(frame)
        doc = self.detector.debug_info()
        stab = self.stability.debug_info()
        camera_healthy = self.health.camera.healthy and not camera_error
        error_msg = camera_error
        if not camera_healthy and error_msg is None:
            error_msg = self.health.camera.message
        available, phase, message = self._manual_status()
        af = self.autofocus.debug
        viewer.publish(
            frame,
            state=self.sm.state.value,
            document_detected=doc.detected,
            document_locked=doc.document_locked,
            motion=doc.motion,
            edge_density=doc.edge_density,
            motion_threshold=doc.motion_threshold,
            edge_density_threshold=doc.edge_density_threshold,
            stable_count=stab.stable_count,
            stable_required=stab.stable_frames_required,
            stable=stab.stable,
            camera_healthy=camera_healthy,
            autofocus_available=self.health.autofocus_available,
            camera_error=error_msg,
            roi_center_fraction=self.config.detection.roi_center_fraction,
            reject_reason=doc.reject_reason,
            roi_width=doc.roi_width,
            roi_height=doc.roi_height,
            gray_std=doc.gray_std,
            gray_mean=doc.gray_mean,
            frame_width=doc.frame_width,
            frame_height=doc.frame_height,
            frame_motion=doc.frame_motion,
            manual_capture_available=available,
            manual_phase=phase,
            manual_message=message,
            removal_detected=self._last_scene_empty,
            removal_count=self._removal_count,
            autofocus_running=af.running,
            autofocus_method=af.method,
            autofocus_attempt=af.attempt,
            focus_before=af.focus_before,
            focus_after=af.focus_after,
            focus_roi=af.preview_roi or af.focus_roi,
            lens_before=af.lens_before,
            lens_after=af.lens_after,
            improvement_ratio=af.improvement_ratio,
            preview_focus_roi=(
                "{},{},{},{}".format(*doc.focus_roi)
                if doc.focus_roi != (0, 0, 0, 0)
                else (af.preview_roi or af.focus_roi)
            ),
            sensor_focus_roi=af.sensor_roi,
            preview_roi_wh=af.preview_roi_wh,
            document_roi="{},{},{},{}".format(*doc.document_roi) if doc.document_roi != (0, 0, 0, 0) else "",
            stability_motion=stab.motion,
            stability_threshold=stab.threshold,
            state_reason=self._last_transition_reason,
            manual_capture_requested=self._manual_capture_requested,
            capture_started=self._capture_started,
            upload_started=self._upload_started,
            upload_result=self._upload_result,
        )

    async def _shutdown(self) -> None:
        self._transition(AppState.SHUTDOWN)
        self._cleanup_transaction_files()
        if self._transaction and not self._transaction.is_terminal:
            self._transaction.mark_failed("shutdown")
        self._finish_manual_request(
            {
                "ok": False,
                "status": "failed",
                "state": "SHUTDOWN",
                "message": "Service shutting down",
            }
        )
        self._clear_transaction()
        if self._debug_viewer is not None:
            self._debug_viewer.stop()
        await asyncio.to_thread(self.led.stop)
        await self.api.stop()
        await asyncio.to_thread(self.camera.shutdown)
        logger.info("Shutdown complete")
