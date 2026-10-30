"""Rapid sequences: the next person swaps the document without the scene ever being empty."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from idcam.app.orchestrator import ApplicationOrchestrator
from idcam.app.state_machine import AppState
from idcam.app.transaction import ActiveCaptureTransaction
from idcam.config.schema import (
    AppConfig,
    CameraConfig,
    DetectionConfig,
    ServerConfig,
    UIConfig,
)
from idcam.system.component_health import ComponentStatus, HealthLevel
from idcam.vision.document_detector import DocumentDetector


def _doc(margin_x=90, margin_y=70, paper=60, texture=0, seed=0):
    frame = np.full((480, 640, 3), 210, dtype=np.uint8)
    x0, y0, x1, y1 = margin_x, margin_y, 640 - margin_x, 480 - margin_y
    frame[y0:y1, x0:x1] = paper
    frame[y0 : y0 + 8, x0:x1] = 0
    frame[y1 - 8 : y1, x0:x1] = 0
    frame[y0:y1, x0 : x0 + 8] = 0
    frame[y0:y1, x1 - 8 : x1] = 0
    if texture:  # blocks of print/photo at seed-dependent places: what makes one document differ from another
        rng = np.random.default_rng(seed)
        for _ in range(14):
            bx = int(rng.integers(x0 + 20, x1 - 70))
            by = int(rng.integers(y0 + 20, y1 - 50))
            frame[by : by + int(rng.integers(15, 45)), bx : bx + int(rng.integers(30, 60))] = int(rng.integers(120, 250))
    return frame


def _empty():
    return np.full((480, 640, 3), 210, dtype=np.uint8)


def _cfg(**kw):
    return DetectionConfig(min_geometry_confidence=0.25, stable_frames_required=2, removal_frames_required=3, **kw)


# --------------------------------------------------------------------------- detector level


class TestDocumentReplaced:
    def test_the_same_document_staying_is_never_a_replacement(self):
        det = DocumentDetector(_cfg())
        a = _doc()
        sig = det.document_signature(a)
        assert sig is not None
        assert not any(det.document_replaced(a.copy(), sig) for _ in range(12))

    def test_another_document_in_another_place_is_a_replacement_once_the_picture_is_still(self):
        det = DocumentDetector(_cfg())
        sig = det.document_signature(_doc())
        results = [det.document_replaced(_doc(margin_x=200, margin_y=150).copy(), sig) for _ in range(10)]
        assert results[0] is False  # the picture was still changing: not believed yet
        assert results[-1] is True

    def test_another_document_in_the_same_outline_is_recognised_by_its_content(self):
        det = DocumentDetector(_cfg())
        sig = det.document_signature(_doc(texture=40, seed=1))
        results = [det.document_replaced(_doc(texture=40, seed=2).copy(), sig) for _ in range(10)]
        assert results[-1] is True

    def test_a_hand_passing_is_not_a_replacement(self):
        det = DocumentDetector(_cfg())
        sig = det.document_signature(_doc())
        frames = [_doc(margin_x=200 + 7 * (i % 3), margin_y=150) for i in range(12)]  # never still
        assert not any(det.document_replaced(f, sig) for f in frames)

    def test_an_empty_scene_is_left_to_the_ordinary_removal_path(self):
        det = DocumentDetector(_cfg())
        sig = det.document_signature(_doc())
        assert not any(det.document_replaced(_empty(), sig) for _ in range(10))

    def test_no_signature_means_no_decision(self):
        det = DocumentDetector(_cfg())
        assert det.document_replaced(_doc(), None) is False

    def test_the_lock_follows_a_page_that_was_swapped_while_waiting(self):
        det = DocumentDetector(_cfg())
        a = _doc()
        det.detect(a)
        assert det.lock_document(a, allow_fallback=False)
        old = det.lock.roi
        assert det.refresh_lock(a.copy()) is False  # same page: nothing to do
        b = _doc(margin_x=220, margin_y=160)
        refreshed = [det.refresh_lock(b.copy()) for _ in range(3)]
        assert any(refreshed)
        assert det.lock.roi != old
        x, y, w, h = det.lock.roi
        assert abs(x - 220) < 30 and abs(y - 160) < 30



# --------------------------------------------------------------------------- orchestrator level


def _orch(tmp_path):
    config = AppConfig(
        device_id="rapid",
        temp_dir=tmp_path / "state" / "tmp",
        server=ServerConfig(url="https://api.example.com"),
        camera=CameraConfig(),
        detection=_cfg(),
        ui=UIConfig(status_file=tmp_path / "run" / "status.json", admin_socket=tmp_path / "run" / "admin.sock"),
    )
    orch = ApplicationOrchestrator(config, use_mock_camera=True)
    orch.health.camera = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
    return orch


class TestRemovalWaitDoesNotGetStuck:
    @pytest.mark.asyncio
    async def test_a_different_document_put_down_at_once_starts_a_new_capture(self, tmp_path):
        orch = _orch(tmp_path)
        a = _doc()
        orch._doc_signature = orch.detector.document_signature(a)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        old_txn = ActiveCaptureTransaction(capture_id="cap-old")
        old_txn.mark_completed()
        orch._transaction = old_txn
        orch._safe_preview = AsyncMock(return_value=_doc(margin_x=200, margin_y=150))

        for _ in range(10):
            await orch._removal_tick()
            if orch.sm.state != AppState.WAITING_FOR_REMOVAL:
                break

        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._transaction is not None and orch._transaction.capture_id != "cap-old"
        assert orch._transaction.trigger == "auto"

    @pytest.mark.asyncio
    async def test_the_same_document_left_in_place_keeps_waiting(self, tmp_path):
        orch = _orch(tmp_path)
        a = _doc()
        orch._doc_signature = orch.detector.document_signature(a)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        txn = ActiveCaptureTransaction(capture_id="cap-same")
        txn.mark_completed()
        orch._transaction = txn
        orch._safe_preview = AsyncMock(return_value=a)
        for _ in range(15):
            await orch._removal_tick()
        assert orch.sm.state == AppState.WAITING_FOR_REMOVAL
        assert orch._transaction.capture_id == "cap-same"

    @pytest.mark.asyncio
    async def test_removing_the_document_still_returns_to_idle(self, tmp_path):
        orch = _orch(tmp_path)
        orch._doc_signature = orch.detector.document_signature(_doc())
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        txn = ActiveCaptureTransaction(capture_id="cap-gone")
        txn.mark_completed()
        orch._transaction = txn
        orch._safe_preview = AsyncMock(return_value=_empty())
        for _ in range(6):
            await orch._removal_tick()
            if orch.sm.state == AppState.IDLE:
                break
        assert orch.sm.state == AppState.IDLE


class TestFailureWaitDoesNotGetStuck:
    @pytest.mark.asyncio
    async def test_a_swap_after_repeated_failures_is_taken_as_a_new_document(self, tmp_path):
        orch = _orch(tmp_path)
        a = _doc()
        orch.detector.detect(a)  # what the idle loop saw before the failures
        orch._doc_signature = orch.detector.document_signature(a)
        orch.sm.state = AppState.IDLE
        orch._awaiting_clear_scene = True
        orch._consecutive_failures = 2
        orch._safe_preview = AsyncMock(return_value=_doc(margin_x=200, margin_y=150))

        for _ in range(14):
            await orch._idle_tick()
            if orch.sm.state != AppState.IDLE:
                break

        assert orch.sm.state == AppState.DOCUMENT_DETECTED
        assert orch._awaiting_clear_scene is False and orch._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_the_same_page_after_repeated_failures_still_waits_for_removal(self, tmp_path):
        orch = _orch(tmp_path)
        a = _doc()
        orch.detector.detect(a)
        orch._doc_signature = orch.detector.document_signature(a)
        orch.sm.state = AppState.IDLE
        orch._awaiting_clear_scene = True
        orch._consecutive_failures = 2
        orch._safe_preview = AsyncMock(return_value=a)
        for _ in range(14):
            await orch._idle_tick()
        assert orch.sm.state == AppState.IDLE and orch._awaiting_clear_scene is True


class TestStabilityFollowsTheSwap:
    @pytest.mark.asyncio
    async def test_a_page_swapped_during_the_stability_wait_is_the_one_that_gets_captured(self, tmp_path):
        orch = _orch(tmp_path)
        a = _doc()
        orch.detector.detect(a)
        assert orch.detector.lock_document(a, allow_fallback=False)
        orch.stability.set_roi(orch.detector.lock.roi)
        orch.sm.state = AppState.WAITING_FOR_STABILITY
        orch._transaction = ActiveCaptureTransaction(capture_id="cap-swap")
        b = _doc(margin_x=220, margin_y=160)
        orch._safe_preview = AsyncMock(return_value=b)
        orch.detector.is_scene_empty = lambda frame: False  # the page is there, just not the same one

        for _ in range(12):
            await orch._stability_tick()
            if orch.sm.state != AppState.WAITING_FOR_STABILITY:
                break

        assert orch.sm.state in {AppState.FOCUSING, AppState.CAPTURING}
        x, y, w, h = orch.detector.lock.roi
        assert abs(x - 220) < 40 and abs(y - 160) < 40
