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


def _same_template_other_fields(a):
    """The same card layout with three fields that differ (NCC with the original ~0.87)."""
    b = a.copy()
    for y, x in ((150, 150), (250, 300), (320, 200)):
        b[y : y + 30, x : x + 60] = 210
    return b


class TestSwapRobustness:
    def test_a_candidate_below_the_detection_threshold_can_never_replace_a_document(self):
        det = DocumentDetector(DetectionConfig(min_geometry_confidence=0.99))  # nothing in view reaches it
        sig = det.document_signature(_doc())
        assert sig is not None
        assert not any(det.document_replaced(_doc(margin_x=200, margin_y=150).copy(), sig) for _ in range(12))

    def test_two_cards_of_the_same_template_are_told_apart_once_the_picture_was_disturbed(self):
        a = _doc(texture=40, seed=1)
        b = _same_template_other_fields(a)
        det = DocumentDetector(_cfg())
        sig = det.document_signature(a)
        strict = [det.document_replaced(b.copy(), sig) for _ in range(10)]
        assert strict[-1] is False  # untouched picture: a modest difference is not believed
        det2 = DocumentDetector(_cfg())
        sig2 = det2.document_signature(a)
        relaxed = [det2.document_replaced(b.copy(), sig2, disturbed=True) for _ in range(10)]
        assert relaxed[-1] is True

    def test_the_same_page_never_counts_as_replaced_even_when_disturbed(self):
        det = DocumentDetector(_cfg())
        a = _doc(texture=40, seed=1)
        sig = det.document_signature(a)
        assert not any(det.document_replaced(a.copy(), sig, disturbed=True) for _ in range(12))

    def test_with_the_background_check_a_candidate_that_is_only_background_never_replaces(self):
        cfg = DetectionConfig(min_geometry_confidence=0.25, background_check=True, background_warmup_frames=4)
        det = DocumentDetector(cfg)
        scene = _doc()
        for _ in range(12):
            det.detect(_empty())  # learns the empty desk as the background
        assert det._bg_thumb is not None
        sig = det.document_signature(scene)  # a document that was captured earlier
        moved = _doc(margin_x=200, margin_y=150)
        det._is_background = lambda *a, **k: True  # whatever stands there still looks like the learned background
        assert not any(det.document_replaced(moved.copy(), sig) for _ in range(12))
        det._is_background = lambda *a, **k: False
        det._gate_small = None  # the scene is analysed afresh (the gate would reuse the empty answer)
        assert any(det.document_replaced(moved.copy(), sig) for _ in range(12))

    @pytest.mark.asyncio
    async def test_a_swap_to_a_card_of_the_same_template_after_a_hand_took_the_page_away(self, tmp_path):
        orch = _orch(tmp_path)
        a = _doc(texture=40, seed=1)
        b = _same_template_other_fields(a)
        orch._doc_signature = orch.detector.document_signature(a)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        txn = ActiveCaptureTransaction(capture_id="cap-tpl")
        txn.mark_completed()
        orch._transaction = txn
        orch._removal_disturbance = 0
        hand = _doc(margin_x=150, margin_y=120, paper=120)
        sequence = [hand, hand.copy(), a] + [b] * 12  # a hand moves in, then the new card stays
        orch._safe_preview = AsyncMock(side_effect=sequence)
        for _ in range(len(sequence)):
            await orch._removal_tick()
            if orch.sm.state != AppState.WAITING_FOR_REMOVAL:
                break
        assert orch.sm.state == AppState.DOCUMENT_DETECTED

    @pytest.mark.asyncio
    async def test_the_same_card_left_untouched_is_not_captured_twice(self, tmp_path):
        orch = _orch(tmp_path)
        a = _doc(texture=40, seed=1)
        orch._doc_signature = orch.detector.document_signature(a)
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        txn = ActiveCaptureTransaction(capture_id="cap-once")
        txn.mark_completed()
        orch._transaction = txn
        orch._safe_preview = AsyncMock(return_value=a)
        for _ in range(20):
            await orch._removal_tick()
        assert orch.sm.state == AppState.WAITING_FOR_REMOVAL


class TestRemovalUsesTheLearnedBackground:
    """The empty desk can score 0.5 on its own (clutter, a mat); then 'confidence < threshold' never says empty."""

    @staticmethod
    def _cfg_bg(**kw):
        return DetectionConfig(
            min_geometry_confidence=0.25, background_check=True, background_warmup_frames=4,
            stable_frames_required=2, removal_frames_required=2, **kw,
        )

    def _learned(self):
        det = DocumentDetector(self._cfg_bg())
        furniture = _doc()  # a rectangle that is always there and scores high on its own
        for _ in range(12):
            det.detect(furniture.copy())
        assert det._bg_thumb is not None
        return det, furniture

    def test_a_confident_outline_that_is_only_the_background_counts_as_empty(self):
        det, furniture = self._learned()
        from idcam.vision import document_geometry as dg

        assert dg.detect_document_geometry(furniture.copy()).confidence > 0.5  # without the background it is a "document"
        assert det.is_scene_empty(furniture.copy()) is True

    def test_the_same_picture_with_a_new_document_is_not_empty(self):
        det, furniture = self._learned()
        bigger = _doc(margin_x=40, margin_y=30, paper=130)
        assert det.is_scene_empty(bigger.copy()) is False

    def test_a_global_exposure_shift_does_not_hide_the_removal(self):
        det, furniture = self._learned()
        brighter = np.clip(furniture.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        assert det.is_scene_empty(brighter) is True

    def test_without_the_background_check_the_old_rule_applies(self):
        det = DocumentDetector(DetectionConfig(min_geometry_confidence=0.25, background_check=False))
        assert det.is_scene_empty(_doc()) is False  # a confident outline is a document

    def test_the_gap_is_reported_for_the_log(self):
        det, furniture = self._learned()
        assert det.background_gap(furniture.copy()) < 2.0
        assert det.background_gap(_doc(margin_x=40, margin_y=30, paper=130)) > 5.0
        assert DocumentDetector(DetectionConfig()).background_gap(furniture) is None

    @pytest.mark.asyncio
    async def test_waiting_for_removal_ends_when_the_scene_is_back_to_the_learned_desk(self, tmp_path):
        config = AppConfig(
            device_id="rm-bg",
            temp_dir=tmp_path / "state" / "tmp",
            server=ServerConfig(url="https://api.example.com"),
            camera=CameraConfig(),
            detection=self._cfg_bg(),
            ui=UIConfig(status_file=tmp_path / "run" / "status.json", admin_socket=tmp_path / "run" / "admin.sock"),
        )
        orch = ApplicationOrchestrator(config, use_mock_camera=True)
        orch.health.camera = ComponentStatus(level=HealthLevel.HEALTHY, details={"healthy": True})
        furniture = _doc()
        for _ in range(12):
            orch.detector.detect(furniture.copy())
        orch._doc_signature = None
        orch.sm.state = AppState.WAITING_FOR_REMOVAL
        txn = ActiveCaptureTransaction(capture_id="cap-bg")
        txn.mark_completed()
        orch._transaction = txn
        orch._safe_preview = AsyncMock(return_value=furniture.copy())
        for _ in range(6):
            await orch._removal_tick()
            if orch.sm.state != AppState.WAITING_FOR_REMOVAL:
                break
        assert orch.sm.state == AppState.IDLE
