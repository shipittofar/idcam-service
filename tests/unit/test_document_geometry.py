"""Document geometry detection tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from idcam.config.schema import DetectionConfig
from idcam.vision import document_detector as det_module
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.document_geometry import (
    corner_motion,
    detect_document_geometry,
    order_corners,
    scan_document_candidates,
    smooth_corners,
)
from idcam.vision.sharpness import inner_roi


def _desk_frame(width=640, height=480, desk=180):
    return np.full((height, width, 3), desk, dtype=np.uint8)


def _document_frame(
    width=640,
    height=480,
    *,
    desk=210,
    paper=60,
    margin_x=80,
    margin_y=60,
    edge_width=8,
    edge_color=0,
):
    frame = np.full((height, width, 3), desk, dtype=np.uint8)
    x0, y0, x1, y1 = margin_x, margin_y, width - margin_x, height - margin_y
    frame[y0:y1, x0:x1] = paper
    if edge_width > 0:
        frame[y0 : y0 + edge_width, x0:x1] = edge_color
        frame[y1 - edge_width : y1, x0:x1] = edge_color
        frame[y0:y1, x0 : x0 + edge_width] = edge_color
        frame[y0:y1, x1 - edge_width : x1] = edge_color
    return frame


def _rotated_document_frame(width=640, height=480, angle_deg=12.0):
    frame = _desk_frame(width, height, desk=200)
    paper = np.full((height - 120, width - 120, 3), 70, dtype=np.uint8)
    cv2.rectangle(paper, (0, 0), (paper.shape[1] - 1, paper.shape[0] - 1), (0, 0, 0), 6)
    center = (paper.shape[1] // 2, paper.shape[0] // 2)
    rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        paper,
        rot,
        (paper.shape[1], paper.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(200, 200, 200),
    )
    y0 = (height - rotated.shape[0]) // 2
    x0 = (width - rotated.shape[1]) // 2
    frame[y0 : y0 + rotated.shape[0], x0 : x0 + rotated.shape[1]] = rotated
    return frame


def _perspective_document_frame(width=640, height=480):
    frame = _desk_frame(width, height, desk=190)
    src = np.array(
        [[0, 0], [width - 160, 20], [width - 100, height - 100], [40, height - 80]],
        dtype=np.float32,
    )
    dst = np.array(
        [[120, 80], [width - 80, 60], [width - 60, height - 60], [80, height - 40]],
        dtype=np.float32,
    )
    h = cv2.getPerspectiveTransform(src, dst)
    paper = np.full((height, width, 3), 55, dtype=np.uint8)
    cv2.rectangle(paper, (0, 0), (width - 1, height - 1), (0, 0, 0), 8)
    warped = cv2.warpPerspective(paper, h, (width, height), borderValue=(190, 190, 190))
    mask = warped[:, :, 0] < 120
    frame[mask] = warped[mask]
    return frame


def _uneven_light_frame(width=640, height=480):
    frame = _document_frame(width, height, desk=160, paper=90)
    gradient = np.linspace(0.55, 1.35, width, dtype=np.float32)
    frame = np.clip(frame.astype(np.float32) * gradient[np.newaxis, :, np.newaxis], 0, 255).astype(
        np.uint8
    )
    return frame


def _weak_edge_frame(width=640, height=480):
    return _document_frame(width, height, desk=175, paper=140, edge_width=0)


class TestDocumentGeometry:
    def test_detects_paper_like_rectangle(self):
        frame = _document_frame()
        geom = detect_document_geometry(frame)
        assert geom is not None
        assert geom.confidence >= 0.35
        assert geom.area_fraction > 0.1
        assert geom.bbox[2] > 100
        assert geom.bbox[3] > 100
        assert geom.detection_method

    def test_uniform_scene_returns_none(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert detect_document_geometry(frame) is None

    def test_order_corners_tl_tr_br_bl(self):
        pts = np.array([[10, 10], [100, 12], [98, 90], [12, 88]], dtype=np.float32)
        ordered = order_corners(pts)
        assert ordered[0][1] <= ordered[2][1]
        assert ordered[0][0] <= ordered[1][0]

    def test_scan_reports_rejected_candidates(self):
        best, reports = scan_document_candidates(np.full((480, 640, 3), 128, dtype=np.uint8))
        assert best is None
        assert isinstance(reports, list)

    def test_scan_accepts_document_frame(self):
        best, reports = scan_document_candidates(_document_frame())
        assert best is not None
        assert any(r.accepted for r in reports)

    def test_rejected_candidates_have_real_bbox(self):
        _, reports = scan_document_candidates(np.full((480, 640, 3), 128, dtype=np.uint8))
        for r in reports:
            if r.rejection_reason == "area_too_small":
                assert r.bbox[2] >= 0 and r.bbox[3] >= 0

    def test_detects_rotated_document(self):
        frame = _rotated_document_frame(angle_deg=10.0)
        geom = detect_document_geometry(frame)
        assert geom is not None
        assert geom.area_fraction > 0.08
        assert abs(geom.rotation_deg) > 2.0

    def test_detects_perspective_document(self):
        frame = _perspective_document_frame()
        geom = detect_document_geometry(frame)
        assert geom is not None
        assert geom.area_fraction > 0.06

    def test_detects_under_uneven_lighting(self):
        frame = _uneven_light_frame()
        geom = detect_document_geometry(frame)
        assert geom is not None
        assert geom.area_fraction > 0.08

    def test_detects_weak_edge_document(self):
        frame = _weak_edge_frame()
        geom = detect_document_geometry(frame)
        assert geom is not None
        assert geom.area_fraction > 0.08

    def test_smooth_corners_reduces_jump(self):
        prev = ((100, 100), (500, 100), (500, 400), (100, 400))
        new = ((110, 105), (510, 95), (505, 410), (95, 395))
        smoothed = smooth_corners(prev, new, alpha=0.25, frame_shape=(480, 640, 3))
        motion = corner_motion(prev, smoothed)
        raw_motion = corner_motion(prev, new)
        assert motion < raw_motion

    def test_inner_focus_roi_inside_document(self):
        frame = _document_frame()
        geom = detect_document_geometry(frame)
        assert geom is not None
        focus = inner_roi(geom.bbox, 0.15, frame.shape)
        fx, fy, fw, fh = focus
        bx, by, bw, bh = geom.bbox
        assert fx >= bx and fy >= by
        assert fx + fw <= bx + bw + 1
        assert fy + fh <= by + bh + 1
        assert fw > 0 and fh > 0


class TestDocumentLockPersistence:
    def test_brightness_change_does_not_drop_detection(self):
        cfg = DetectionConfig(min_geometry_confidence=0.25, detect_frames_required=1)
        det = DocumentDetector(cfg)
        base = _document_frame()
        brighter = np.clip(base.astype(np.int16) + 35, 0, 255).astype(np.uint8)

        assert det.detect(base)
        det.lock_document(base)
        assert det.lock.is_locked

        for _ in range(5):
            assert det.detect(brighter)
        assert det.lock.is_locked

    def test_small_movement_smooths_not_replaces_lock(self):
        cfg = DetectionConfig(min_geometry_confidence=0.25, detect_frames_required=1, roi_smooth_alpha=0.2)
        det = DocumentDetector(cfg)
        frame_a = _document_frame(margin_x=90, margin_y=70)
        frame_b = _document_frame(margin_x=95, margin_y=75)

        det.detect(frame_a)
        det.lock_document(frame_a)
        roi_before = det.lock.roi

        det.detect(frame_b)
        if det._last_geometry:
            det.lock.update(frame_b, det._last_geometry)

        roi_after = det.lock.roi
        assert det.lock.is_locked
        assert abs(roi_after[0] - roi_before[0]) <= 20
        assert abs(roi_after[1] - roi_before[1]) <= 20


class TestDownscaledDetection:
    def test_1080p_frame_detected_at_reduced_resolution_and_mapped_back(self):
        frame = np.full((1080, 1920, 3), (70, 80, 75), dtype=np.uint8)
        cv2.rectangle(frame, (600, 200), (1300, 900), (228, 226, 220), -1)
        for i in range(18):
            cv2.putText(frame, "NAME SURNAME 123456", (650, 260 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 1)

        geom = detect_document_geometry(frame)

        assert geom is not None
        x, y, w, h = geom.bbox
        # Coordinates are full-frame pixels even though segmentation ran on a 640 px image.
        assert abs(x - 600) <= 25 and abs(y - 200) <= 25
        assert abs(w - 700) <= 40 and abs(h - 700) <= 40
        assert all(0 <= cx < 1920 and 0 <= cy < 1080 for cx, cy in geom.corners)

    def test_region_glued_to_frame_borders_is_not_a_document(self):
        frame = np.full((1080, 1920, 3), (70, 80, 75), dtype=np.uint8)
        cv2.rectangle(frame, (230, 0), (1650, 1079), (228, 226, 220), -1)

        geom = detect_document_geometry(frame)

        assert geom is None or geom.confidence < 0.42


class TestInscribedFocusRoi:
    @staticmethod
    def _inside(corners, roi):
        poly = np.array(corners, dtype=np.float32)
        x, y, w, h = roi
        return all(
            cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0
            for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
        )

    def test_rotated_page_focus_roi_stays_on_paper_not_desk(self):
        from idcam.vision.sharpness import inscribed_focus_roi

        corners = ((600, 200), (1300, 330), (1150, 950), (480, 780))
        roi = inscribed_focus_roi(corners, (1080, 1920, 3), 0.12)
        assert self._inside(corners, roi)
        assert roi[2] > 250 and roi[3] > 250  # still a useful window, not a sliver

    def test_axis_aligned_page_focus_roi_matches_inset_bbox(self):
        from idcam.vision.sharpness import inscribed_focus_roi

        corners = ((500, 100), (1200, 100), (1200, 900), (500, 900))
        x, y, w, h = inscribed_focus_roi(corners, (1080, 1920, 3), 0.12)
        assert abs(x - 584) <= 20 and abs(y - 196) <= 20
        assert abs(w - 532) <= 40 and abs(h - 608) <= 40

    def test_degenerate_quad_falls_back_to_bbox_inset(self):
        from idcam.vision.sharpness import inscribed_focus_roi

        roi = inscribed_focus_roi(((10, 10), (10, 10), (10, 10), (10, 10)), (480, 640, 3), 0.12)
        assert roi[2] >= 1 and roi[3] >= 1


class TestOutlineAlignment:
    def test_rotated_page_outline_is_tight(self):
        """The detected box follows the page itself (scored by how exactly the outline sits on its edge)."""
        frame = np.full((1080, 1920, 3), (70, 80, 75), dtype=np.uint8)
        pts = np.array([[600, 200], [1300, 330], [1150, 950], [480, 780]], np.int32)
        cv2.fillPoly(frame, [pts], (228, 226, 220))
        for i in range(18):
            cv2.putText(frame, "NAME SURNAME 1234567890", (680, 330 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 1)

        geom = detect_document_geometry(frame)

        assert geom is not None
        gt = (480, 200, 820, 750)
        x, y, w, h = geom.bbox
        inter = max(0, min(x + w, gt[0] + gt[2]) - max(x, gt[0])) * max(0, min(y + h, gt[1] + gt[3]) - max(y, gt[1]))
        iou = inter / (w * h + gt[2] * gt[3] - inter)
        assert iou > 0.85

    def test_empty_desk_gives_no_document(self):
        frame = np.full((1080, 1920, 3), (70, 80, 75), dtype=np.uint8)
        noise = np.random.default_rng(0).normal(0, 4, frame.shape)
        assert detect_document_geometry((frame + noise).clip(0, 255).astype(np.uint8)) is None


class TestRobustCornerMotion:
    def test_one_jittering_corner_on_frame_border_is_not_motion(self):
        prev = ((603, 393), (1293, 492), (1209, 1077), (501, 810))
        curr = ((604, 394), (1292, 491), (1209, 1000), (502, 811))  # clipped corner moved 77 px
        assert corner_motion(prev, curr) > 15.0
        assert corner_motion(prev, curr, robust=True) < 3.0

    def test_real_shift_of_the_whole_page_is_still_motion(self):
        prev = ((600, 400), (1300, 400), (1300, 900), (600, 900))
        curr = tuple((x + 60, y + 40) for x, y in prev)
        assert corner_motion(prev, curr, robust=True) > 50.0


class TestSceneEmpty:
    @staticmethod
    def _detector_with(confidence):
        from types import SimpleNamespace

        det = DocumentDetector(DetectionConfig())
        geom = None if confidence is None else SimpleNamespace(confidence=confidence, bbox=(0, 0, 10, 10),
                                                              corners=((0, 0), (10, 0), (10, 10), (0, 10)),
                                                              edge_strength=10.0)
        det._find_geometry = lambda frame: geom
        return det

    def test_weak_desk_candidate_counts_as_empty(self):
        """Empty desks produce outlines with confidence ~0.2; the removal wait must still end."""
        assert self._detector_with(0.22).is_scene_empty(_desk_frame())
        assert self._detector_with(0.30).is_scene_empty(_desk_frame())
        assert self._detector_with(None).is_scene_empty(_desk_frame())

    def test_page_in_view_is_not_empty(self):
        assert not self._detector_with(0.60).is_scene_empty(_desk_frame())
        assert not self._detector_with(0.50).is_scene_empty(_desk_frame())

    def test_a_cluttered_bare_desk_scoring_about_040_counts_as_empty(self):
        # Real Pi desk: 0.39-0.41 with the page gone; it must not keep "remove the document" up.
        for conf in (0.38, 0.40, 0.41, 0.44):
            assert self._detector_with(conf).is_scene_empty(_desk_frame())


class TestBorderAndRebaseline:
    def test_page_cut_off_by_the_top_edge_is_demoted(self):
        inside = np.full((1080, 1920, 3), (70, 80, 75), dtype=np.uint8)
        cv2.rectangle(inside, (600, 200), (1300, 900), (228, 226, 220), -1)
        cut = np.full((1080, 1920, 3), (70, 80, 75), dtype=np.uint8)
        cv2.rectangle(cut, (600, 0), (1300, 700), (228, 226, 220), -1)  # same page, pushed past the top edge

        g_in = detect_document_geometry(inside)
        g_cut = detect_document_geometry(cut)

        assert g_in is not None
        assert g_cut is None or g_cut.confidence < g_in.confidence * 0.8

    def test_rebaseline_removes_defocus_induced_corner_shift(self):
        cfg = DetectionConfig(min_geometry_confidence=0.25, detect_frames_required=1)
        det = DocumentDetector(cfg)
        frame = _document_frame(margin_x=90, margin_y=70)
        det.detect(frame)
        det.lock_document(frame)
        det._prev_corners = tuple((x + 70, y + 40) for x, y in det._prev_corners)  # stale lock-time outline

        det.rebaseline_motion(frame)

        assert det.structural_motion_in_lock(frame) < 5.0


class TestGeometryReuse:
    def test_pre_capture_check_segments_the_frame_only_once(self):
        from unittest.mock import patch

        det = DocumentDetector(DetectionConfig(min_geometry_confidence=0.25))
        frame = _document_frame(margin_x=90, margin_y=70)
        det.detect(frame)
        det.lock_document(frame)

        real = det_module.detect_document_geometry
        with patch.object(det_module, "detect_document_geometry", wraps=real) as spy:
            det.is_scene_empty(frame)  # what _abort_if_document_gone does
            det.structural_motion_in_lock(frame)  # what the motion check does next
            # at most one segmentation (none when the static-scene gate already knows this scene)
            assert spy.call_count <= 1
            before = spy.call_count
            det.structural_motion_in_lock(_document_frame(margin_x=90, margin_y=70))  # a new frame is analysed
            assert spy.call_count == before + 1


class TestStaticSceneGate:
    @staticmethod
    def _counting(det_frames):
        from unittest.mock import patch

        real = det_module.detect_document_geometry
        return patch.object(det_module, "detect_document_geometry", wraps=real)

    def test_an_unchanged_scene_is_segmented_once_then_reused(self):
        det = DocumentDetector(DetectionConfig())
        frame = _desk_frame()
        with self._counting(frame) as spy:
            for _ in range(6):
                det.detect(frame.copy())
            assert spy.call_count == 1
        assert det.gate_hits == 5

    def test_sensor_noise_alone_does_not_force_a_new_analysis(self):
        det = DocumentDetector(DetectionConfig())
        base = _desk_frame().astype(np.int16)
        rng = np.random.default_rng(3)
        with self._counting(base) as spy:
            for _ in range(5):
                noisy = np.clip(base + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
                det.detect(noisy)
            assert spy.call_count == 1

    def test_a_document_entering_is_analysed_at_once(self):
        det = DocumentDetector(DetectionConfig(min_geometry_confidence=0.25))
        desk = _desk_frame()
        page = _document_frame(margin_x=90, margin_y=70)
        with self._counting(desk) as spy:
            det.detect(desk)
            det.detect(page)
            assert spy.call_count == 2
        assert det._last_geometry is not None

    def test_a_static_scene_is_still_re_analysed_now_and_then(self):
        det = DocumentDetector(DetectionConfig())
        frame = _desk_frame()
        with self._counting(frame) as spy:
            for _ in range(20):
                det.detect(frame.copy())
            assert 2 <= spy.call_count <= 3  # every 9th frame


class TestSameObjectDetection:
    """A detection needs the same outline in consecutive frames, not just a high score each time."""

    @staticmethod
    def _detector(**kwargs):
        from types import SimpleNamespace

        det = DocumentDetector(DetectionConfig(detect_frames_required=3, **kwargs))

        def feed(bbox, conf=0.6):
            geom = SimpleNamespace(confidence=conf, bbox=bbox, corners=((0, 0),) * 4, edge_strength=0.0, area_fraction=0.1)
            det._find_geometry = lambda frame: geom  # type: ignore[method-assign]
            return det.detect(np.zeros((120, 160, 3), np.uint8))

        return det, feed

    def test_a_different_best_candidate_every_frame_never_latches(self):
        det, feed = self._detector()
        boxes = [(100, 100, 400, 300), (900, 600, 300, 200), (300, 50, 500, 150), (1000, 800, 200, 250), (50, 700, 600, 200)]
        assert not any(feed(b) for b in boxes * 3)

    def test_the_same_outline_latches_after_the_required_frames(self):
        det, feed = self._detector()
        results = [feed((100, 100, 400, 300)) for _ in range(3)]
        assert results == [False, False, True]

    def test_small_jitter_and_slow_drift_still_count_as_one_object(self):
        det, feed = self._detector()
        results = [feed(b) for b in [(100, 100, 400, 300), (106, 98, 402, 301), (112, 104, 398, 297)]]
        assert results[-1] is True

    def test_one_jump_restarts_the_count(self):
        det, feed = self._detector()
        feed((100, 100, 400, 300))
        feed((100, 100, 400, 300))
        assert feed((900, 600, 300, 200)) is False  # would have been the third frame
        assert feed((900, 600, 300, 200)) is False
        assert feed((900, 600, 300, 200)) is True

    def test_the_check_can_be_switched_off(self):
        det, feed = self._detector(detect_min_iou=0.0)
        boxes = [(100, 100, 400, 300), (900, 600, 300, 200), (300, 50, 500, 150)]
        assert [feed(b) for b in boxes][-1] is True

    def test_a_low_confidence_frame_in_between_resets_everything(self):
        det, feed = self._detector()
        feed((100, 100, 400, 300))
        feed((100, 100, 400, 300))
        feed((100, 100, 400, 300), conf=0.3)
        assert feed((100, 100, 400, 300)) is False


class TestEmptySceneBackground:
    """With background_check a candidate must differ from the learned empty scene."""

    @staticmethod
    def _furniture():
        return _document_frame(desk=210, paper=60, margin_x=90, margin_y=70)  # looks like a document, always there

    @staticmethod
    def _cfg(**kwargs):
        return DetectionConfig(min_geometry_confidence=0.25, background_check=True, background_warmup_frames=4, **kwargs)

    def test_the_furniture_scene_is_detected_without_the_check(self):
        det = DocumentDetector(DetectionConfig(min_geometry_confidence=0.25, background_check=False))
        assert any(det.detect(self._furniture().copy()) for _ in range(10))

    def test_furniture_that_was_there_from_the_start_is_never_a_document(self):
        det = DocumentDetector(self._cfg())
        results = [det.detect(self._furniture().copy()) for _ in range(40)]
        assert not any(results)
        # the furniture is dropped inside the candidate search now, so nothing is left to reject afterwards
        assert det._last_reject_reason in {"static_background", "no_document_geometry"}

    def test_a_document_placed_on_the_learned_desk_is_detected(self):
        det = DocumentDetector(self._cfg())
        desk = _desk_frame(desk=210)
        for _ in range(12):
            det.detect(desk.copy())  # learn the empty desk
        results = [det.detect(self._furniture().copy()) for _ in range(8)]
        assert results[-1] is True

    def test_a_larger_document_placed_over_the_furniture_is_detected(self):
        det = DocumentDetector(self._cfg())
        for _ in range(12):
            det.detect(self._furniture().copy())
        bigger = _document_frame(desk=210, paper=120, margin_x=50, margin_y=40)
        results = [det.detect(bigger.copy()) for _ in range(8)]
        assert results[-1] is True

    def test_nothing_is_reported_while_the_background_is_still_being_learned(self):
        det = DocumentDetector(self._cfg())
        assert det.detect(self._furniture().copy()) is False
        assert det._last_reject_reason == "learning_background"

    def test_a_scene_that_never_calms_down_does_not_block_detection_forever(self):
        det = DocumentDetector(self._cfg(background_give_up_frames=20))
        rng = np.random.default_rng(2)
        page = self._furniture()
        for _ in range(30):
            noisy = np.clip(page.astype(np.int16) + rng.integers(-60, 61, page.shape), 0, 255).astype(np.uint8)
            det.detect(noisy)
        assert det._bg_given_up is True

    def test_global_exposure_change_is_not_a_new_object(self):
        det = DocumentDetector(self._cfg())
        for _ in range(12):
            det.detect(self._furniture().copy())
        brighter = np.clip(self._furniture().astype(np.int16) + 18, 0, 255).astype(np.uint8)
        assert not any(det.detect(brighter.copy()) for _ in range(10))
