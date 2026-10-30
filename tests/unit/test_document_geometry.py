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
            assert spy.call_count == 1
            det.structural_motion_in_lock(_document_frame(margin_x=90, margin_y=70))  # a new frame is analysed
            assert spy.call_count == 2
