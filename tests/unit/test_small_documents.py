"""Passport-sized documents: small in the frame and often dark-on-dark (cover on a desk)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from idcam.config.schema import DetectionConfig
from idcam.vision import document_geometry as dg

W, H = 1920, 1080

NAVY = (95, 55, 30)  # RGB-ish triplets as used by the synthetic scenes (blue cover)
BURGUNDY = (45, 35, 95)
WHITE_PAGE = (235, 235, 232)
BEIGE_PAGE = (205, 215, 225)


def _desk(kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "wood":
        img = np.tile(np.array([70, 95, 120], np.float32), (H, W, 1))
        img += rng.normal(0, 9, (H, 1, 1)).repeat(W, 1)  # grain
        img += rng.normal(0, 4, img.shape)
    elif kind == "dark":
        img = np.full((H, W, 3), (45, 48, 50), np.float32) + rng.normal(0, 3, (H, W, 3))
    else:
        img = np.full((H, W, 3), (215, 215, 210), np.float32) + rng.normal(0, 3, (H, W, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _scene(kind: str, cover, width: int, angle: float, seed: int = 1):
    """A 125x88 mm booklet ``width`` px wide with printed detail, rotated by ``angle``."""
    img = _desk(kind, seed)
    height = width * 88 / 125
    r = np.deg2rad(angle)
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    corners = np.array([[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]])
    quad = corners @ rot.T + [930, 520]
    cv2.fillPoly(img, [quad.astype(np.int32)], cover)
    x0, y0 = quad.min(0).astype(int)
    for i in range(int(height / 14)):
        cv2.putText(img, "PASSPORT 123456 <<<<", (x0 + 12, y0 + 20 + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    tuple(int(c * 0.6) for c in cover), 1)
    x1, y1 = quad.max(0)
    return img, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def _detect(img):
    return dg.detect_document_geometry(
        img,
        min_area_fraction=DetectionConfig().geometry_min_area_fraction,
        max_area_fraction=DetectionConfig().geometry_max_area_fraction,
    )


def _found(img, truth) -> bool:
    geom = _detect(img)
    return geom is not None and geom.confidence >= DetectionConfig().min_geometry_confidence and dg._bbox_iou(truth, geom.bbox) >= 0.7


class TestPassportSizedDocuments:
    def test_default_minimum_area_admits_a_passport_that_fills_three_percent_of_the_frame(self):
        assert DetectionConfig().geometry_min_area_fraction <= 0.03

    # A white page on a white desk scores ~0.46, below the stricter default threshold (0.50) that keeps
    # hands/shadows from triggering a capture; it is detected again with min_geometry_confidence <= 0.42.
    @pytest.mark.parametrize("desk", ["wood", "dark"])
    def test_white_page_at_3_7_percent_of_the_frame_is_detected(self, desk):
        img, truth = _scene(desk, WHITE_PAGE, 330, 12)  # 3.7 % of the frame: failed at the old 5 % minimum
        assert _found(img, truth)

    def test_small_page_at_2_3_percent_is_still_detected(self):
        img, truth = _scene("dark", WHITE_PAGE, 260, 0)
        assert _found(img, truth)

    @pytest.mark.parametrize("angle", [0, 12, -20])
    def test_dark_blue_cover_on_a_wooden_desk(self, angle):
        img, truth = _scene("wood", NAVY, 330, angle)
        assert _found(img, truth)

    @pytest.mark.parametrize("angle", [0, 12])
    def test_blue_cover_on_wood_at_6_percent_needs_the_colour_variant(self, angle, monkeypatch):
        img, truth = _scene("wood", NAVY, 420, angle)
        assert _found(img, truth)
        monkeypatch.setattr(dg, "_USE_COLOR", False)
        assert not _found(img, truth)

    @pytest.mark.parametrize("width", [330, 420, 520])
    def test_burgundy_cover_on_a_dark_desk_needs_the_colour_variant(self, width, monkeypatch):
        """Almost no luminance contrast (cover ~54 vs desk ~48 grey levels) but a clear hue difference."""
        img, truth = _scene("dark", BURGUNDY, width, 12)
        assert _found(img, truth)
        monkeypatch.setattr(dg, "_USE_COLOR", False)
        assert not _found(img, truth)  # grey-level analysis alone cannot see it

    @pytest.mark.parametrize("desk", ["wood", "dark", "white"])
    def test_empty_desk_is_not_a_document(self, desk):
        assert _detect(_desk(desk, 5)) is None or _detect(_desk(desk, 5)).confidence < 0.42

    def test_too_small_to_capture_is_not_reported(self):
        img, _truth = _scene("dark", WHITE_PAGE, 90, 0)  # 0.3 % of the frame: a coin, not a document
        geom = _detect(img)
        assert geom is None or geom.confidence < 0.42

    def test_large_pages_are_unaffected(self):
        img = _desk("wood", 2)
        cv2.rectangle(img, (500, 100), (1300, 950), (228, 226, 220), -1)
        for i in range(20):
            cv2.putText(img, "NAME SURNAME 123456", (540, 170 + i * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 1)
        geom = _detect(img)
        assert geom is not None
        assert dg._bbox_iou((500, 100, 800, 850), geom.bbox) >= 0.9


def _scene_at(kind: str, cover, width: int, cx: int, cy: int, angle: float = 8.0):
    """Like ``_scene`` but with the booklet centred anywhere in the frame."""
    img = _desk(kind, 1)
    height = width * 88 / 125
    r = np.deg2rad(angle)
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    corners = np.array([[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]])
    quad = corners @ rot.T + [cx, cy]
    cv2.fillPoly(img, [quad.astype(np.int32)], cover)
    x0, y0 = quad.min(0).astype(int)
    for i in range(int(height / 14)):
        cv2.putText(img, "PASSPORT 123456 <<<<", (x0 + 12, y0 + 20 + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    tuple(int(c * 0.6) for c in cover), 1)
    x1, y1 = quad.max(0)
    return img, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


POSITIONS = {
    "centre": (960, 540), "left": (260, 540), "right": (1660, 540), "top": (960, 170), "bottom": (960, 910),
    "top-left": (260, 170), "top-right": (1660, 170), "bottom-left": (260, 910), "bottom-right": (1660, 910),
}


class TestDocumentNearTheFrameEdges:
    """A fully visible page in a corner or against a side must be detected like one in the middle
    (the background estimate used to be dragged towards the page there, so a dark cover on a dark
    desk faded out)."""

    @pytest.mark.parametrize("position", list(POSITIONS))
    @pytest.mark.parametrize("desk,cover", [("dark", BURGUNDY), ("dark", NAVY), ("wood", NAVY), ("wood", BURGUNDY)])
    def test_detected_with_enough_confidence_anywhere_in_view(self, desk, cover, position):
        img, truth = _scene_at(desk, cover, 330, *POSITIONS[position])
        geom = _detect(img)
        assert geom is not None, position
        assert geom.confidence >= DetectionConfig().min_geometry_confidence, (position, geom.confidence)
        assert dg._bbox_iou(truth, geom.bbox) >= 0.6

    @pytest.mark.parametrize("desk", ["wood", "dark", "white"])
    def test_an_empty_desk_never_yields_a_document_sized_candidate(self, desk):
        geom = _detect(_desk(desk, 1))
        # weak outlines of the noise may exist now that ID-card sized areas are admitted, far below the threshold
        assert geom is None or geom.confidence < DetectionConfig().min_geometry_confidence * 0.6


def _card_scene(kind, color, width, cx, cy, angle=6.0):
    """An ID-1 card (85.6 x 54 mm) with a photo block and text lines."""
    img = _desk(kind, 1)
    height = width * 54 / 85.6
    r = np.deg2rad(angle)
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    corners = np.array([[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]])
    quad = corners @ rot.T + [cx, cy]
    cv2.fillPoly(img, [quad.astype(np.int32)], color)
    x0, y0 = quad.min(0).astype(int)
    dark = tuple(int(v * 0.6) for v in color)
    cv2.rectangle(img, (x0 + int(width * 0.08), y0 + int(height * 0.25)), (x0 + int(width * 0.3), y0 + int(height * 0.7)), dark, -1)
    for i in range(4):
        y = y0 + int(height * (0.3 + i * 0.12))
        cv2.line(img, (x0 + int(width * 0.4), y), (x0 + int(width * 0.9), y), tuple(int(v * 0.5) for v in color), 2)
    x1, y1 = quad.max(0)
    return img, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


CARD_POSITIONS = {
    "centre": (960, 540), "left": (190, 540), "right": (1730, 540), "top": (960, 110), "bottom": (960, 970),
    "top-left": (190, 110), "top-right": (1730, 110), "bottom-left": (190, 970), "bottom-right": (1730, 970),
}


class TestIdCardSizedDocuments:
    """A national ID card is ~1.5 % of a 1080p frame at 220 px wide; the old 2 % floor dropped it everywhere."""

    @pytest.mark.parametrize("position", list(CARD_POSITIONS))
    @pytest.mark.parametrize("desk,color", [("wood", (190, 150, 90)), ("wood", WHITE_PAGE), ("dark", (190, 150, 90)), ("dark", WHITE_PAGE)])
    def test_detected_anywhere_in_view(self, desk, color, position):
        img, truth = _card_scene(desk, color, 220, *CARD_POSITIONS[position])
        geom = _detect(img)
        assert geom is not None, position
        assert geom.confidence >= DetectionConfig().min_geometry_confidence, (position, geom.confidence)
        assert dg._bbox_iou(truth, geom.bbox) >= 0.6

    def test_the_default_minimum_area_admits_an_id_card(self):
        assert DetectionConfig().geometry_min_area_fraction <= 0.012
