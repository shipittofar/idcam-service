"""Sharp page corners from fitted edge lines; "the captured page is still there" for the removal wait."""

from __future__ import annotations

import cv2
import numpy as np

from idcam.config.schema import DetectionConfig
from idcam.vision.document_detector import DocumentDetector
from idcam.vision.document_geometry import _contour_to_corners

from test_hand_held import GREEN, _card, _desk


def _rounded_quad_contour(box):
    img = np.zeros((400, 600), np.uint8)
    cv2.fillPoly(img, [np.array(box, np.int32)], 255)
    img = cv2.GaussianBlur(img, (0, 0), 9)  # rounds the corners like a real card / a soft lens
    img = (img > 128).astype(np.uint8) * 255
    return max(cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)


def test_corners_sit_on_the_real_corner_of_a_rounded_page():
    box = [[100, 80], [480, 100], [470, 300], [90, 280]]
    corners = _contour_to_corners(_rounded_quad_contour(box))
    truth = sorted(map(tuple, box))
    got = sorted(corners)
    for (tx, ty), (gx, gy) in zip(truth, got):
        assert abs(tx - gx) <= 3 and abs(ty - gy) <= 3, (truth, got)


class TestDocumentStillThere:
    def _setup(self):
        det = DocumentDetector(DetectionConfig())
        desk = _desk("dark")
        page, _ = _card("dark", GREEN, None)
        for _ in range(3):
            det.detect(page.copy())
        sig = det.document_signature(page)
        assert sig is not None
        return det, desk, page, sig

    def test_same_picture_is_still_there(self):
        det, _desk_img, page, sig = self._setup()
        assert det.document_still_there(page.copy(), sig) is True

    def test_bare_desk_is_not(self):
        det, desk, _page, sig = self._setup()
        assert det.document_still_there(desk.copy(), sig) is False

    def test_no_signature_is_not(self):
        det = DocumentDetector(DetectionConfig())
        assert det.document_still_there(_desk("dark"), None) is False
