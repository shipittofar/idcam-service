"""Upload image preparation: crop to the document, limit size, re-encode."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from idcam.camera.upload_image import prepare_upload_image
from idcam.config.schema import UploadImageConfig


@pytest.fixture
def capture(tmp_path):
    """5800x4400 'desk' with a text page at (1800, 1200) sized 2200x1700."""
    img = np.full((4400, 5800, 3), (70, 80, 75), dtype=np.uint8)
    cv2.rectangle(img, (1800, 1200), (4000, 2900), (225, 224, 218), -1)
    for i in range(30):
        cv2.putText(img, "NAME SURNAME 1234567890", (1900, 1300 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (30, 30, 30), 3)
    path = tmp_path / "capture.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


DOC_ROI = (1800, 1200, 2200, 1700)


class TestPrepareUploadImage:
    def test_crops_to_document_and_limits_size(self, capture, tmp_path):
        out = prepare_upload_image(capture, tmp_path / "up.jpg", DOC_ROI, UploadImageConfig())
        assert out.cropped
        assert max(out.width, out.height) <= 2400
        assert out.file_bytes < capture.stat().st_size / 2  # synthetic source is flat, real photos shrink far more
        with Image.open(out.path) as img:
            assert img.size == (out.width, out.height)

    def test_crop_keeps_margin_so_the_whole_page_is_inside(self, capture, tmp_path):
        cfg = UploadImageConfig(max_dimension=0, crop_margin=0.10)
        out = prepare_upload_image(capture, tmp_path / "up.jpg", DOC_ROI, cfg)
        # native size kept: document 2200x1700 plus 10% each side
        assert out.width == 2200 + 2 * 220 and out.height == 1700 + 2 * 170

    def test_no_resize_when_max_dimension_zero_and_no_crop(self, capture, tmp_path):
        cfg = UploadImageConfig(crop_to_document=False, max_dimension=0)
        out = prepare_upload_image(capture, tmp_path / "up.jpg", DOC_ROI, cfg)
        assert (out.width, out.height) == (5800, 4400)
        assert not out.cropped

    def test_implausible_roi_keeps_the_whole_frame(self, capture, tmp_path):
        out = prepare_upload_image(capture, tmp_path / "up.jpg", (10, 10, 20, 20), UploadImageConfig())
        assert not out.cropped
        assert max(out.width, out.height) <= 2400

    def test_missing_roi_only_resizes(self, capture, tmp_path):
        out = prepare_upload_image(capture, tmp_path / "up.jpg", None, UploadImageConfig())
        assert not out.cropped
        assert out.width == 2400

    def test_document_text_stays_legible_after_shrinking(self, capture, tmp_path):
        out = prepare_upload_image(capture, tmp_path / "up.jpg", DOC_ROI, UploadImageConfig())
        gray = cv2.cvtColor(cv2.imread(str(out.path)), cv2.COLOR_BGR2GRAY)
        # text strokes (dark) and paper (bright) are both still present
        assert (gray < 90).mean() > 0.01 and (gray > 200).mean() > 0.3
