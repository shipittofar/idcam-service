"""Image validation tests."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from idcam.camera.quality import ImageValidator
from idcam.config.schema import QualityConfig
from idcam.exceptions import ImageValidationError


@pytest.fixture
def quality_config():
    return QualityConfig(min_sharpness=10.0, min_file_bytes=1000, min_width=100, min_height=100)


@pytest.fixture
def valid_image(tmp_path):
    path = tmp_path / "valid.jpg"
    # Create image with edges for sharpness
    arr = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path, format="JPEG", quality=90)
    return path


class TestImageValidator:
    def test_valid_image(self, quality_config, valid_image):
        validator = ImageValidator(quality_config)
        result = validator.validate(valid_image)
        assert result.width == 640
        assert result.height == 480

    def test_missing_file(self, quality_config, tmp_path):
        validator = ImageValidator(quality_config)
        with pytest.raises(ImageValidationError, match="not found"):
            validator.validate(tmp_path / "missing.jpg")

    def test_empty_file(self, quality_config, tmp_path):
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        validator = ImageValidator(quality_config)
        with pytest.raises(ImageValidationError):
            validator.validate(path)

    def test_corrupt_file(self, quality_config, tmp_path):
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"not a jpeg at all" * 100)
        validator = ImageValidator(quality_config)
        with pytest.raises(ImageValidationError):
            validator.validate(path)

    def test_too_small_dimensions(self, quality_config, tmp_path):
        path = tmp_path / "tiny.jpg"
        img = Image.new("RGB", (10, 10), color=(128, 128, 128))
        img.save(path)
        validator = ImageValidator(quality_config)
        with pytest.raises(ImageValidationError, match="too small"):
            validator.validate(path)

    def test_uniform_image_rejected(self, quality_config, tmp_path):
        path = tmp_path / "uniform.jpg"
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        img.save(path, quality=90)
        validator = ImageValidator(quality_config)
        with pytest.raises(ImageValidationError):
            validator.validate(path)


class TestLargeStillValidation:
    """64 MP stills are measured at 1/4 scale; reported size stays the full-resolution size."""

    @staticmethod
    def _page(path, blur_sigma=0.0):
        import cv2

        rng = np.random.default_rng(0)
        img = np.full((4400, 5800, 3), 170, dtype=np.uint8)
        for i in range(40):
            cv2.putText(img, "NAME SURNAME 1234567890 ABC", (400, 300 + i * 90), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (20, 20, 20), 4)
        img = (img + rng.normal(0, 3, img.shape)).clip(0, 255).astype(np.uint8)
        if blur_sigma:
            img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def test_sharp_large_still_passes_and_keeps_full_size(self, tmp_path):
        path = tmp_path / "big_sharp.jpg"
        self._page(path)
        result = ImageValidator(QualityConfig(min_file_bytes=1000)).validate(path)
        assert (result.width, result.height) == (5800, 4400)

    def test_blurry_large_still_is_rejected(self, tmp_path):
        path = tmp_path / "big_blur.jpg"
        self._page(path, blur_sigma=8.0)
        with pytest.raises(ImageValidationError, match="blurry"):
            ImageValidator(QualityConfig(min_file_bytes=1000)).validate(path)

    def test_blurry_still_is_rejected_even_when_it_matches_the_focus_time_sharpness(self, tmp_path):
        """Focus can converge on a soft peak; matching that peak must not turn blur into an upload."""
        path = tmp_path / "big_soft.jpg"
        self._page(path, blur_sigma=8.0)
        validator = ImageValidator(QualityConfig(min_file_bytes=1000))
        with pytest.raises(ImageValidationError, match="blurry"):
            validator.validate(path, reference_sharpness=1.0)

    def test_sharp_still_reports_focus_ratio(self, tmp_path):
        path = tmp_path / "big_sharp2.jpg"
        self._page(path)
        result = ImageValidator(QualityConfig(min_file_bytes=1000)).validate(path, reference_sharpness=100.0)
        assert result.focus_ratio is not None and result.focus_ratio > 0
