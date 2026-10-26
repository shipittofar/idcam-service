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
