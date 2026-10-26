"""Camera diagnostic helper tests."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from idcam.camera.diagnostic import analyze_channels, validate_jpeg_file


def test_analyze_rgb_channels():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(50, 200, 120, dtype=np.uint8)
    frame[:, :, 1] = 100
    frame[:, :, 2] = 50
    info = analyze_channels(frame, "test_rgb")
    assert info["interpreted_as_rgb"]["R"] > 100
    assert info["gray_std"] > 0


def test_validate_jpeg(tmp_path):
    path = tmp_path / "test.jpg"
    arr = np.random.randint(20, 200, (240, 320, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=90)
    result = validate_jpeg_file(path, expected_width=320, expected_height=240, tolerance=8)
    assert result.ok
    assert result.file_bytes > 0
