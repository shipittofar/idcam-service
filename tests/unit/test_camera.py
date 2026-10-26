"""Camera manager preview format and mock-camera tests."""

from __future__ import annotations

import numpy as np

from idcam.camera.manager import (
    CameraManager,
    describe_preview_frame,
    map_preview_roi_to_sensor,
    normalize_picamera_preview_frame,
)
from idcam.config.schema import CameraConfig


class TestPicameraRgb888Normalize:
    def test_bgr_memory_becomes_rgb(self):
        # Picamera2 RGB888 ndarray: channel 0 is Blue.
        native = np.zeros((8, 12, 3), dtype=np.uint8)
        native[:, :] = (10, 20, 200)  # B, G, R
        rgb = normalize_picamera_preview_frame(native)
        assert rgb.shape == (8, 12, 3)
        assert rgb.dtype == np.uint8
        assert rgb[0, 0, 0] == 200  # R
        assert rgb[0, 0, 1] == 20  # G
        assert rgb[0, 0, 2] == 10  # B
        assert rgb.flags["C_CONTIGUOUS"]

    def test_gray_passthrough(self):
        gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
        out = normalize_picamera_preview_frame(gray)
        assert np.array_equal(out, gray)

    def test_xrgb_four_channel_drops_padding(self):
        native = np.zeros((2, 2, 4), dtype=np.uint8)
        native[:, :] = (10, 20, 200, 255)  # B, G, R, X
        rgb = normalize_picamera_preview_frame(native)
        assert rgb.shape == (2, 2, 3)
        assert rgb[0, 0, 0] == 200
        assert rgb[0, 0, 2] == 10

    def test_describe_reports_channel_means(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[..., 0] = 10
        frame[..., 1] = 20
        frame[..., 2] = 30
        info = describe_preview_frame(frame)
        assert info["shape"] == [4, 4, 3]
        assert info["dtype"] == "uint8"
        assert info["channel_means"]["c0"] == 10.0
        assert info["channel_means"]["c2"] == 30.0


class TestMockCamera:
    def test_mock_preview_is_rgb_shaped(self):
        mgr = CameraManager(CameraConfig(preview_width=320, preview_height=240), use_mock=True)
        mgr.initialize()
        frame = mgr.capture_preview_array()
        assert frame.shape == (240, 320, 3)
        meta = mgr.capture_metadata()
        assert meta["AfState"] == 2
        advertised = mgr.get_controls()
        assert "AfMode" in advertised
        assert "AfState" not in advertised
        mapping = mgr.apply_document_focus_window((288, 162, 1344, 756), (1920, 1080))
        assert mapping["preview_roi"] == (288, 162, 1344, 756)
        assert mapping["preview_size"] == (1920, 1080)
        assert mapping["sensor_roi"][2] > 0
        mgr.shutdown()


class TestPreviewToSensorMap:
    def test_16_by_9_preview_not_70_percent_of_4_by_3_sensor(self):
        preview_roi = (288, 162, 1344, 756)
        mapped = map_preview_roi_to_sensor(preview_roi, (1920, 1080), (0, 898, 9152, 5148))
        assert mapped != (1936, 1960, 5376, 3024)
        assert mapped[1] >= 898
