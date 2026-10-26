"""Vision pipeline tests."""

import cv2
import numpy as np
import pytest

from idcam.config.schema import DetectionConfig
from idcam.vision.document_detector import DocumentDetector, DocumentDetectionDebug, DocumentPresence, gradient_edge_density
from idcam.vision.sharpness import laplacian_variance, mean_brightness
from idcam.vision.stability_detector import StabilityDetector


@pytest.fixture
def detection_config():
    return DetectionConfig(
        motion_threshold=5.0,
        edge_density_threshold=0.01,
        stable_frames_required=3,
        removal_frames_required=3,
        detect_frames_required=1,
        lose_frames_required=2,
    )


class TestDocumentDetector:
    def test_no_document_on_uniform_frame(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        detector.update_baseline(frame)
        assert not detector.detect(frame)

    def test_document_detected_on_change(self, detection_config):
        detector = DocumentDetector(detection_config)
        baseline = np.full((480, 640, 3), 200, dtype=np.uint8)
        detector.update_baseline(baseline)
        frame = baseline.copy()
        # Add high-contrast document region
        frame[100:380, 100:540] = 50
        frame[100:120, 100:540] = 0
        frame[360:380, 100:540] = 0
        frame[100:380, 100:120] = 0
        frame[100:380, 520:540] = 0
        assert detector.detect(frame)

    def test_scene_empty_after_removal(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = np.full((480, 640, 3), 150, dtype=np.uint8)
        for _ in range(20):
            detector.update_baseline(frame)
        assert detector.is_scene_empty(frame)


class TestDocumentDetectionDebug:
    def test_debug_values_match_detection_without_changing_result(self, detection_config):
        detector = DocumentDetector(detection_config)
        baseline = np.full((480, 640, 3), 200, dtype=np.uint8)
        detector.update_baseline(baseline)
        frame = baseline.copy()
        frame[100:380, 100:540] = 50
        frame[100:120, 100:540] = 0
        frame[360:380, 100:540] = 0
        frame[100:380, 100:120] = 0
        frame[100:380, 520:540] = 0
        result = detector.detect(frame)
        info = detector.debug_info()
        assert isinstance(info, DocumentDetectionDebug)
        assert result is True
        assert info.detected is True
        assert info.geometry_confidence >= detection_config.min_geometry_confidence
        assert detector.detect(frame) is True

    def test_debug_info_does_not_run_detection(self, detection_config):
        detector = DocumentDetector(detection_config)
        before = detector.debug_info()
        again = detector.debug_info()
        assert before == again
        assert before.detected is False
        assert before.motion == 0.0

    def test_reject_reason_on_uniform_scene(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        detector.detect(frame)
        info = detector.debug_info()
        assert info.detected is False
        assert info.reject_reason
        assert info.frame_width == 640

    def test_rgb_preview_format_detects_document(self, detection_config):
        detector = DocumentDetector(detection_config)
        frame = _document_frame_like()
        assert detector.detect(frame)
        info = detector.debug_info()
        assert info.edge_density > 0
        assert info.reject_reason == ""


def _document_frame_like(width=320, height=240):
    frame = np.full((height, width, 3), 210, dtype=np.uint8)
    frame[30:210, 40:280] = 50
    frame[30:38, 40:280] = 0
    frame[202:210, 40:280] = 0
    frame[30:210, 40:48] = 0
    frame[30:210, 272:280] = 0
    return frame

    def test_static_document_is_not_moving(self, detection_config):
        detector = DocumentDetector(detection_config)
        empty = np.full((240, 320, 3), 200, dtype=np.uint8)
        doc = empty.copy()
        doc[40:200, 40:280] = 50
        doc[40:50, 40:280] = 0
        doc[190:200, 40:280] = 0
        detector.update_baseline(empty)
        first = detector.classify(doc)
        assert first != DocumentPresence.NO_DOCUMENT
        still = detector.classify(doc)
        assert still != DocumentPresence.DOCUMENT_MOVING

    def test_textured_empty_table_is_scene_empty(self, detection_config):
        detector = DocumentDetector(detection_config)
        yy, xx = np.indices((240, 320))
        table = np.stack(
            [
                (80 + (xx % 16)).astype(np.uint8),
                (70 + (yy % 16)).astype(np.uint8),
                np.full((240, 320), 60, dtype=np.uint8),
            ],
            axis=-1,
        )
        detector.update_baseline(table)
        assert detector.is_scene_empty(table)
        assert detector.last_edge_density > 0.0

    def test_mild_defocus_has_gradient_edges(self, detection_config):
        sharp = np.full((240, 320, 3), 200, dtype=np.uint8)
        sharp[40:200, 40:280] = 40
        for y in range(50, 190, 8):
            sharp[y : y + 2, 50:270] = 0
        mild = cv2.GaussianBlur(sharp, (9, 9), 0)
        gray = cv2.cvtColor(mild, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, 50, 150)
        canny_density = float(np.count_nonzero(canny)) / canny.size
        sobel_density = gradient_edge_density(gray)
        assert sobel_density > canny_density
        detector = DocumentDetector(detection_config)
        detector.update_baseline(np.full((240, 320, 3), 180, dtype=np.uint8))
        assert detector.detect(mild)


class TestStabilityDetector:
    def test_not_stable_initially(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert not sd.update(frame)

    def test_stable_after_consecutive_frames(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = False
        for _ in range(10):
            result = sd.update(frame)
        assert result

    def test_reset_on_motion(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        for _ in range(2):
            sd.update(frame)
        moving = frame.copy()
        moving[100:400, 100:500] = 255
        sd.update(moving)
        assert sd.stable_count == 0

    def test_stability_debug_info(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        results = [sd.update(frame) for _ in range(10)]
        info = sd.debug_info()
        assert results[-1] is True
        assert info.stable is True
        assert info.stable_count == sd.stable_count
        assert info.stable_frames_required == detection_config.stable_frames_required
        assert info.threshold == detection_config.motion_threshold
        assert sd.update(frame) is True

    def test_small_preview_noise_keeps_stable_document(self, detection_config):
        sd = StabilityDetector(detection_config)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        for _ in range(10):
            sd.update(frame)
        assert sd.stable_count >= detection_config.stable_frames_required
        noisy = frame.copy()
        noisy[200:210, 200:210] = 129
        assert sd.update(noisy) is True
        assert sd.stable_count >= detection_config.stable_frames_required

class TestDetectionHysteresis:
    def test_requires_consecutive_frames_to_detect(self):
        cfg = DetectionConfig(
            motion_threshold=5.0,
            edge_density_threshold=0.01,
            detect_frames_required=3,
            lose_frames_required=3,
        )
        detector = DocumentDetector(cfg)
        baseline = np.full((480, 640, 3), 200, dtype=np.uint8)
        detector.update_baseline(baseline)
        frame = baseline.copy()
        frame[100:380, 100:540] = 50
        frame[100:120, 100:540] = 0
        frame[360:380, 100:540] = 0
        frame[100:380, 100:120] = 0
        frame[100:380, 520:540] = 0
        assert not detector.detect(frame)
        assert not detector.detect(frame)
        assert detector.detect(frame)
        assert detector.is_latched

    def test_holds_detection_through_noise(self):
        cfg = DetectionConfig(
            motion_threshold=5.0,
            edge_density_threshold=0.01,
            detect_frames_required=2,
            lose_frames_required=5,
            lose_motion_factor=0.4,
            lose_edge_factor=0.4,
        )
        detector = DocumentDetector(cfg)
        baseline = np.full((240, 320, 3), 200, dtype=np.uint8)
        detector.update_baseline(baseline)
        doc = baseline.copy()
        doc[40:200, 40:280] = 50
        doc[40:50, 40:280] = 0
        for _ in range(3):
            detector.detect(doc)
        assert detector.is_latched
        noisy = doc.copy()
        noisy[100:110, 100:110] = 195
        assert detector.detect(noisy)
        assert detector.is_latched


    def test_roi_is_preview_pixels_inside_frame(self, detection_config):
        detector = DocumentDetector(detection_config)
        empty = np.full((1080, 1920, 3), 200, dtype=np.uint8)
        doc = empty.copy()
        doc[200:880, 400:1520] = 40
        doc[200:230, 400:1520] = 0
        doc[850:880, 400:1520] = 0
        detector.update_baseline(empty)
        detector.detect(doc)
        roi = detector.document_preview_roi(doc)
        x, y, w, h = roi
        assert x >= 0 and y >= 0
        assert x + w <= 1920
        assert y + h <= 1080
        assert w > 100 and h > 100
        # Must not look like the old sensor-space debug window.
        assert roi != (1936, 1960, 5376, 3024)
        assert x + w <= 1920


class TestSharpness:
    def test_blurry_lower_than_sharp(self):
        blurry = np.full((480, 640, 3), 128, dtype=np.uint8)
        sharp = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        assert laplacian_variance(blurry) < laplacian_variance(sharp)

    def test_brightness(self):
        dark = np.full((100, 100), 30, dtype=np.uint8)
        bright = np.full((100, 100), 200, dtype=np.uint8)
        assert mean_brightness(bright) > mean_brightness(dark)


class TestBlurKillsEdgeDensity:
    def test_heavy_blur_reduces_textured_edge_density(self):
        """Sobel density of a printed card drops under heavy defocus; a solid
        rectangle's outline can *spread* after Gaussian blur, so this uses texture."""
        h, w = 240, 320
        yy, xx = np.indices((h, w))
        checker = np.where(((xx // 8) + (yy // 8)) % 2 == 0, 220, 40).astype(np.uint8)
        textured = np.stack([checker, checker, checker], axis=-1)
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        frame[40:200, 40:280] = textured[40:200, 40:280]
        blurred = cv2.GaussianBlur(frame, (51, 51), 18)
        sharp = gradient_edge_density(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        blur = gradient_edge_density(cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY))
        assert sharp > 0.15
        assert blur < sharp * 0.25

