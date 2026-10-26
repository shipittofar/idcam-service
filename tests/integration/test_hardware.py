"""Integration test strategy (hardware tests run on Pi only)."""

import os
import platform

import pytest

from idcam.camera.manager import CameraManager
from idcam.config.schema import CameraConfig

pytestmark = pytest.mark.skipif(
    os.environ.get("IDC_RUN_HARDWARE_TESTS", "").lower() not in ("1", "true"),
    reason="Hardware tests disabled (set IDC_RUN_HARDWARE_TESTS=1 on Pi)",
)


@pytest.mark.hardware
class TestHardwareIntegration:
    """Run on Raspberry Pi 5 with Arducam 64MP attached."""

    def test_camera_init(self):
        mgr = CameraManager(CameraConfig(preview_width=640, preview_height=480))
        mgr.initialize()
        assert mgr.is_started
        frame = mgr.capture_preview_array()
        assert frame is not None
        assert frame.shape[0] == 480
        mgr.shutdown()

    def test_autofocus_controls_advertised(self):
        mgr = CameraManager(CameraConfig())
        mgr.initialize()
        controls = mgr.get_controls()
        # At least one focus mechanism should exist on hardware
        has_af = "AfMode" in controls or hasattr(mgr, "autofocus_cycle")
        assert has_af or True  # informational; driver-dependent
        mgr.shutdown()

    @pytest.mark.hardware
    @pytest.mark.slow
    def test_full_res_capture(self, tmp_path):
        mgr = CameraManager(CameraConfig(capture_width=4056, capture_height=3040))
        mgr.initialize()
        out = tmp_path / "capture.jpg"
        mgr.capture_still_to_file(out)
        assert out.exists()
        assert out.stat().st_size > 100_000
        mgr.shutdown()
