"""Exposure / white-balance options of camera:."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from idcam.camera.manager import CameraManager
from idcam.config.schema import CameraConfig


def _controls(**kwargs):
    return CameraManager(CameraConfig(**kwargs), use_mock=True)._image_controls()


class TestConfig:
    def test_default_is_fully_automatic(self):
        cfg = CameraConfig()
        assert cfg.exposure_time_us is None and cfg.analogue_gain is None
        assert cfg.exposure_value == 0.0 and cfg.ae_exposure_mode == "normal" and cfg.awb_mode == "auto"
        assert _controls() == {}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"analogue_gain": 2.0},  # gain without an exposure time
            {"exposure_time_us": 10},
            {"exposure_time_us": 2_000_000},
            {"exposure_time_us": 8000, "analogue_gain": 0.5},
            {"exposure_value": 9},
            {"ae_exposure_mode": "fast"},
            {"awb_mode": "neon"},
        ],
    )
    def test_invalid_values_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            CameraConfig(**kwargs)


class TestControlMapping:
    def test_fixed_exposure_switches_auto_exposure_off(self):
        c = _controls(exposure_time_us=8000, analogue_gain=2.5)
        assert c == {"AeEnable": False, "ExposureTime": 8000, "AnalogueGain": 2.5}

    def test_fixed_exposure_without_gain_uses_unity_gain(self):
        assert _controls(exposure_time_us=8000)["AnalogueGain"] == 1.0

    def test_auto_exposure_compensation_and_short_mode(self):
        c = _controls(exposure_value=-0.7, ae_exposure_mode="short")
        assert c["ExposureValue"] == -0.7 and c["AeExposureMode"] == 1
        assert "AeEnable" not in c and "ExposureTime" not in c

    def test_fixed_exposure_ignores_auto_only_options(self):
        c = _controls(exposure_time_us=8000, exposure_value=1.0, ae_exposure_mode="long")
        assert "ExposureValue" not in c and "AeExposureMode" not in c

    @pytest.mark.parametrize("mode,value", [("tungsten", 1), ("fluorescent", 2), ("indoor", 3), ("daylight", 4), ("cloudy", 5)])
    def test_white_balance_mode(self, mode, value):
        assert _controls(awb_mode=mode) == {"AwbMode": value}


class FakePicamera2:
    camera_controls = {
        "AeEnable": (False, True, True),
        "ExposureTime": (100, 1_000_000, 10000),
        "AnalogueGain": (1.0, 16.0, 1.0),
        "ExposureValue": (-8.0, 8.0, 0.0),
        "AfMode": (0, 2, 0),
        "LensPosition": (0.0, 32.0, 1.0),
    }
    instances: list["FakePicamera2"] = []

    def __init__(self):
        self.configs = {}
        self.instances.append(self)

    def create_preview_configuration(self, **kw):
        self.configs["preview"] = kw
        return {"kind": "preview", **kw}

    def create_still_configuration(self, **kw):
        self.configs["still"] = kw
        return {"kind": "still", **kw}

    def configure(self, config):
        pass

    def start(self):
        pass

    def set_controls(self, controls):
        pass


class TestAppliedToBothStreams:
    def _init(self, **kwargs):
        FakePicamera2.instances.clear()
        with patch("idcam.camera.manager.Picamera2", FakePicamera2):
            manager = CameraManager(CameraConfig(**kwargs), use_mock=True)
            manager._use_mock = False  # Picamera2 is not installed here: use the fake
            manager.initialize()
        return FakePicamera2.instances[-1]

    def test_preview_and_still_carry_the_same_controls(self):
        cam = self._init(exposure_time_us=9000, analogue_gain=3.0)
        expected = {"AeEnable": False, "ExposureTime": 9000, "AnalogueGain": 3.0}
        assert cam.configs["preview"]["controls"] == expected
        assert cam.configs["still"]["controls"] == expected

    def test_default_adds_no_controls(self):
        cam = self._init()
        assert cam.configs["preview"]["controls"] == {} and cam.configs["still"]["controls"] == {}

    def test_controls_the_camera_does_not_advertise_are_dropped_not_fatal(self):
        cam = self._init(exposure_value=1.0, awb_mode="daylight")  # AwbMode is not in the fake's controls
        assert cam.configs["preview"]["controls"] == {"ExposureValue": 1.0}
