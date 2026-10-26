"""Autofocus controller uses live metadata, not advertised control ranges."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from idcam.camera.autofocus import AutofocusController
from idcam.camera.manager import map_preview_roi_to_sensor, parse_xywh
from idcam.config.schema import CameraConfig
from idcam.exceptions import AutofocusError
from idcam.vision.sharpness import accept_preview_focus


def _sharp_frame() -> np.ndarray:
    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    frame[20:100, 20:30] = 0
    frame[20:30, 20:140] = 0
    frame[90:100, 20:140] = 0
    return frame


class FakeCamera:
    def __init__(
        self,
        *,
        advertised: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        cycle_result: Any = True,
        has_cycle: bool = True,
    ) -> None:
        self.advertised = advertised or {
            "AfMode": (0, 2, 0),
            "AfTrigger": (0, 1, 0),
            "LensPosition": (0.0, 10.0, 1.0),
        }
        self.metadata = metadata or {"AfState": 2, "LensPosition": 1.25}
        self.cycle_result = cycle_result
        self.has_cycle = has_cycle
        self.cycle_calls = 0
        self.set_calls: list[dict[str, Any]] = []
        self.metadata_calls = 0
        self.controls_calls = 0

    def get_controls(self) -> dict[str, Any]:
        self.controls_calls += 1
        return dict(self.advertised)

    def capture_metadata(self) -> dict[str, Any]:
        self.metadata_calls += 1
        return dict(self.metadata)

    def set_controls(self, controls: dict[str, Any]) -> None:
        self.set_calls.append(dict(controls))

    def apply_document_focus_window(
        self,
        preview_roi: tuple[int, int, int, int],
        preview_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        self.applied_roi = tuple(int(v) for v in preview_roi)
        size = preview_size or (160, 120)
        return {
            "preview_roi": self.applied_roi,
            "preview_size": size,
            "scaler_crop": (0, 0, int(size[0]), int(size[1])),
            "sensor_roi": self.applied_roi,
        }

    def autofocus_cycle(self) -> Any:
        if not self.has_cycle:
            raise AssertionError("autofocus_cycle should not be called")
        self.cycle_calls += 1
        return self.cycle_result

    def capture_preview_array(self) -> np.ndarray:
        return _sharp_frame()


class TestAutofocusController:
    @pytest.mark.asyncio
    async def test_sharp_preview_still_runs_hardware_af(self):
        """AF always runs once: a frame compared with itself proves nothing about the lens."""
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert camera.cycle_calls >= 1
        assert result.method != "already_sharp"
        assert result.roi[2] > 0

    @pytest.mark.asyncio
    async def test_prefers_autofocus_cycle_when_not_yet_sharp(self):
        class NeedsCycle(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                if self.cycle_calls:
                    return _sharp_frame()
                return np.full((120, 160, 3), 128, dtype=np.uint8)

        camera = NeedsCycle()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert camera.cycle_calls == 1
        assert result.method == "hardware_cycle"
        assert result.improved is True
        assert result.sharpness >= 1.0

    @pytest.mark.asyncio
    async def test_wait_reads_af_state_from_metadata_not_control_ranges(self):
        class NeedsTrigger(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                if self.set_calls:
                    return _sharp_frame()
                return np.full((120, 160, 3), 128, dtype=np.uint8)

            def trigger_autofocus(self) -> None:
                self.set_controls({"AfMode": 1, "AfTrigger": 0})

            def wait_for_lens_settle(self, timeout: float = 3.5) -> None:
                return None

        camera = NeedsTrigger(has_cycle=False)
        camera.advertised["AfState"] = (0, 3, 0)
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        caps = ctl.probe_capabilities(camera)
        caps.autofocus_cycle = False
        ctl.capabilities = caps
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert camera.metadata_calls >= 1
        assert camera.set_calls
        assert camera.set_calls[0]["AfMode"] == 1
        assert camera.set_calls[0]["AfTrigger"] == 0

    @pytest.mark.asyncio
    async def test_missing_af_state_does_not_block_until_timeout(self):
        class NeedsTrigger(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                if self.set_calls:
                    return _sharp_frame()
                return np.full((120, 160, 3), 128, dtype=np.uint8)

            def trigger_autofocus(self) -> None:
                self.set_controls({"AfMode": 1, "AfTrigger": 0})

            def wait_for_lens_settle(self, timeout: float = 3.5) -> None:
                return None

        camera = NeedsTrigger(has_cycle=False, metadata={"LensPosition": 0.8})
        ctl = AutofocusController(CameraConfig(af_timeout_sec=2.0, af_max_retries=1))
        caps = ctl.probe_capabilities(camera)
        caps.autofocus_cycle = False
        ctl.capabilities = caps
        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert camera.metadata_calls >= 1

    @pytest.mark.asyncio
    async def test_verify_focus_uses_metadata(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig())
        ctl.probe_capabilities(camera)
        result = await ctl.verify_focus(camera, sharpness_threshold=1.0)
        assert result.success is True
        assert result.af_state == 2
        assert camera.metadata_calls >= 1

    @pytest.mark.asyncio
    async def test_low_sharpness_retries_then_fails(self):
        class Blurry(FakeCamera):
            def capture_preview_array(self) -> np.ndarray:
                return np.full((80, 80, 3), 128, dtype=np.uint8)

        camera = Blurry()
        ctl = AutofocusController(CameraConfig(af_max_retries=2, af_timeout_sec=1.0))
        ctl.probe_capabilities(camera)
        with pytest.raises(AutofocusError, match="did not improve"):
            await ctl.run_autofocus(camera, sharpness_threshold=500.0)
        assert camera.cycle_calls == 2
        err = None
        try:
            await ctl.run_autofocus(camera, sharpness_threshold=500.0)
        except AutofocusError as exc:
            err = str(exc)
        assert err is not None
        assert "before=" in err
        assert "after=" in err
        assert "preview_roi=" in err


class TestPreviewRoiMapping:
    def test_preview_roi_maps_through_live_scaler_crop(self):
        preview_size = (1920, 1080)
        preview_roi = (288, 162, 1344, 756)
        # 16:9 crop on a 4:3 9152x6944 sensor — not ScalerCropMaximum 70%.
        scaler = (0, 898, 9152, 5148)
        mapped = map_preview_roi_to_sensor(preview_roi, preview_size, scaler)
        independent_70_of_sensor = (1373, 1042, 6406, 4860)
        assert mapped != independent_70_of_sensor
        assert mapped[0] >= scaler[0]
        assert mapped[1] >= scaler[1]
        assert mapped[0] + mapped[2] <= scaler[0] + scaler[2] + 1
        assert mapped[1] + mapped[3] <= scaler[1] + scaler[3] + 1
        # 70% of 1920x1080 stays inside the 1920x1080 preview, unlike 1936,1960,5376,3024.
        assert preview_roi[0] + preview_roi[2] <= preview_size[0]
        assert preview_roi[1] + preview_roi[3] <= preview_size[1]
        reported_sensor_window = (1936, 1960, 5376, 3024)
        assert reported_sensor_window[2] > preview_size[0]
        assert mapped != reported_sensor_window

    def test_parse_xywh_tuple(self):
        assert parse_xywh((10, 20, 30, 40)) == (10, 20, 30, 40)
        assert parse_xywh(None) is None


class TestPreviewFocusAcceptance:
    def test_improvement_10_to_30_succeeds_below_floor(self):
        assert accept_preview_focus(before=10.0, after=30.0, absolute_floor=48.0) is True

    def test_degradation_10_to_8_fails(self):
        assert accept_preview_focus(before=10.0, after=8.0, absolute_floor=48.0) is False

    def test_absolute_floor_does_not_veto_improvement(self):
        assert accept_preview_focus(before=11.8, after=11.1, absolute_floor=48.0) is False
        assert accept_preview_focus(before=10.0, after=35.0, absolute_floor=48.0) is True

    @pytest.mark.asyncio
    async def test_controller_accepts_10_to_30_despite_threshold_48(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=1))
        ctl.probe_capabilities(camera)
        scores = iter([10.0, 30.0])
        with patch("idcam.camera.autofocus.preview_sharpness", side_effect=lambda *_a, **_k: next(scores)):
            result = await ctl.run_autofocus(camera, sharpness_threshold=48.0)
        assert result.success is True
        assert result.before_sharpness == 10.0
        assert result.sharpness == 30.0
        assert result.roi[2] > 0
        assert result.roi[0] + result.roi[2] <= 160

    @pytest.mark.asyncio
    async def test_controller_rejects_10_to_8(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig(af_timeout_sec=1.0, af_max_retries=2))
        ctl.probe_capabilities(camera)
        scores = iter([10.0, 8.0, 8.0, 8.0])
        with patch("idcam.camera.autofocus.preview_sharpness", side_effect=lambda *_a, **_k: next(scores)):
            with pytest.raises(AutofocusError, match="after=8.0"):
                await ctl.run_autofocus(camera, sharpness_threshold=48.0, preview_roi=(20, 20, 80, 60))
        assert camera.cycle_calls == 2

    @pytest.mark.asyncio
    async def test_verify_does_not_reject_improved_preview_below_48(self):
        camera = FakeCamera()
        ctl = AutofocusController(CameraConfig())
        ctl.probe_capabilities(camera)
        ctl.debug.focus_before = 10.0
        ctl.debug.focus_after = 30.0
        ctl._last_preview_roi = (20, 20, 80, 60)
        with patch("idcam.camera.autofocus.preview_sharpness", return_value=28.0):
            result = await ctl.verify_focus(camera, sharpness_threshold=48.0)
        assert result.success is True
        assert result.sharpness == 28.0



class _LensCamera(FakeCamera):
    """Preview sharpness peaks at lens position ``peak`` (dioptres)."""

    def __init__(self, *, start: float, peak: float) -> None:
        super().__init__(metadata={"AfState": 2, "LensPosition": start})
        self.lens = start
        self.peak = peak
        self.lens_calls: list[float] = []

    def set_lens_position(self, position: float) -> None:
        self.lens = float(position)
        self.lens_calls.append(self.lens)

    def lens_position_range(self) -> tuple[float, float]:
        return (0.0, 32.0)

    def capture_preview_array(self) -> np.ndarray:
        import cv2

        sigma = 0.3 + 2.0 * abs(self.lens - self.peak)
        return cv2.GaussianBlur(_sharp_frame(), (0, 0), sigma)


class TestFocusRefinement:
    @pytest.mark.asyncio
    async def test_hill_climb_finds_sharper_lens_position_outside_af_result(self):
        camera = _LensCamera(start=10.0, peak=8.2)
        cfg = CameraConfig(af_timeout_sec=1.0, af_max_retries=1, af_refine_settle_sec=0.05)
        ctl = AutofocusController(cfg)
        ctl.probe_capabilities(camera)

        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)

        assert result.success is True
        assert "refine" in result.method
        assert abs(camera.lens - 8.2) <= 0.5
        assert abs(result.lens_after - 8.2) <= 0.5

    @pytest.mark.asyncio
    async def test_refinement_can_be_disabled(self):
        camera = _LensCamera(start=10.0, peak=8.2)
        cfg = CameraConfig(af_timeout_sec=1.0, af_max_retries=1, af_refine=False)
        ctl = AutofocusController(cfg)
        ctl.probe_capabilities(camera)

        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)

        assert camera.lens_calls == []
        assert "refine" not in result.method


class _NarrowPeakCamera(_LensCamera):
    """Sharp only within ~0.5 dioptre of the peak; flat (fully blurred) elsewhere."""

    def capture_preview_array(self) -> np.ndarray:
        import cv2

        sigma = min(7.0, 0.3 + 12.0 * abs(self.lens - self.peak))
        return cv2.GaussianBlur(_sharp_frame(), (0, 0), sigma)


class TestWideSweep:
    @pytest.mark.asyncio
    async def test_plateau_at_af_result_triggers_wide_sweep_that_finds_the_real_peak(self):
        camera = _NarrowPeakCamera(start=8.0, peak=13.6)
        cfg = CameraConfig(af_timeout_sec=1.0, af_max_retries=1, af_refine_settle_sec=0.05)
        ctl = AutofocusController(cfg)
        ctl.probe_capabilities(camera)

        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)

        assert "refine" in result.method
        assert abs(camera.lens - 13.6) <= 0.6

    @pytest.mark.asyncio
    async def test_wide_sweep_can_be_disabled(self):
        camera = _NarrowPeakCamera(start=8.0, peak=13.6)
        cfg = CameraConfig(af_timeout_sec=1.0, af_max_retries=1, af_refine_settle_sec=0.05, af_wide_sweep=False)
        ctl = AutofocusController(cfg)
        ctl.probe_capabilities(camera)

        await ctl.run_autofocus(camera, sharpness_threshold=1.0)

        assert abs(camera.lens - 13.6) > 1.0  # stayed on the plateau near the AF result


class _FlatCamera(_LensCamera):
    def capture_preview_array(self) -> np.ndarray:
        return _sharp_frame()  # same sharpness at every lens position: nothing to focus on


class TestNoFocusPeak:
    @pytest.mark.asyncio
    async def test_flat_sharpness_curve_fails_focus_instead_of_capturing_blind(self):
        camera = _FlatCamera(start=8.0, peak=0.0)
        cfg = CameraConfig(af_timeout_sec=1.0, af_max_retries=1, af_refine_settle_sec=0.05)
        ctl = AutofocusController(cfg)
        ctl.probe_capabilities(camera)

        with pytest.raises(AutofocusError, match="No focus peak"):
            await ctl.run_autofocus(camera, sharpness_threshold=1.0)


class _ScenechangeCamera(_LensCamera):
    """Page is sharp until the confirming reading, then it moves (everything turns soft)."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.frames = 0

    def capture_preview_array(self) -> np.ndarray:
        import cv2

        self.frames += 1
        sigma = 0.3 + 2.0 * abs(self.lens - self.peak)
        if self.frames > 60:
            sigma += 6.0
        return cv2.GaussianBlur(_sharp_frame(), (0, 0), sigma)


class TestUnstableFocus:
    @pytest.mark.asyncio
    async def test_reading_that_collapses_at_confirmation_fails_focus(self):
        camera = _ScenechangeCamera(start=8.0, peak=8.6)
        cfg = CameraConfig(af_timeout_sec=1.0, af_max_retries=1, af_refine_settle_sec=0.05, af_refine_discard_frames=1)
        ctl = AutofocusController(cfg)
        ctl.probe_capabilities(camera)
        # frames before ~60 are consistent; make the confirmation land after the page "moved"
        camera.frames = 55

        with pytest.raises(AutofocusError, match="unstable|No focus peak"):
            await ctl.run_autofocus(camera, sharpness_threshold=1.0)
