"""Faster autofocus: lens memory, metadata settling, adaptive refinement, motion detection."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import pytest

from idcam.camera.autofocus import AutofocusController
from idcam.config.schema import CameraConfig
from idcam.exceptions import AutofocusError, FocusMotionError


def _page() -> np.ndarray:
    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    frame[20:100, 20:30] = 0
    frame[20:30, 20:140] = 0
    frame[90:100, 20:140] = 0
    frame[40:50, 40:120] = 0
    frame[60:70, 40:120] = 0
    return frame


class LensCamera:
    """Preview camera whose sharpness peaks at ``peak`` dioptres.

    The physical lens follows a command only after ``lag_frames`` fresh frames; the metadata
    reports the physical lens (plus ``meta_offset``), or nothing when ``meta`` is False.
    """

    def __init__(
        self,
        *,
        peak: float,
        start: float,
        hw_af_lens: float | None = None,
        lag_frames: int = 0,
        meta: bool = True,
        meta_offset: float = 0.0,
    ) -> None:
        self.peak = peak
        self.cmd_lens = start
        self.phys_lens = start
        self.hw_af_lens = start if hw_af_lens is None else hw_af_lens
        self.lag_frames = lag_frames
        self.meta = meta
        self.meta_offset = meta_offset
        self._since_cmd = 10**6
        self.cycle_calls = 0
        self.lens_calls: list[float] = []
        self.fresh_calls = 0
        self.last_fresh_lens: float | None = None
        self.last_fresh_meta: dict[str, Any] = {"exposure_us": 20000, "analogue_gain": 2.0}

    # -- optics ---------------------------------------------------------------------------------
    def _frame(self) -> np.ndarray:
        sigma = min(7.0, 0.3 + 4.0 * abs(self.phys_lens - self.peak))
        return cv2.GaussianBlur(_page(), (0, 0), sigma)

    # -- camera API used by the controller ---------------------------------------------------
    def get_controls(self) -> dict[str, Any]:
        return {"AfMode": (0, 2, 0), "AfTrigger": (0, 1, 0), "LensPosition": (0.0, 32.0, 1.0)}

    def capture_metadata(self) -> dict[str, Any]:
        return {"LensPosition": self.phys_lens}

    def set_controls(self, controls: dict[str, Any]) -> None:
        pass

    def apply_document_focus_window(self, preview_roi, preview_size=None) -> dict[str, Any]:
        size = preview_size or (160, 120)
        return {"preview_roi": preview_roi, "preview_size": size, "scaler_crop": (0, 0, *size), "sensor_roi": preview_roi}

    def autofocus_cycle(self) -> bool:
        self.cycle_calls += 1
        self.cmd_lens = self.phys_lens = self.hw_af_lens
        return True

    def capture_preview_array(self) -> np.ndarray:
        return self._frame()

    def lens_position_range(self) -> tuple[float, float]:
        return (0.0, 32.0)

    def set_lens_position(self, position: float) -> None:
        self.lens_calls.append(round(position, 3))
        self.cmd_lens = position
        self._since_cmd = 0

    def capture_fresh_preview_array(self) -> np.ndarray:
        self.fresh_calls += 1
        self._since_cmd += 1
        if self._since_cmd > self.lag_frames:
            self.phys_lens = self.cmd_lens
        self.last_fresh_lens = (self.phys_lens + self.meta_offset) if self.meta else None
        return self._frame()


def _cfg(**kw: Any) -> CameraConfig:
    base = dict(
        af_timeout_sec=1.0,
        af_max_retries=1,
        af_refine_settle_sec=0.05,
        af_refine_min_settle_sec=0.005,
        af_settle_tolerance=0.06,
    )
    base.update(kw)
    return CameraConfig(**base)


async def _run(camera: LensCamera, **kw: Any) -> tuple[Any, AutofocusController]:
    ctl = AutofocusController(_cfg(**kw))
    ctl.probe_capabilities(camera)
    result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)
    return result, ctl


# --------------------------------------------------------------------- lens metadata settling


class TestMetadataSettle:
    @pytest.mark.asyncio
    async def test_waits_for_the_lens_not_for_the_full_fixed_delay(self):
        camera = LensCamera(peak=8.4, start=8.0, lag_frames=3)
        started = time.monotonic()

        result, _ = await _run(camera, af_refine_settle_sec=0.6)  # fixed mode would need ~0.6 s per sample

        assert abs(result.lens_after - 8.4) <= 0.3
        assert time.monotonic() - started < 3.0

    @pytest.mark.asyncio
    async def test_frames_from_before_the_lens_arrived_are_never_measured(self):
        camera = LensCamera(peak=8.4, start=8.0, lag_frames=4)

        result, _ = await _run(camera)

        # with a 4-frame lag a naive "measure the first frame" finds a peak one step behind
        assert abs(result.lens_after - 8.4) <= 0.3

    @pytest.mark.asyncio
    async def test_metadata_that_never_matches_is_bounded_by_the_settle_time(self):
        camera = LensCamera(peak=8.4, start=8.0, meta_offset=0.5)  # reports 0.5 off: never within tolerance
        started = time.monotonic()

        result, _ = await _run(camera, af_refine_settle_sec=0.05)

        assert abs(result.lens_after - 8.4) <= 0.3  # still found, using the settle upper bound
        assert time.monotonic() - started < 6.0

    @pytest.mark.asyncio
    async def test_missing_metadata_falls_back_to_the_fixed_settle(self):
        camera = LensCamera(peak=8.4, start=8.0, meta=False)

        result, _ = await _run(camera)

        assert abs(result.lens_after - 8.4) <= 0.3

    @pytest.mark.asyncio
    async def test_metadata_settling_can_be_turned_off(self):
        camera = LensCamera(peak=8.4, start=8.0)

        result, _ = await _run(camera, af_settle_by_metadata=False)

        assert abs(result.lens_after - 8.4) <= 0.3


# --------------------------------------------------------------------------- lens memory


class TestLensMemory:
    @pytest.mark.asyncio
    async def test_remembered_lens_skips_hardware_af_and_needs_fewer_samples(self, caplog):
        full = LensCamera(peak=8.4, start=6.0, hw_af_lens=8.0)
        await _run(full)
        full_samples = len(full.lens_calls)

        camera = LensCamera(peak=8.4, start=6.0, hw_af_lens=8.0)
        ctl = AutofocusController(_cfg())
        ctl.probe_capabilities(camera)
        ctl.remember_lens(8.4)  # the validated position of the previous document
        with caplog.at_level("INFO"):
            result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)

        assert result.method == "memory+refine"
        assert camera.cycle_calls == 0
        assert abs(result.lens_after - 8.4) <= 0.3
        assert len(camera.lens_calls) < full_samples
        assert "path=memory" in caplog.text

    @pytest.mark.asyncio
    async def test_wrong_memory_falls_back_to_hardware_af_and_still_focuses(self):
        camera = LensCamera(peak=9.0, start=6.0, hw_af_lens=8.7)
        ctl = AutofocusController(_cfg())
        ctl.probe_capabilities(camera)
        ctl.remember_lens(3.0)  # document height changed: nothing sharp near 3.0

        result = await ctl.run_autofocus(camera, sharpness_threshold=1.0)

        assert camera.cycle_calls >= 1
        assert "memory" not in result.method
        assert abs(result.lens_after - 9.0) <= 0.3
        assert ctl.remembered_lens is None  # the stale position was dropped

    @pytest.mark.asyncio
    async def test_memory_is_ignored_when_disabled_or_expired(self):
        ctl = AutofocusController(_cfg(af_lens_memory=False))
        ctl.remember_lens(8.0)
        assert ctl.remembered_lens is None

        ctl = AutofocusController(_cfg(af_memory_max_age_sec=60))
        ctl.remember_lens(8.0)
        assert ctl.remembered_lens == 8.0
        ctl._memory = (8.0, time.monotonic() - 3600)
        assert ctl.remembered_lens is None

    @pytest.mark.asyncio
    async def test_forget_lens_clears_the_memory(self):
        ctl = AutofocusController(_cfg())
        ctl.remember_lens(8.0)
        ctl.forget_lens("test")
        assert ctl.remembered_lens is None


# ------------------------------------------------------------------ adaptive refinement


class TestAdaptiveRefinement:
    @pytest.mark.asyncio
    async def test_clean_peak_needs_fewer_measurements_than_exhaustive_search(self):
        # hardware AF already lands on the peak (the common case once the setup is stable)
        adaptive = LensCamera(peak=8.4, start=8.4)
        exhaustive = LensCamera(peak=8.4, start=8.4)

        r1, _ = await _run(adaptive, af_adaptive=True)
        r2, _ = await _run(exhaustive, af_adaptive=False)

        assert abs(r1.lens_after - 8.4) <= 0.3 and abs(r2.lens_after - 8.4) <= 0.3
        assert len(adaptive.lens_calls) <= len(exhaustive.lens_calls) - 2  # finest probe + confirmation

    @pytest.mark.asyncio
    async def test_peak_found_by_a_fine_probe_still_gets_the_finest_refinement(self):
        """AF stopped 0.2 dioptre off the peak: the first fine probe moves the best position, so the
        finest stage must still run (its neighbours are not yet bracketing the new best)."""
        camera = LensCamera(peak=8.4, start=8.2)
        result, _ = await _run(camera)
        assert abs(result.lens_after - 8.4) <= 0.12

    @pytest.mark.asyncio
    async def test_clean_peak_is_reported_as_skipped_confirmation(self, caplog):
        camera = LensCamera(peak=8.4, start=8.4)
        with caplog.at_level("INFO"):
            await _run(camera)
        assert "confirm=skipped_clean_last" in caplog.text or "confirm=settled_clean" in caplog.text

    @pytest.mark.asyncio
    async def test_exhaustive_mode_always_confirms(self, caplog):
        camera = LensCamera(peak=8.4, start=8.2)
        with caplog.at_level("INFO"):
            await _run(camera, af_adaptive=False)
        assert "confirm=measured" in caplog.text

    def test_clean_peak_rules(self):
        clean = AutofocusController._clean_peak
        good = {8.0: 100.0, 8.2: 300.0, 8.4: 900.0, 8.6: 400.0, 8.9: 150.0}
        assert clean(good, 8.4, unstable=0)
        # two bumps (glare / reflection): not a single peak, so the winner must be re-measured
        assert not clean({8.0: 900.0, 8.2: 300.0, 8.4: 700.0, 8.6: 400.0, 8.9: 150.0}, 8.0, unstable=0)
        # winner at the edge of what was sampled: the real peak may lie beyond it
        assert not clean({8.0: 100.0, 8.2: 300.0, 8.4: 900.0}, 8.4, unstable=0)
        # neighbours barely lower: no clear peak
        assert not clean({8.0: 880.0, 8.2: 890.0, 8.4: 900.0, 8.6: 895.0, 8.9: 885.0}, 8.4, unstable=0)
        # any motion hint disables the shortcut
        assert not clean(good, 8.4, unstable=1)
        # too few samples
        assert not clean({8.0: 100.0, 8.4: 900.0}, 8.4, unstable=0)

    @pytest.mark.asyncio
    async def test_one_hardware_pass_is_enough_when_the_lens_search_follows(self):
        # hardware AF lands off the peak and *worse* than where Continuous AF was: before the
        # change this triggered a second hardware cycle; now the lens search recovers directly.
        camera = LensCamera(peak=8.4, start=8.4, hw_af_lens=7.6)

        result, _ = await _run(camera)

        assert camera.cycle_calls == 1
        assert abs(result.lens_after - 8.4) <= 0.3


# --------------------------------------------------------------------------- motion


class TestMotionDetection:
    @pytest.mark.asyncio
    async def test_inconsistent_frames_at_one_lens_position_abort_as_motion(self):
        class Shaking(LensCamera):
            """Every other fresh frame is smeared, as when a hand moves the page."""

            def _frame(self) -> np.ndarray:
                frame = super()._frame()
                self._n = getattr(self, "_n", 0) + 1
                return cv2.GaussianBlur(frame, (0, 0), 6.0) if self._n % 2 else frame

        camera = Shaking(peak=8.4, start=8.4)

        with pytest.raises(FocusMotionError, match="moving"):
            await _run(camera)

    @pytest.mark.asyncio
    async def test_motion_error_is_an_autofocus_error(self):
        assert issubclass(FocusMotionError, AutofocusError)

    @pytest.mark.asyncio
    async def test_steady_scene_never_flags_motion(self):
        camera = LensCamera(peak=8.4, start=8.0)
        result, ctl = await _run(camera)
        assert ctl.last_refine_stats["unstable"] == 0
        assert result.success

    @pytest.mark.asyncio
    async def test_moving_lens_is_not_mistaken_for_page_motion(self):
        """While the lens is still travelling two frames differ: that must not count as motion."""
        camera = LensCamera(peak=8.4, start=8.0, lag_frames=6)
        result, ctl = await _run(camera, af_refine_settle_sec=0.3)
        assert ctl.last_refine_stats["unstable"] == 0
        assert abs(result.lens_after - 8.4) <= 0.3


# ------------------------------------------------------------------ config + timing log


class TestConfigAndLogging:
    def test_experimental_speed_profile_is_accepted_by_the_config_schema(self):
        from idcam.config.schema import DetectionConfig, LedConfig

        camera = CameraConfig(preview_fps=10, af_refine_settle_sec=0.2, af_refine_max_points=5)
        detection = DetectionConfig(detect_frames_required=2, stable_frames_required=4, removal_frames_required=6)
        led = LedConfig(brightness=0.2)
        assert (camera.preview_fps, detection.stable_frames_required, led.brightness) == (10, 4, 0.2)

    def test_new_settings_default_to_safe_values(self):
        cfg = CameraConfig()
        assert cfg.af_adaptive and cfg.af_lens_memory and cfg.af_settle_by_metadata
        assert cfg.af_refine_min_settle_sec < cfg.af_refine_settle_sec  # metadata mode never waits longer than fixed
        assert cfg.af_refine_settle_sec == 0.3 and cfg.af_refine_max_points == 7  # existing values untouched

    @pytest.mark.asyncio
    async def test_every_sample_and_the_focus_timing_are_logged(self, caplog):
        camera = LensCamera(peak=8.4, start=8.0)
        with caplog.at_level("INFO"):
            await _run(camera)
        assert "event=focus_sample" in caplog.text and "exposure_us=20000" in caplog.text
        assert "event=focus_timing path=hardware" in caplog.text and "refine_ms=" in caplog.text
