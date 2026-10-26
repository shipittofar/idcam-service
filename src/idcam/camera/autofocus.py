"""Camera autofocus — one hardware pass, preview sharpness verified."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

from idcam.config.schema import CameraConfig, QualityConfig, QualityConfig
from idcam.exceptions import AutofocusError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import accept_preview_focus, clip_roi, preview_sharpness, roi_rect

logger = get_logger("camera.autofocus")


class FocusCapableCamera(Protocol):
    def get_controls(self) -> dict[str, Any]: ...
    def set_controls(self, controls: dict[str, Any]) -> None: ...
    def autofocus_cycle(self) -> Any: ...
    def capture_preview_array(self) -> Any: ...
    def capture_metadata(self) -> dict[str, Any]: ...


@dataclass
class AutofocusCapabilities:
    af_mode: bool = False
    af_trigger: bool = False
    af_state: bool = False
    lens_position: bool = False
    autofocus_cycle: bool = False
    af_windows: bool = False
    af_range: bool = False
    af_metering: bool = False

    @property
    def hardware_af_available(self) -> bool:
        return self.af_mode or self.autofocus_cycle

    @property
    def degraded(self) -> bool:
        return not self.hardware_af_available

    def to_dict(self) -> dict[str, bool]:
        return {
            "AfMode": self.af_mode,
            "AfTrigger": self.af_trigger,
            "AfState": self.af_state,
            "LensPosition": self.lens_position,
            "hardware_af_available": self.hardware_af_available,
            "degraded": self.degraded,
        }


@dataclass
class AutofocusResult:
    success: bool
    sharpness: float
    attempts: int
    af_state: int | None
    method: str
    min_required_sharpness: float = 0.0
    before_sharpness: float = 0.0
    improved: bool = False
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    lens_before: float | None = None
    lens_after: float | None = None
    improvement_ratio: float = 0.0
    sensor_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    preview_size: tuple[int, int] = (0, 0)


@dataclass
class AutofocusDebug:
    running: bool = False
    method: str = ""
    attempt: int = 0
    focus_before: float = 0.0
    focus_after: float = 0.0
    focus_roi: str = ""
    preview_roi: str = ""
    sensor_roi: str = ""
    preview_roi_wh: str = ""
    lens_before: float | None = None
    lens_after: float | None = None
    improvement_ratio: float = 0.0
    scaler_crop: str = ""
    preview_size: str = ""


def _fmt_roi(roi: tuple[int, int, int, int] | None) -> str:
    if not roi:
        return ""
    return "{},{},{},{}".format(*(int(v) for v in roi))


class AutofocusController:
    """Hardware AF on document ROI with Tenengrad before/after verification."""

    def __init__(self, config: CameraConfig, quality: QualityConfig | None = None) -> None:
        self.config = config
        self.quality = quality or QualityConfig()
        self.capabilities = AutofocusCapabilities()
        self.debug = AutofocusDebug()
        self._last_preview_roi: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def hardware_af_available(self) -> bool:
        return self.capabilities.hardware_af_available

    def probe_capabilities(self, camera: FocusCapableCamera) -> AutofocusCapabilities:
        controls = camera.get_controls()
        self.capabilities = AutofocusCapabilities(
            af_mode="AfMode" in controls,
            af_trigger="AfTrigger" in controls,
            af_state="AfState" in controls,
            lens_position="LensPosition" in controls,
            autofocus_cycle=hasattr(camera, "autofocus_cycle"),
            af_windows="AfWindows" in controls,
            af_range="AfRange" in controls,
            af_metering="AfMetering" in controls,
        )
        logger.info("Autofocus capabilities: %s", self.capabilities.to_dict())
        return self.capabilities

    def _resolve_roi(
        self,
        frame: Any,
        roi_fraction: float,
        preview_roi: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int]:
        if preview_roi is not None:
            return clip_roi(preview_roi, frame.shape)
        return clip_roi(roi_rect(frame.shape, roi_fraction), frame.shape)

    async def run_autofocus(
        self,
        camera: FocusCapableCamera,
        *,
        roi_fraction: float = 0.7,
        preview_roi: tuple[int, int, int, int] | None = None,
        sharpness_threshold: float | None = None,
    ) -> AutofocusResult:
        floor = float(
            sharpness_threshold if sharpness_threshold is not None else self.quality.preview_min_sharpness
        )
        min_ratio = float(self.quality.preview_focus_min_ratio)
        min_delta = float(self.quality.preview_focus_min_delta)

        preview = await asyncio.to_thread(camera.capture_preview_array)
        if preview is None or getattr(preview, "size", 1) == 0:
            raise AutofocusError("Invalid preview frame")

        h, w = int(preview.shape[0]), int(preview.shape[1])
        preview_size = (w, h)
        roi = self._resolve_roi(preview, roi_fraction, preview_roi)
        mapping = await self._apply_document_roi(camera, roi, preview_size)
        sensor = tuple(mapping.get("sensor_roi", roi))

        before = preview_sharpness(preview, roi=roi)
        meta_before = self._read_metadata(camera)
        lens_before = meta_before.get("LensPosition")
        self._last_preview_roi = roi
        self._fill_debug(roi, sensor, preview_size, mapping, before, lens_before)

        logger.info(
            "event=pipeline_stage stage=focusing focus_method=measure "
            "sharpness_before=%.1f preview_floor=%.1f roi=%s sensor_roi=%s lens=%s",
            before,
            floor,
            self.debug.preview_roi,
            self.debug.sensor_roi,
            lens_before,
        )

        # No "already sharp" shortcut: comparing a frame with itself always passes, so AF
        # never ran and the lens stayed wherever Continuous AF last drifted (it moved
        # between this check and the still). Always run a one-shot AF on the document
        # ROI, then lock the lens (pause_autofocus below) before the still capture.
        after = before
        lens_after = lens_before
        method = "none"
        ratio = 0.0

        # At least two passes: the first can land worse than the drifting Continuous AF.
        for attempt in range(1, max(2, self.config.af_max_retries) + 1):
            self.debug.running = True
            self.debug.attempt = attempt
            method, _ = await self._run_hardware_af(camera, attempt)
            self.debug.method = method

            after_frame = await asyncio.to_thread(camera.capture_preview_array)
            after = preview_sharpness(after_frame, roi=roi)
            meta_after = self._read_metadata(camera)
            lens_after = meta_after.get("LensPosition")
            ratio = after / before if before > 1e-6 else (2.0 if after > 0 else 0.0)
            improved = after > before
            accepted = accept_preview_focus(
                before=before,
                after=after,
                absolute_floor=floor,
                min_ratio=min_ratio,
                min_delta=min_delta,
            )

            self.debug.focus_after = after
            self.debug.lens_after = float(lens_after) if lens_after is not None else None
            self.debug.improvement_ratio = round(ratio, 3)
            self.debug.running = False

            logger.info(
                "event=autofocus_result focus_method=%s success=%s sharpness_before=%.1f "
                "sharpness_after=%.1f sharpness_ratio=%.3f improved=%s attempt=%d "
                "lens_before=%s lens_after=%s roi=%s",
                method,
                accepted,
                before,
                after,
                ratio,
                improved,
                attempt,
                lens_before,
                lens_after,
                self.debug.preview_roi,
            )

            if accepted:
                logger.info("event=pipeline_stage stage=focus_verified method=%s sharpness=%.1f", method, after)
                pause = getattr(camera, "pause_autofocus", None)
                if callable(pause):
                    await asyncio.to_thread(pause)
                return self._result(
                    True, after, attempt, method, floor, before, after, ratio, roi, sensor, preview_size, lens_before, lens_after
                )
            before = after
            lens_before = lens_after

        raise AutofocusError(
            f"Autofocus did not improve: before={before:.1f} after={after:.1f} "
            f"ratio={ratio:.2f} method={method} preview_roi={_fmt_roi(roi)}"
        )

    async def verify_focus(
        self,
        camera: FocusCapableCamera,
        *,
        roi_fraction: float = 0.7,
        preview_roi: tuple[int, int, int, int] | None = None,
        sharpness_threshold: float | None = None,
    ) -> AutofocusResult:
        floor = float(
            sharpness_threshold if sharpness_threshold is not None else self.quality.preview_min_sharpness
        )
        min_ratio = float(self.quality.preview_focus_min_ratio)
        min_delta = float(self.quality.preview_focus_min_delta)

        preview = await asyncio.to_thread(camera.capture_preview_array)
        if preview is None or getattr(preview, "size", 1) == 0:
            raise AutofocusError("Invalid preview frame")

        h, w = int(preview.shape[0]), int(preview.shape[1])
        preview_size = (w, h)
        roi = self._resolve_roi(
            preview,
            roi_fraction,
            preview_roi or (self._last_preview_roi if self._last_preview_roi[2] > 0 else None),
        )
        sharp = preview_sharpness(preview, roi=roi)
        before = self.debug.focus_before if self.debug.focus_before > 0 else sharp
        meta = self._read_metadata(camera)
        af_state = meta.get("AfState")
        lens = meta.get("LensPosition")

        self.debug.focus_after = sharp
        self.debug.lens_after = float(lens) if lens is not None else None
        if not accept_preview_focus(
            before=before,
            after=sharp,
            absolute_floor=floor,
            min_ratio=min_ratio,
            min_delta=min_delta,
        ):
            raise AutofocusError(
                f"Focus verification failed: sharpness={sharp:.1f} before={before:.1f} floor={floor:.1f}"
            )

        sensor = tuple(self._last_preview_roi if self._last_preview_roi[2] > 0 else roi)
        return AutofocusResult(
            success=True,
            sharpness=sharp,
            attempts=0,
            af_state=int(af_state) if af_state is not None else None,
            method="verify",
            min_required_sharpness=floor,
            before_sharpness=before,
            improved=sharp >= before,
            roi=roi,
            lens_before=self.debug.lens_before,
            lens_after=float(lens) if lens is not None else None,
            improvement_ratio=sharp / before if before > 1e-6 else 1.0,
            sensor_roi=sensor,
            preview_size=preview_size,
        )

    def _result(
        self,
        success: bool,
        sharpness: float,
        attempts: int,
        method: str,
        floor: float,
        before: float,
        after: float,
        ratio: float,
        roi: tuple[int, int, int, int],
        sensor: tuple[int, int, int, int],
        preview_size: tuple[int, int],
        lens_before: Any,
        lens_after: Any,
    ) -> AutofocusResult:
        lb = float(lens_before) if lens_before is not None else None
        la = float(lens_after) if lens_after is not None else None
        return AutofocusResult(
            success=success,
            sharpness=sharpness,
            attempts=attempts,
            af_state=None,
            method=method,
            min_required_sharpness=floor,
            before_sharpness=before,
            improved=after > before,
            roi=roi,
            lens_before=lb,
            lens_after=la,
            improvement_ratio=ratio,
            sensor_roi=sensor,
            preview_size=preview_size,
        )

    def _fill_debug(
        self,
        roi: tuple[int, int, int, int],
        sensor: tuple[int, int, int, int],
        preview_size: tuple[int, int],
        mapping: dict,
        before: float,
        lens_before: Any,
    ) -> None:
        self.debug.focus_before = before
        self.debug.lens_before = float(lens_before) if lens_before is not None else None
        self.debug.focus_roi = _fmt_roi(roi)
        self.debug.preview_roi = _fmt_roi(roi)
        self.debug.sensor_roi = _fmt_roi(sensor)
        self.debug.preview_roi_wh = f"{roi[2]}x{roi[3]}"
        sc = mapping.get("scaler_crop", (0, 0, preview_size[0], preview_size[1]))
        self.debug.scaler_crop = _fmt_roi(tuple(sc))
        self.debug.preview_size = f"{preview_size[0]}x{preview_size[1]}"

    async def _apply_document_roi(
        self,
        camera: FocusCapableCamera,
        preview_roi: tuple[int, int, int, int],
        preview_size: tuple[int, int],
    ) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "preview_roi": preview_roi,
            "preview_size": preview_size,
            "scaler_crop": (0, 0, preview_size[0], preview_size[1]),
            "sensor_roi": preview_roi,
        }
        applier = getattr(camera, "apply_document_focus_window", None)
        if not callable(applier):
            return mapping
        try:
            result = await asyncio.to_thread(applier, preview_roi, preview_size)
        except TypeError:
            result = await asyncio.to_thread(applier, 0.7)
        if isinstance(result, dict):
            mapping.update(result)
            if result.get("sensor_roi"):
                mapping["sensor_roi"] = tuple(int(v) for v in result["sensor_roi"])
        return mapping

    def _read_metadata(self, camera: FocusCapableCamera) -> dict[str, Any]:
        reader = getattr(camera, "capture_metadata", None)
        if reader is None:
            return {}
        try:
            return dict(reader() or {})
        except Exception:
            return {}

    async def _run_hardware_af(self, camera: FocusCapableCamera, attempt: int) -> tuple[str, int | None]:
        trigger = getattr(camera, "trigger_autofocus", None)
        settle = getattr(camera, "wait_for_lens_settle", None)
        if callable(trigger) and callable(settle):
            logger.info("event=autofocus_triggered method=hardware_trigger attempt=%d", attempt)
            await asyncio.to_thread(trigger)
            await asyncio.to_thread(settle, min(float(self.config.af_timeout_sec), 3.5))
            return "hardware_trigger", None
        if self.capabilities.autofocus_cycle:
            await asyncio.wait_for(
                asyncio.to_thread(camera.autofocus_cycle),
                timeout=self.config.af_timeout_sec,
            )
            return "hardware_cycle", None
        return "sharpness_only", None
