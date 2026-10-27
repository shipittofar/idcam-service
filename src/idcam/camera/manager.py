"""Picamera2 camera manager with mock fallback for development."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from idcam.config.schema import CameraConfig
from idcam.exceptions import CameraError, CameraUnavailableError
from idcam.logging.setup import get_logger

logger = get_logger("camera.manager")

try:
    from picamera2 import Picamera2
    from libcamera import controls

    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None  # type: ignore
    controls = None  # type: ignore

# libcamera AfMode / AfTrigger integer values (stable across bindings).
_AF_MODE_MANUAL = 0
_AF_MODE_AUTO = 1
_AF_MODE_CONTINUOUS = 2
_AF_TRIGGER_START = 0
_AF_STATE_FOCUSED = 2
_AF_STATE_FAILED = 3
_AF_METERING_WINDOWS = 1
_AF_RANGE_FULL = 2


def describe_preview_frame(frame: np.ndarray) -> dict[str, Any]:
    """Return shape/dtype/channel statistics for diagnostics. Does not mutate ``frame``."""
    info: dict[str, Any] = {
        "shape": [int(x) for x in frame.shape],
        "dtype": str(frame.dtype),
        "ndim": int(frame.ndim),
    }
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        means = frame.mean(axis=(0, 1))
        info["channel_means"] = {
            "c0": round(float(means[0]), 2),
            "c1": round(float(means[1]), 2),
            "c2": round(float(means[2]), 2),
        }
    elif frame.ndim == 2:
        info["channel_means"] = {"c0": round(float(frame.mean()), 2)}
    return info


def parse_xywh(value: Any) -> tuple[int, int, int, int] | None:
    """Normalize libcamera Rectangle / tuple to (x, y, width, height)."""
    if value is None:
        return None
    if hasattr(value, "x") and hasattr(value, "width"):
        return int(value.x), int(value.y), int(value.width), int(value.height)
    try:
        if len(value) >= 4:
            return int(value[0]), int(value[1]), int(value[2]), int(value[3])
    except TypeError:
        return None
    return None


def map_preview_roi_to_sensor(
    preview_roi: tuple[int, int, int, int],
    preview_size: tuple[int, int],
    scaler_crop: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Map a preview-pixel ROI into AfWindows / ScalerCrop coordinates.

    Preview is typically 1920x1080 (16:9). The sensor/ScalerCropMaximum is
    ~9152x6944 (≈4:3). Taking 70% of ScalerCropMaximum independently does
    **not** match 70% of the preview. Use the live ScalerCrop of the current
    stream so AF and Laplacian see the same document region.
    """
    px, py, pw, ph = (int(v) for v in preview_roi)
    fw, fh = int(preview_size[0]), int(preview_size[1])
    sx, sy, sw, sh = (int(v) for v in scaler_crop)
    if fw <= 0 or fh <= 0 or sw <= 0 or sh <= 0:
        return sx, sy, sw, sh
    return (
        sx + int(round(px * sw / fw)),
        sy + int(round(py * sh / fh)),
        max(1, int(round(pw * sw / fw))),
        max(1, int(round(ph * sh / fh))),
    )


def clip_preview_roi(
    roi: tuple[int, int, int, int],
    preview_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Clip an (x, y, w, h) ROI to preview pixel bounds (width, height)."""
    fw, fh = int(preview_size[0]), int(preview_size[1])
    x, y, rw, rh = (int(v) for v in roi)
    if fw <= 0 or fh <= 0:
        return 0, 0, 1, 1
    x = max(0, min(x, max(0, fw - 1)))
    y = max(0, min(y, max(0, fh - 1)))
    rw = max(1, min(rw, fw - x))
    rh = max(1, min(rh, fh - y))
    return x, y, rw, rh


def normalize_picamera_preview_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a Picamera2 RGB888/XRGB8888 ndarray to RGB channel order.

    Picamera2 requests libcamera/DRM ``RGB888``, which is B,G,R in memory.
    Picamera2's own JPEG encoder documents this as ``FORMAT_TABLE['RGB888'] = 'BGR'``.
    idCam preview consumers (detector, sharpness, debug viewer) assume RGB.
    """
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return np.ascontiguousarray(frame[..., ::-1])
    if frame.ndim == 3 and frame.shape[-1] == 4:
        return np.ascontiguousarray(frame[..., 2::-1])
    return frame


def _boottime_ns() -> int | None:
    """Now on the clock libcamera uses for SensorTimestamp (None when unavailable, e.g. Windows)."""
    clock = getattr(time, "CLOCK_BOOTTIME", None)
    if clock is None:
        return None
    try:
        return time.clock_gettime_ns(clock)
    except (OSError, AttributeError):
        return None


class CameraManager:
    """
    Manages Picamera2 lifecycle: preview stream and full-res capture.

  Thread-safe via lock. All public methods may be called from executor threads.
    """

    def __init__(self, config: CameraConfig, *, use_mock: bool = False) -> None:
        self.config = config
        self._use_mock = use_mock or not PICAMERA2_AVAILABLE
        self._lock = threading.RLock()
        self._picam2: Any = None
        self._started = False
        self._preview_config: Any = None
        self._still_config: Any = None
        self._logged_preview_format = False
        self._preview_af_mode: str | None = None
        # Lens position we last commanded manually. capture_metadata() lags the command, so the
        # still must use this value, not a metadata read (logs: still at 8.38 after parking at 8.17).
        self._held_lens: float | None = None
        # Manual focus mode: the lens is driven to this value and never auto-focused.
        self._manual_lens: float | None = (
            float(config.manual_lens_position) if getattr(config, "focus_mode", "auto") == "manual" else None
        )
        # LensPosition reported in the metadata of the last frame from capture_fresh_preview_array.
        self.last_fresh_lens: float | None = None
        # Exposure details of that frame (None entries = the metadata field was not reported).
        self.last_fresh_meta: dict[str, Any] = {}

    @property
    def is_mock(self) -> bool:
        return self._use_mock

    @property
    def is_started(self) -> bool:
        return self._started

    def initialize(self) -> None:
        with self._lock:
            if self._started:
                return
            if self._use_mock:
                logger.warning("Using mock camera (Picamera2 not available)")
                self._started = True
                return

            try:
                self._picam2 = Picamera2()
                self._preview_config = self._picam2.create_preview_configuration(
                    main={"size": (self.config.preview_width, self.config.preview_height), "format": "RGB888"},
                    buffer_count=4,
                )
                self._still_config = self._picam2.create_still_configuration(
                    main={
                        "size": (self.config.capture_width, self.config.capture_height),
                        "format": "RGB888",
                    },
                    buffer_count=2,
                )
                self._picam2.configure(self._preview_config)
                self._picam2.start()
                self._started = True
                self._logged_preview_format = False
                logger.info(
                    "Camera initialized: preview=%dx%d capture=%dx%d",
                    self.config.preview_width,
                    self.config.preview_height,
                    self.config.capture_width,
                    self.config.capture_height,
                )
                try:
                    self._enable_preview_autofocus(wait=False)
                except Exception as exc:
                    logger.warning("event=autofocus_preview_failed error=%s", exc)
            except Exception as exc:
                self._started = False
                self._picam2 = None
                raise CameraUnavailableError(f"Camera initialization failed: {exc}") from exc

    def shutdown(self) -> None:
        with self._lock:
            if not self._started:
                return
            if self._picam2 is not None:
                try:
                    self._picam2.stop()
                    self._picam2.close()
                except Exception as exc:
                    logger.error("Error closing camera: %s", exc)
                finally:
                    self._picam2 = None
            self._started = False
            self._logged_preview_format = False
            self._preview_af_mode = None
            logger.info("Camera shut down")

    def restart(self) -> None:
        logger.info("Restarting camera")
        self.shutdown()
        time.sleep(1.0)
        self.initialize()

    def capture_preview_array(self) -> np.ndarray:
        with self._lock:
            if not self._started:
                raise CameraError("Camera not started")
            if self._use_mock:
                return self._mock_frame()
            assert self._picam2 is not None
            native = self._picam2.capture_array("main")
            frame = normalize_picamera_preview_frame(native)
            if not self._logged_preview_format:
                self._log_preview_format(native, frame)
                self._logged_preview_format = True
            return frame

    def capture_fresh_preview_array(self, timeout_sec: float = 2.0) -> np.ndarray:
        """First preview frame whose exposure started after this call.

        capture_array() returns the oldest queued frame, i.e. one exposed before a lens move that
        was just commanded. Focus measurements on such frames lag one position behind (a sweep
        then reported 4 dioptres of lens travel with flat sharpness).
        """
        with self._lock:
            if not self._started:
                raise CameraError("Camera not started")
            if self._use_mock:
                return self._mock_frame()
            assert self._picam2 is not None
            t0 = _boottime_ns()
            if t0 is None:
                return self.capture_preview_array()
            deadline = time.monotonic() + timeout_sec
            native = None
            try:
                while native is None:
                    request = self._picam2.capture_request()
                    try:
                        meta = request.get_metadata()
                        stamp = meta.get("SensorTimestamp")
                        if stamp is None or stamp >= t0 or time.monotonic() >= deadline:
                            native = np.array(request.make_array("main"))
                            lens = meta.get("LensPosition")
                            self.last_fresh_lens = float(lens) if lens is not None else None
                            self.last_fresh_meta = {
                                "exposure_us": meta.get("ExposureTime"),
                                "analogue_gain": meta.get("AnalogueGain"),
                                "frame_age_ms": int((stamp - t0) / 1e6) if stamp is not None else None,
                            }
                    finally:
                        request.release()
            except CameraError:
                raise
            except Exception:
                logger.warning("event=fresh_preview_unavailable using=capture_array", exc_info=True)
                return self.capture_preview_array()
            return normalize_picamera_preview_frame(native)

    def capture_still_to_file(self, path: Path) -> None:
        with self._lock:
            if not self._started:
                raise CameraError("Camera not started")
            if self._use_mock:
                self._mock_still(path)
                return

            assert self._picam2 is not None
            lens_position = self._held_lens if self._held_lens is not None else self._read_lens_position()
            t_begin = time.monotonic()
            timings: dict[str, Any] = {}
            try:
                self._picam2.stop()
                self._picam2.configure(self._still_config)
                if lens_position is not None:
                    # Queue the lens command before streaming starts so the first frames already
                    # have it; setting it only after start() let the lens travel during the capture.
                    self._set_controls_unlocked(
                        {
                            "AfMode": self._af_mode_manual(),
                            "LensPosition": float(lens_position),
                        }
                    )
                self._picam2.start()
                ready_ns = _boottime_ns()
                timings["reconfigure_ms"] = int((time.monotonic() - t_begin) * 1000)
                if lens_position is not None:
                    self._set_controls_unlocked(
                        {
                            "AfMode": self._af_mode_manual(),
                            "LensPosition": float(lens_position),
                        }
                    )
                    logger.info(
                        "event=autofocus_result method=hold_lens_position lens_position=%s",
                        lens_position,
                    )
                if not self._save_fresh_still(path, ready_ns, timings):
                    t_fallback = time.monotonic()
                    self._picam2.capture_file(str(path), format="jpeg")
                    timings["fallback_capture_file_ms"] = int((time.monotonic() - t_fallback) * 1000)
            finally:
                try:
                    t_restore = time.monotonic()
                    self._picam2.stop()
                    self._picam2.configure(self._preview_config)
                    self._picam2.start()
                    try:
                        self._enable_preview_autofocus(wait=False)
                    except Exception as exc:
                        logger.warning("event=autofocus_preview_failed error=%s", exc)
                    timings["restore_preview_ms"] = int((time.monotonic() - t_restore) * 1000)
                    # Which part of the ~2.4 s still capture is where (all monotonic, measured here).
                    logger.info(
                        "event=still_timing reconfigure_ms=%s frame_wait_ms=%s encode_ms=%s fallback_ms=%s "
                        "restore_preview_ms=%s total_ms=%d",
                        timings.get("reconfigure_ms"),
                        timings.get("frame_wait_ms"),
                        timings.get("encode_ms"),
                        timings.get("fallback_capture_file_ms"),
                        timings.get("restore_preview_ms"),
                        int((time.monotonic() - t_begin) * 1000),
                    )
                except Exception as exc:
                    logger.error("Failed to restore preview mode: %s", exc)
                    raise CameraError(f"Failed to restore preview after capture: {exc}") from exc

    def _save_fresh_still(self, path: Path, ready_ns: int | None, timings: dict[str, Any] | None = None) -> bool:
        """Save the first still exposed after the lens had ``still_lens_settle_sec`` to arrive.

        Returns False when the request API is unavailable, so the caller falls back to
        ``capture_file``.
        """
        if ready_ns is None or self._picam2 is None:
            return False
        settle_ns = int(float(self.config.still_lens_settle_sec) * 1e9)
        t_wait = time.monotonic()
        deadline = t_wait + 8.0 + float(self.config.still_lens_settle_sec)
        try:
            while True:
                request = self._picam2.capture_request()
                try:
                    meta = request.get_metadata()
                    stamp = meta.get("SensorTimestamp")
                    if stamp is None or stamp >= ready_ns + settle_ns or time.monotonic() >= deadline:
                        t_save = time.monotonic()
                        request.save("main", str(path), format="jpeg")
                        if timings is not None:
                            timings["frame_wait_ms"] = int((t_save - t_wait) * 1000)
                            timings["encode_ms"] = int((time.monotonic() - t_save) * 1000)
                        logger.info(
                            "event=still_capture_metadata exposure_us=%s analogue_gain=%s lens_position=%s "
                            "frame_age_ms=%s",
                            meta.get("ExposureTime"),
                            meta.get("AnalogueGain"),
                            meta.get("LensPosition"),
                            int((stamp - ready_ns) / 1e6) if stamp is not None else None,
                        )
                        return True
                finally:
                    request.release()
        except Exception:
            logger.warning("event=still_fresh_capture_failed using=capture_file", exc_info=True)
            return False

    def get_controls(self) -> dict[str, Any]:
        """Return advertised control *capabilities* (typically min/max/default), not live values."""
        with self._lock:
            if self._use_mock:
                return {"AfMode": (0, 2, 0), "AfTrigger": (0, 1, 0), "LensPosition": (0.0, 10.0, 1.0)}
            assert self._picam2 is not None
            return dict(self._picam2.camera_controls)

    def capture_metadata(self) -> dict[str, Any]:
        """Return live per-frame metadata (AfState, LensPosition, etc.)."""
        with self._lock:
            if self._use_mock:
                return {"AfMode": 1, "AfState": 2, "AfTrigger": 0, "LensPosition": 1.0}
            assert self._picam2 is not None
            return dict(self._picam2.capture_metadata())

    def set_controls(self, control_dict: dict[str, Any]) -> None:
        with self._lock:
            self._set_controls_unlocked(control_dict)

    def _set_controls_unlocked(self, control_dict: dict[str, Any]) -> None:
        if self._use_mock:
            return
        assert self._picam2 is not None
        advertised = set(self._picam2.camera_controls.keys())
        filtered = {k: v for k, v in control_dict.items() if k in advertised}
        if not filtered:
            logger.warning("None of controls %s are advertised", list(control_dict.keys()))
            return
        self._picam2.set_controls(filtered)

    def current_scaler_crop(self) -> tuple[int, int, int, int]:
        """Live stream crop on the sensor, else ScalerCropMaximum, else preview size."""
        with self._lock:
            return self._current_scaler_crop_unlocked()

    def _current_scaler_crop_unlocked(self) -> tuple[int, int, int, int]:
        fallback = (0, 0, int(self.config.preview_width), int(self.config.preview_height))
        if self._use_mock or self._picam2 is None:
            return fallback
        try:
            meta = dict(self._picam2.capture_metadata())
        except Exception:
            meta = {}
        crop = parse_xywh(meta.get("ScalerCrop"))
        if crop is None:
            props = getattr(self._picam2, "camera_properties", None) or {}
            crop = parse_xywh(props.get("ScalerCropMaximum"))
        return crop or fallback

    def apply_document_focus_window(
        self,
        preview_roi: tuple[int, int, int, int],
        preview_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Point hardware AF at a preview-pixel ROI mapped through live ScalerCrop."""
        with self._lock:
            fw, fh = int(preview_size[0]), int(preview_size[1])
            clipped = clip_preview_roi(preview_roi, (fw, fh))
            scaler = self._current_scaler_crop_unlocked()
            sensor = map_preview_roi_to_sensor(clipped, (fw, fh), scaler)
            if not self._use_mock and self._picam2 is not None:
                x, y, w, h = sensor
                self._set_controls_unlocked(
                    {
                        "AfMetering": self._af_metering_windows(),
                        "AfWindows": [(int(x), int(y), int(w), int(h))],
                        "AfRange": self._af_range_full(),
                    }
                )
            logger.info(
                "event=autofocus_roi preview=%d,%d,%d,%d preview_size=%dx%d "
                "scaler_crop=%d,%d,%d,%d sensor=%d,%d,%d,%d",
                clipped[0],
                clipped[1],
                clipped[2],
                clipped[3],
                preview_size[0],
                preview_size[1],
                *scaler,
                *sensor,
            )
            return {
                "preview_roi": clipped,
                "preview_size": (fw, fh),
                "scaler_crop": scaler,
                "sensor_roi": sensor,
            }

    def trigger_autofocus(self) -> None:
        """Start a one-shot Auto cycle. Do not wait on AfState (often unavailable)."""
        with self._lock:
            if self._use_mock:
                return
            assert self._picam2 is not None
            # Do not call picam2.autofocus_cycle(): with AfState unadvertised it
            # often returns immediately, and a blocking wait holds the camera lock.
            self._held_lens = None
            # Mode first, trigger in the next request: sent together, libcamera sometimes
            # rejects the trigger ("no AF algorithm or not Auto") while still in Continuous.
            self._set_controls_unlocked(
                {
                    "AfMode": self._af_mode_auto(),
                    "AfSpeed": self._af_speed_fast(),
                    "AfRange": self._af_range_full(),
                }
            )
            time.sleep(0.1)
            self._set_controls_unlocked({"AfTrigger": self._af_trigger_start()})
            logger.info("event=autofocus_triggered method=af_controls AfMode=Auto AfTrigger=Start wait=lens_settle")

    def wait_for_lens_settle(self, timeout_sec: float) -> float | None:
        """Wait until LensPosition stops changing. AfState is not required."""
        if self._use_mock:
            time.sleep(0.05)
            return 1.0
        deadline = time.monotonic() + max(0.4, float(timeout_sec))
        time.sleep(0.25)
        last: float | None = None
        stable = 0
        while time.monotonic() < deadline:
            meta = self.capture_metadata()
            raw = meta.get("LensPosition")
            try:
                lens = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                lens = None
            if lens is None:
                time.sleep(0.1)
                continue
            if last is not None and abs(lens - last) < 0.05:
                stable += 1
                if stable >= 3:
                    logger.info("event=autofocus_result method=lens_settle lens_position=%s", lens)
                    return lens
            else:
                stable = 0
            last = lens
            time.sleep(0.1)
        logger.info("event=autofocus_result method=lens_settle_timeout lens_position=%s", last)
        return last

    def autofocus_cycle(self) -> bool:
        """Trigger AF and wait for the lens to settle (not AfState)."""
        self.trigger_autofocus()
        self.wait_for_lens_settle(min(float(self.config.af_timeout_sec), 3.0))
        return True

    def lens_position_range(self) -> tuple[float, float]:
        """Advertised LensPosition range in dioptres (0 = infinity)."""
        if self._use_mock or self._picam2 is None:
            return (0.0, 32.0)  # same as the real Arducam 64MP range
        info = self._picam2.camera_controls.get("LensPosition")
        try:
            return (float(info[0]), float(info[1]))
        except (TypeError, IndexError, ValueError):
            return (0.0, 10.0)

    def set_lens_position(self, position: float) -> None:
        """Drive the lens manually (AfMode=Manual), used by focus refinement."""
        with self._lock:
            if self._use_mock:
                return
            self._set_controls_unlocked(
                {"AfMode": self._af_mode_manual(), "LensPosition": float(position)}
            )
            self._held_lens = float(position)

    def pause_autofocus(self) -> None:
        """Lock the current lens position so Continuous AF does not hunt during still capture."""
        with self._lock:
            if self._use_mock or self._picam2 is None:
                return
            advertised = self._picam2.camera_controls
            if "AfPause" in advertised:
                self._set_controls_unlocked({"AfPause": 0})
                logger.info("event=autofocus_triggered method=pause")
            elif "AfMode" in advertised and "LensPosition" in advertised:
                lens = self._read_lens_position()
                if lens is not None:
                    self._set_controls_unlocked({"AfMode": self._af_mode_manual(), "LensPosition": lens})
                    self._held_lens = float(lens)
                    logger.info("event=autofocus_result method=hold_lens_position lens_position=%s", lens)

    def _af_enum(self, group: str, name: str, fallback: int) -> int:
        if controls is None:
            return fallback
        enum_group = getattr(controls, group, None)
        if enum_group is None:
            return fallback
        value = getattr(enum_group, name, None)
        return fallback if value is None else int(value)

    def _af_mode_manual(self) -> int:
        return self._af_enum("AfModeEnum", "Manual", _AF_MODE_MANUAL)

    def _af_mode_auto(self) -> int:
        return self._af_enum("AfModeEnum", "Auto", _AF_MODE_AUTO)

    def _af_mode_continuous(self) -> int:
        return self._af_enum("AfModeEnum", "Continuous", _AF_MODE_CONTINUOUS)

    def _af_trigger_start(self) -> int:
        return self._af_enum("AfTriggerEnum", "Start", _AF_TRIGGER_START)

    def _af_speed_fast(self) -> int:
        return self._af_enum("AfSpeedEnum", "Fast", 1)

    def _af_metering_windows(self) -> int:
        return self._af_enum("AfMeteringEnum", "Windows", _AF_METERING_WINDOWS)

    def _af_range_full(self) -> int:
        return self._af_enum("AfRangeEnum", "Full", _AF_RANGE_FULL)

    def _af_mode_max(self) -> int | None:
        if self._picam2 is None:
            return None
        info = self._picam2.camera_controls.get("AfMode")
        if info is None:
            return None
        try:
            return int(info[1])
        except (TypeError, IndexError, ValueError):
            return self._af_mode_continuous()

    def _read_lens_position(self) -> float | None:
        if self._picam2 is None:
            return None
        try:
            meta = dict(self._picam2.capture_metadata())
        except Exception:
            return None
        value = meta.get("LensPosition")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # -- manual focus ---------------------------------------------------------------------------

    @property
    def is_manual_focus(self) -> bool:
        return self._manual_lens is not None

    def get_manual_lens(self) -> float | None:
        with self._lock:
            return self._manual_lens

    def set_manual_lens(self, position: float) -> float:
        """Set (and apply right away) the manual lens position; returns the value used."""
        lo, hi = self.lens_position_range()
        value = float(position)
        if not lo <= value <= hi:
            raise CameraError(f"Manual lens position {value:.2f} outside {lo:.1f}..{hi:.1f}")
        with self._lock:
            self._manual_lens = value
            self._held_lens = value
            if self._started and not self._use_mock and self._picam2 is not None:
                self._set_controls_unlocked({"AfMode": self._af_mode_manual(), "LensPosition": value})
        logger.info("event=manual_lens_set lens=%.3f", value)
        return value

    def _apply_manual_lens(self) -> None:
        """Park the lens on the manual value (instead of starting autofocus)."""
        if self._manual_lens is None or self._picam2 is None:
            return
        self._set_controls_unlocked({"AfMode": self._af_mode_manual(), "LensPosition": float(self._manual_lens)})
        self._held_lens = float(self._manual_lens)
        self._preview_af_mode = "Manual"
        logger.info("event=autofocus_disabled mode=manual lens=%.3f", self._manual_lens)

    def _enable_preview_autofocus(self, *, wait: bool = True) -> None:
        """Start real AF on the preview stream (Continuous, else one-shot Auto)."""
        if self._use_mock or self._picam2 is None:
            return
        if self._manual_lens is not None:
            self._apply_manual_lens()
            return
        advertised = self._picam2.camera_controls
        if "AfMode" not in advertised:
            logger.info("event=autofocus_preview_skipped reason=AfMode_not_advertised")
            return

        self._held_lens = None
        continuous = self._af_mode_continuous()
        max_mode = self._af_mode_max()
        if max_mode is not None and max_mode >= continuous:
            self.set_controls({"AfMode": continuous, "AfSpeed": self._af_speed_fast()})
            mode_name = "Continuous"
            logger.info("event=autofocus_triggered mode=%s speed=Fast", mode_name)
        else:
            self.set_controls(
                {
                    "AfMode": self._af_mode_auto(),
                    "AfTrigger": self._af_trigger_start(),
                    "AfSpeed": self._af_speed_fast(),
                }
            )
            mode_name = "Auto"
            logger.info("event=autofocus_triggered mode=%s trigger=Start", mode_name)
        self._preview_af_mode = mode_name
        if not wait:
            return

        deadline = time.monotonic() + min(float(self.config.af_timeout_sec), 3.0)
        last_state: Any = None
        last_lens: Any = None
        while time.monotonic() < deadline:
            try:
                meta = dict(self._picam2.capture_metadata())
            except Exception as exc:
                logger.debug("event=autofocus_metadata_unavailable error=%s", exc)
                break
            last_state = meta.get("AfState")
            last_lens = meta.get("LensPosition")
            if last_state is None:
                logger.info(
                    "event=autofocus_result mode=%s af_state=unavailable lens_position=%s",
                    mode_name,
                    last_lens,
                )
                return
            if int(last_state) in (_AF_STATE_FOCUSED, _AF_STATE_FAILED):
                logger.info(
                    "event=autofocus_result mode=%s af_state=%s lens_position=%s",
                    mode_name,
                    last_state,
                    last_lens,
                )
                return
            time.sleep(0.05)
        logger.warning(
            "event=autofocus_timeout mode=%s last_af_state=%s lens_position=%s",
            mode_name,
            last_state,
            last_lens,
        )

    def _log_preview_format(self, native: np.ndarray, rgb: np.ndarray) -> None:
        native_info = describe_preview_frame(native)
        rgb_info = describe_preview_frame(rgb)
        logger.info(
            "event=preview_frame_format requested=RGB888 "
            "native_order=BGR native_shape=%s native_dtype=%s native_means=%s "
            "rgb_shape=%s rgb_means=%s",
            native_info.get("shape"),
            native_info.get("dtype"),
            native_info.get("channel_means"),
            rgb_info.get("shape"),
            rgb_info.get("channel_means"),
        )

    def _mock_frame(self) -> np.ndarray:
        h, w = self.config.preview_height, self.config.preview_width
        frame = np.full((h, w, 3), 200, dtype=np.uint8)
        # Add some edges in center for detection tests
        ch, cw = h // 3, w // 3
        y0, x0 = h // 3, w // 3
        frame[y0 : y0 + ch, x0 : x0 + 10] = 50
        frame[y0 : y0 + 10, x0 : x0 + cw] = 50
        return frame

    def _mock_still(self, path: Path) -> None:
        from PIL import Image

        img = Image.new("RGB", (self.config.capture_width, self.config.capture_height), color=(220, 220, 220))
        # Draw document-like rectangle
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        margin = 200
        draw.rectangle(
            [margin, margin, self.config.capture_width - margin, self.config.capture_height - margin],
            outline=(30, 30, 30),
            width=20,
        )
        img.save(path, format="JPEG", quality=self.config.jpeg_quality)
