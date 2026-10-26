"""Isolated camera diagnostic helpers — used only by scripts/, not the production pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from idcam.camera.manager import CameraManager, describe_preview_frame, normalize_picamera_preview_frame
from idcam.vision.sharpness import laplacian_variance, preview_sharpness, tenengrad_score


@dataclass
class JpegValidation:
    ok: bool
    path: str
    exists: bool
    file_bytes: int
    width: int
    height: int
    pil_ok: bool
    opencv_ok: bool
    sharpness_laplacian: float
    sha256_prefix: str
    error: str | None = None
    expected_width: int | None = None
    expected_height: int | None = None
    dimension_match: bool | None = None


def capture_native_preview(mgr: CameraManager) -> np.ndarray | None:
    """Return the raw Picamera2 main stream array (before RGB normalization)."""
    if mgr.is_mock or not mgr.is_started:
        return None
    picam2 = mgr._picam2  # diagnostic access to hardware backend
    if picam2 is None:
        return None
    with mgr._lock:
        return picam2.capture_array("main")


def analyze_channels(frame: np.ndarray, label: str) -> dict[str, Any]:
    """Channel statistics for diagnosing RGB vs BGR vs malformed data."""
    info: dict[str, Any] = {
        "label": label,
        "shape": [int(x) for x in frame.shape],
        "dtype": str(frame.dtype),
        "ndim": int(frame.ndim),
    }
    if frame.size == 0:
        info["error"] = "empty_frame"
        return info

    info["min_pixel"] = float(np.min(frame))
    info["max_pixel"] = float(np.max(frame))

    if frame.ndim == 3 and frame.shape[-1] >= 3:
        means = frame.mean(axis=(0, 1))
        c0, c1, c2 = float(means[0]), float(means[1]), float(means[2])
        info["channel_0_mean"] = round(c0, 2)
        info["channel_1_mean"] = round(c1, 2)
        info["channel_2_mean"] = round(c2, 2)
        info["interpreted_as_rgb"] = {
            "R": round(c0, 2),
            "G": round(c1, 2),
            "B": round(c2, 2),
            "R_over_G": round(c0 / max(c1, 1e-6), 4),
            "B_over_G": round(c2 / max(c1, 1e-6), 4),
        }
        info["interpreted_as_bgr"] = {
            "B": round(c0, 2),
            "G": round(c1, 2),
            "R": round(c2, 2),
            "R_over_G": round(c2 / max(c1, 1e-6), 4),
            "B_over_G": round(c0 / max(c1, 1e-6), 4),
        }
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if "rgb" in label.lower() else cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )
    elif frame.ndim == 2:
        gray = frame
        info["gray_mean"] = round(float(np.mean(gray)), 2)
        info["gray_std"] = round(float(np.std(gray)), 2)
        return info
    else:
        info["error"] = "unsupported_channels"
        return info

    info["gray_mean"] = round(float(np.mean(gray)), 2)
    info["gray_std"] = round(float(np.std(gray)), 2)
    return info


def save_rgb_jpeg(frame_rgb: np.ndarray, path: Path, quality: int = 92) -> None:
    """Save an RGB ndarray as JPEG (OpenCV expects BGR)."""
    if frame_rgb.ndim == 2:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"failed to write JPEG: {path}")


def validate_jpeg_file(
    path: Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    tolerance: int = 64,
    min_bytes: int = 1,
) -> JpegValidation:
    result = JpegValidation(
        ok=False,
        path=str(path),
        exists=path.exists(),
        file_bytes=0,
        width=0,
        height=0,
        pil_ok=False,
        opencv_ok=False,
        sharpness_laplacian=0.0,
        sha256_prefix="",
        expected_width=expected_width,
        expected_height=expected_height,
    )
    if not path.exists():
        result.error = "file_not_found"
        return result

    result.file_bytes = path.stat().st_size
    if result.file_bytes < min_bytes:
        result.error = f"file_too_small_{result.file_bytes}"
        return result

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    result.sha256_prefix = digest.hexdigest()[:16]

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            result.width, result.height = img.size
            result.pil_ok = True
    except Exception as exc:
        result.error = f"pil_decode_failed:{exc}"
        return result

    bgr = cv2.imread(str(path))
    if bgr is None:
        result.error = "opencv_decode_failed"
        return result
    result.opencv_ok = True
    if bgr.shape[1] != result.width or bgr.shape[0] != result.height:
        result.error = "pil_opencv_dimension_mismatch"
        return result

    result.sharpness_laplacian = round(laplacian_variance(bgr), 2)

    if expected_width is not None and expected_height is not None:
        result.dimension_match = (
            abs(result.width - expected_width) <= tolerance
            and abs(result.height - expected_height) <= tolerance
        )
        if not result.dimension_match:
            result.error = (
                f"dimension_mismatch got={result.width}x{result.height} "
                f"expected={expected_width}x{expected_height} tol={tolerance}"
            )
            return result

    result.ok = True
    result.error = None
    return result


def preview_sharpness_report(frame_rgb: np.ndarray, roi: tuple[int, int, int, int] | None) -> dict[str, float]:
    return {
        "tenengrad": round(preview_sharpness(frame_rgb, roi=roi), 3),
        "laplacian": round(laplacian_variance(frame_rgb, roi=roi), 3),
    }


def camera_hardware_info(mgr: CameraManager) -> dict[str, Any]:
    info: dict[str, Any] = {
        "is_mock": mgr.is_mock,
        "preview_config": {
            "width": mgr.config.preview_width,
            "height": mgr.config.preview_height,
        },
        "capture_config": {
            "width": mgr.config.capture_width,
            "height": mgr.config.capture_height,
        },
        "jpeg_quality": mgr.config.jpeg_quality,
    }
    if mgr.is_mock or mgr._picam2 is None:
        return info

    picam2 = mgr._picam2
    props = dict(getattr(picam2, "camera_properties", {}) or {})
    info["camera_properties"] = {k: str(v) for k, v in props.items()}
    controls = mgr.get_controls()
    info["controls"] = {k: list(v) if isinstance(v, tuple) else v for k, v in controls.items()}
    info["scaler_crop"] = mgr.current_scaler_crop()
    try:
        info["live_metadata"] = mgr.capture_metadata()
    except Exception as exc:
        info["live_metadata_error"] = str(exc)
    return info


def center_focus_roi(frame_shape: tuple[int, ...], fraction: float = 0.5) -> tuple[int, int, int, int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    rw, rh = max(1, int(w * fraction)), max(1, int(h * fraction))
    x0 = (w - rw) // 2
    y0 = (h - rh) // 2
    return x0, y0, rw, rh
