"""Image sharpness measurement — preview vs still-image scales differ."""

from __future__ import annotations

import cv2
import numpy as np


def roi_rect(frame_shape: tuple[int, ...], roi_fraction: float = 0.7) -> tuple[int, int, int, int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    frac = min(max(float(roi_fraction), 0.05), 1.0)
    rh, rw = max(1, int(h * frac)), max(1, int(w * frac))
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    return int(x0), int(y0), int(rw), int(rh)


def clip_roi(roi: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x, y, rw, rh = (int(v) for v in roi)
    x = max(0, min(x, max(0, w - 1)))
    y = max(0, min(y, max(0, h - 1)))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return x, y, rw, rh


def inner_roi(
    roi: tuple[int, int, int, int],
    inset_fraction: float = 0.12,
    frame_shape: tuple[int, ...] | None = None,
) -> tuple[int, int, int, int]:
    x, y, rw, rh = (int(v) for v in roi)
    inset_fraction = min(max(float(inset_fraction), 0.0), 0.4)
    dx = max(1, int(rw * inset_fraction))
    dy = max(1, int(rh * inset_fraction))
    inner = (x + dx, y + dy, max(1, rw - 2 * dx), max(1, rh - 2 * dy))
    if frame_shape is not None:
        return clip_roi(inner, frame_shape)
    return inner


def _gray_patch(frame: np.ndarray, roi: tuple[int, int, int, int] | None, roi_fraction: float) -> np.ndarray:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    if roi is None:
        x0, y0, rw, rh = roi_rect(gray.shape, roi_fraction)
    else:
        x0, y0, rw, rh = clip_roi(roi, gray.shape)
    return gray[y0 : y0 + rh, x0 : x0 + rw]


def laplacian_variance(
    frame: np.ndarray,
    roi_fraction: float = 0.7,
    roi: tuple[int, int, int, int] | None = None,
) -> float:
    patch = _gray_patch(frame, roi, roi_fraction)
    laplacian = cv2.Laplacian(patch, cv2.CV_64F)
    return float(laplacian.var())


def tenengrad_score(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    roi_fraction: float = 0.7,
) -> float:
    """Gradient-energy sharpness for 1080p preview. Correlates better than Laplacian var at low res."""
    patch = _gray_patch(frame, roi, roi_fraction)
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def preview_sharpness(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    roi_fraction: float = 0.7,
) -> float:
    """Primary preview focus metric (Tenengrad mean energy)."""
    return tenengrad_score(frame, roi=roi, roi_fraction=roi_fraction)


def accept_preview_focus(
    *,
    before: float,
    after: float,
    absolute_floor: float,
    min_ratio: float = 1.15,
    min_delta: float = 1.5,
) -> bool:
    """Accept preview AF from meaningful improvement — not still-image thresholds."""
    if after <= 0:
        return False
    if before <= 0:
        return after >= absolute_floor
    ratio = after / before
    if ratio >= min_ratio and (after - before) >= min_delta:
        return True
    if after >= absolute_floor and after >= before * 0.92:
        return True
    return False


def scale_preview_roi_to_image(
    roi: tuple[int, int, int, int],
    preview_size: tuple[int, int],
    image_size: tuple[int, int],
    scaler_crop: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Map preview ROI to full-resolution still coordinates via ScalerCrop."""
    from idcam.camera.manager import map_preview_roi_to_sensor

    pw, ph = int(preview_size[0]), int(preview_size[1])
    iw, ih = int(image_size[0]), int(image_size[1])
    if scaler_crop is not None:
        sx, sy, sw, sh = (int(v) for v in scaler_crop)
        mapped = map_preview_roi_to_sensor(roi, (pw, ph), scaler_crop)
        return clip_roi(mapped, (ih, iw))
    sx = int(round(roi[0] * iw / max(pw, 1)))
    sy = int(round(roi[1] * ih / max(ph, 1)))
    sw = max(1, int(round(roi[2] * iw / max(pw, 1))))
    sh = max(1, int(round(roi[3] * ih / max(ph, 1))))
    return clip_roi((sx, sy, sw, sh), (ih, iw))


def still_sharpness(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> float:
    """Laplacian variance on full-resolution still (BGR or RGB)."""
    return laplacian_variance(frame, roi=roi)


def combined_preview_sharpness(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[float, float]:
    """Return Tenengrad and Laplacian on preview document ROI."""
    ten = preview_sharpness(frame, roi=roi)
    lap = laplacian_variance(frame, roi=roi)
    return ten, lap


def mean_brightness(frame: np.ndarray) -> float:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    return float(np.mean(gray))
