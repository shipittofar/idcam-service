"""Illumination-normalized motion metrics for preview frames."""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    """Scale gray so global brightness changes do not dominate diff metrics."""
    work = np.ascontiguousarray(gray)
    if work.dtype != np.uint8:
        work = np.clip(work, 0, 255).astype(np.uint8)
    mean = max(float(np.mean(work)), 1.0)
    scaled = (work.astype(np.float32) * (128.0 / mean)).clip(0, 255)
    return scaled.astype(np.uint8)


def structural_motion(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    use_gradients: bool = True,
) -> float:
    """Mean structural change between frames (AE-resistant)."""
    if prev.shape != curr.shape:
        return 255.0
    p = normalize_gray(prev)
    c = normalize_gray(curr)
    if use_gradients:
        pgx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
        pgy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
        cgx = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3)
        cgy = cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
        pm = cv2.magnitude(pgx, pgy)
        cm = cv2.magnitude(cgx, cgy)
        return float(np.mean(cv2.absdiff(pm, cm)))
    return float(np.mean(cv2.absdiff(p, c)))


def illumination_change(prev: np.ndarray, curr: np.ndarray) -> float:
    """Global mean brightness delta — not used as document motion."""
    return abs(float(np.mean(curr)) - float(np.mean(prev)))
