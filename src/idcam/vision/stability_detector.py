"""Scene stability detection on locked document ROI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from idcam.config.schema import DetectionConfig
from idcam.vision.motion import normalize_gray, structural_motion, to_gray
from idcam.vision.sharpness import clip_roi


@dataclass(frozen=True)
class StabilityDebug:
    motion: float
    threshold: float
    stable_count: int
    stable_frames_required: int
    stable: bool


class StabilityDetector:
    """Tracks physical stability using AE-resistant structural motion on document ROI."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._stable_count = 0
        self._stable_latched = False
        self._prev_norm: np.ndarray | None = None
        self._last_motion = 0.0
        self._roi: tuple[int, int, int, int] | None = None

    def reset(self) -> None:
        self._stable_count = 0
        self._stable_latched = False
        self._prev_norm = None
        self._last_motion = 0.0
        self._roi = None

    def set_roi(self, roi: tuple[int, int, int, int]) -> None:
        self._roi = tuple(int(v) for v in roi)
        self._prev_norm = None
        self._stable_count = 0
        self._stable_latched = False

    def _patch(self, frame: np.ndarray) -> np.ndarray:
        if self._roi is not None:
            x, y, rw, rh = clip_roi(self._roi, frame.shape)
            patch = frame[y : y + rh, x : x + rw]
        else:
            h, w = frame.shape[:2]
            frac = self.config.roi_center_fraction
            rh, rw = int(h * frac), int(w * frac)
            y0, x0 = (h - rh) // 2, (w - rw) // 2
            patch = frame[y0 : y0 + rh, x0 : x0 + rw]
        return to_gray(patch)

    def update(self, frame: np.ndarray) -> bool:
        gray = self._patch(frame)
        norm = normalize_gray(gray)

        if self._prev_norm is None:
            self._prev_norm = norm
            self._stable_count = 0
            return False

        motion = structural_motion(self._prev_norm, norm)
        self._last_motion = motion
        self._prev_norm = norm

        enter = self.config.motion_threshold
        exit_threshold = enter * self.config.stability_exit_factor

        if motion < enter:
            self._stable_count += 1
            if self._stable_count >= self.config.stable_frames_required:
                self._stable_latched = True
        elif motion >= exit_threshold:
            self._stable_count = 0
            self._stable_latched = False
        else:
            self._stable_count = 0
            self._stable_latched = False

        return self._stable_latched

    @property
    def stable_count(self) -> int:
        return self._stable_count

    def debug_info(self) -> StabilityDebug:
        return StabilityDebug(
            motion=self._last_motion,
            threshold=self.config.motion_threshold,
            stable_count=self._stable_count,
            stable_frames_required=self.config.stable_frames_required,
            stable=self._stable_latched,
        )
