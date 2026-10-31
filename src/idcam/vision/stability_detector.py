"""Scene stability detection on locked document ROI."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from idcam.config.schema import DetectionConfig
from idcam.vision.document_geometry import corner_motion
from idcam.vision.motion import normalize_gray, structural_motion, to_gray
from idcam.vision.sharpness import clip_roi


@dataclass(frozen=True)
class StabilityDebug:
    motion: float
    threshold: float
    stable_count: int
    stable_frames_required: int
    stable: bool


_SKIN_LO = (70, 135, 85)  # YCrCb, same range as the geometry detector's skin mask
_SKIN_HI = (255, 180, 135)


class StabilityDetector:
    """Tracks physical stability using AE-resistant structural motion on document ROI."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._stable_count = 0
        self._stable_latched = False
        self._prev_norm: np.ndarray | None = None
        self._last_motion = 0.0
        self._roi: tuple[int, int, int, int] | None = None
        self._hand = False  # skin next to the page: it is held, so a little tremor is not movement
        self._prev_corners: tuple[tuple[int, int], ...] | None = None

    def reset(self) -> None:
        self._stable_count = 0
        self._stable_latched = False
        self._prev_norm = None
        self._last_motion = 0.0
        self._roi = None
        self._hand = False
        self._prev_corners = None

    def _hand_near_page(self, frame: np.ndarray) -> bool:
        """True when a hand-sized amount of skin is in or around the page's box."""
        if self._roi is None or frame.ndim != 3 or frame.shape[2] != 3:
            return False
        x, y, rw, rh = self._roi
        pad_x, pad_y = int(rw * 0.25), int(rh * 0.25)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        region = frame[y0 : y + rh + pad_y, x0 : x + rw + pad_x]
        if region.size == 0:
            return False
        small = cv2.resize(region, (96, max(8, int(96 * region.shape[0] / max(region.shape[1], 1)))), interpolation=cv2.INTER_AREA)
        mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb), _SKIN_LO, _SKIN_HI)
        return float(np.count_nonzero(mask)) / mask.size > 0.03

    def set_roi(self, roi: tuple[int, int, int, int]) -> None:
        self._roi = tuple(int(v) for v in roi)
        self._prev_norm = None
        self._prev_corners = None
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

    def _update_by_corners(self, corners: tuple[tuple[int, int], ...]) -> bool:
        """Held page: stable while its corners move less than hand_corner_motion_px between frames."""
        previous, self._prev_corners = self._prev_corners, corners
        if previous is None:
            self._stable_count = 0
            self._stable_latched = False
            return False
        motion = corner_motion(previous, corners, robust=True)
        self._last_motion = motion
        if motion < self.config.hand_corner_motion_px:
            self._stable_count += 1
            if self._stable_count >= self.config.stable_frames_required:
                self._stable_latched = True
        else:
            self._stable_count = 0
            self._stable_latched = False
        return self._stable_latched

    def update(self, frame: np.ndarray, corners: tuple[tuple[int, int], ...] | None = None) -> bool:
        """``corners`` (the page's outline in this frame) lets a page held in a hand be judged by pixels."""
        self._hand = self._hand_near_page(frame)
        if self._hand and corners is not None and len(corners) == 4:
            self._prev_norm = None  # the picture-difference path restarts if the hand leaves
            return self._update_by_corners(corners)
        self._prev_corners = None
        gray = self._patch(frame)
        norm = normalize_gray(gray)

        if self._prev_norm is None:
            self._prev_norm = norm
            self._stable_count = 0
            return False

        motion = structural_motion(self._prev_norm, norm)
        self._last_motion = motion
        self._prev_norm = norm

        enter = self.effective_motion_threshold
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
    def effective_motion_threshold(self) -> float:
        """Motion below this counts as still: the base threshold relaxed by the sensitivity setting."""
        return self.config.motion_threshold / self.config.stability_sensitivity

    @property
    def hand_present(self) -> bool:
        return self._hand

    @property
    def stable_count(self) -> int:
        return self._stable_count

    def debug_info(self) -> StabilityDebug:
        return StabilityDebug(
            motion=self._last_motion,
            threshold=self.config.hand_corner_motion_px if self._hand and self._prev_corners else self.effective_motion_threshold,
            stable_count=self._stable_count,
            stable_frames_required=self.config.stable_frames_required,
            stable=self._stable_latched,
        )
