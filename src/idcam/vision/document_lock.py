"""Stable document lock with temporally smoothed quadrilateral."""

from __future__ import annotations

from dataclasses import dataclass
import time

from idcam.config.schema import DetectionConfig
from idcam.logging.setup import get_logger
from idcam.vision.document_geometry import DocumentGeometry, smooth_corners
from idcam.vision.sharpness import clip_roi, inner_roi

logger = get_logger("vision.document_lock")


@dataclass
class DocumentLockState:
    locked: bool = False
    corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] = (
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    )
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    focus_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    locked_at: float = 0.0
    updates: int = 0
    aspect_ratio: float = 0.0
    rotation_deg: float = 0.0
    area_fraction: float = 0.0


class DocumentLock:
    """One smoothed document quadrilateral for the active capture pipeline."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._state = DocumentLockState()

    @property
    def state(self) -> DocumentLockState:
        return self._state

    @property
    def is_locked(self) -> bool:
        return self._state.locked

    @property
    def corners(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        return self._state.corners

    @property
    def roi(self) -> tuple[int, int, int, int]:
        return self._state.roi

    @property
    def focus_roi(self) -> tuple[int, int, int, int]:
        return self._state.focus_roi

    def lock(self, frame, geometry: DocumentGeometry) -> None:
        roi = clip_roi(geometry.bbox, frame.shape)
        focus = inner_roi(
            roi,
            inset_fraction=self.config.focus_roi_inset_fraction,
            frame_shape=frame.shape,
        )
        self._state = DocumentLockState(
            locked=True,
            corners=geometry.corners,
            roi=roi,
            focus_roi=focus,
            confidence=geometry.confidence,
            locked_at=time.monotonic(),
            updates=1,
            aspect_ratio=geometry.aspect_ratio,
            rotation_deg=geometry.rotation_deg,
            area_fraction=geometry.area_fraction,
        )
        logger.info(
            "event=pipeline_stage stage=document_locked "
            "corners=%s roi=%d,%d,%d,%d focus_roi=%d,%d,%d,%d confidence=%.2f",
            geometry.corners,
            *roi,
            *focus,
            geometry.confidence,
        )

    def update(self, frame, geometry: DocumentGeometry) -> None:
        if not self._state.locked:
            return
        smoothed_corners = smooth_corners(
            self._state.corners,
            geometry.corners,
            self.config.roi_smooth_alpha,
            frame.shape,
        )
        xs = [c[0] for c in smoothed_corners]
        ys = [c[1] for c in smoothed_corners]
        roi = clip_roi((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)), frame.shape)
        focus = inner_roi(
            roi,
            inset_fraction=self.config.focus_roi_inset_fraction,
            frame_shape=frame.shape,
        )
        self._state.corners = smoothed_corners
        self._state.roi = roi
        self._state.focus_roi = focus
        self._state.confidence = max(self._state.confidence * 0.9, geometry.confidence)
        self._state.aspect_ratio = geometry.aspect_ratio
        self._state.rotation_deg = geometry.rotation_deg
        self._state.area_fraction = geometry.area_fraction
        self._state.updates += 1

    def release(self) -> None:
        if self._state.locked:
            logger.info("event=pipeline_stage stage=document_lock_released")
        self._state = DocumentLockState()
