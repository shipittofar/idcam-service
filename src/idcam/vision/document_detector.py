"""Document presence detection with lock and AE-resistant motion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

from idcam.config.schema import DetectionConfig
from idcam.logging.setup import get_logger
from idcam.vision.document_geometry import corner_motion, detect_document_geometry
from idcam.vision.document_lock import DocumentLock
from idcam.vision.motion import illumination_change, normalize_gray, structural_motion, to_gray
from idcam.vision.sharpness import clip_roi, inner_roi, inscribed_focus_roi, roi_rect

logger = get_logger("vision.document_detector")


class DocumentPresence(str, Enum):
    NO_DOCUMENT = "no_document"
    DOCUMENT_ENTERING = "document_entering"
    DOCUMENT_MOVING = "document_moving"
    DOCUMENT_STABLE = "document_stable"
    DOCUMENT_READY = "document_ready"


@dataclass(frozen=True)
class DocumentDetectionDebug:
    motion: float
    edge_density: float
    motion_threshold: float
    edge_density_threshold: float
    presence_frames: int
    stable_frames_required: int
    detected: bool
    reject_reason: str = ""
    roi_width: int = 0
    roi_height: int = 0
    frame_width: int = 0
    frame_height: int = 0
    gray_std: float = 0.0
    gray_mean: float = 0.0
    frame_motion: float = 0.0
    illumination_change: float = 0.0
    document_locked: bool = False
    document_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    focus_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    lock_confidence: float = 0.0
    geometry_confidence: float = 0.0
    document_corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] = (
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    )


def gradient_edge_density(gray: np.ndarray, magnitude_threshold: float = 30.0) -> float:
    if gray.size == 0:
        return 0.0
    work = np.ascontiguousarray(gray)
    if work.dtype != np.uint8:
        work = np.clip(work, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.count_nonzero(mag > magnitude_threshold)) / mag.size


class DocumentDetector:
    """
    AE-resistant document presence + temporally smoothed document lock.

    One authoritative ROI while locked. Lighting changes do not release the lock.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.lock = DocumentLock(config)
        self._baseline_norm: np.ndarray | None = None
        self._prev_norm: np.ndarray | None = None
        self._present_frames = 0
        self._absent_frames = 0
        self._detected_latched = False
        self._last_motion = 0.0
        self._last_edge_density = 0.0
        self._last_frame_motion = 0.0
        self._last_illumination = 0.0
        self._last_reject_reason = "no_frame"
        self._last_roi_size = (0, 0)
        self._last_frame_size = (0, 0)
        self._last_gray_std = 0.0
        self._last_gray_mean = 0.0
        self._last_document_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._last_focus_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._lock_prev_norm: np.ndarray | None = None
        self._last_geometry: Any = None
        self._prev_corners: tuple[tuple[int, int], ...] | None = None

    @property
    def last_motion(self) -> float:
        return self._last_motion

    @property
    def last_edge_density(self) -> float:
        return self._last_edge_density

    @property
    def is_latched(self) -> bool:
        return self._detected_latched or self.lock.is_locked

    def roi_rect(self, frame_shape: tuple[int, ...] | None = None) -> tuple[int, int, int, int]:
        if frame_shape is not None and len(frame_shape) >= 2:
            h, w = int(frame_shape[0]), int(frame_shape[1])
        else:
            w, h = self._last_frame_size
        return roi_rect((h, w), self.config.roi_center_fraction)

    def _find_geometry(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        self._last_frame_size = (int(w), int(h))
        geom = detect_document_geometry(
            frame,
            min_area_fraction=self.config.geometry_min_area_fraction,
            max_area_fraction=self.config.geometry_max_area_fraction,
        )
        self._last_geometry = geom
        if geom:
            self._last_document_roi = geom.bbox
            self._last_focus_roi = inscribed_focus_roi(
                geom.corners,
                frame.shape,
                self.config.focus_roi_inset_fraction,
            )
            self._last_roi_size = (geom.bbox[2], geom.bbox[3])
            self._last_edge_density = min(1.0, geom.edge_strength / 120.0)
        return geom

    def document_preview_roi(self, frame: np.ndarray | None = None) -> tuple[int, int, int, int]:
        if self.lock.is_locked:
            return self.lock.roi
        if self._last_document_roi != (0, 0, 0, 0):
            return self._last_document_roi
        if frame is not None:
            geom = self._find_geometry(frame)
            if geom:
                return geom.bbox
        return self.roi_rect(frame.shape if frame is not None else None)

    def focus_preview_roi(self, frame: np.ndarray | None = None) -> tuple[int, int, int, int]:
        if self.lock.is_locked:
            return self.lock.focus_roi
        if self._last_focus_roi != (0, 0, 0, 0):
            return self._last_focus_roi
        if frame is not None:
            roi = self.document_preview_roi(frame)
            return inner_roi(roi, self.config.focus_roi_inset_fraction, frame.shape)
        return self.roi_rect()

    def lock_document(self, frame: np.ndarray, allow_fallback: bool = True) -> bool:
        """Lock the detected document. Without geometry: fake centre region, or False if not allowed.

        Automatic captures must pass ``allow_fallback=False``: locking an invented region would
        focus and capture an empty table. Only a user-requested manual capture may fall back.
        """
        geom = self._last_geometry or self._find_geometry(frame)
        if geom is None and not allow_fallback:
            logger.warning("event=document_lock_failed reason=no_geometry")
            return False
        if geom is None:
            logger.warning("event=document_lock_fallback reason=no_geometry")
            h, w = frame.shape[:2]
            margin_x, margin_y = int(w * 0.15), int(h * 0.15)
            corners = (
                (margin_x, margin_y),
                (w - margin_x, margin_y),
                (w - margin_x, h - margin_y),
                (margin_x, h - margin_y),
            )
            from idcam.vision.document_geometry import DocumentGeometry, corners_to_bbox

            bbox = corners_to_bbox(corners, frame.shape)
            geom = DocumentGeometry(
                corners=corners,
                bbox=bbox,
                center=(bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2),
                width=bbox[2],
                height=bbox[3],
                rotation_deg=0.0,
                aspect_ratio=bbox[2] / max(bbox[3], 1),
                area_fraction=(bbox[2] * bbox[3]) / max(w * h, 1),
                confidence=0.35,
                edge_strength=0.0,
                rectangularity=0.9,
            )
        self.lock.lock(frame, geom)
        self._last_document_roi = self.lock.roi
        self._last_focus_roi = self.lock.focus_roi
        self._prev_corners = self.lock.corners
        # No pixel baseline here: focusing changes sharpness, which the pixel metric reads as motion.
        self._lock_prev_norm = None
        self._detected_latched = True
        self._present_frames = self.config.detect_frames_required
        self._absent_frames = 0
        return True

    def release_lock(self) -> None:
        self.lock.release()
        self._detected_latched = False
        self._present_frames = 0
        self._absent_frames = 0

    def _roi_gray(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._last_frame_size = (int(w), int(h))
        x0, y0, rw, rh = self.roi_rect(frame.shape)
        self._last_roi_size = (int(rw), int(rh))
        gray = to_gray(frame[y0 : y0 + rh, x0 : x0 + rw])
        return np.ascontiguousarray(gray)

    def _patch_gray(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        x, y, rw, rh = clip_roi(roi, frame.shape)
        return to_gray(frame[y : y + rh, x : x + rw])

    def _metrics(self, frame: np.ndarray) -> tuple[float, float]:
        gray = self._roi_gray(frame)
        norm = normalize_gray(gray)
        edge_density = gradient_edge_density(gray)
        self._last_gray_std = float(np.std(gray))
        self._last_gray_mean = float(np.mean(gray))
        self._last_edge_density = edge_density

        if self._prev_norm is not None and self._prev_norm.shape == norm.shape:
            self._last_frame_motion = structural_motion(self._prev_norm, norm)
            self._last_illumination = illumination_change(self._prev_norm, norm)
        else:
            self._last_frame_motion = 0.0
            self._last_illumination = 0.0
        self._prev_norm = norm.copy()

        if self._baseline_norm is None:
            self._baseline_norm = norm.copy()
            self._last_motion = 0.0
            self._last_reject_reason = "baseline_initialized"
            return 0.0, edge_density

        motion = structural_motion(self._baseline_norm, norm)
        self._last_motion = motion

        strong = self._strong_present(motion, edge_density)
        if not strong:
            if motion <= self.config.motion_threshold and edge_density <= self.config.edge_density_threshold:
                self._last_reject_reason = "below_motion_and_edge"
            elif motion <= self.config.motion_threshold:
                self._last_reject_reason = "below_motion_threshold"
            else:
                self._last_reject_reason = "below_edge_density_threshold"
        else:
            self._last_reject_reason = ""

        center = self.roi_rect(frame.shape)
        found_roi = None
        if self.lock.is_locked and self._last_geometry:
            found_roi = self._last_geometry.bbox

        return motion, edge_density

    def _strong_present(self, motion: float, edge_density: float) -> bool:
        return (
            motion > self.config.motion_threshold
            and edge_density > self.config.edge_density_threshold
        )

    def _clearly_absent(self, motion: float, edge_density: float) -> bool:
        return (
            motion < self.config.motion_threshold * self.config.lose_motion_factor
            and edge_density < self.config.edge_density_threshold * self.config.lose_edge_factor
        )

    def _update_latch(self, motion: float, edge_density: float) -> bool:
        if self.lock.is_locked:
            return True
        strong = self._strong_present(motion, edge_density)
        if strong:
            self._present_frames += 1
            self._absent_frames = 0
        else:
            self._present_frames = 0
            if self._detected_latched:
                if self._clearly_absent(motion, edge_density):
                    self._absent_frames += 1
            else:
                self._absent_frames += 1

        if not self._detected_latched and self._present_frames >= self.config.detect_frames_required:
            self._detected_latched = True
            self._absent_frames = 0

        if self._detected_latched and not self.lock.is_locked and self._absent_frames >= self.config.lose_frames_required:
            self._detected_latched = False
            self._present_frames = 0

        return self._detected_latched or self.lock.is_locked

    def debug_info(self) -> DocumentDetectionDebug:
        detected = self.is_latched
        roi = self.lock.roi if self.lock.is_locked else self._last_document_roi
        focus = self.lock.focus_roi if self.lock.is_locked else self._last_focus_roi
        corners = self.lock.corners if self.lock.is_locked else (
            self._last_geometry.corners if self._last_geometry else ((0, 0), (0, 0), (0, 0), (0, 0))
        )
        geom_conf = self._last_geometry.confidence if self._last_geometry else 0.0
        return DocumentDetectionDebug(
            motion=self._last_motion,
            edge_density=self._last_edge_density,
            motion_threshold=self.config.motion_threshold,
            edge_density_threshold=self.config.edge_density_threshold,
            presence_frames=self._present_frames,
            stable_frames_required=self.config.stable_frames_required,
            detected=detected,
            reject_reason="" if detected else self._last_reject_reason,
            roi_width=self._last_roi_size[0],
            roi_height=self._last_roi_size[1],
            frame_width=self._last_frame_size[0],
            frame_height=self._last_frame_size[1],
            gray_std=self._last_gray_std,
            gray_mean=self._last_gray_mean,
            frame_motion=self._last_frame_motion,
            illumination_change=self._last_illumination,
            document_locked=self.lock.is_locked,
            document_roi=roi,
            focus_roi=focus,
            lock_confidence=self.lock.state.confidence,
            geometry_confidence=geom_conf,
            document_corners=corners,
        )

    def reset_baseline(self) -> None:
        self._baseline_norm = None
        self._prev_norm = None
        self._present_frames = 0
        self._absent_frames = 0
        self._detected_latched = False
        self.release_lock()
        self._last_document_roi = (0, 0, 0, 0)
        self._last_focus_roi = (0, 0, 0, 0)
        self._lock_prev_norm = None
        self._last_reject_reason = "no_frame"

    def update_baseline(self, frame: np.ndarray) -> None:
        gray = self._roi_gray(frame)
        norm = normalize_gray(gray)
        if self._baseline_norm is None:
            self._baseline_norm = norm.copy()
        else:
            alpha = self.config.baseline_alpha
            self._baseline_norm = cv2.addWeighted(norm, alpha, self._baseline_norm, 1 - alpha, 0)

    def classify(self, frame: np.ndarray) -> DocumentPresence:
        if self.lock.is_locked:
            motion = self.structural_motion_in_lock(frame)
            self._last_frame_motion = motion
            if motion > self.config.motion_threshold * 2.0:
                return DocumentPresence.DOCUMENT_MOVING
            return DocumentPresence.DOCUMENT_STABLE

        motion, edge_density = self._metrics(frame)
        present = self._update_latch(motion, edge_density)
        if not present:
            if self._clearly_absent(motion, edge_density):
                gray = self._roi_gray(frame)
                norm = normalize_gray(gray)
                if self._baseline_norm is None:
                    self._baseline_norm = norm.copy()
                else:
                    alpha = self.config.baseline_alpha
                    self._baseline_norm = cv2.addWeighted(norm, alpha, self._baseline_norm, 1 - alpha, 0)
            return DocumentPresence.NO_DOCUMENT

        if self._present_frames < self.config.detect_frames_required:
            return DocumentPresence.DOCUMENT_ENTERING
        if self._last_frame_motion > self.config.motion_threshold * 2.0:
            return DocumentPresence.DOCUMENT_MOVING
        return DocumentPresence.DOCUMENT_STABLE

    def detect(self, frame: np.ndarray) -> bool:
        gray = to_gray(frame)
        self._last_gray_std = float(np.std(gray))
        self._last_gray_mean = float(np.mean(gray))

        geom = self._find_geometry(frame)
        if self.lock.is_locked:
            if geom and geom.confidence >= self.config.min_geometry_confidence * 0.65:
                self.lock.update(frame, geom)
                self._prev_corners = self.lock.corners
            return True

        if geom and geom.confidence >= self.config.min_geometry_confidence:
            self._present_frames += 1
            self._absent_frames = 0
            self._last_reject_reason = ""
            if not self._detected_latched and self._present_frames >= self.config.detect_frames_required:
                self._detected_latched = True
        else:
            self._present_frames = 0
            if self._detected_latched:
                self._absent_frames += 1
                if self._absent_frames >= self.config.lose_frames_required:
                    self._detected_latched = False
            else:
                self._absent_frames += 1
            if geom is None:
                self._last_reject_reason = "no_document_geometry"
            else:
                self._last_reject_reason = f"low_geometry_confidence_{geom.confidence:.2f}"

        return self._detected_latched or self.lock.is_locked

    def is_scene_empty(self, frame: np.ndarray) -> bool:
        """True when no document is in view.

        An empty desk still yields weak candidate outlines (confidence ~0.2 in logs), so "empty"
        must mean "below the presence threshold", not "no candidate at all": the old rule
        (empty only below 0.35x the detection threshold) never fired and the removal wait hung.
        A small margin under the detection threshold keeps a present page from flickering to empty.
        """
        geom = self._find_geometry(frame)
        return geom is None or geom.confidence < self.config.min_geometry_confidence * 0.8

    def rebaseline_motion(self, frame: np.ndarray) -> None:
        """Re-measure the locked outline on a frame taken after focusing.

        The lock-time corners come from a defocused frame; sharp frames outline the same page a
        few tens of pixels differently (up to ~80 px for a large page), which the pre-capture
        movement check then misread as the document moving.
        """
        geom = self._find_geometry(frame)
        if geom is not None and (self._prev_corners is None or self._same_document(self._prev_corners, geom.corners)):
            self._prev_corners = geom.corners
            if self.lock.is_locked:
                self.lock.update(frame, geom)
        self._lock_prev_norm = None

    @staticmethod
    def _same_document(prev: tuple[tuple[int, int], ...], new: tuple[tuple[int, int], ...]) -> bool:
        pa = np.array(prev, dtype=np.float32)
        pb = np.array(new, dtype=np.float32)
        area_a = float(cv2.contourArea(pa))
        area_b = float(cv2.contourArea(pb))
        if area_a <= 0 or area_b <= 0 or not 0.7 <= area_b / area_a <= 1.43:
            return False
        shift = float(np.linalg.norm(pa.mean(axis=0) - pb.mean(axis=0)))
        return shift < 0.25 * float(np.sqrt(area_a))

    def structural_motion_in_lock(self, frame: np.ndarray) -> float:
        if not self.lock.is_locked:
            return self._last_frame_motion
        geom = detect_document_geometry(
            frame,
            min_area_fraction=self.config.geometry_min_area_fraction,
            max_area_fraction=self.config.geometry_max_area_fraction,
        )
        # Re-detection can pick a different quadrilateral (e.g. whole frame vs the paper) after
        # focus changes; corner distance between different objects is not document movement
        # (observed motion=380 with a still document). Only compare corners of the same object.
        if geom and self._prev_corners is not None and self._same_document(self._prev_corners, geom.corners):
            motion = corner_motion(self._prev_corners, geom.corners, robust=True)
            self._prev_corners = geom.corners
            self._last_frame_motion = motion
            return motion
        gray = self._patch_gray(frame, self.lock.roi)
        norm = normalize_gray(gray)
        if self._lock_prev_norm is None or self._lock_prev_norm.shape != norm.shape:
            self._lock_prev_norm = norm.copy()
            return 0.0
        motion = structural_motion(self._lock_prev_norm, norm)
        self._lock_prev_norm = norm.copy()
        self._last_frame_motion = motion
        return motion
