"""Document presence detection with lock and AE-resistant motion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import time

import cv2
import numpy as np

from idcam.config.schema import DetectionConfig
from idcam.logging.setup import get_logger
from idcam.vision.document_geometry import _bbox_iou, corner_motion, detect_document_geometry
from idcam.vision.document_lock import DocumentLock
from idcam.vision.motion import illumination_change, normalize_gray, structural_motion, to_gray
from idcam.vision.sharpness import clip_roi, inner_roi, inscribed_focus_roi, roi_rect

logger = get_logger("vision.document_detector")

# Static-scene gate: segmenting a frame costs ~0.1 s on a PC (several times that on the Pi) and the
# idle loop, the stability wait and the pre-capture check all ask for the geometry of a scene that
# has not changed. A frame that matches the last analysed one on a tiny thumbnail reuses its result.
_GATE_SIZE = (96, 54)
_GATE_MEAN_DIFF = 0.8  # grey levels, averaged over the thumbnail
_GATE_PIXEL_DIFF = 10  # a thumbnail pixel that changed by more than this ...
_GATE_MAX_CHANGED = 2  # ... may occur at most this often
_GATE_MAX_REUSE = 8  # consecutive reuses before the frame is analysed again anyway
_BG_THUMB = (160, 90)  # colour thumbnail used to learn / compare the empty background
_SIG_SIZE = (64, 48)  # grey patch of a captured document, to recognise a different one in its place
_REPLACE_CALM_FRAMES = 5  # the picture must have been still this long before "replaced" is believed
_REPLACE_MIN_IOU = 0.5  # a new outline overlapping the old one less than this is another document
_REPLACE_MIN_NCC = 0.8  # correlation of the old document's area with itself now; lower = different content
# After the picture was disturbed (a hand took the document away) the same page rarely comes back looking
# identical, but two cards of the same template do look alike: then a smaller change is enough.
_REPLACE_MIN_IOU_DISTURBED = 0.9
_REPLACE_MIN_NCC_DISTURBED = 0.95


class DocumentPresence(str, Enum):
    NO_DOCUMENT = "no_document"
    DOCUMENT_ENTERING = "document_entering"
    DOCUMENT_MOVING = "document_moving"
    DOCUMENT_STABLE = "document_stable"
    DOCUMENT_READY = "document_ready"


_ASCII_RAMP = " .:-=+*#%@"


def ascii_snapshot(frame: np.ndarray, quad: Any = None, cols: int = 64) -> tuple[str, str]:
    """Two small text renderings of a frame, for a log line (rows joined by '|').

    ``gray``: brightness as characters, the detected outline drawn with 'Q'. ``cr``: the red chroma channel
    as digits 0-9 (skin and other warm colours are high, 4-9), so a hand shows up whatever the skin
    classifier thinks. For judging a detection from a log when no image can be sent.
    """
    h, w = frame.shape[:2]
    rows = max(8, int(round(cols * h / max(w, 1) * 0.5)))  # characters are about twice as tall as wide
    small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
    gray = to_gray(small)
    grid = [[_ASCII_RAMP[min(9, int(v) * 10 // 256)] for v in row] for row in gray]
    if quad is not None and len(quad) == 4:
        pts = [(x * cols / max(w, 1), y * rows / max(h, 1)) for x, y in quad]
        for i in range(4):
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % 4]
            steps = int(max(abs(x1 - x0), abs(y1 - y0), 1)) * 2
            for t in range(steps + 1):
                x = int(round(x0 + (x1 - x0) * t / steps))
                y = int(round(y0 + (y1 - y0) * t / steps))
                if 0 <= x < cols and 0 <= y < rows:
                    grid[y][x] = "Q"
    cr = cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb)[..., 1].astype(np.int16) if small.ndim == 3 else np.zeros_like(gray)
    digits = np.clip((cr - 100) // 8, 0, 9)
    return "|".join("".join(r) for r in grid), "|".join("".join(str(int(d)) for d in row) for row in digits)


@dataclass(frozen=True)
class DocumentSignature:
    """What a captured document looked like: its box and a small grey patch of that area."""

    bbox: tuple[int, int, int, int]
    patch: np.ndarray


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
        self._first_seen_at: float | None = None
        # Frame the cached ``_last_geometry`` was computed on (identity check, see _geometry_for).
        self._geometry_frame: np.ndarray | None = None
        self._gate_small: np.ndarray | None = None
        self._gate_reuse = 0
        self._candidate_bbox: tuple[int, int, int, int] | None = None
        # Empty-scene background (see DetectionConfig.background_check)
        self._bg_thumb: np.ndarray | None = None
        self._calm_streak = 0  # consecutive frames that look like the previous analysed one
        self._bg_frames_seen = 0
        self._bg_given_up = False
        self.gate_hits = 0  # frames answered from the previous analysis (diagnostics / tests)

    @property
    def detection_latency_ms(self) -> int:
        """Time since the first frame that showed the document (0 when none is being tracked)."""
        if self._first_seen_at is None:
            return 0
        return int((time.monotonic() - self._first_seen_at) * 1000)

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

    def _scene_unchanged(self, frame: np.ndarray) -> bool:
        """True (and the thumbnail kept) when ``frame`` matches the last analysed one."""
        small = cv2.resize(to_gray(frame), _GATE_SIZE, interpolation=cv2.INTER_AREA)
        previous = self._gate_small
        similar = False
        if previous is not None:
            diff = cv2.absdiff(small, previous)
            similar = float(diff.mean()) < _GATE_MEAN_DIFF and int(np.count_nonzero(diff > _GATE_PIXEL_DIFF)) <= _GATE_MAX_CHANGED
        self._calm_streak = self._calm_streak + 1 if similar else 0
        if similar and self._gate_reuse < _GATE_MAX_REUSE:
            return True
        self._gate_small = small
        self._gate_reuse = 0
        return False

    # -- empty-scene background ---------------------------------------------------------------

    @staticmethod
    def _bg_thumbnail(frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, _BG_THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)

    def _update_background(self, thumb: np.ndarray) -> bool:
        """Learn / refresh the empty background. False while it is still being learned at start."""
        cfg = self.config
        if self._bg_given_up:
            return True
        self._bg_frames_seen += 1
        if self._bg_thumb is None:
            if self._calm_streak >= cfg.background_warmup_frames:
                self._bg_thumb = thumb
                logger.info("event=background_learned frames_seen=%d", self._bg_frames_seen)
                return True
            if self._bg_frames_seen >= cfg.background_give_up_frames:
                self._bg_given_up = True
                logger.warning("event=background_check_skipped reason=scene_never_calm")
                return True
            return False
        if not self.is_latched and self._calm_streak and self._calm_streak % cfg.background_refresh_frames == 0:
            self._bg_thumb = thumb  # follow slow light changes while nothing is detected
        return True

    def _is_background(self, thumb: np.ndarray, bbox: tuple[int, int, int, int], frame_size: tuple[int, int]) -> bool:
        """True when the candidate's area still looks like the learned empty background."""
        base = self._bg_thumb
        if base is None or base.shape != thumb.shape:
            return False
        fw, fh = frame_size
        tw, th = _BG_THUMB
        x, y, w, h = bbox
        x0 = max(0, min(tw - 1, int(x * tw / fw)))
        y0 = max(0, min(th - 1, int(y * th / fh)))
        x1 = max(x0 + 1, min(tw, int(np.ceil((x + w) * tw / fw))))
        y1 = max(y0 + 1, min(th, int(np.ceil((y + h) * th / fh))))
        diff = thumb - base
        outside = np.ones(diff.shape[:2], bool)
        outside[y0:y1, x0:x1] = False
        offset = diff[outside].mean(axis=0) if outside.sum() >= 0.1 * outside.size else 0.0  # global exposure shift
        inner = float(np.abs(diff[y0:y1, x0:x1] - offset).mean())
        return inner < self.config.background_min_diff

    def _find_geometry(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        self._last_frame_size = (int(w), int(h))
        if self._scene_unchanged(frame):
            self._gate_reuse += 1
            self.gate_hits += 1
            self._geometry_frame = frame
            return self._last_geometry
        geom = detect_document_geometry(
            frame,
            min_area_fraction=self.config.geometry_min_area_fraction,
            max_area_fraction=self.config.geometry_max_area_fraction,
            touch_penalty_vertical=self.config.geometry_touch_penalty_vertical,
            change_mask=self._change_mask(frame),
        )
        self._last_geometry = geom
        self._geometry_frame = frame
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

    # -- recognising a different document in the same place ------------------------------------------

    @staticmethod
    def _signature_patch(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x, y, bw, bh = (int(v) for v in bbox)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        gray = to_gray(frame[y0:y1, x0:x1])
        return cv2.resize(gray, _SIG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)

    def document_signature(self, frame: np.ndarray) -> DocumentSignature | None:
        """Remember the document that is in view now (box of its outline + a small patch)."""
        geom = self._find_geometry(frame)
        bbox = tuple(geom.bbox) if geom is not None else (self.lock.roi if self.lock.is_locked else None)
        if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
            return None
        patch = self._signature_patch(frame, bbox)  # type: ignore[arg-type]
        return None if patch is None else DocumentSignature(tuple(int(v) for v in bbox), patch)  # type: ignore[arg-type]

    def _present_threshold(self, geom: Any, factor: float = 1.0) -> float:
        """Confidence a candidate needs; less when it was found with a hand's pixels cut out."""
        threshold = self.config.min_geometry_confidence * factor
        if geom is not None and str(getattr(geom, "detection_method", "")).endswith("_noskin"):
            threshold -= self.config.hand_confidence_relax
        return threshold

    def current_corners(self, frame: np.ndarray) -> tuple[tuple[int, int], ...] | None:
        """Outline of the page in ``frame`` (analysed already for this frame in the stability wait), if confident."""
        geom = self._find_geometry(frame)
        if geom is None or geom.confidence < self._present_threshold(geom, 0.9):
            return None
        return tuple(geom.corners)

    def background_gap(self, frame: np.ndarray) -> float | None:
        """Mean colour difference of the picture to the learned empty scene (None while not learned); for logs."""
        if self._bg_thumb is None:
            return None
        thumb = self._bg_thumbnail(frame)
        if self._bg_thumb.shape != thumb.shape:
            return None
        diff = thumb - self._bg_thumb
        offset = np.median(diff.reshape(-1, diff.shape[-1]), axis=0)
        return float(np.abs(diff - offset).mean())

    @property
    def calm_streak(self) -> int:
        """Consecutive frames that looked like the previous analysed one (0 = the picture just changed)."""
        return self._calm_streak

    def document_replaced(
        self, frame: np.ndarray, signature: DocumentSignature | None, *, disturbed: bool = False
    ) -> bool:
        """True when the document in view is a different one than ``signature`` (swapped by the next person).

        Only believed when the picture has been still for a few frames (a hand passing is not a swap) and
        a real document is present: the same bar as for a first detection (confidence, and not part of the
        learned empty background), so a bare desk that scores 0.45-0.5 can never "replace" a document.
        Either its outline barely overlaps the old one, or the old area no longer correlates with what it
        showed. ``disturbed`` (the picture changed since the capture, e.g. a hand took the page away) makes
        smaller differences count, which tells two cards of the same template apart.
        """
        if signature is None:
            return False
        geom = self._find_geometry(frame)  # also advances the calm counter for this frame
        if self._calm_streak < _REPLACE_CALM_FRAMES:
            return False
        if geom is None or geom.confidence < self._present_threshold(geom):
            return False  # nothing there: the ordinary removal path handles it
        bbox = tuple(geom.bbox)
        if self.config.background_check and self._bg_thumb is not None:
            h, w = frame.shape[:2]
            if self._is_background(self._bg_thumbnail(frame), bbox, (w, h)):
                return False
        min_iou = _REPLACE_MIN_IOU_DISTURBED if disturbed else _REPLACE_MIN_IOU
        min_ncc = _REPLACE_MIN_NCC_DISTURBED if disturbed else _REPLACE_MIN_NCC
        if _bbox_iou(signature.bbox, bbox) < min_iou:
            return True
        now = self._signature_patch(frame, signature.bbox)
        if now is None:
            return False
        a = signature.patch - float(signature.patch.mean())
        b = now - float(now.mean())
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if denom < 1e-6:
            return False  # a flat area carries no content to compare
        return float((a * b).sum()) / denom < min_ncc

    def refresh_lock(self, frame: np.ndarray) -> bool:
        """Re-lock when the outline in view no longer matches the lock (the page was swapped or moved)."""
        if not self.lock.is_locked:
            return False
        geom = self._find_geometry(frame)
        if geom is None or geom.confidence < self._present_threshold(geom, 0.9):
            return False
        if _bbox_iou(self.lock.roi, tuple(geom.bbox)) >= 0.7:
            return False
        self.lock.release()
        return self.lock_document(frame, allow_fallback=False)

    def release_lock(self) -> None:
        self._gate_small = None
        self._candidate_bbox = None
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
        self._gate_small = None
        self._candidate_bbox = None
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
            if geom and geom.confidence >= self._present_threshold(geom, 0.65):
                self.lock.update(frame, geom)
                self._prev_corners = self.lock.corners
            return True

        background_reason = ""
        if self.config.background_check:
            thumb = self._bg_thumbnail(frame)
            if not self._update_background(thumb):
                background_reason = "learning_background"
            elif geom and geom.confidence >= self._present_threshold(geom):
                h, w = frame.shape[:2]
                if self._is_background(thumb, tuple(geom.bbox), (w, h)):
                    background_reason = "static_background"
        if background_reason:
            geom = None  # nothing new in the picture: not a document (the cached outline stays for reuse)

        if geom and geom.confidence >= self._present_threshold(geom):
            same_object = (
                self._candidate_bbox is None
                or self.config.detect_min_iou <= 0.0
                or _bbox_iou(self._candidate_bbox, geom.bbox) >= self.config.detect_min_iou
            )
            self._candidate_bbox = tuple(geom.bbox)  # type: ignore[assignment]
            if same_object:
                self._present_frames += 1
            else:
                self._present_frames = 1  # a different best candidate: start counting again
            if self._present_frames == 1:
                self._first_seen_at = time.monotonic()
            self._absent_frames = 0
            self._last_reject_reason = ""
            # The page together with the hand holding it is a wrong outline; the hand-free candidate usually
            # appears once the hand settles, so a winner that swallowed a hand has to persist longer first.
            required = self.config.detect_frames_required
            if getattr(geom, "swallows_hand", False):
                required *= self.config.hand_detect_patience
                self._last_reject_reason = "waiting_for_hand_free_outline" if self._present_frames < required else ""
            if not self._detected_latched and self._present_frames >= required:
                self._detected_latched = True
        else:
            self._present_frames = 0
            self._candidate_bbox = None
            self._first_seen_at = None
            if self._detected_latched:
                self._absent_frames += 1
                if self._absent_frames >= self.config.lose_frames_required:
                    self._detected_latched = False
            else:
                self._absent_frames += 1
            if background_reason:
                self._last_reject_reason = background_reason
            elif geom is None:
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
        if geom is None or geom.confidence < self._present_threshold(geom, self.config.scene_empty_factor):
            return True
        if self.config.background_check and self._bg_thumb is not None:
            # A confident outline is still "empty" when it is the learned empty background (furniture, a mat
            # that scores 0.5 on its own) or when the whole picture is back to what the empty desk looked like.
            thumb = self._bg_thumbnail(frame)
            h, w = frame.shape[:2]
            if self._is_background(thumb, tuple(geom.bbox), (w, h)) or self._looks_like_background(thumb):
                return True
        return False

    def _change_mask(self, frame: np.ndarray) -> np.ndarray | None:
        """Where the picture differs from the learned empty scene (small mask; None while not learned)."""
        if not self.config.background_check or self._bg_thumb is None:
            return None
        thumb = self._bg_thumbnail(frame)
        if thumb.shape != self._bg_thumb.shape:
            return None
        diff = thumb - self._bg_thumb
        offset = np.median(diff.reshape(-1, diff.shape[-1]), axis=0)  # global exposure shift
        changed = (np.abs(diff - offset).max(axis=2) > 14.0).astype(np.uint8) * 255
        return cv2.dilate(changed, np.ones((3, 3), np.uint8))

    def _looks_like_background(self, thumb: np.ndarray) -> bool:
        """The whole picture matches the learned empty scene (after removing a global exposure shift)."""
        base = self._bg_thumb
        if base is None or base.shape != thumb.shape:
            return False
        diff = thumb - base
        offset = np.median(diff.reshape(-1, diff.shape[-1]), axis=0)
        return float(np.abs(diff - offset).mean()) < 0.5 * self.config.background_min_diff

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
        # The pre-capture check runs is_scene_empty() on this very frame first: reuse that result
        # instead of segmenting the same frame a second time (~0.1 s on the Pi).
        if self._geometry_frame is frame:
            geom = self._last_geometry
        else:
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
