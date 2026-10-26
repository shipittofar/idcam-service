"""Geometric document quadrilateral detection on preview frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from idcam.camera.diagnostic import save_rgb_jpeg
from idcam.vision.motion import normalize_gray, to_gray
from idcam.vision.sharpness import clip_roi


@dataclass(frozen=True)
class DocumentGeometry:
    """Detected document quadrilateral in preview pixel coordinates."""

    corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    width: int
    height: int
    rotation_deg: float
    aspect_ratio: float
    area_fraction: float
    confidence: float
    edge_strength: float
    rectangularity: float
    detection_method: str = "unknown"
    contrast_score: float = 0.0


@dataclass
class GeometryCandidateDiag:
    accepted: bool
    bbox: tuple[int, int, int, int]
    corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None
    area_fraction: float
    aspect_ratio: float
    rectangularity: float
    edge_strength: float
    final_score: float
    rejection_reason: str | None = None
    source_method: str = ""
    contrast_score: float = 0.0


def order_corners(pts: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    work = pts.reshape(4, 2).astype(np.float32)
    s = work.sum(axis=1)
    diff = np.diff(work, axis=1).ravel()
    tl = work[np.argmin(s)]
    br = work[np.argmax(s)]
    tr = work[np.argmin(diff)]
    bl = work[np.argmax(diff)]
    return (
        (int(round(tl[0])), int(round(tl[1]))),
        (int(round(tr[0])), int(round(tr[1]))),
        (int(round(br[0])), int(round(br[1]))),
        (int(round(bl[0])), int(round(bl[1]))),
    )


def corners_to_bbox(
    corners: tuple[tuple[int, int], ...],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return clip_roi((x0, y0, x1 - x0, y1 - y0), frame_shape)


def smooth_corners(
    prev: tuple[tuple[int, int], ...] | None,
    new: tuple[tuple[int, int], ...],
    alpha: float,
    frame_shape: tuple[int, ...],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    if prev is None or len(prev) != 4:
        return order_corners(np.array(new, dtype=np.float32))
    a = min(max(float(alpha), 0.05), 1.0)
    smoothed: list[tuple[int, int]] = []
    for i in range(4):
        px, py = prev[i]
        nx, ny = new[i]
        sx = int(round(px * (1 - a) + nx * a))
        sy = int(round(py * (1 - a) + ny * a))
        smoothed.append((sx, sy))
    return order_corners(np.array(smoothed, dtype=np.float32))


def corner_motion(
    prev: tuple[tuple[int, int], ...],
    curr: tuple[tuple[int, int], ...],
) -> float:
    if len(prev) != 4 or len(curr) != 4:
        return 0.0
    dist = 0.0
    for i in range(4):
        dist += math.hypot(curr[i][0] - prev[i][0], curr[i][1] - prev[i][1])
    return dist / 4.0


def _prepare_gray(frame: np.ndarray) -> np.ndarray:
    """Illumination-normalized grayscale with local contrast enhancement."""
    gray = to_gray(frame)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    norm = normalize_gray(gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(norm)


def _odd(value: float, minimum: int) -> int:
    n = max(minimum, int(round(value)))
    return n if n % 2 else n + 1


def _binary_variants(gray: np.ndarray, scale: float = 1.0) -> list[tuple[str, np.ndarray]]:
    """Multiple segmentation strategies for uneven lighting and weak edges.

    ``scale`` is the downscale factor applied to the frame (1.0 = untouched): structuring
    elements and adaptive-threshold windows were tuned at full preview size, so they shrink with it.
    """
    h, w = gray.shape
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(5 * scale, 3),) * 2)
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(9 * scale, 3),) * 2)
    k15 = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(15 * scale, 5),) * 2)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    variants: list[tuple[str, np.ndarray]] = []

    def _clean(mask: np.ndarray, kernel: np.ndarray, it: int = 2) -> np.ndarray:
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=it)
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, k5, iterations=1)

    for block in (21, 35, 51):
        block = _odd(block * scale, 7)
        for c in (7, 12):
            for inv, tag in ((False, "norm"), (True, "inv")):
                flag = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
                mask = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, block, c
                )
                variants.append((f"adaptive_{tag}_b{block}_c{c}", _clean(mask, k15)))

    sigma_bg = max(w, h) / 10.0
    # sigma ~ max(w, h)/10 is huge at full resolution; the background is smooth, so
    # blur a 1/8 downscale and upsample (same result within ~1 grey level, ~300x faster).
    ds = 8
    small = cv2.resize(gray, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma_bg / ds, sigmaY=sigma_bg / ds)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    diff = cv2.absdiff(gray, bg)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    for tag, tflag in (("bg_otsu", cv2.THRESH_BINARY), ("bg_otsu_inv", cv2.THRESH_BINARY_INV)):
        _, mask = cv2.threshold(diff, 0, 255, tflag + cv2.THRESH_OTSU)
        variants.append((tag, _clean(mask, k15)))

    for tag, tflag in (("otsu", cv2.THRESH_BINARY), ("otsu_inv", cv2.THRESH_BINARY_INV)):
        _, mask = cv2.threshold(blur, 0, 255, tflag + cv2.THRESH_OTSU)
        variants.append((tag, _clean(mask, k9)))

    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k9)
    _, gmask = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("morph_gradient", _clean(gmask, k9)))

    median = float(np.median(blur))
    for scale, tag in ((0.5, "wide"), (0.33, "mid")):
        lower = int(max(0, (1.0 - scale) * median))
        upper = int(min(255, (1.0 + scale) * median))
        if upper <= lower:
            upper = lower + 1
        edges = cv2.Canny(blur, lower, upper)
        variants.append((f"canny_{tag}", _clean(edges, k9)))

    return variants


def _contrast_score(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """How much the region differs from its immediate surroundings (paper vs desk)."""
    h, w = gray.shape
    x, y, rw, rh = clip_roi(bbox, (h, w))
    if rw < 4 or rh < 4:
        return 0.0
    pad = max(4, int(min(rw, rh) * 0.08))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + rw + pad)
    y1 = min(h, y + rh + pad)
    ring = gray[y0:y1, x0:x1].copy()
    inner = gray[y : y + rh, x : x + rw]
    ring[y - y0 : y - y0 + rh, x - x0 : x - x0 + rw] = 0
    inner_mean = float(np.mean(inner))
    ring_vals = ring[ring > 0]
    if ring_vals.size < 16:
        return 0.0
    outer_mean = float(np.mean(ring_vals))
    return abs(inner_mean - outer_mean) / 255.0


def _edge_strength_in_region(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, rw, rh = clip_roi(bbox, gray.shape)
    patch = gray[y : y + rh, x : x + rw]
    if patch.size == 0:
        return 0.0
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(cv2.magnitude(gx, gy)))


def _edge_alignment(corners: tuple[tuple[int, int], ...], grad_near: np.ndarray, grad_local_max: np.ndarray) -> float:
    """0..1: how exactly the quad outline sits on the strongest local edge (1 = on the edge).

    A candidate that includes a halo/shadow around the page has its outline a few pixels off the
    real paper edge, so it scores lower than the tight outline of the page itself.
    """
    h, w = grad_near.shape
    on_edge = 0.0
    peak = 0.0
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        n = max(8, int(math.hypot(x1 - x0, y1 - y0) / 2))
        ts = np.linspace(0.0, 1.0, n)
        xs = np.clip(np.rint(x0 + (x1 - x0) * ts).astype(int), 0, w - 1)
        ys = np.clip(np.rint(y0 + (y1 - y0) * ts).astype(int), 0, h - 1)
        on_edge += float(np.sum(grad_near[ys, xs]))
        peak += float(np.sum(grad_local_max[ys, xs]))
    return on_edge / peak if peak > 1e-6 else 0.0


def _contour_to_corners(contour: np.ndarray) -> tuple[tuple[int, int], ...]:
    peri = cv2.arcLength(contour, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx)
    rect = cv2.minAreaRect(contour)
    return order_corners(cv2.boxPoints(rect))


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _score_geometry(
    corners: tuple[tuple[int, int], ...],
    contour_area: float,
    gray: np.ndarray,
    frame_shape: tuple[int, ...],
    method: str,
    *,
    min_area_fraction: float,
    max_area_fraction: float,
    min_rectangularity: float,
    min_contrast: float,
    edge_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[DocumentGeometry | None, GeometryCandidateDiag]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    frame_area = float(w * h)
    bbox = corners_to_bbox(corners, frame_shape)
    bx, by, bw, bh = bbox
    area_fraction = contour_area / frame_area
    aspect = float(bw) / max(float(bh), 1.0)
    rectangularity = contour_area / max(float(bw * bh), 1.0)
    edge_strength = _edge_strength_in_region(gray, bbox)
    contrast = _contrast_score(gray, bbox)

    def _reject(reason: str) -> tuple[None, GeometryCandidateDiag]:
        return None, GeometryCandidateDiag(
            False,
            bbox,
            corners,  # type: ignore[arg-type]
            area_fraction,
            aspect,
            rectangularity,
            edge_strength,
            0.0,
            rejection_reason=reason,
            source_method=method,
            contrast_score=contrast,
        )

    if area_fraction < min_area_fraction:
        return _reject("area_too_small")
    if area_fraction > max_area_fraction:
        return _reject("area_too_large")
    if bw < w * 0.08 or bh < h * 0.08:
        return _reject("bbox_too_small")
    if aspect < 0.3 or aspect > 3.2:
        return _reject("aspect_ratio_out_of_range")
    if rectangularity < min_rectangularity:
        return _reject("low_rectangularity")
    if contrast < min_contrast and edge_strength < 6.0:
        return _reject("low_contrast_and_weak_edges")

    cx = bx + bw // 2
    cy = by + bh // 2
    center_dist = math.hypot(cx - w / 2.0, cy - h / 2.0) / max(math.hypot(w / 2.0, h / 2.0), 1.0)
    center_score = max(0.0, 1.0 - center_dist * 0.85)
    aspect_target = 1.414
    aspect_score = max(0.0, 1.0 - min(abs(math.log(max(aspect, 1e-3) / aspect_target)), 1.2))

    confidence = min(
        1.0,
        max(
            0.0,
            0.28 * min(1.0, area_fraction / 0.20)
            + 0.22 * min(1.0, rectangularity)
            + 0.18 * min(1.0, contrast / 0.12)
            + 0.12 * min(1.0, edge_strength / 60.0)
            + 0.12 * center_score
            + 0.08 * aspect_score,
        ),
    )

    if edge_maps is not None:
        confidence *= _ALIGN_FLOOR + (1.0 - _ALIGN_FLOOR) * min(1.0, _edge_alignment(corners, *edge_maps))

    # A region glued to two or more frame borders is the table/background (or a document that
    # is cut off), not a fully visible document: logs showed locks on the whole frame
    # (roi=227,0,1406,1079) with confidence 0.85. Demote it so a real paper candidate wins.
    edge = 3
    touching = (bx <= edge) + (by <= edge) + (bx + bw >= w - edge) + (by + bh >= h - edge)
    if touching >= 2:
        confidence *= 0.4

    tl, tr, br, bl = corners
    width_top = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    width_bottom = math.hypot(br[0] - bl[0], br[1] - bl[1])
    height_left = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
    height_right = math.hypot(br[0] - tr[0], br[1] - tr[1])
    rotation = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))

    geom = DocumentGeometry(
        corners=corners,  # type: ignore[arg-type]
        bbox=bbox,
        center=(cx, cy),
        width=int(round(max(width_top, width_bottom))),
        height=int(round(max(height_left, height_right))),
        rotation_deg=float(rotation),
        aspect_ratio=float(aspect),
        area_fraction=float(area_fraction),
        confidence=float(confidence),
        edge_strength=float(edge_strength),
        rectangularity=float(rectangularity),
        detection_method=method,
        contrast_score=float(contrast),
    )
    diag = GeometryCandidateDiag(
        True,
        bbox,
        corners,  # type: ignore[arg-type]
        area_fraction,
        aspect,
        rectangularity,
        edge_strength,
        confidence,
        rejection_reason=None,
        source_method=method,
        contrast_score=contrast,
    )
    return geom, diag


def _collect_from_mask(
    mask: np.ndarray,
    gray: np.ndarray,
    frame_shape: tuple[int, ...],
    method: str,
    *,
    min_area_fraction: float,
    max_area_fraction: float,
    min_rectangularity: float,
    min_contrast: float,
    edge_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list[tuple[DocumentGeometry, GeometryCandidateDiag]], list[GeometryCandidateDiag]]:
    h, w = frame_shape[:2]
    frame_area = float(w * h)
    min_area = frame_area * min_area_fraction
    results: list[tuple[DocumentGeometry, GeometryCandidateDiag]] = []
    reports: list[GeometryCandidateDiag] = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

    for contour in contours:
        area = float(cv2.contourArea(contour))
        bx, by, bw, bh = cv2.boundingRect(contour)
        bbox_raw = (bx, by, bw, bh)
        area_fraction = area / frame_area

        if area < min_area:
            reports.append(
                GeometryCandidateDiag(
                    False,
                    bbox_raw,
                    None,
                    area_fraction,
                    float(bw) / max(bh, 1),
                    area / max(float(bw * bh), 1.0),
                    0.0,
                    0.0,
                    rejection_reason="area_too_small",
                    source_method=method,
                )
            )
            continue

        corners = _contour_to_corners(contour)
        geom, diag = _score_geometry(
            corners,
            area,
            gray,
            frame_shape,
            method,
            min_area_fraction=min_area_fraction,
            max_area_fraction=max_area_fraction,
            min_rectangularity=min_rectangularity,
            min_contrast=min_contrast,
            edge_maps=edge_maps,
        )
        reports.append(diag)
        if geom is not None:
            results.append((geom, diag))

    return results, reports


_DETECT_MAX_DIM = 640
_ALIGN_WINDOW = 15  # px at working resolution: search radius for the true edge next to the outline
_ALIGN_FLOOR = 0.35  # confidence multiplier for an outline that is nowhere on an edge


def _rescale_geometry(geom: DocumentGeometry, factor: float, frame_shape: tuple[int, ...]) -> DocumentGeometry:
    """Map a geometry found on a downscaled frame back to full-frame pixels."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    corners = tuple(
        (min(max(int(round(x * factor)), 0), w - 1), min(max(int(round(y * factor)), 0), h - 1))
        for x, y in geom.corners
    )
    bbox = corners_to_bbox(corners, frame_shape)
    return DocumentGeometry(
        corners=corners,  # type: ignore[arg-type]
        bbox=bbox,
        center=(bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2),
        width=int(round(geom.width * factor)),
        height=int(round(geom.height * factor)),
        rotation_deg=geom.rotation_deg,
        aspect_ratio=geom.aspect_ratio,
        area_fraction=geom.area_fraction,
        confidence=geom.confidence,
        edge_strength=geom.edge_strength,
        rectangularity=geom.rectangularity,
        detection_method=geom.detection_method,
        contrast_score=geom.contrast_score,
    )


def _detect_all(
    frame: np.ndarray,
    *,
    min_area_fraction: float = 0.03,
    max_area_fraction: float = 0.90,
    min_rectangularity: float = 0.42,
    min_contrast: float = 0.018,
) -> tuple[DocumentGeometry | None, list[GeometryCandidateDiag], str | None]:
    if frame is None or frame.size == 0:
        return None, [], "empty_frame"
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if h < 32 or w < 32:
        return None, [], "frame_too_small"

    # Segmenting 1080p costs ~0.65 s per call (19 mask variants); running it on every preview
    # frame capped detection at ~1.4 fps on the Pi. Work at <= _DETECT_MAX_DIM and map back.
    long_side = max(h, w)
    ratio = min(1.0, _DETECT_MAX_DIM / long_side)
    work = frame
    if ratio < 1.0:
        work = cv2.resize(frame, (max(1, round(w * ratio)), max(1, round(h * ratio))), interpolation=cv2.INTER_AREA)
    scale = ratio  # kernels shrink only by the factor the frame was shrunk

    gray = _prepare_gray(work)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    edge_maps = (
        cv2.dilate(grad, np.ones((3, 3), np.uint8)),
        cv2.dilate(grad, np.ones((_ALIGN_WINDOW, _ALIGN_WINDOW), np.uint8)),
    )
    all_geoms: list[DocumentGeometry] = []
    all_reports: list[GeometryCandidateDiag] = []

    for method, mask in _binary_variants(gray, scale):
        geoms, reports = _collect_from_mask(
            mask,
            gray,
            work.shape,
            method,
            min_area_fraction=min_area_fraction,
            max_area_fraction=max_area_fraction,
            min_rectangularity=min_rectangularity,
            min_contrast=min_contrast,
            edge_maps=edge_maps,
        )
        all_reports.extend(reports)
        if geoms:
            best_local = max((g[0] for g in geoms), key=lambda g: g.confidence)
            all_geoms.append(best_local)

    if not all_geoms:
        return None, all_reports, "no_contour_passed_filters"

    # Deduplicate by bbox IoU — keep highest confidence
    all_geoms.sort(key=lambda g: g.confidence, reverse=True)
    unique: list[DocumentGeometry] = []
    for geom in all_geoms:
        if all(_bbox_iou(geom.bbox, u.bbox) < 0.55 for u in unique):
            unique.append(geom)
    best = unique[0]
    if ratio < 1.0:
        best = _rescale_geometry(best, 1.0 / ratio, frame.shape)
    return best, all_reports, best.detection_method


def detect_document_geometry(
    frame: np.ndarray,
    *,
    min_area_fraction: float = 0.03,
    max_area_fraction: float = 0.90,
    min_rectangularity: float = 0.42,
    min_edge_strength: float = 8.0,  # kept for API compat; contrast path used internally
    min_contrast: float = 0.018,
) -> DocumentGeometry | None:
    """Detect document quadrilateral using multi-strategy segmentation."""
    _ = min_edge_strength
    best, _, _ = _detect_all(
        frame,
        min_area_fraction=min_area_fraction,
        max_area_fraction=max_area_fraction,
        min_rectangularity=min_rectangularity,
        min_contrast=min_contrast,
    )
    return best


def scan_document_candidates(
    frame: np.ndarray,
    *,
    min_area_fraction: float = 0.03,
    max_area_fraction: float = 0.90,
    min_rectangularity: float = 0.42,
    min_edge_strength: float = 8.0,
    min_contrast: float = 0.018,
) -> tuple[DocumentGeometry | None, list[GeometryCandidateDiag]]:
    """Diagnostic scan with per-candidate rejection reasons."""
    _ = min_edge_strength
    best, reports, _ = _detect_all(
        frame,
        min_area_fraction=min_area_fraction,
        max_area_fraction=max_area_fraction,
        min_rectangularity=min_rectangularity,
        min_contrast=min_contrast,
    )
    # Sort reports: accepted first, then by area
    reports.sort(key=lambda r: (not r.accepted, -r.area_fraction))
    return best, reports[:40]


def save_segmentation_debug_mask(frame: np.ndarray, path: Path) -> str | None:
    """Save the mask from the winning detection method (diagnostic only)."""
    gray = _prepare_gray(frame)
    best, _, method = _detect_all(frame)
    if method is None:
        return None
    for name, mask in _binary_variants(gray):
        if name == method:
            save_rgb_jpeg(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), path)
            return name
    return None


def draw_geometry_debug_image(
    frame_rgb: np.ndarray,
    geometry: DocumentGeometry | None,
    path: Path,
    *,
    reason: str = "",
    near_miss: GeometryCandidateDiag | None = None,
) -> None:
    """Save preview with exactly ONE document polygon when detected."""
    canvas = frame_rgb.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    lines: list[str] = []
    if geometry is not None:
        pts = np.array(geometry.corners, dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(80, 180, 255), thickness=3)
        for i, (cx, cy) in enumerate(geometry.corners):
            cv2.circle(canvas, (int(cx), int(cy)), 7, (0, 220, 255), -1)
            cv2.putText(
                canvas,
                str(i),
                (int(cx) + 5, int(cy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        lines = [
            "document_detected=true",
            f"method={geometry.detection_method}",
            f"confidence={geometry.confidence:.3f}",
            f"area_pct={geometry.area_fraction * 100:.1f}%",
            f"aspect={geometry.aspect_ratio:.2f}",
            f"rotation={geometry.rotation_deg:.1f}",
            f"contrast={geometry.contrast_score:.3f}",
        ]
    else:
        lines = ["document_detected=false", f"reason={reason or 'no_candidate'}"]
        if near_miss is not None and near_miss.bbox[2] > 0:
            x, y, rw, rh = near_miss.bbox
            cv2.rectangle(canvas, (x, y), (x + rw, y + rh), (0, 180, 255), 2)
            lines.append(f"near_miss={near_miss.rejection_reason}")
            lines.append(f"near_area_pct={near_miss.area_fraction * 100:.1f}%")
            lines.append(f"near_method={near_miss.source_method}")

    y = 26
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24

    save_rgb_jpeg(canvas, path)
