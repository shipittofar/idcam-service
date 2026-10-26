"""Post-capture image validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from idcam.config.schema import CameraConfig, QualityConfig
from idcam.exceptions import ImageValidationError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import (
    clip_roi,
    laplacian_variance,
    mean_brightness,
    preview_sharpness,
    roi_rect,
    scale_preview_roi_to_image,
    still_sharpness,
)

logger = get_logger("camera.quality")


@dataclass
class ValidationResult:
    width: int
    height: int
    file_bytes: int
    sharpness: float
    document_sharpness: float
    brightness: float
    sha256: str
    document_roi: tuple[int, int, int, int] | None = None
    # Still Tenengrad on the document divided by the preview sharpness measured at focus time.
    focus_ratio: float | None = None


class ImageValidator:
    """Validates captured JPEG files before upload."""

    def __init__(self, quality_config: QualityConfig, camera_config: CameraConfig | None = None) -> None:
        self.config = quality_config
        self.camera_config = camera_config

    def validate(
        self,
        path: Path,
        *,
        document_roi: tuple[int, int, int, int] | None = None,
        preview_size: tuple[int, int] | None = None,
        scaler_crop: tuple[int, int, int, int] | None = None,
        reference_sharpness: float | None = None,
    ) -> ValidationResult:
        if not path.exists():
            raise ImageValidationError(f"Image file not found: {path}")
        if not path.is_file():
            raise ImageValidationError(f"Not a regular file: {path}")

        file_bytes = path.stat().st_size
        if file_bytes < self.config.min_file_bytes:
            raise ImageValidationError(f"File too small: {file_bytes} bytes")
        if file_bytes > self.config.max_file_bytes:
            raise ImageValidationError(f"File too large: {file_bytes} bytes")

        sha256 = self._compute_sha256(path)

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            raise ImageValidationError(f"JPEG corrupt or undecodable: {exc}") from exc

        try:
            with Image.open(path) as img:
                width, height = img.size
                # 1/8-scale decode still detects truncated data; cv2.imread below does the full decode.
                img.draft("RGB", (max(1, width // 8), max(1, height // 8)))
                img.load()
        except Exception as exc:
            raise ImageValidationError(f"JPEG truncated or incomplete: {exc}") from exc

        # 64 MP stills: decode at 1/4 size (JPEG DCT scaling, ~2x faster and far less memory) and
        # measure sharpness there. At 9152x6944 the lens resolves less than the sensor, so
        # pixel-level Laplacian variance is tiny even for a well-focused page (observed 9-13);
        # at 1/4 scale it is comparable with the thresholds. Smaller images decode in full.
        reduce = 4 if max(width, height) >= 4000 else 1
        flag = cv2.IMREAD_REDUCED_COLOR_4 if reduce == 4 else cv2.IMREAD_COLOR
        frame = cv2.imread(str(path), flag)
        if frame is None:
            raise ImageValidationError("OpenCV could not decode image")

        cv_height, cv_width = frame.shape[:2]
        if abs(cv_width * reduce - width) > reduce or abs(cv_height * reduce - height) > reduce:
            raise ImageValidationError("Image dimension mismatch between PIL and OpenCV")

        if width < self.config.min_width or height < self.config.min_height:
            raise ImageValidationError(f"Dimensions too small: {width}x{height}")

        if self.config.require_expected_dimensions and self.camera_config is not None:
            tol = self.config.dimension_tolerance
            ew, eh = self.camera_config.capture_width, self.camera_config.capture_height
            if abs(width - ew) > tol or abs(height - eh) > tol:
                raise ImageValidationError(
                    f"Dimensions {width}x{height} do not match expected {ew}x{eh} (tol={tol})"
                )

        sharpness = laplacian_variance(frame)
        brightness = mean_brightness(frame)

        still_doc_roi: tuple[int, int, int, int] | None = None
        document_sharpness = sharpness
        if document_roi is not None and preview_size is not None:
            still_doc_roi = scale_preview_roi_to_image(
                document_roi,
                preview_size,
                (width, height),
                scaler_crop=scaler_crop,
            )
            roi_for_frame = still_doc_roi
            if reduce > 1:
                rx, ry, rw, rh = still_doc_roi
                roi_for_frame = (rx // reduce, ry // reduce, max(1, rw // reduce), max(1, rh // reduce))
            document_sharpness = still_sharpness(frame, roi=roi_for_frame)

        # Diagnostic only: same metric and pixel scale as the autofocus check (the preview covers
        # the sensor at 1/4 size, and so does this decode). A ratio near 1 with a failing absolute
        # sharpness means focus converged on a soft peak, not that the thresholds are wrong; it
        # must never turn a blurry image into an upload.
        focus_ratio: float | None = None
        still_tenengrad = 0.0
        if reference_sharpness is not None and reference_sharpness > 0:
            if still_doc_roi is not None:
                x, y, rw, rh = clip_roi(roi_for_frame, frame.shape)
            else:
                x, y, rw, rh = clip_roi(roi_rect(frame.shape, 0.7), frame.shape)
            patch = np.ascontiguousarray(frame[y : y + rh, x : x + rw, ::-1])
            still_tenengrad = preview_sharpness(patch, roi=(0, 0, patch.shape[1], patch.shape[0]))
            focus_ratio = still_tenengrad / reference_sharpness

        logger.info(
            "event=pipeline_stage stage=image_validated "
            "sharpness_global=%.1f sharpness_document=%.1f min_global=%.1f min_document=%.1f "
            "document_roi=%s still_tenengrad=%.1f focus_reference=%s focus_ratio=%s",
            sharpness,
            document_sharpness,
            self.config.min_sharpness,
            self.config.min_document_sharpness,
            still_doc_roi,
            still_tenengrad,
            f"{reference_sharpness:.1f}" if reference_sharpness else None,
            f"{focus_ratio:.2f}" if focus_ratio is not None else None,
        )

        if sharpness < self.config.min_sharpness:
            raise ImageValidationError(
                f"Image too blurry (global): sharpness={sharpness:.1f} min={self.config.min_sharpness:.1f}"
            )
        if document_roi is not None and document_sharpness < self.config.min_document_sharpness:
            raise ImageValidationError(
                f"Document region too blurry: sharpness={document_sharpness:.1f} "
                f"min={self.config.min_document_sharpness:.1f} roi={still_doc_roi}"
            )
        if brightness < self.config.min_brightness:
            raise ImageValidationError(f"Image too dark: brightness={brightness:.1f}")
        if brightness > self.config.max_brightness:
            raise ImageValidationError(f"Image too bright: brightness={brightness:.1f}")

        if float(np.std(frame[::8, ::8])) < 5.0:  # subsampled: whole-frame std was ~5 s at 64 MP
            raise ImageValidationError("Image appears uniform/corrupted")

        result = ValidationResult(
            width=width,
            height=height,
            file_bytes=file_bytes,
            sharpness=sharpness,
            document_sharpness=document_sharpness,
            brightness=brightness,
            sha256=sha256,
            document_roi=still_doc_roi,
            focus_ratio=focus_ratio,
        )
        logger.info(
            "Image validated: %dx%d bytes=%d sharpness=%.1f doc_sharpness=%.1f sha256=%s path=%s",
            width,
            height,
            file_bytes,
            sharpness,
            document_sharpness,
            sha256[:16],
            path,
        )
        return result

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
