"""Shrink a validated capture before upload: crop to the document, limit size, re-encode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from idcam.config.schema import UploadImageConfig


@dataclass(frozen=True)
class PreparedImage:
    path: Path
    width: int
    height: int
    file_bytes: int
    cropped: bool
    source_width: int
    source_height: int


def _crop_box(
    roi: tuple[int, int, int, int] | None,
    size: tuple[int, int],
    margin: float,
) -> tuple[int, int, int, int] | None:
    """Document ROI expanded by ``margin`` on every side, in (left, top, right, bottom)."""
    if roi is None:
        return None
    width, height = size
    x, y, w, h = (int(v) for v in roi)
    # Passport-size pages are ~2-4 % of the frame (the detector's minimum is 2 %); anything under
    # 1 % is not a document box, so keep the whole frame rather than cut it.
    if w <= 0 or h <= 0 or w * h < 0.01 * width * height:
        return None
    pad_x, pad_y = int(w * margin), int(h * margin)
    left, top = max(0, x - pad_x), max(0, y - pad_y)
    right, bottom = min(width, x + w + pad_x), min(height, y + h + pad_y)
    if right - left < 16 or bottom - top < 16:
        return None
    return left, top, right, bottom


def prepare_upload_image(
    src: Path,
    dst: Path,
    crop_roi: tuple[int, int, int, int] | None,
    cfg: UploadImageConfig,
) -> PreparedImage:
    """Write a smaller JPEG of ``src`` to ``dst``.

    ``crop_roi`` is the document bounding box in full-resolution pixels. The crop keeps a margin
    around it so a slightly-off detection never cuts the page; the result is then limited to
    ``cfg.max_dimension`` on its long side (0 = keep native size) and re-encoded at
    ``cfg.jpeg_quality``.
    """
    with Image.open(src) as img:
        src_w, src_h = img.size
        box = _crop_box(crop_roi, (src_w, src_h), cfg.crop_margin) if cfg.crop_to_document else None
        long_side = max(box[2] - box[0], box[3] - box[1]) if box else max(src_w, src_h)

        # JPEG decodes at 1/2, 1/4 or 1/8 size for free; only use it while the target stays covered.
        reduction = 1
        if cfg.max_dimension:
            for candidate in (8, 4, 2):
                if long_side / candidate >= cfg.max_dimension:
                    reduction = candidate
                    break
        if reduction > 1:
            img.draft("RGB", (max(1, src_w // reduction), max(1, src_h // reduction)))
        img.load()
        factor = img.size[0] / src_w

        image = img.convert("RGB")
        if box:
            image = image.crop(tuple(int(round(v * factor)) for v in box))
        if cfg.max_dimension and max(image.size) > cfg.max_dimension:
            scale = cfg.max_dimension / max(image.size)
            image = image.resize(
                (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))),
                Image.LANCZOS,
            )

        image.save(dst, format="JPEG", quality=cfg.jpeg_quality)  # no optimize: a second pass for ~2 % size
        out_w, out_h = image.size

    return PreparedImage(
        path=dst,
        width=out_w,
        height=out_h,
        file_bytes=dst.stat().st_size,
        cropped=box is not None,
        source_width=src_w,
        source_height=src_h,
    )
