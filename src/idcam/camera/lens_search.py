"""Find the sharpest lens position by stepping the lens and measuring the document area.

Used by the panel's "Find best focus" button (manual focus mode). Unlike the autofocus controller it
does no hardware AF and no hill-climbing with short settle times: it moves the lens, waits long enough
for the voice-coil to arrive, takes a frame exposed after that and measures it. Coarse steps over the
whole range first, then fine steps around the best one, because the depth of field of this camera is
only a few tenths of a dioptre. ``scripts/lens_sweep.py`` does the same on the bench.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from idcam.exceptions import AutofocusError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import preview_sharpness

logger = get_logger("camera.lens_search")


@dataclass(frozen=True)
class LensSearchResult:
    lens: float
    sharpness: float
    peak_ratio: float  # best sharpness divided by the median of all samples
    samples: int
    meta_followed: bool  # the frame metadata reported the commanded lens position


def find_sharpest_lens(
    camera: Any,
    roi: tuple[int, int, int, int] | None,
    lo: float,
    hi: float,
    *,
    settle_sec: float = 0.7,
    coarse_step: float = 0.5,
    fine_step: float = 0.1,
    min_peak_ratio: float = 1.3,
    start: float = 1.0,
    sharpness_fn: Callable[..., float] = preview_sharpness,
    max_seconds: float = 120.0,
) -> LensSearchResult:
    """Return the lens position with the sharpest ``roi``; raises AutofocusError without a clear peak.

    ``camera.probe_lens(position, settle_sec)`` must move the lens, wait, and return ``(frame, metadata)``
    for a frame exposed after the wait.
    """
    t_begin = time.monotonic()
    points: dict[float, float] = {}
    followed = 0

    def measure(pos: float) -> float:
        nonlocal followed
        pos = round(min(max(pos, lo), hi), 3)
        if pos in points:
            return points[pos]
        if time.monotonic() - t_begin > max_seconds:
            raise AutofocusError("lens search took too long")
        frame, meta = camera.probe_lens(pos, settle_sec)
        value = float(sharpness_fn(frame, roi=roi))
        points[pos] = value
        reported = meta.get("LensPosition") if isinstance(meta, dict) else None
        if isinstance(reported, (int, float)) and abs(float(reported) - pos) <= 0.1:
            followed += 1
        logger.info(
            "event=lens_search_sample pos=%.3f sharpness=%.1f lens_meta=%s exposure_us=%s gain=%s",
            pos,
            value,
            f"{float(reported):.3f}" if isinstance(reported, (int, float)) else None,
            meta.get("ExposureTime") if isinstance(meta, dict) else None,
            meta.get("AnalogueGain") if isinstance(meta, dict) else None,
        )
        return value

    # 1) the whole range in coarse steps
    pos = max(lo, start)
    while pos <= hi + 1e-9:
        measure(pos)
        pos += coarse_step
    best = max(points, key=points.__getitem__)

    # 2) fine steps around the best coarse point
    pos = best - coarse_step
    while pos <= best + coarse_step + 1e-9:
        measure(pos)
        pos += fine_step
    best = max(points, key=points.__getitem__)

    median = statistics.median(points.values())
    peak_ratio = points[best] / median if median > 1e-9 else float("inf")
    meta_followed = followed >= max(1, len(points) // 2)
    if not meta_followed:
        logger.warning(
            "event=lens_search_meta_not_following samples=%d followed=%d "
            "(the frame metadata does not show the commanded lens position)",
            len(points),
            followed,
        )
    logger.info(
        "event=lens_search_done best=%.3f sharpness=%.1f peak_ratio=%.1f samples=%d seconds=%d",
        best,
        points[best],
        peak_ratio,
        len(points),
        int(time.monotonic() - t_begin),
    )
    if peak_ratio < min_peak_ratio:
        raise AutofocusError(
            f"no clear sharpness peak (best is only x{peak_ratio:.1f} over the median): "
            "check the light, the distance and that the document is in view"
        )
    return LensSearchResult(best, points[best], peak_ratio, len(points), meta_followed)
