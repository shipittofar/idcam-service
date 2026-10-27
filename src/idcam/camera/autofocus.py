"""Camera autofocus — one hardware pass, preview sharpness verified."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

from idcam.config.schema import CameraConfig, QualityConfig, QualityConfig
from idcam.exceptions import AutofocusError, FocusMotionError
from idcam.logging.setup import get_logger
from idcam.vision.sharpness import accept_preview_focus, clip_roi, preview_sharpness, roi_rect

logger = get_logger("camera.autofocus")


class FocusCapableCamera(Protocol):
    def get_controls(self) -> dict[str, Any]: ...
    def set_controls(self, controls: dict[str, Any]) -> None: ...
    def autofocus_cycle(self) -> Any: ...
    def capture_preview_array(self) -> Any: ...
    def capture_metadata(self) -> dict[str, Any]: ...


@dataclass
class AutofocusCapabilities:
    af_mode: bool = False
    af_trigger: bool = False
    af_state: bool = False
    lens_position: bool = False
    autofocus_cycle: bool = False
    af_windows: bool = False
    af_range: bool = False
    af_metering: bool = False

    @property
    def hardware_af_available(self) -> bool:
        return self.af_mode or self.autofocus_cycle

    @property
    def degraded(self) -> bool:
        return not self.hardware_af_available

    def to_dict(self) -> dict[str, bool]:
        return {
            "AfMode": self.af_mode,
            "AfTrigger": self.af_trigger,
            "AfState": self.af_state,
            "LensPosition": self.lens_position,
            "hardware_af_available": self.hardware_af_available,
            "degraded": self.degraded,
        }


@dataclass
class AutofocusResult:
    success: bool
    sharpness: float
    attempts: int
    af_state: int | None
    method: str
    min_required_sharpness: float = 0.0
    before_sharpness: float = 0.0
    improved: bool = False
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    lens_before: float | None = None
    lens_after: float | None = None
    improvement_ratio: float = 0.0
    sensor_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    preview_size: tuple[int, int] = (0, 0)


@dataclass
class AutofocusDebug:
    running: bool = False
    method: str = ""
    attempt: int = 0
    focus_before: float = 0.0
    focus_after: float = 0.0
    focus_roi: str = ""
    preview_roi: str = ""
    sensor_roi: str = ""
    preview_roi_wh: str = ""
    lens_before: float | None = None
    lens_after: float | None = None
    improvement_ratio: float = 0.0
    scaler_crop: str = ""
    preview_size: str = ""


def _fmt_roi(roi: tuple[int, int, int, int] | None) -> str:
    if not roi:
        return ""
    return "{},{},{},{}".format(*(int(v) for v in roi))


class AutofocusController:
    """Hardware AF on document ROI with Tenengrad before/after verification."""

    def __init__(self, config: CameraConfig, quality: QualityConfig | None = None) -> None:
        self.config = config
        self.quality = quality or QualityConfig()
        self.capabilities = AutofocusCapabilities()
        self.debug = AutofocusDebug()
        self._last_preview_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
        # (lens position, monotonic time) of the last capture that passed validation.
        self._memory: tuple[float, float] | None = None
        self.last_refine_stats: dict[str, Any] = {}

    # -- lens memory ---------------------------------------------------------------------------

    @property
    def remembered_lens(self) -> float | None:
        memory = self._memory
        if memory is None or not self.config.af_lens_memory:
            return None
        if time.monotonic() - memory[1] > float(self.config.af_memory_max_age_sec):
            return None
        return memory[0]

    def remember_lens(self, lens: float | None) -> None:
        """Called after a capture taken at ``lens`` passed validation (a known-good position)."""
        if lens is not None and self.config.af_lens_memory:
            self._memory = (float(lens), time.monotonic())

    def forget_lens(self, reason: str = "") -> None:
        if self._memory is not None:
            logger.info("event=autofocus_memory_cleared lens=%.2f reason=%s", self._memory[0], reason)
        self._memory = None

    @property
    def hardware_af_available(self) -> bool:
        return self.capabilities.hardware_af_available

    def probe_capabilities(self, camera: FocusCapableCamera) -> AutofocusCapabilities:
        controls = camera.get_controls()
        self.capabilities = AutofocusCapabilities(
            af_mode="AfMode" in controls,
            af_trigger="AfTrigger" in controls,
            af_state="AfState" in controls,
            lens_position="LensPosition" in controls,
            autofocus_cycle=hasattr(camera, "autofocus_cycle"),
            af_windows="AfWindows" in controls,
            af_range="AfRange" in controls,
            af_metering="AfMetering" in controls,
        )
        logger.info("Autofocus capabilities: %s", self.capabilities.to_dict())
        return self.capabilities

    def _resolve_roi(
        self,
        frame: Any,
        roi_fraction: float,
        preview_roi: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int]:
        if preview_roi is not None:
            return clip_roi(preview_roi, frame.shape)
        return clip_roi(roi_rect(frame.shape, roi_fraction), frame.shape)

    def _refine_available(self, camera: FocusCapableCamera) -> bool:
        return bool(self.config.af_refine) and callable(getattr(camera, "set_lens_position", None))

    async def run_autofocus(
        self,
        camera: FocusCapableCamera,
        *,
        roi_fraction: float = 0.7,
        preview_roi: tuple[int, int, int, int] | None = None,
        sharpness_threshold: float | None = None,
    ) -> AutofocusResult:
        t_start = time.monotonic()
        floor = float(
            sharpness_threshold if sharpness_threshold is not None else self.quality.preview_min_sharpness
        )
        min_ratio = float(self.quality.preview_focus_min_ratio)
        min_delta = float(self.quality.preview_focus_min_delta)

        preview = await asyncio.to_thread(camera.capture_preview_array)
        if preview is None or getattr(preview, "size", 1) == 0:
            raise AutofocusError("Invalid preview frame")

        h, w = int(preview.shape[0]), int(preview.shape[1])
        preview_size = (w, h)
        roi = self._resolve_roi(preview, roi_fraction, preview_roi)
        mapping = await self._apply_document_roi(camera, roi, preview_size)
        sensor = tuple(mapping.get("sensor_roi", roi))

        before = preview_sharpness(preview, roi=roi)
        meta_before = self._read_metadata(camera)
        lens_before = meta_before.get("LensPosition")
        self._last_preview_roi = roi
        self._fill_debug(roi, sensor, preview_size, mapping, before, lens_before)
        t_prep = time.monotonic()

        logger.info(
            "event=pipeline_stage stage=focusing focus_method=measure "
            "sharpness_before=%.1f preview_floor=%.1f roi=%s sensor_roi=%s lens=%s",
            before,
            floor,
            self.debug.preview_roi,
            self.debug.sensor_roi,
            lens_before,
        )

        refine_ok = self._refine_available(camera)

        # Fast path: same setup as the last validated capture, so start the lens search there and
        # skip the hardware AF cycle. A bounded local search must find a clear peak; otherwise the
        # memory is dropped and the full procedure below runs (hardware AF + refinement + sweep).
        remembered = self.remembered_lens if refine_ok else None
        if remembered is not None:
            try:
                refined = await self._refine_focus(camera, roi, remembered, before, local_only=True)
            except FocusMotionError:
                raise
            except AutofocusError as exc:
                logger.info("event=autofocus_memory_miss lens=%.2f reason=%s", remembered, exc)
                self.forget_lens("local_search_failed")
                refined = None
            if refined is not None:
                lens_after, after = refined
                ratio = after / before if before > 1e-6 else 1.0
                self.debug.method = "memory+refine"
                self.debug.lens_after = float(lens_after)
                self.debug.focus_after = after
                self.debug.improvement_ratio = round(ratio, 3)
                self._log_focus_timing("memory", t_start, t_prep, None, time.monotonic())
                logger.info("event=pipeline_stage stage=focus_verified method=memory+refine sharpness=%.1f", after)
                return self._result(
                    True, after, 0, "memory+refine", floor, before, after, ratio, roi, sensor,
                    preview_size, lens_before, lens_after,
                )

        # No "already sharp" shortcut: comparing a frame with itself always passes, so AF
        # never ran and the lens stayed wherever Continuous AF last drifted (it moved
        # between this check and the still). Always run a one-shot AF on the document
        # ROI, then lock the lens (pause_autofocus below) before the still capture.
        after = before
        lens_after = lens_before
        method = "none"
        ratio = 0.0
        t_hw_start = time.monotonic()

        # At least two passes: the first can land worse than the drifting Continuous AF. When the
        # lens search follows anyway, one hardware pass is enough (its result is only a start).
        for attempt in range(1, max(2, self.config.af_max_retries) + 1):
            self.debug.running = True
            self.debug.attempt = attempt
            method, _ = await self._run_hardware_af(camera, attempt)
            self.debug.method = method

            fresh = getattr(camera, "capture_fresh_preview_array", None)
            after_frame = await asyncio.to_thread(fresh if callable(fresh) else camera.capture_preview_array)
            after = preview_sharpness(after_frame, roi=roi)
            meta_after = self._read_metadata(camera)
            lens_after = meta_after.get("LensPosition")
            ratio = after / before if before > 1e-6 else (2.0 if after > 0 else 0.0)
            improved = after > before
            accepted = accept_preview_focus(
                before=before,
                after=after,
                absolute_floor=floor,
                min_ratio=min_ratio,
                min_delta=min_delta,
            )

            self.debug.focus_after = after
            self.debug.lens_after = float(lens_after) if lens_after is not None else None
            self.debug.improvement_ratio = round(ratio, 3)
            self.debug.running = False
            t_hw_end = time.monotonic()

            logger.info(
                "event=autofocus_result focus_method=%s success=%s sharpness_before=%.1f "
                "sharpness_after=%.1f sharpness_ratio=%.3f improved=%s attempt=%d "
                "lens_before=%s lens_after=%s roi=%s",
                method,
                accepted,
                before,
                after,
                ratio,
                improved,
                attempt,
                lens_before,
                lens_after,
                self.debug.preview_roi,
            )

            if accepted or (refine_ok and self.config.af_adaptive):
                refined = await self._refine_focus(camera, roi, lens_after, after)
                if refined is None and not accepted:
                    before = after
                    lens_before = lens_after
                    continue
                if refined is not None:
                    lens_after, after = refined
                    self.debug.lens_after = float(lens_after)
                    self.debug.focus_after = after
                    ratio = after / before if before > 1e-6 else ratio
                    method = f"{method}+refine"
                else:
                    pause = getattr(camera, "pause_autofocus", None)
                    if callable(pause):
                        await asyncio.to_thread(pause)
                self._log_focus_timing("hardware", t_start, t_prep, (t_hw_start, t_hw_end), time.monotonic())
                logger.info("event=pipeline_stage stage=focus_verified method=%s sharpness=%.1f", method, after)
                return self._result(
                    True, after, attempt, method, floor, before, after, ratio, roi, sensor, preview_size, lens_before, lens_after
                )
            before = after
            lens_before = lens_after

        raise AutofocusError(
            f"Autofocus did not improve: before={before:.1f} after={after:.1f} "
            f"ratio={ratio:.2f} method={method} preview_roi={_fmt_roi(roi)}"
        )

    def _log_focus_timing(
        self,
        path: str,
        t_start: float,
        t_prep: float,
        hw: tuple[float, float] | None,
        t_end: float,
    ) -> None:
        stats = self.last_refine_stats
        logger.info(
            "event=focus_timing path=%s prep_ms=%d hw_af_ms=%s refine_ms=%d total_ms=%d samples=%s "
            "settle=%s confirm=%s unstable_samples=%s",
            path,
            int((t_prep - t_start) * 1000),
            int((hw[1] - hw[0]) * 1000) if hw else "skipped",
            int(stats.get("refine_ms", 0)),
            int((t_end - t_start) * 1000),
            stats.get("samples"),
            stats.get("settle"),
            stats.get("confirm"),
            stats.get("unstable"),
        )

    @staticmethod
    def _bracketed(pts: dict[float, float], best: float, margin: float) -> bool:
        """The best position has a measured neighbour on each side, both clearly lower."""
        below = [p for p in pts if p < best]
        above = [p for p in pts if p > best]
        if not below or not above:
            return False
        lower, upper = max(below), min(above)
        return pts[lower] <= pts[best] * (1.0 - margin) and pts[upper] <= pts[best] * (1.0 - margin)

    @classmethod
    def _clean_peak(cls, pts: dict[float, float], best: float, unstable: int) -> bool:
        """A single, bracketed peak without motion hints: safe to skip re-confirming it."""
        if unstable or len(pts) < 3 or not cls._bracketed(pts, best, 0.04):
            return False
        ordered = [pts[p] for p in sorted(pts)]
        peak = ordered.index(pts[best])
        rising = all(ordered[i] <= ordered[i + 1] * 1.03 for i in range(peak))
        falling = all(ordered[i + 1] <= ordered[i] * 1.03 for i in range(peak, len(ordered) - 1))
        return rising and falling

    async def _refine_focus(
        self,
        camera: FocusCapableCamera,
        roi: tuple[int, int, int, int],
        lens_start: Any,
        start_sharpness: float,
        *,
        local_only: bool = False,
    ) -> tuple[float, float] | None:
        """Find the lens position with the sharpest document ROI and park the lens there.

        Hill-climb from ``lens_start`` (coarse, then fine steps). If the sharpness curve has no
        clear peak the AF stopped on a plateau, so scan a wide lens range and climb again from
        the best point. ``local_only`` (used from the remembered position) never sweeps: it
        raises AutofocusError unless the bounded search finds a clearly bracketed peak.

        Returns (lens_position, sharpness), or None when refinement is unavailable or failed
        (the caller then locks the lens as before).
        """
        setter = getattr(camera, "set_lens_position", None)
        if not (self.config.af_refine and callable(setter)) or lens_start is None:
            return None
        cfg = self.config
        range_fn = getattr(camera, "lens_position_range", None)
        lo, hi = range_fn() if callable(range_fn) else (0.0, 32.0)
        step = float(cfg.af_local_step if local_only else cfg.af_refine_step)
        settle = float(cfg.af_refine_settle_sec)
        min_settle = min(float(cfg.af_refine_min_settle_sec), settle)
        tolerance = float(cfg.af_settle_tolerance)
        discard = int(cfg.af_refine_discard_frames)
        adaptive = bool(cfg.af_adaptive)
        max_points = int(cfg.af_refine_max_points)
        if local_only:
            max_points = min(max_points, 5)
        start = min(max(float(lens_start), lo), hi)
        pts: dict[float, float] = {}
        order: list[float] = []
        fresh_frame = getattr(camera, "capture_fresh_preview_array", None)
        stats = {"unstable": 0, "meta_settled": 0, "meta_timeout": 0, "meta_missing": 0}
        t_refine = time.monotonic()

        def clamp(pos: float) -> float:
            return round(min(max(pos, lo), hi), 3)

        def sample_frames(pos: float, full_settle: bool) -> tuple[float, float | None, dict[str, Any]]:
            """Runs in a worker thread: wait for the lens, then measure fresh frames."""
            info: dict[str, Any] = {"lens_ok": None, "lens_meta": None}
            t_cmd = time.monotonic()
            if not callable(fresh_frame):
                time.sleep(settle)
                for _ in range(discard):
                    camera.capture_preview_array()
                frame = camera.capture_preview_array()
                info["waited_ms"] = int((time.monotonic() - t_cmd) * 1000)
                return float(preview_sharpness(frame, roi=roi)), None, info

            use_meta = bool(cfg.af_settle_by_metadata) and not full_settle
            time.sleep(min_settle if use_meta else settle)
            frame = fresh_frame()
            lens_meta = getattr(camera, "last_fresh_lens", None)
            if use_meta:
                if lens_meta is None:
                    # No lens metadata from this camera: wait out the full settle time as before.
                    info["lens_ok"] = None
                    remaining = settle - (time.monotonic() - t_cmd)
                    if remaining > 0:
                        time.sleep(remaining)
                        frame = fresh_frame()
                        lens_meta = getattr(camera, "last_fresh_lens", None)
                elif abs(lens_meta - pos) <= tolerance:
                    info["lens_ok"] = True
                else:
                    info["lens_ok"] = False
                    while time.monotonic() - t_cmd < settle:
                        frame = fresh_frame()
                        lens_meta = getattr(camera, "last_fresh_lens", None)
                        if lens_meta is not None and abs(lens_meta - pos) <= tolerance:
                            info["lens_ok"] = True
                            break
            info["lens_meta"] = lens_meta
            info["waited_ms"] = int((time.monotonic() - t_cmd) * 1000)
            frames = [frame, fresh_frame()]  # best of two: see the spread check in measure()
            values = [float(preview_sharpness(f, roi=roi)) for f in frames]
            exposure = getattr(camera, "last_fresh_meta", None) or {}
            info.update(exposure_us=exposure.get("exposure_us"), gain=exposure.get("analogue_gain"))
            top = max(values)
            spread = (min(values) / top) if top > 0 else None
            return top, spread, info

        async def measure(pos: float, *, full_settle: bool = False) -> float:
            pos = clamp(pos)
            await asyncio.to_thread(setter, pos)
            sharp, spread, info = await asyncio.to_thread(sample_frames, pos, full_settle)
            pts[pos] = sharp
            order.append(pos)
            if info.get("lens_ok") is True:
                stats["meta_settled"] += 1
            elif info.get("lens_ok") is False:
                stats["meta_timeout"] += 1
            elif info.get("lens_meta") is None and callable(fresh_frame):
                stats["meta_missing"] += 1
            # Two fresh frames at the same lens position that disagree strongly mean motion (or
            # flicker); not when the lens was still travelling (lens_ok False).
            if spread is not None and spread < float(cfg.af_motion_spread_min) and info.get("lens_ok") is not False:
                stats["unstable"] += 1
            logger.info(
                "event=focus_sample pos=%.3f sharpness=%.0f spread=%s lens_meta=%s lens_ok=%s waited_ms=%s "
                "exposure_us=%s gain=%s",
                pos,
                sharp,
                f"{spread:.2f}" if spread is not None else None,
                f"{info['lens_meta']:.3f}" if info.get("lens_meta") is not None else None,
                info.get("lens_ok"),
                info.get("waited_ms"),
                info.get("exposure_us"),
                f"{info['gain']:.2f}" if isinstance(info.get("gain"), (int, float)) else info.get("gain"),
            )
            if stats["unstable"] >= int(cfg.af_motion_max_unstable):
                raise FocusMotionError(
                    f"Page or camera moving during focus: {stats['unstable']} samples with inconsistent "
                    f"sharpness (spread < {cfg.af_motion_spread_min:.2f})"
                )
            return sharp

        async def climb(origin: float, *, fine_probes: bool = True) -> None:
            """Coarse hill-climb from ``origin`` in `step` increments, then fine probes."""
            moved_in_first_fine = False
            origin = clamp(origin)
            if origin not in pts:
                await measure(origin)
            budget = len(pts) + max_points
            for direction in (1.0, -1.0):
                pos = origin
                improved = False
                while len(pts) < budget:
                    nxt = clamp(pos + direction * step)
                    if nxt in pts:
                        break
                    await measure(nxt)
                    if pts[nxt] <= pts[pos] * 1.02:
                        break
                    improved = True
                    pos = nxt
                if improved:
                    break
            # The peak at close range is narrower than the coarse step (0.12 dioptre changed the
            # metric 2.4x in logs), so probe both sides at smaller steps. (Not from the remembered
            # lens: its step is already fine, and that position was itself a fine-tuned best.)
            best = max(pts, key=lambda p: pts[p])
            if not fine_probes:
                return
            for level, fine in enumerate((step * 0.4, step * 0.16)):
                if level == 1 and adaptive and not moved_in_first_fine and self._bracketed(pts, best, 0.04):
                    break  # the peak is already bracketed by the first fine probes: stop here
                moved = False
                for cand in (clamp(best + fine), clamp(best - fine)):
                    if cand in pts:
                        continue
                    await measure(cand)
                    if pts[cand] > pts[best] * 1.01:
                        best = cand
                        moved = True
                if level == 0:
                    moved_in_first_fine = moved

        def prominent() -> bool:
            """A real focus peak is clearly above the rest of the sampled curve."""
            values = sorted(pts.values())
            return len(values) < 3 or values[-1] >= 1.3 * values[0]

        try:
            await climb(start, fine_probes=not local_only)
            if local_only:
                best_local = max(pts, key=lambda p: pts[p])
                if not self._bracketed(pts, best_local, 0.03):
                    raise AutofocusError(
                        f"No bracketed focus peak around the remembered lens {start:.2f} "
                        f"(best {best_local:.2f}, {len(pts)} samples)"
                    )
                floor = float(cfg.af_local_min_ratio) * float(start_sharpness)
                if pts[best_local] < floor:
                    # The live (continuous-AF) preview was sharper than anything near the
                    # remembered lens: the document is at another height now.
                    raise AutofocusError(
                        f"Remembered lens {best_local:.2f} reads {pts[best_local]:.0f}, below "
                        f"{cfg.af_local_min_ratio:.0%} of the live preview ({start_sharpness:.0f})"
                    )
            elif cfg.af_wide_sweep and not prominent():
                wide = float(cfg.af_wide_step)
                pos = max(lo, float(cfg.af_wide_min))
                logger.info(
                    "event=autofocus_wide_sweep reason=no_clear_peak range=%.1f..%.1f step=%.1f",
                    pos,
                    min(hi, float(cfg.af_wide_max)),
                    wide,
                )
                while pos <= min(hi, float(cfg.af_wide_max)) + 1e-6:
                    if clamp(pos) not in pts:
                        await measure(pos)
                    pos += wide
                await climb(max(pts, key=lambda p: pts[p]))
                if not prominent():
                    values = sorted(pts.values())
                    raise AutofocusError(
                        f"No focus peak found over lens {cfg.af_wide_min:.0f}-"
                        f"{cfg.af_wide_max:.0f}: sharpness {values[0]:.0f}..{values[-1]:.0f} "
                        f"(document blank, too far/near, or moving)"
                    )

            best = max(pts, key=lambda p: pts[p])
            clean = adaptive and self._clean_peak(pts, best, stats["unstable"])
            if clean and order[-1] == best:
                confirm = "skipped_clean_last"  # the freshest reading is at the winner already
            elif clean:
                # Clean, bracketed peak: put the lens on it and wait for it to settle (metadata
                # settle: ~0.2 s) but skip the "reading collapsed" check. The lens must be at rest
                # before the outline is re-measured for the pre-capture motion check.
                await measure(best)
                confirm = "settled_clean"
            else:
                # Park on the winner and confirm with a fresh reading at the full settle time.
                claimed = pts[best]
                await measure(best, full_settle=True)
                confirm = "measured"
                if pts[best] < 0.6 * claimed:
                    # The same lens position now reads far softer than when it won: the page moved
                    # (or the earlier reading was stale), so this focus cannot be trusted.
                    raise AutofocusError(
                        f"Focus unstable at lens {best:.2f}: sharpness {claimed:.0f} then {pts[best]:.0f} "
                        f"(document moved?)"
                    )
            settle_mode = "metadata" if cfg.af_settle_by_metadata and callable(fresh_frame) else "fixed"
            self.last_refine_stats = {
                "refine_ms": (time.monotonic() - t_refine) * 1000,
                "samples": len(order),
                "settle": settle_mode,
                "confirm": confirm,
                "unstable": stats["unstable"],
            }
            logger.info(
                "event=autofocus_refined start=%.2f final=%.2f sharpness_start=%.1f sharpness_final=%.1f "
                "samples=%d path=%s confirm=%s meta_settled=%d meta_timeout=%d meta_missing=%d curve=%s",
                start,
                best,
                start_sharpness,
                pts[best],
                len(order),
                "local" if local_only else "full",
                confirm,
                stats["meta_settled"],
                stats["meta_timeout"],
                stats["meta_missing"],
                ",".join(f"{p:.2f}:{v:.0f}" for p, v in sorted(pts.items())),
            )
            return float(best), float(pts[best])
        except AutofocusError:
            raise
        except Exception:
            logger.warning("event=autofocus_refine_failed", exc_info=True)
            return None

    async def verify_focus(
        self,
        camera: FocusCapableCamera,
        *,
        roi_fraction: float = 0.7,
        preview_roi: tuple[int, int, int, int] | None = None,
        sharpness_threshold: float | None = None,
    ) -> AutofocusResult:
        floor = float(
            sharpness_threshold if sharpness_threshold is not None else self.quality.preview_min_sharpness
        )
        min_ratio = float(self.quality.preview_focus_min_ratio)
        min_delta = float(self.quality.preview_focus_min_delta)

        preview = await asyncio.to_thread(camera.capture_preview_array)
        if preview is None or getattr(preview, "size", 1) == 0:
            raise AutofocusError("Invalid preview frame")

        h, w = int(preview.shape[0]), int(preview.shape[1])
        preview_size = (w, h)
        roi = self._resolve_roi(
            preview,
            roi_fraction,
            preview_roi or (self._last_preview_roi if self._last_preview_roi[2] > 0 else None),
        )
        sharp = preview_sharpness(preview, roi=roi)
        before = self.debug.focus_before if self.debug.focus_before > 0 else sharp
        meta = self._read_metadata(camera)
        af_state = meta.get("AfState")
        lens = meta.get("LensPosition")

        self.debug.focus_after = sharp
        self.debug.lens_after = float(lens) if lens is not None else None
        if not accept_preview_focus(
            before=before,
            after=sharp,
            absolute_floor=floor,
            min_ratio=min_ratio,
            min_delta=min_delta,
        ):
            raise AutofocusError(
                f"Focus verification failed: sharpness={sharp:.1f} before={before:.1f} floor={floor:.1f}"
            )

        sensor = tuple(self._last_preview_roi if self._last_preview_roi[2] > 0 else roi)
        return AutofocusResult(
            success=True,
            sharpness=sharp,
            attempts=0,
            af_state=int(af_state) if af_state is not None else None,
            method="verify",
            min_required_sharpness=floor,
            before_sharpness=before,
            improved=sharp >= before,
            roi=roi,
            lens_before=self.debug.lens_before,
            lens_after=float(lens) if lens is not None else None,
            improvement_ratio=sharp / before if before > 1e-6 else 1.0,
            sensor_roi=sensor,
            preview_size=preview_size,
        )

    def _result(
        self,
        success: bool,
        sharpness: float,
        attempts: int,
        method: str,
        floor: float,
        before: float,
        after: float,
        ratio: float,
        roi: tuple[int, int, int, int],
        sensor: tuple[int, int, int, int],
        preview_size: tuple[int, int],
        lens_before: Any,
        lens_after: Any,
    ) -> AutofocusResult:
        lb = float(lens_before) if lens_before is not None else None
        la = float(lens_after) if lens_after is not None else None
        return AutofocusResult(
            success=success,
            sharpness=sharpness,
            attempts=attempts,
            af_state=None,
            method=method,
            min_required_sharpness=floor,
            before_sharpness=before,
            improved=after > before,
            roi=roi,
            lens_before=lb,
            lens_after=la,
            improvement_ratio=ratio,
            sensor_roi=sensor,
            preview_size=preview_size,
        )

    def _fill_debug(
        self,
        roi: tuple[int, int, int, int],
        sensor: tuple[int, int, int, int],
        preview_size: tuple[int, int],
        mapping: dict,
        before: float,
        lens_before: Any,
    ) -> None:
        self.debug.focus_before = before
        self.debug.lens_before = float(lens_before) if lens_before is not None else None
        self.debug.focus_roi = _fmt_roi(roi)
        self.debug.preview_roi = _fmt_roi(roi)
        self.debug.sensor_roi = _fmt_roi(sensor)
        self.debug.preview_roi_wh = f"{roi[2]}x{roi[3]}"
        sc = mapping.get("scaler_crop", (0, 0, preview_size[0], preview_size[1]))
        self.debug.scaler_crop = _fmt_roi(tuple(sc))
        self.debug.preview_size = f"{preview_size[0]}x{preview_size[1]}"

    async def _apply_document_roi(
        self,
        camera: FocusCapableCamera,
        preview_roi: tuple[int, int, int, int],
        preview_size: tuple[int, int],
    ) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "preview_roi": preview_roi,
            "preview_size": preview_size,
            "scaler_crop": (0, 0, preview_size[0], preview_size[1]),
            "sensor_roi": preview_roi,
        }
        applier = getattr(camera, "apply_document_focus_window", None)
        if not callable(applier):
            return mapping
        try:
            result = await asyncio.to_thread(applier, preview_roi, preview_size)
        except TypeError:
            result = await asyncio.to_thread(applier, 0.7)
        if isinstance(result, dict):
            mapping.update(result)
            if result.get("sensor_roi"):
                mapping["sensor_roi"] = tuple(int(v) for v in result["sensor_roi"])
        return mapping

    def _read_metadata(self, camera: FocusCapableCamera) -> dict[str, Any]:
        reader = getattr(camera, "capture_metadata", None)
        if reader is None:
            return {}
        try:
            return dict(reader() or {})
        except Exception:
            return {}

    async def _run_hardware_af(self, camera: FocusCapableCamera, attempt: int) -> tuple[str, int | None]:
        trigger = getattr(camera, "trigger_autofocus", None)
        settle = getattr(camera, "wait_for_lens_settle", None)
        if callable(trigger) and callable(settle):
            logger.info("event=autofocus_triggered method=hardware_trigger attempt=%d", attempt)
            await asyncio.to_thread(trigger)
            await asyncio.to_thread(settle, min(float(self.config.af_timeout_sec), 3.5))
            return "hardware_trigger", None
        if self.capabilities.autofocus_cycle:
            await asyncio.wait_for(
                asyncio.to_thread(camera.autofocus_cycle),
                timeout=self.config.af_timeout_sec,
            )
            return "hardware_cycle", None
        return "sharpness_only", None
