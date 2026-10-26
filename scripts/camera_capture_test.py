#!/usr/bin/env python3
"""
Isolated Arducam capture test — NO orchestrator, state machine, detector, or upload.

Phase 1 (--raw-only):
  init camera → preview → capture full-res JPEG → validate → JSON

Phase 6 (default with --af-center):
  init → preview diagnostics → center AF → capture → validate → JSON

Usage on Pi:
  sudo systemctl stop idcam
  /opt/idcam/venv/bin/python scripts/camera_capture_test.py \
      --config /etc/idcam/config.yaml \
      --output /tmp/idcam-camera-test.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from idcam.camera.diagnostic import (
    analyze_channels,
    camera_hardware_info,
    capture_native_preview,
    center_focus_roi,
    preview_sharpness_report,
    save_rgb_jpeg,
    validate_jpeg_file,
)
from idcam.camera.manager import CameraManager, normalize_picamera_preview_frame
from idcam.config.loader import load_camera_config_for_diagnostic


def _candidate_to_dict(c) -> dict:
    return {
        "accepted": c.accepted,
        "bbox": list(c.bbox),
        "corners": [list(p) for p in c.corners] if c.corners else None,
        "area_fraction": round(c.area_fraction, 4),
        "aspect_ratio": round(c.aspect_ratio, 3),
        "rectangularity": round(c.rectangularity, 3),
        "edge_strength": round(c.edge_strength, 2),
        "final_score": round(c.final_score, 3),
        "rejection_reason": c.rejection_reason,
        "source_method": c.source_method,
        "contrast_score": round(c.contrast_score, 4),
    }


def run_preview_phase(mgr: CameraManager, report: dict, preview_path: Path) -> tuple:
    """Phase 2: capture native + RGB preview, save JPEG, return (native, rgb)."""
    native = capture_native_preview(mgr)
    rgb = mgr.capture_preview_array()

    report["preview_native"] = analyze_channels(native, "native_from_picamera2") if native is not None else None
    report["preview_rgb"] = analyze_channels(rgb, "idcam_rgb_normalized")
    report["preview_rgb"]["describe"] = __import__(
        "idcam.camera.manager", fromlist=["describe_preview_frame"]
    ).describe_preview_frame(rgb)

    if native is not None:
        report["preview_native_bgr_view"] = analyze_channels(native, "native_as_bgr")
        # If native is BGR in memory, interpreting as BGR for gray stats
        import cv2
        import numpy as np

        bgr_view = np.ascontiguousarray(native[..., :3])
        report["preview_native_bgr_view"] = analyze_channels(bgr_view, "native_bgr_view")

    save_rgb_jpeg(rgb, preview_path)
    report["preview_jpeg_path"] = str(preview_path)
    report["preview_jpeg_bytes"] = preview_path.stat().st_size if preview_path.exists() else 0
    return native, rgb


def run_geometry_phase(rgb, report: dict, geometry_debug_path: Path) -> None:
    """Phase 3-4: scan candidates, save debug image."""
    from idcam.vision.document_geometry import draw_geometry_debug_image, scan_document_candidates

    best, candidates = scan_document_candidates(rgb)
    report["geometry_candidates"] = [_candidate_to_dict(c) for c in candidates]
    if best is None:
        reason = "no_contour_passed_filters"
        if not candidates:
            reason = "no_contours_found"
        near_miss = next((c for c in candidates if c.bbox[2] > 0 and c.bbox[3] > 0), None)
        report["document_detected"] = False
        report["document_reason"] = reason
        draw_geometry_debug_image(rgb, None, geometry_debug_path, reason=reason, near_miss=near_miss)
    else:
        report["document_detected"] = True
        report["document"] = {
            "document_bbox": list(best.bbox),
            "document_corners": [list(p) for p in best.corners],
            "confidence": round(best.confidence, 3),
            "area_fraction": round(best.area_fraction, 4),
            "aspect_ratio": round(best.aspect_ratio, 3),
            "rotation": round(best.rotation_deg, 2),
            "detection_method": best.detection_method,
        }
        draw_geometry_debug_image(rgb, best, geometry_debug_path)
    report["geometry_debug_path"] = str(geometry_debug_path)


def run_af_center(mgr: CameraManager, rgb, report: dict, timeout: float) -> tuple[int, int, int, int]:
    """Phase 5 variant: AF on center region — no document detector."""
    focus_roi = center_focus_roi(rgb.shape, 0.45)
    fw, fh = int(rgb.shape[1]), int(rgb.shape[0])
    sharp_before = preview_sharpness_report(rgb, focus_roi)
    lens_before = mgr.capture_metadata().get("LensPosition")

    mapping = mgr.apply_document_focus_window(focus_roi, (fw, fh))
    mgr.trigger_autofocus()
    lens_after = mgr.wait_for_lens_settle(timeout)
    rgb_after = mgr.capture_preview_array()
    sharp_after = preview_sharpness_report(rgb_after, focus_roi)

    ratio = sharp_after["tenengrad"] / max(sharp_before["tenengrad"], 1e-6)
    report["autofocus"] = {
        "mode": "center_roi",
        "focus_roi": list(focus_roi),
        "sensor_af_window": list(mapping.get("sensor_roi", (0, 0, 0, 0))),
        "scaler_crop": list(mapping.get("scaler_crop", (0, 0, 0, 0))),
        "lens_before": lens_before,
        "lens_after": lens_after,
        "sharpness_before": sharp_before,
        "sharpness_after": sharp_after,
        "sharpness_ratio": round(ratio, 3),
        "improved": ratio >= 1.05 or sharp_after["tenengrad"] > sharp_before["tenengrad"] + 1.0,
    }
    mgr.pause_autofocus()
    return focus_roi


def run_af_document(mgr: CameraManager, rgb, report: dict, timeout: float) -> tuple[int, int, int, int] | None:
    """Phase 5: AF on detected document inner ROI."""
    from idcam.vision.document_geometry import scan_document_candidates
    from idcam.vision.sharpness import inner_roi

    best, _ = scan_document_candidates(rgb)
    if best is None:
        report["autofocus"] = {"mode": "document_roi", "error": "no_document_geometry"}
        return None

    focus_roi = inner_roi(best.bbox, 0.15, rgb.shape)
    fw, fh = int(rgb.shape[1]), int(rgb.shape[0])
    sharp_before = preview_sharpness_report(rgb, focus_roi)
    lens_before = mgr.capture_metadata().get("LensPosition")
    mapping = mgr.apply_document_focus_window(focus_roi, (fw, fh))
    mgr.trigger_autofocus()
    lens_after = mgr.wait_for_lens_settle(timeout)
    rgb_after = mgr.capture_preview_array()
    sharp_after = preview_sharpness_report(rgb_after, focus_roi)
    ratio = sharp_after["tenengrad"] / max(sharp_before["tenengrad"], 1e-6)

    report["autofocus"] = {
        "mode": "document_roi",
        "focus_roi": list(focus_roi),
        "document_bbox": list(best.bbox),
        "sensor_af_window": list(mapping.get("sensor_roi", (0, 0, 0, 0))),
        "lens_before": lens_before,
        "lens_after": lens_after,
        "sharpness_before": sharp_before,
        "sharpness_after": sharp_after,
        "sharpness_ratio": round(ratio, 3),
        "improved": ratio >= 1.05 or sharp_after["tenengrad"] > sharp_before["tenengrad"] + 1.0,
        "focus_failed": ratio < 0.95 and sharp_after["tenengrad"] < sharp_before["tenengrad"],
    }
    mgr.pause_autofocus()
    return focus_roi


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated idCam camera capture test")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML (e.g. /etc/idcam/config.yaml)")
    parser.add_argument("--output", type=Path, default=Path("/tmp/idcam-camera-test.jpg"))
    parser.add_argument("--preview-output", type=Path, default=Path("/tmp/idcam-preview.jpg"))
    parser.add_argument("--geometry-debug", type=Path, default=Path("/tmp/idcam-geometry-debug.jpg"))
    parser.add_argument("--raw-only", action="store_true", help="Phase 1: capture only, no autofocus")
    parser.add_argument("--af-center", action="store_true", help="AF on center ROI (no document detector)")
    parser.add_argument("--af-document", action="store_true", help="AF on detected document ROI")
    parser.add_argument("--geometry", action="store_true", help="Run geometry candidate scan + debug image")
    parser.add_argument("--mock", action="store_true", help="Mock camera (dev only)")
    args = parser.parse_args()

    report: dict = {"ok": False, "phase": "raw_only" if args.raw_only else "capture_test"}

    camera_cfg, config_warnings = load_camera_config_for_diagnostic(args.config)
    if config_warnings:
        report.update(config_warnings)

    mgr = CameraManager(camera_cfg, use_mock=args.mock)

    try:
        mgr.initialize()
        report["hardware"] = camera_hardware_info(mgr)

        _, rgb = run_preview_phase(mgr, report, args.preview_output)

        if args.geometry or args.af_document:
            run_geometry_phase(rgb, report, args.geometry_debug)

        if args.raw_only:
            report["autofocus"] = {"skipped": True, "reason": "raw_only"}
        elif args.af_document:
            run_af_document(mgr, rgb, report, camera_cfg.af_timeout_sec)
        elif args.af_center or not args.raw_only:
            # Default: center AF when not raw-only
            run_af_center(mgr, rgb, report, camera_cfg.af_timeout_sec)

        # Phase 1 / 6: full-resolution JPEG
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        mgr.capture_still_to_file(output)

        validation = validate_jpeg_file(
            output,
            expected_width=camera_cfg.capture_width,
            expected_height=camera_cfg.capture_height,
            tolerance=128,
            min_bytes=1000,
        )
        report["capture"] = {
            "path": str(output),
            "validation": {
                "ok": validation.ok,
                "file_bytes": validation.file_bytes,
                "width": validation.width,
                "height": validation.height,
                "expected_width": validation.expected_width,
                "expected_height": validation.expected_height,
                "dimension_match": validation.dimension_match,
                "sharpness_laplacian": validation.sharpness_laplacian,
                "sha256_prefix": validation.sha256_prefix,
                "error": validation.error,
            },
        }

        report["ok"] = validation.ok
        if not validation.ok:
            report["failure_reason"] = validation.error or "jpeg_validation_failed"

        af = report.get("autofocus", {})
        if isinstance(af, dict) and af.get("focus_failed"):
            report["autofocus_warning"] = "sharpness_degraded_after_af"

    except Exception as exc:
        report["ok"] = False
        report["failure_reason"] = str(exc)
        report["traceback"] = traceback.format_exc()
        print(json.dumps(report, indent=2, default=str))
        mgr.shutdown()
        return 1

    mgr.shutdown()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
