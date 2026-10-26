#!/usr/bin/env python3
"""Print camera capabilities and perform one diagnostic capture on Raspberry Pi."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from idcam.camera.autofocus import AutofocusController
from idcam.camera.manager import CameraManager, describe_preview_frame, map_preview_roi_to_sensor
from idcam.camera.quality import ImageValidator
from idcam.config.loader import load_config
from idcam.config.schema import CameraConfig
from idcam.vision.document_geometry import detect_document_geometry
from idcam.vision.sharpness import inner_roi, laplacian_variance, preview_sharpness, tenengrad_score


def _print_preview_diagnostics(frame) -> dict:
    import numpy as np

    info = describe_preview_frame(frame)
    info["min_pixel"] = float(np.min(frame))
    info["max_pixel"] = float(np.max(frame))
    info["channel_order_assumed"] = "RGB"
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="idCam camera hardware diagnostic")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument("--output", type=Path, default=None, help="JPEG output path")
    parser.add_argument("--mock", action="store_true", help="Use mock camera (dev only)")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"config_error: {exc}", file=sys.stderr)
        config = None
        camera_cfg = CameraConfig()
        quality_cfg = None
    else:
        camera_cfg = config.camera
        quality_cfg = config.quality

    mgr = CameraManager(camera_cfg, use_mock=args.mock)
    result: dict = {"ok": False}

    try:
        mgr.initialize()
        result["camera_started"] = True
        result["is_mock"] = mgr.is_mock

        controls = mgr.get_controls()
        result["controls"] = {k: list(v) if isinstance(v, tuple) else v for k, v in controls.items()}

        props = {}
        if not mgr.is_mock and mgr._picam2 is not None:
            props = dict(getattr(mgr._picam2, "camera_properties", {}) or {})
        result["camera_properties"] = {k: str(v) for k, v in list(props.items())[:40]}

        preview = mgr.capture_preview_array()
        preview_info = _print_preview_diagnostics(preview)
        result["preview"] = preview_info

        geom = detect_document_geometry(preview)
        if geom:
            focus_roi = inner_roi(geom.bbox, 0.15, preview.shape)
            result["document"] = {
                "bbox": geom.bbox,
                "corners": geom.corners,
                "confidence": geom.confidence,
                "area_fraction": geom.area_fraction,
                "aspect_ratio": geom.aspect_ratio,
                "rotation_deg": geom.rotation_deg,
            }
            preview_roi = focus_roi
        else:
            result["document"] = None
            h, w = preview.shape[:2]
            preview_roi = (w // 8, h // 8, w // 2, h // 2)

        sharp_before = preview_sharpness(preview, roi=preview_roi)
        tenengrad_before = tenengrad_score(preview, roi=preview_roi)
        result["sharpness_before"] = {
            "tenengrad": round(sharp_before, 2),
            "laplacian_preview_roi": round(laplacian_variance(preview, roi=preview_roi), 2),
        }

        scaler = mgr.current_scaler_crop()
        fw, fh = int(preview.shape[1]), int(preview.shape[0])
        sensor_roi = map_preview_roi_to_sensor(preview_roi, (fw, fh), scaler)
        result["af_mapping"] = {
            "preview_roi": preview_roi,
            "preview_size": [fw, fh],
            "scaler_crop": scaler,
            "sensor_roi": sensor_roi,
        }

        af = AutofocusController(camera_cfg, quality_cfg)
        caps = af.probe_capabilities(mgr)
        result["af_capabilities"] = caps.to_dict()

        lens_before = mgr.capture_metadata().get("LensPosition")
        result["lens_before"] = lens_before

        mapping = mgr.apply_document_focus_window(preview_roi, (fw, fh))
        result["af_window_applied"] = mapping

        mgr.trigger_autofocus()
        lens_after = mgr.wait_for_lens_settle(min(camera_cfg.af_timeout_sec, 6.0))
        result["lens_after"] = lens_after

        preview2 = mgr.capture_preview_array()
        sharp_after = preview_sharpness(preview2, roi=preview_roi)
        result["sharpness_after"] = {
            "tenengrad": round(sharp_after, 2),
            "laplacian_preview_roi": round(laplacian_variance(preview2, roi=preview_roi), 2),
            "ratio": round(sharp_after / max(sharp_before, 1e-6), 3),
        }

        mgr.pause_autofocus()

        out = args.output
        if out is None:
            out = Path(tempfile.gettempdir()) / "idcam-diagnose-capture.jpg"
        mgr.capture_still_to_file(out)
        result["capture_path"] = str(out)
        result["capture_bytes"] = out.stat().st_size if out.exists() else 0

        if quality_cfg is not None:
            validator = ImageValidator(quality_cfg, camera_cfg)
            validation = validator.validate(out, document_roi=preview_roi, preview_size=(fw, fh))
            result["validation"] = {
                "width": validation.width,
                "height": validation.height,
                "file_bytes": validation.file_bytes,
                "sharpness_global": round(validation.sharpness, 2),
                "sharpness_document_roi": round(validation.document_sharpness, 2),
                "brightness": round(validation.brightness, 2),
                "sha256": validation.sha256,
            }
        else:
            import cv2

            img = cv2.imread(str(out))
            if img is not None:
                result["validation"] = {
                    "width": img.shape[1],
                    "height": img.shape[0],
                    "sharpness_global": round(laplacian_variance(img), 2),
                }

        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
        print(json.dumps(result, indent=2))
        mgr.shutdown()
        return 1

    mgr.shutdown()
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
