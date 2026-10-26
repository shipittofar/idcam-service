#!/usr/bin/env python3
"""Minimal camera capture test — no state machine, no upload."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from idcam.camera.autofocus import AutofocusController
from idcam.camera.manager import CameraManager, describe_preview_frame
from idcam.camera.quality import ImageValidator
from idcam.config.loader import load_config
from idcam.config.schema import CameraConfig
from idcam.vision.document_geometry import detect_document_geometry
from idcam.vision.sharpness import inner_roi, preview_sharpness


def main() -> int:
    parser = argparse.ArgumentParser(description="idCam isolated capture test")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        camera_cfg = config.camera
        quality_cfg = config.quality
    except Exception:
        camera_cfg = CameraConfig()
        quality_cfg = None

    mgr = CameraManager(camera_cfg, use_mock=args.mock)
    out = args.output or Path(tempfile.gettempdir()) / "idcam-test-capture.jpg"
    report: dict = {"ok": False, "output": str(out)}

    try:
        mgr.initialize()
        preview = mgr.capture_preview_array()
        report["preview"] = describe_preview_frame(preview)

        geom = detect_document_geometry(preview)
        focus_roi = inner_roi(geom.bbox, 0.15, preview.shape) if geom else None
        report["document"] = {
            "bbox": geom.bbox,
            "corners": geom.corners,
            "confidence": geom.confidence,
        } if geom else None

        af = AutofocusController(camera_cfg, quality_cfg)
        af.probe_capabilities(mgr)
        result = __import__("asyncio").run(
            af.run_autofocus(mgr, preview_roi=focus_roi)
        )
        report["autofocus"] = {
            "method": result.method,
            "sharpness": result.sharpness,
            "before": result.before_sharpness,
            "ratio": result.improvement_ratio,
        }

        mgr.capture_still_to_file(out)
        report["capture_bytes"] = out.stat().st_size

        if quality_cfg is not None:
            fw, fh = int(preview.shape[1]), int(preview.shape[0])
            validation = ImageValidator(quality_cfg, camera_cfg).validate(
                out,
                document_roi=focus_roi,
                preview_size=(fw, fh),
                scaler_crop=mgr.current_scaler_crop(),
            )
            report["validation"] = {
                "width": validation.width,
                "height": validation.height,
                "sharpness": validation.sharpness,
                "document_sharpness": validation.document_sharpness,
                "sha256": validation.sha256,
            }

        report["ok"] = True
    except Exception as exc:
        report["error"] = str(exc)
        print(json.dumps(report, indent=2))
        mgr.shutdown()
        return 1

    mgr.shutdown()
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
