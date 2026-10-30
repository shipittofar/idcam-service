#!/usr/bin/env python3
"""Does the lens really move, and does ONE lens position fit the whole field of view?

Sweeps LensPosition and measures the sharpness of a 3x3 grid of tiles at every position. The summary
tells whether the best lens position is the same everywhere (one saved manual value is enough) or
differs by place (tilted camera / curved field: use focus_mode auto, which focuses on each document).

Stop the service first (it holds the camera):

    sudo systemctl stop idcam-local
    .venv/bin/python scripts/lens_sweep.py            # default 1.5 .. 12
    .venv/bin/python scripts/lens_sweep.py --min 2 --max 8 --step 0.5 --save /tmp/sweep

Cover the WHOLE view with something full of fine detail (a newspaper page or a printed text sheet laid flat
on the table, edge to edge). The tiles are only comparable when each one sees texture.
A column that never changes (sharpness flat, ``lens_meta`` constant) means the lens does not respond.
Nothing is sent anywhere; ``--save DIR`` writes one JPEG per position for you to look at, then delete.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def normalised(gray: np.ndarray) -> float:
    """Tenengrad divided by the squared mean: independent of the auto-exposure level."""
    return tenengrad(gray) / (float(np.mean(gray)) ** 2 + 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min", type=float, default=1.5)
    ap.add_argument("--max", type=float, default=12.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--settle", type=float, default=0.7, help="seconds to wait after each move")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--save", type=Path, default=None, help="directory for one JPEG per position")
    args = ap.parse_args()

    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 is not available here; run this on the Pi with the project venv.")
        return 2

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": (args.width, args.height), "format": "RGB888"}, buffer_count=4))
    cam.start()
    controls = cam.camera_controls
    print("LensPosition range:", controls.get("LensPosition"))
    if "LensPosition" not in controls:
        print("The camera does not advertise LensPosition: manual focus cannot work with this driver.")
        cam.stop()
        return 1
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)

    rows = []
    grid: list[tuple[float, list[float]]] = []
    pos = args.min
    print(f"{'lens':>6} {'lens_meta':>10} {'sharp_centre':>13} {'norm_centre':>12} {'exp_us':>8} {'gain':>5}")
    while pos <= args.max + 1e-9:
        cam.set_controls({"AfMode": 0, "LensPosition": float(pos)})
        time.sleep(args.settle)
        cam.capture_metadata()  # drop a frame exposed before the move
        request = cam.capture_request()
        try:
            frame = request.make_array("main")
            meta = request.get_metadata()
            if args.save:
                request.save("main", str(args.save / f"lens_{pos:05.2f}.jpg"))
        finally:
            request.release()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        centre = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        tiles = [
            normalised(gray[r * h // 3 : (r + 1) * h // 3, c * w // 3 : (c + 1) * w // 3]) for r in range(3) for c in range(3)
        ]
        grid.append((pos, tiles))
        row = (pos, meta.get("LensPosition"), tenengrad(centre), normalised(centre), meta.get("ExposureTime"), meta.get("AnalogueGain"))
        rows.append(row)
        lm = f"{row[1]:.2f}" if isinstance(row[1], (int, float)) else "n/a"
        gain = f"{row[5]:.1f}" if isinstance(row[5], (int, float)) else "n/a"
        print(f"{pos:6.2f} {lm:>10} {row[2]:13.1f} {row[3] * 1e4:12.2f} {row[4] or 0:8d} {gain:>5}")
        pos += args.step
    cam.stop()
    cam.close()

    best = max(rows, key=lambda r: r[3])  # exposure-normalised, see the norm_centre column
    values = [r[3] for r in rows]
    spread = (max(values) - min(values)) / max(float(np.mean(values)), 1e-6)
    print()
    print(f"sharpest centre: lens {best[0]:.2f} (norm {best[3] * 1e4:.2f}, raw {best[2]:.1f})")
    if best[0] >= rows[-1][0] - 1e-6 or best[0] <= rows[0][0] + 1e-6:
        print("The best value is at the END of the swept range: the real peak may lie outside it. Widen --min/--max.")
    metas = {round(r[1], 2) for r in rows if isinstance(r[1], (int, float))}
    if len(metas) <= 1:
        print("lens_meta never changes: the driver may not report the lens position (that alone is not an error).")
    if spread < 0.15:
        print("Sharpness hardly changes with the lens (<15 %): the lens is probably NOT moving, or the scene has no")
        print("detail at this distance. Check the cable/driver, the libcamera version and the distance to the document.")
    else:
        print(f"Sharpness varies by {spread * 100:.0f} % across the sweep: the lens responds. Save {best[0]:.2f} in the panel.")

    names = ["top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right"]
    print()
    print("best lens position per area of the view:")
    best_per_tile = []
    for i, name in enumerate(names):
        series = [(p, t[i]) for p, t in grid]
        top = max(series, key=lambda x: x[1])
        floor = min(v for _, v in series)
        best_per_tile.append((name, top[0], top[1] / max(floor, 1e-6)))
        print(f"  {name:>13}: lens {top[0]:5.2f}   (peak is {top[1] / max(floor, 1e-6):4.1f}x its weakest position)")
    lenses = [b[1] for b in best_per_tile if b[2] >= 1.3]  # tiles with a real peak only
    if len(lenses) < 5:
        print("Fewer than 5 areas show a real peak: the view has too little detail or the lens does not respond.")
    else:
        spread_lens = max(lenses) - min(lenses)
        if spread_lens <= 1.0:
            print(f"The best position is the same everywhere (spread {spread_lens:.2f} dioptres): one manual value fits the whole view.")
        else:
            print(f"The best position differs by {spread_lens:.2f} dioptres across the view: one fixed lens value cannot be sharp")
            print("everywhere. Use focus_mode: auto (focus on each document) or make the camera look straight down at the table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
