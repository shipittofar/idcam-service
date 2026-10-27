"""Live Debug Viewer — observer of the existing preview pipeline.

ONE producer (the orchestrator preview path) publishes the latest frame.
Multiple HTTP clients consume the latest JPEG. There is no per-client camera
capture and no unbounded queue.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from idcam.camera.manager import describe_preview_frame
from idcam.config.schema import DebugConfig
from idcam.logging.setup import get_logger

logger = get_logger("preview.live_viewer")

MJPEG_BOUNDARY = "frame"
PLACEHOLDER_WIDTH = 1280
PLACEHOLDER_HEIGHT = 720
_MAX_CLIENT_LOGS_PER_MINUTE = 20


@dataclass
class DebugSnapshot:
    """Latest debug view of the capture pipeline. Frame may be None."""

    frame: np.ndarray | None = None
    timestamp: float = 0.0
    state: str = "BOOT"
    document_detected: bool = False
    document_locked: bool = False
    motion: float = 0.0
    edge_density: float = 0.0
    motion_threshold: float = 8.0
    edge_density_threshold: float = 0.02
    stable_count: int = 0
    stable_required: int = 8
    stable: bool = False
    camera_healthy: bool = False
    autofocus_available: bool = False
    camera_error: str | None = None
    roi_center_fraction: float = 0.7
    waiting_for_camera: bool = True
    reject_reason: str = ""
    roi_width: int = 0
    roi_height: int = 0
    gray_std: float = 0.0
    gray_mean: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    frame_motion: float = 0.0
    manual_capture_available: bool = False
    manual_phase: str = "idle"
    manual_message: str = ""
    removal_detected: bool = False
    removal_count: int = 0
    autofocus_running: bool = False
    autofocus_method: str = ""
    autofocus_attempt: int = 0
    focus_before: float = 0.0
    focus_after: float = 0.0
    focus_roi: str = ""
    lens_before: float | None = None
    lens_after: float | None = None
    improvement_ratio: float = 0.0
    preview_focus_roi: str = ""
    sensor_focus_roi: str = ""
    preview_roi_wh: str = ""
    document_roi: str = ""
    stability_motion: float = 0.0
    stability_threshold: float = 0.0
    state_reason: str = ""
    manual_capture_requested: bool = False
    capture_started: bool = False
    upload_started: bool = False
    upload_result: str = ""


class LiveFrameBuffer:
    """Thread-safe latest-slot buffer. Slow consumers skip to the newest item."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._seq = 0
        self._payload: Any = None

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._seq

    def publish(self, payload: Any) -> None:
        with self._cond:
            self._payload = payload
            self._seq += 1
            self._cond.notify_all()

    def latest(self) -> tuple[int, Any]:
        with self._lock:
            return self._seq, self._payload

    def wait_latest(self, last_seq: int, timeout: float) -> tuple[int, Any]:
        """Return the newest payload. Never queues older frames."""
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            return self._seq, self._payload

    def notify_all(self) -> None:
        with self._cond:
            self._cond.notify_all()


def resize_debug_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Resize only a debug copy. Never upscale."""
    if frame.ndim < 2:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width or max_width <= 0:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB preview frame to BGR for OpenCV JPEG encoding.

    CameraManager.normalize_picamera_preview_frame() delivers RGB (or gray).
    OpenCV imencode expects BGR.
    """
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    bgr = _to_bgr(frame)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _draw_panel(image: np.ndarray, lines: list[str], *, banner: str | None = None) -> None:
    overlay = image.copy()
    line_h = 22
    pad = 12
    panel_w = 620
    panel_h = pad * 2 + line_h * len(lines)
    if banner:
        panel_h += line_h + 8
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, dst=image)

    y = 8 + pad + 18
    if banner:
        cv2.putText(
            image,
            banner,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 80, 80),
            2,
            cv2.LINE_AA,
        )
        y += line_h + 4
    for line in lines:
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_h


def draw_overlay(frame: np.ndarray, snapshot: DebugSnapshot, fps: float) -> np.ndarray:
    """Draw diagnostics on a copy of ``frame``. The input array is not modified."""
    debug_frame = frame.copy()
    if debug_frame.ndim == 2:
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2RGB)

    doc_roi = _parse_roi(snapshot.document_roi)
    if doc_roi is not None and snapshot.document_locked:
        dx, dy, dw, dh = doc_roi
        cv2.rectangle(debug_frame, (dx, dy), (dx + dw, dy + dh), (80, 180, 255), 2)

    camera = "OK" if snapshot.camera_healthy else "ERROR"
    if not snapshot.camera_healthy and snapshot.waiting_for_camera and not snapshot.camera_error:
        camera = "WAITING"
    document = "YES" if snapshot.document_detected else "NO"
    locked = "YES" if snapshot.document_locked else "NO"
    banner = None
    if snapshot.waiting_for_camera and not snapshot.camera_error:
        banner = "WAITING FOR CAMERA..."
    if snapshot.camera_error or (not snapshot.camera_healthy and not snapshot.waiting_for_camera):
        banner = "CAMERA ERROR"

    focus_status = snapshot.autofocus_method or "-"
    if snapshot.autofocus_running:
        focus_status = "running"
    capture_status = "—"
    if snapshot.capture_started:
        capture_status = "started"
    if snapshot.upload_started:
        capture_status = "uploading"
    if snapshot.upload_result:
        capture_status = snapshot.upload_result

    lines = [
        f"STATE: {snapshot.state}",
        f"DOCUMENT: {document}  LOCKED: {locked}",
        f"FOCUS: {focus_status} {snapshot.focus_before:.1f}->{snapshot.focus_after:.1f}",
        f"CAPTURE: {capture_status}",
        f"FPS: {fps:.1f}  CAMERA: {camera}",
    ]
    if snapshot.camera_error and not snapshot.camera_healthy:
        err = snapshot.camera_error.replace("\n", " ")[:60]
        lines.append(f"DETAIL: {err}")
    _draw_panel(debug_frame, lines, banner=banner)
    return debug_frame


def _parse_roi(value: str) -> tuple[int, int, int, int] | None:
    if not value or "," not in value:
        return None
    parts = value.replace(" ", "").split(",")
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (int(float(p)) for p in parts)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def make_placeholder(message: str) -> np.ndarray:
    image = np.full((PLACEHOLDER_HEIGHT, PLACEHOLDER_WIDTH, 3), 28, dtype=np.uint8)
    cv2.putText(
        image,
        message,
        (48, PLACEHOLDER_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return image


_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>idCam V3 — Live Debug Viewer</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif;
           background: #111; color: #eee; }
    header { padding: 16px 20px; background: #1b1b1b; border-bottom: 1px solid #333; }
    h1 { margin: 0; font-size: 1.25rem; font-weight: 600; }
    .note { color: #aaa; font-size: 0.85rem; margin-top: 4px; }
    main { display: grid; grid-template-columns: 1fr 320px; gap: 16px; padding: 16px; }
    img { width: 100%; max-width: 1280px; background: #000; border: 1px solid #333; }
    dl { margin: 0; }
    dt { color: #888; font-size: 0.75rem; text-transform: uppercase; margin-top: 12px; }
    dd { margin: 2px 0 0; font-variant-numeric: tabular-nums; font-size: 1.05rem; }
    button {
      margin-top: 16px; width: 100%; padding: 12px 16px; font-size: 1rem;
      font-weight: 600; border: 0; border-radius: 8px; cursor: pointer;
      background: #3b82f6; color: #fff;
    }
    button:disabled { background: #334155; color: #94a3b8; cursor: not-allowed; }
    .manual { margin-top: 8px; font-size: 0.9rem; color: #ddd; min-height: 2.4em; }
    .manual.error { color: #f87171; }
    .manual.ok { color: #4ade80; }
    .focuscard { margin-top: 20px; padding: 12px; border: 1px solid #334155; border-radius: 8px; background: #161b22; }
    .focuscard h2 { margin: 0 0 8px; font-size: 1rem; }
    .focuscard .row { display: flex; gap: 8px; align-items: center; }
    .focuscard input[type=range] { flex: 1; }
    .focuscard input[type=number] { width: 84px; padding: 4px; background: #0d1117; color: #eee; border: 1px solid #334155; border-radius: 4px; }
    .focuscard small { color: #94a3b8; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>idCam V3 — Live Debug Viewer</h1>
    <div class="note">Debug only. Live camera preview — do not enable on untrusted networks.</div>
  </header>
  <main>
    <section>
      <img src="/preview/live.mjpg" alt="Live preview">
    </section>
    <aside>
      <dl>
        <dt>State</dt><dd id="state">—</dd>
        <dt>Document</dt><dd id="detection">—</dd>
        <dt>Locked</dt><dd id="locked">—</dd>
        <dt>Focus</dt><dd id="af">—</dd>
        <dt>Capture</dt><dd id="capture">—</dd>
        <dt>Upload</dt><dd id="upload">—</dd>
        <dt>FPS</dt><dd id="fps">—</dd>
      </dl>
      <button id="capture" type="button">Capture &amp; Send</button>
      <div id="manual" class="manual">Idle</div>
      <section id="focuscard" class="focuscard" hidden>
        <h2>Manual focus</h2>
        <div class="row">
          <input id="lens_slider" type="range" min="0" max="32" step="0.05" value="8">
          <input id="lens_num" type="number" min="0" max="32" step="0.05" value="8">
        </div>
        <small>Move the slider while watching the preview and the sharpness value; the highest sharpness is the best focus.</small>
        <dl>
          <dt>Sharpness now</dt><dd id="focus_sharp">—</dd>
          <dt>Saved value</dt><dd id="focus_saved">not saved yet</dd>
        </dl>
        <button id="focus_save" type="button">Confirm &amp; save</button>
        <div id="focus_msg" class="manual"></div>
      </section>
    </aside>
  </main>
  <script>
    const el = (id) => document.getElementById(id);
    let capturing = false;
    function phaseLabel(d) {
      const phase = d.manual_phase || "";
      const map = {
        queued: "Queued...",
        starting: "Starting...",
        detected: "Document detected...",
        focusing: "Focusing...",
        capturing: "Capturing...",
        validating: "Validating...",
        uploading: "Uploading...",
        success: "Success",
        failed: "Failure",
        timeout: "Timed out",
        busy: "Busy",
        idle: "Idle"
      };
      return map[phase] || phase || "Idle";
    }
    let lensTouched = false;
    let lensTimer = null;
    function focusUI(f) {
      const card = el("focuscard");
      if (!f || f.mode !== "manual") { card.hidden = true; return; }
      card.hidden = false;
      const range = f.range || [0, 32];
      for (const id of ["lens_slider", "lens_num"]) { el(id).min = range[0]; el(id).max = range[1]; }
      if (!lensTouched && typeof f.lens_position === "number") {
        el("lens_slider").value = f.lens_position;
        el("lens_num").value = Number(f.lens_position).toFixed(2);
      }
      el("focus_sharp").textContent = f.sharpness == null ? "—" : Number(f.sharpness).toFixed(0);
      el("focus_saved").textContent = typeof f.saved_lens === "number" ? Number(f.saved_lens).toFixed(2) : "not saved yet";
      el("focus_save").disabled = !!f.busy;
    }
    async function sendFocus(save) {
      const box = el("focus_msg");
      try {
        const r = await fetch("/preview/focus", {
          method: "POST", cache: "no-store",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ lens_position: Number(el("lens_num").value), save: save })
        });
        const d = await r.json();
        box.className = "manual" + (d.ok ? " ok" : " error");
        box.textContent = d.ok ? (save ? "Saved " + Number(d.lens_position).toFixed(2) + " for all later captures" : "") : (d.message || "Failed");
        if (d.ok && save) lensTouched = false;
      } catch (err) {
        box.className = "manual error";
        box.textContent = "Request failed";
      }
    }
    function onLensInput(value) {
      lensTouched = true;
      el("lens_slider").value = value;
      el("lens_num").value = value;
      clearTimeout(lensTimer);
      lensTimer = setTimeout(() => sendFocus(false), 200);
    }
    el("lens_slider").addEventListener("input", (e) => onLensInput(e.target.value));
    el("lens_num").addEventListener("change", (e) => onLensInput(e.target.value));
    el("focus_save").addEventListener("click", async () => {
      const v = Number(el("lens_num").value).toFixed(2);
      if (!confirm("Save lens position " + v + " as the manual focus for all later captures?")) return;
      await sendFocus(true);
    });
    async function refresh() {
      try {
        const r = await fetch("/preview/status", { cache: "no-store" });
        if (!r.ok) throw new Error("status " + r.status);
        const d = await r.json();
        focusUI(d.focus);
        el("state").textContent = d.state || "—";
        el("detection").textContent = d.document_detected ? "detected" : "not detected";
        el("locked").textContent = d.document_locked ? "yes" : "no";
        el("af").textContent = (d.autofocus_method || "—") + " " + Number(d.focus_before || 0).toFixed(1) + "→" + Number(d.focus_after || 0).toFixed(1);
        el("capture").textContent = d.capture_started ? "started" : "—";
        el("upload").textContent = d.upload_started ? (d.upload_result || "uploading") : (d.upload_result || "—");
        el("fps").textContent = Number(d.fps).toFixed(1);
        const btn = el("capture");
        btn.disabled = capturing || d.manual_capture_available === false;
        const box = el("manual");
        const live = d.manual_message || phaseLabel(d);
        if (capturing) {
          if (live && live !== "Idle") box.textContent = live;
        } else {
          box.textContent = live;
          box.className = "manual" + (d.manual_phase === "failed" ? " error" : d.manual_phase === "success" ? " ok" : "");
        }
      } catch (err) {
        el("fps").textContent = "—";
      }
    }
    el("capture").addEventListener("click", async () => {
      if (capturing) return;
      capturing = true;
      const btn = el("capture");
      const box = el("manual");
      btn.disabled = true;
      box.className = "manual";
      box.textContent = "Queued...";
      const poll = setInterval(refresh, 300);
      try {
        const r = await fetch("/preview/capture", { method: "POST", cache: "no-store" });
        const d = await r.json();
        if (d.ok) {
          box.className = "manual ok";
          box.textContent = "Success" + (d.capture_id ? " (" + d.capture_id + ")" : "");
        } else {
          box.className = "manual error";
          box.textContent = (d.status === "busy" ? "Busy" : "Failure") + ": " + (d.message || "unknown error");
        }
      } catch (err) {
        box.className = "manual error";
        box.textContent = "Failure: request failed";
      } finally {
        clearInterval(poll);
        capturing = false;
        refresh();
      }
    });
    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""


def _make_handler(viewer: LiveDebugViewer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/preview/live":
                self._serve_html()
            elif path == "/preview/live.mjpg":
                self._serve_mjpeg()
            elif path == "/preview/status":
                self._serve_status()
            elif path == "/preview/focus":
                self._json_bytes(viewer.focus_payload())
            else:
                self.send_error(404, "Not Found")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/preview/capture":
                self._serve_capture()
            elif path == "/preview/focus":
                self._serve_focus()
            else:
                self.send_error(404, "Not Found")

        def _json_bytes(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_focus(self) -> None:
            """POST {"lens_position": 8.4, "save": false|true}: set (and optionally save) manual focus."""
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(min(length, 4096)) if length > 0 else b""
            # application/json only: a cross-site form post (text/plain) must not move the lens.
            if "application/json" not in (self.headers.get("Content-Type") or "").lower():
                self._json_bytes({"ok": False, "status": "invalid", "message": "Content-Type must be application/json"}, 415)
                return
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be an object")
            except ValueError:
                self._json_bytes({"ok": False, "status": "invalid", "message": "Invalid JSON body"}, 400)
                return
            handler = viewer.focus_set_handler
            if handler is None:
                self._json_bytes({"ok": False, "status": "unavailable", "message": "Manual focus is not attached"}, 503)
                return
            try:
                result = handler(body.get("lens_position"), bool(body.get("save", False)))
            except Exception:
                logger.exception("event=debug_manual_focus_failed")
                self._json_bytes({"ok": False, "status": "error", "message": "Focus request failed"}, 500)
                return
            safe = {
                "ok": bool(result.get("ok")),
                "status": str(result.get("status") or ""),
                "message": str(result.get("message") or ""),
            }
            for key in ("lens_position", "saved_lens"):
                if isinstance(result.get(key), (int, float)):
                    safe[key] = float(result[key])
            code = {"ok": 200, "applied": 200, "saved": 200, "invalid": 400, "busy": 409, "disabled": 409}.get(
                safe["status"], 200 if safe["ok"] else 500
            )
            self._json_bytes(safe, code)

        def _serve_capture(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 0:
                remaining = min(length, 65536)
                self.rfile.read(remaining)
            handler = viewer.capture_handler
            if handler is None:
                self._json_bytes(
                    {"ok": False, "status": "unavailable", "message": "Manual capture is not attached"},
                    503,
                )
                return
            try:
                result = handler()
            except Exception:
                logger.exception("event=debug_manual_capture_failed")
                self._json_bytes(
                    {"ok": False, "status": "error", "message": "Capture request failed"},
                    500,
                )
                return
            if not isinstance(result, dict):
                self._json_bytes({"ok": False, "status": "error", "message": "Invalid capture result"}, 500)
                return
            safe = {
                "ok": bool(result.get("ok")),
                "status": str(result.get("status") or ""),
                "state": str(result.get("state") or ""),
                "message": str(result.get("message") or ""),
            }
            if result.get("capture_id"):
                safe["capture_id"] = str(result["capture_id"])
            code = 200
            if safe["status"] == "busy":
                code = 409
            elif safe["status"] == "timeout":
                code = 504
            elif not safe["ok"] and safe["status"] not in {"failed"}:
                code = 500
            self._json_bytes(safe, code)

        def _serve_html(self) -> None:
            body = _HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_status(self) -> None:
            payload = json.dumps(viewer.status_dict(), separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _serve_mjpeg(self) -> None:
            viewer.client_connected(self.client_address)
            try:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
                )
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                last_seq = -1
                while viewer.is_running:
                    seq, jpeg = viewer.wait_jpeg(last_seq, timeout=1.0)
                    if jpeg is None or seq == last_seq:
                        continue
                    last_seq = seq
                    try:
                        self.wfile.write(
                            (
                                f"--{MJPEG_BOUNDARY}\r\n"
                                "Content-Type: image/jpeg\r\n"
                                f"Content-Length: {len(jpeg)}\r\n"
                                "\r\n"
                            ).encode("ascii")
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                        break
                    except OSError:
                        break
            finally:
                viewer.client_disconnected(self.client_address)

    return Handler


class LiveDebugViewer:
    """Background JPEG encoder + HTTP observer. Never captures from the camera."""

    def __init__(self, config: DebugConfig) -> None:
        self.config = config
        self._meta_lock = threading.Lock()
        self._frame_seq = 0
        self._latest_frame: np.ndarray | None = None
        self._snapshot = DebugSnapshot()
        self._jpeg_buffer = LiveFrameBuffer()
        self._stop = threading.Event()
        self._encoder_thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None
        self._server: ThreadingHTTPServer | None = None
        self._clients = 0
        self._clients_lock = threading.Lock()
        self._encode_times: deque[float] = deque(maxlen=12)
        self._actual_fps = 0.0
        self._client_log_times: deque[float] = deque(maxlen=_MAX_CLIENT_LOGS_PER_MINUTE)
        self._started = False
        self._frame_info: dict[str, Any] | None = None
        self._logged_encode_format = False
        self._capture_handler: Any = None
        self._focus_status_fn: Any = None
        self._focus_set_fn: Any = None

    @property
    def is_running(self) -> bool:
        return self._started and not self._stop.is_set()

    @property
    def capture_handler(self) -> Any:
        return self._capture_handler

    def set_capture_handler(self, handler: Any) -> None:
        self._capture_handler = handler

    def set_focus_handlers(self, status_fn: Any, set_fn: Any) -> None:
        """Manual focus panel: ``status_fn() -> dict`` and ``set_fn(lens, save) -> dict``."""
        self._focus_status_fn = status_fn
        self._focus_set_fn = set_fn

    @property
    def focus_set_handler(self) -> Any:
        return self._focus_set_fn

    def focus_payload(self) -> dict[str, Any]:
        """Focus mode/values for the panel, with the live sharpness of the document area."""
        fn = self._focus_status_fn
        if fn is None:
            return {"mode": "auto"}
        try:
            payload = dict(fn())
        except Exception:
            logger.exception("event=debug_focus_status_failed")
            return {"mode": "auto"}
        if payload.get("mode") == "manual":
            payload["sharpness"] = self._live_sharpness()
        return payload

    def _live_sharpness(self) -> float | None:
        """Tenengrad of the document focus area (centre of the frame when none is locked)."""
        frame = self.latest_frame_copy()
        if frame is None or frame.ndim < 2:
            return None
        roi = None
        parts = (self.snapshot_meta().focus_roi or "").split(",")
        if len(parts) == 4:
            try:
                roi = tuple(int(float(v)) for v in parts)
            except ValueError:
                roi = None
        try:
            from idcam.vision.sharpness import clip_roi, preview_sharpness, roi_rect

            if roi is None or roi[2] < 8 or roi[3] < 8:
                roi = roi_rect(frame.shape, 0.4)
            return round(float(preview_sharpness(frame, roi=clip_roi(roi, frame.shape))), 1)
        except Exception:
            return None

    def start(self) -> None:
        if self._started:
            return
        handler = _make_handler(self)
        try:
            self._server = ThreadingHTTPServer((self.config.host, self.config.port), handler)
            self._server.daemon_threads = True
            self._server.timeout = 0.5
        except OSError as exc:
            logger.error(
                "event=debug_viewer_bind_failed host=%s port=%s error=%s",
                self.config.host,
                self.config.port,
                exc,
            )
            self._server = None
            return

        self._stop.clear()
        self._encoder_thread = threading.Thread(
            target=self._encoder_loop, name="idcam-debug-encoder", daemon=True
        )
        self._http_thread = threading.Thread(
            target=self._http_loop, name="idcam-debug-http", daemon=True
        )
        self._encoder_thread.start()
        self._http_thread.start()
        self._started = True
        logger.info(
            "Live Debug Viewer enabled: http://%s:%s/preview/live",
            self.config.host,
            self.config.port,
        )
        logger.info(
            "event=debug_viewer_started host=%s port=%s fps=%.1f overlay=%s",
            self.config.host,
            self.config.port,
            self.config.fps,
            self.config.overlay,
        )

    def stop(self) -> None:
        if not self._started and self._server is None:
            return
        self._stop.set()
        self._jpeg_buffer.notify_all()
        server = self._server
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                logger.debug("event=debug_viewer_shutdown_error", exc_info=True)
            try:
                server.server_close()
            except Exception:
                logger.debug("event=debug_viewer_close_error", exc_info=True)
        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=2.0)
        self._server = None
        self._started = False
        logger.info("event=debug_viewer_stopped")

    def publish(
        self,
        frame: np.ndarray | None = None,
        *,
        state: str,
        document_detected: bool,
        document_locked: bool = False,
        motion: float,
        edge_density: float,
        motion_threshold: float,
        edge_density_threshold: float,
        stable_count: int,
        stable_required: int,
        stable: bool,
        camera_healthy: bool,
        autofocus_available: bool,
        camera_error: str | None = None,
        roi_center_fraction: float = 0.7,
        reject_reason: str = "",
        roi_width: int = 0,
        roi_height: int = 0,
        gray_std: float = 0.0,
        gray_mean: float = 0.0,
        frame_width: int = 0,
        frame_height: int = 0,
        frame_motion: float = 0.0,
        manual_capture_available: bool = False,
        manual_phase: str = "idle",
        manual_message: str = "",
        removal_detected: bool = False,
        removal_count: int = 0,
        autofocus_running: bool = False,
        autofocus_method: str = "",
        autofocus_attempt: int = 0,
        focus_before: float = 0.0,
        focus_after: float = 0.0,
        focus_roi: str = "",
        lens_before: float | None = None,
        lens_after: float | None = None,
        improvement_ratio: float = 0.0,
        preview_focus_roi: str = "",
        sensor_focus_roi: str = "",
        preview_roi_wh: str = "",
        document_roi: str = "",
        stability_motion: float = 0.0,
        stability_threshold: float = 0.0,
        state_reason: str = "",
        manual_capture_requested: bool = False,
        capture_started: bool = False,
        upload_started: bool = False,
        upload_result: str = "",
    ) -> None:
        """Store the latest preview copy and metadata. Never blocks on HTTP clients."""
        with self._meta_lock:
            if frame is not None:
                self._latest_frame = frame.copy()
                self._frame_seq += 1
                info = describe_preview_frame(self._latest_frame)
                if self._latest_frame.ndim == 3:
                    info["channel_order"] = "RGB"
                elif self._latest_frame.ndim == 2:
                    info["channel_order"] = "GRAY"
                self._frame_info = info
            self._snapshot = DebugSnapshot(
                frame=None,
                timestamp=time.time(),
                state=state,
                document_detected=document_detected,
                document_locked=document_locked,
                motion=motion,
                edge_density=edge_density,
                motion_threshold=motion_threshold,
                edge_density_threshold=edge_density_threshold,
                stable_count=stable_count,
                stable_required=stable_required,
                stable=stable,
                camera_healthy=camera_healthy,
                autofocus_available=autofocus_available,
                camera_error=camera_error,
                roi_center_fraction=roi_center_fraction,
                waiting_for_camera=self._latest_frame is None,
                reject_reason=reject_reason,
                roi_width=roi_width,
                roi_height=roi_height,
                gray_std=gray_std,
                gray_mean=gray_mean,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_motion=frame_motion,
                manual_capture_available=manual_capture_available,
                manual_phase=manual_phase,
                manual_message=manual_message,
                removal_detected=removal_detected,
                removal_count=removal_count,
                autofocus_running=autofocus_running,
                autofocus_method=autofocus_method,
                autofocus_attempt=autofocus_attempt,
                focus_before=focus_before,
                focus_after=focus_after,
                focus_roi=preview_focus_roi or focus_roi,
                lens_before=lens_before,
                lens_after=lens_after,
                improvement_ratio=improvement_ratio,
                preview_focus_roi=preview_focus_roi or focus_roi,
                sensor_focus_roi=sensor_focus_roi,
                preview_roi_wh=preview_roi_wh,
                document_roi=document_roi,
                stability_motion=stability_motion,
                stability_threshold=stability_threshold,
                state_reason=state_reason,
                manual_capture_requested=manual_capture_requested,
                capture_started=capture_started,
                upload_started=upload_started,
                upload_result=upload_result,
            )

    def snapshot_meta(self) -> DebugSnapshot:
        with self._meta_lock:
            return self._snapshot

    def latest_frame_copy(self) -> np.ndarray | None:
        with self._meta_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def status_dict(self) -> dict[str, Any]:
        snap = self.snapshot_meta()
        with self._meta_lock:
            frame_info = dict(self._frame_info) if self._frame_info else None
        return {
            "enabled": True,
            "state": snap.state,
            "document_detected": snap.document_detected,
            "document_locked": snap.document_locked,
            "motion": snap.motion,
            "edge_density": snap.edge_density,
            "motion_threshold": snap.motion_threshold,
            "edge_density_threshold": snap.edge_density_threshold,
            "stable_count": snap.stable_count,
            "stable_required": snap.stable_required,
            "stable": snap.stable,
            "camera_healthy": snap.camera_healthy,
            "autofocus_available": snap.autofocus_available,
            "fps": round(self._actual_fps or self.config.fps, 2),
            "timestamp": snap.timestamp,
            "frame": frame_info,
            "reject_reason": snap.reject_reason,
            "roi_width": snap.roi_width,
            "roi_height": snap.roi_height,
            "gray_std": snap.gray_std,
            "gray_mean": snap.gray_mean,
            "frame_width": snap.frame_width,
            "frame_height": snap.frame_height,
            "frame_motion": snap.frame_motion,
            "manual_capture_available": snap.manual_capture_available,
            "manual_phase": snap.manual_phase,
            "manual_message": snap.manual_message,
            "removal_detected": snap.removal_detected,
            "removal_count": snap.removal_count,
            "autofocus_running": snap.autofocus_running,
            "autofocus_method": snap.autofocus_method,
            "autofocus_attempt": snap.autofocus_attempt,
            "focus_before": snap.focus_before,
            "focus_after": snap.focus_after,
            "focus_roi": snap.preview_focus_roi or snap.focus_roi,
            "lens_before": snap.lens_before,
            "lens_after": snap.lens_after,
            "improvement_ratio": snap.improvement_ratio,
            "preview_focus_roi": snap.preview_focus_roi or snap.focus_roi,
            "sensor_focus_roi": snap.sensor_focus_roi,
            "preview_roi_wh": snap.preview_roi_wh,
            "document_roi": snap.document_roi,
            "stability_motion": snap.stability_motion,
            "stability_threshold": snap.stability_threshold,
            "state_reason": snap.state_reason,
            "manual_capture_requested": snap.manual_capture_requested,
            "capture_started": snap.capture_started,
            "upload_started": snap.upload_started,
            "upload_result": snap.upload_result,
            "focus": self.focus_payload(),
        }

    def wait_jpeg(self, last_seq: int, timeout: float) -> tuple[int, bytes | None]:
        seq, payload = self._jpeg_buffer.wait_latest(last_seq, timeout)
        jpeg = payload if isinstance(payload, (bytes, bytearray)) else None
        return seq, jpeg

    def client_connected(self, address: tuple[str, int]) -> None:
        with self._clients_lock:
            self._clients += 1
            count = self._clients
        self._log_client("event=debug_client_connected", address, count)

    def client_disconnected(self, address: tuple[str, int]) -> None:
        with self._clients_lock:
            self._clients = max(0, self._clients - 1)
            count = self._clients
        self._log_client("event=debug_client_disconnected", address, count)

    def _log_client(self, event: str, address: tuple[str, int], count: int) -> None:
        now = time.monotonic()
        recent = sum(1 for t in self._client_log_times if now - t < 60.0)
        if recent >= _MAX_CLIENT_LOGS_PER_MINUTE:
            logger.debug("%s clients=%d peer=%s", event, count, address[0])
        else:
            logger.info("%s clients=%d peer=%s", event, count, address[0])
        self._client_log_times.append(now)

    def _http_loop(self) -> None:
        server = self._server
        if server is None:
            return
        try:
            server.serve_forever(poll_interval=0.3)
        except Exception:
            if not self._stop.is_set():
                logger.exception("event=debug_http_server_failed")

    def _encoder_loop(self) -> None:
        interval = 1.0 / max(self.config.fps, 0.1)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                jpeg = self._encode_latest()
                if jpeg is not None:
                    self._jpeg_buffer.publish(jpeg)
                    now = time.monotonic()
                    self._encode_times.append(now)
                    if len(self._encode_times) >= 2:
                        span = self._encode_times[-1] - self._encode_times[0]
                        if span > 0:
                            self._actual_fps = (len(self._encode_times) - 1) / span
            except Exception:
                logger.exception("event=debug_encode_failed")
            remaining = interval - (time.monotonic() - t0)
            self._stop.wait(timeout=max(0.0, remaining))

    def _encode_latest(self) -> bytes | None:
        with self._meta_lock:
            frame = self._latest_frame
            snap = self._snapshot
        if frame is None:
            message = "WAITING FOR CAMERA..."
            if snap.camera_error:
                message = "CAMERA ERROR"
            image = make_placeholder(message)
            if self.config.overlay:
                image = draw_overlay(image, snap, self._actual_fps or self.config.fps)
            else:
                image = resize_debug_frame(image, self.config.max_width)
            return encode_jpeg(image, self.config.jpeg_quality)

        debug_copy = frame.copy()
        debug_copy = resize_debug_frame(debug_copy, self.config.max_width)
        if not self._logged_encode_format:
            info = describe_preview_frame(debug_copy)
            logger.info(
                "event=preview_encode_format shape=%s dtype=%s means=%s assumed_order=RGB",
                info.get("shape"),
                info.get("dtype"),
                info.get("channel_means"),
            )
            self._logged_encode_format = True
        if self.config.overlay:
            debug_copy = draw_overlay(debug_copy, snap, self._actual_fps or self.config.fps)
        return encode_jpeg(debug_copy, self.config.jpeg_quality)
