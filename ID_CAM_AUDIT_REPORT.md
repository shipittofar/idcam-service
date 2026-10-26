# ID Cam (idCam v1.0.0) — Codebase Audit, Upload Forensics and Performance Report

Scope: every file under `src/`, `deploy/`, `scripts/install.sh`, `docs/`, `tests/`, `config.example.yaml`, `pyproject.toml`.
Method: full read of the implementation, `grep` sweeps for networking/file/image APIs, the existing unit suite executed, and three small isolated micro-benchmarks run in memory on the audit host (details in §15).
No project file was modified, no package installed, no external service contacted, no identity data used. This report is the only file added.

**Evidence labels used throughout:** *Confirmed* = read in source or measured. *Likely* = strong code evidence, effect depends on hardware/library behaviour. *Needs verification* = requires the Pi, the camera or the backend.

**Important limit:** this was a static audit plus host-side micro-benchmarks. The audit host is a Windows x86 PC, not a Raspberry Pi 5 with the Arducam 64MP Hawkeye. No camera, autofocus, capture or live network behaviour was observed. The upload description below is the request *as constructed by the source code*, not a network capture.

---

## 1. Executive Summary

ID Cam is a headless, Python 3.11+ kiosk service for a Raspberry Pi 5 + Arducam 64MP Hawkeye. It watches a capture area through a 1080p preview stream, detects a document, waits for it to be still, autofocuses, switches the sensor to a 9152×6944 still mode, saves one JPEG, validates it, uploads it **unmodified and uncropped** as a single multipart POST, deletes it, and waits for the document to be removed. One transaction at a time; no queue; no offline retry.

Headline findings:

1. **Upload content (Confirmed from source):** one file part `image` (the full-frame JPEG, filename `capture-<32 hex>.jpg`, content type hard-coded `image/jpeg`) plus headers `X-Device-ID`, `X-Capture-ID`, `Idempotency-Key`, and `Authorization: Bearer …` when a token is set. No OCR, no metadata fields, no crop, no thumbnails, no telemetry. Destination is whatever `server.url` + `server.path` is in `/etc/idcam/config.yaml`; the repository only contains a placeholder host.
2. **Biggest speed bottleneck (Confirmed, measured):** document detection spends ~90% of its time in one `GaussianBlur` with σ≈192 (`document_geometry.py:143-146`): 7.6 s per 1080p call on the audit host, versus 21 ms for an equivalent downscaled blur (mean pixel difference 0.24/255). This detector is run synchronously on the asyncio loop in IDLE, before capture, and during removal.
3. **Second biggest (Confirmed, measured):** post-capture validation decodes the 64MP JPEG three times and calls `np.std()` over the whole 190 MB BGR array (4.8 s on the audit host; it also materialises a float64 temporary of about 1.5 GB — computed, not measured). It runs on the event loop.
4. **Retry is dead code (Confirmed):** the default `RetryPolicy` never retries (`attempt < 1`), contradicting `docs/SECURITY.md` and `docs/API_PROTOCOL.md`. The documented unit test passes only because it builds a non-default policy.
5. **`camera.jpeg_quality` is never applied to real captures** (`manager.py:261` vs mock-only `:591`).
6. **Optional debug viewer** (off by default) is unauthenticated, binds `0.0.0.0` by default, streams preview frames and exposes `POST /preview/capture`, which triggers a real capture and upload.
7. **Existing tests:** 280 collected; the 277 unit tests I ran all pass. Nothing exercises the real camera path, the 64MP pipeline timing, or the default retry policy.

---

## 2. Project Purpose and Architecture

Purpose: unattended document capture appliance (README: "Production document capture … 24/7").

```
                       ┌──────────────────────── ApplicationOrchestrator (asyncio, single loop) ────────────────────────┐
 Pi CSI camera         │  main loop: health refresh → watchdog → state timeout → one state tick → sleep(1/preview_fps)  │
 (Picamera2/libcamera) │                                                                                                  │
      │                │  StateMachine   Watchdog   StatusPublisher(/run/idcam/status.json)   LedIndicator (thread)     │
      │ 1920x1080 RGB888 preview (4 buffers)                                                                              │
      ▼                └──────────────────────────────────────────────────────────────────────────────────────────────────┘
 CameraManager ──capture_array──► normalize BGR→RGB copy ──► DocumentDetector.detect (geometry, ~19 segmentation variants)
      │                                   │                         │ lock ROI           StabilityDetector (8 still frames)
      │                                   ▼                         ▼
      │                         LiveDebugViewer (optional HTTP, 0.0.0.0:8088)      AutofocusController
      │                                                               (AfWindows on doc ROI → AfMode=Auto+Trigger → lens-settle poll → Tenengrad check)
      │  stop → still config 9152x6944 RGB888 → start → hold LensPosition manual → capture_file(jpeg) → stop → preview config → start
      ▼
 CaptureService (asyncio.to_thread, 45 s timeout) → TempFileStore: capture-<uuid>.partial.jpg → fsync → os.replace → capture-<uuid>.jpg (0640)
      ▼
 ImageValidator (sync, on loop): sha256, PIL verify, PIL load, cv2.imread, Laplacian, brightness, np.std
      ▼
 ApiClient (httpx.AsyncClient, one POST multipart) ──HTTPS──► {server.url}{server.path}  ──► JSON {success, capture_id, result, message}
      ▼
 ServerResponse parse → delete local JPEG (success AND every failure path) → SUCCESS → WAITING_FOR_REMOVAL → IDLE
```

---

## 3. Technology Stack

| Item | Value | Source |
|---|---|---|
| Language/runtime | Python ≥ 3.11 (tests ran on 3.14.1) | `pyproject.toml` |
| Camera | `picamera2` + `libcamera` (apt, optional extra `pi`), mock fallback when import fails | `camera/manager.py:18-26` |
| Vision | `opencv-python-headless ≥ 4.9`, `numpy ≥ 1.26`, `Pillow ≥ 10.2` | `pyproject.toml` |
| Network | `httpx ≥ 0.27` (async) | `network/api_client.py` |
| Config | `pydantic ≥ 2.5`, `pyyaml` | `config/` |
| LED (optional) | Adafruit Blinka / pixelbuf (WS2812) | `ui/led.py` |
| Debug HTTP | stdlib `http.server.ThreadingHTTPServer` | `preview/live_viewer.py:15` |
| Service | systemd unit, user `idcam`, hardening flags | `deploy/idcam.service` |
| Tests | pytest, pytest-asyncio, respx | `pyproject.toml` |
| Lockfile | none (only lower-bound pins) | — |

No analytics, crash-reporting, telemetry, WebSocket, gRPC, WebView or cloud-storage SDK exists in `src/` (grep for `requests|urllib|socket|websocket|sentry|telemetry|analytics|boto|smtp` returned only `urllib.parse` in the viewer and the `admin_socket` config path, which no code opens).

---

## 4. Important Files and Directory Structure

```
src/idcam/
  app/main.py            entry point (CLI args, config load, asyncio.run)
  app/orchestrator.py    1119 lines: state machine driver, every pipeline stage handler, transaction lifecycle
  app/state_machine.py   AppState enum, allowed transitions, per-state timeouts
  app/transaction.py     ActiveCaptureTransaction (capture_id = "cap-" + uuid4 hex)
  app/admin.py           CLI: version/status/config(redacted)/validate/health/storage
  app/long_run.py        long-run harness (not on production path)
  camera/manager.py      Picamera2 lifecycle, preview stream, AF controls, still capture
  camera/autofocus.py    AutofocusController: ROI → trigger → lens settle → Tenengrad accept
  camera/capture.py      CaptureService: to_thread + timeout + atomic finalize
  camera/quality.py      ImageValidator (post-capture checks)
  camera/diagnostic.py   hardware diagnostics helpers
  vision/document_detector.py / document_geometry.py / document_lock.py   detection, quad geometry, smoothed lock
  vision/stability_detector.py / motion.py / sharpness.py                  motion and focus metrics
  network/api_client.py  the only outbound request code (upload + startup HEAD)
  network/protocol.py    UploadRequest, ServerResponse parsing
  network/retry_policy.py
  storage/temp_files.py  temp paths, atomic write, permissions, stale cleanup
  storage/disk_monitor.py, storage/paths.py
  preview/live_viewer.py optional MJPEG/JSON/POST debug server
  ui/status.py, ui/led.py, ui/messages.py
  system/watchdog.py, health.py, component_health.py
  config/schema.py, config/loader.py
deploy/ idcam.service, config.yaml.template, environment, install.sh (wrapper)
scripts/ install.sh (796 lines), uninstall.sh, camera/geometry diagnostics, ws2812_test.py
docs/    ARCHITECTURE.md, API_PROTOCOL.md, SECURITY.md, TROUBLESHOOTING.md
tests/   22 unit modules + 1 hardware integration module
tmp-test-capture.jpg   1.1 MB, 9152×6944, JFIF, no EXIF (see §8)
```

---

## 5. End-to-End Execution Flow

| # | Stage | Trigger | Function (file:line) | Thread | Notes / errors |
|---|---|---|---|---|---|
| 1 | Startup | `idcam` CLI | `main()` `app/main.py:17`; `_startup` `orchestrator.py:160` | main | Config errors exit code 2. LED start, stale temp cleanup (`:167`), disk check, `api.start()`, HEAD connectivity check (`:175`). |
| 2 | Permission | n/a | no runtime permission prompt; Linux device access comes from the service user/groups | — | — |
| 3 | Camera init | `_startup` | `CameraManager.initialize` `manager.py:153` via `to_thread` | worker | `Picamera2()`, preview config 1920×1080 RGB888 ×4 buffers, still config 9152×6944 RGB888 ×2 buffers, start, request Continuous AF (Fast). Failure → RECOVERY. |
| 4 | Preview | every loop tick | `_safe_preview` `orchestrator.py:316` → `capture_preview_array` `manager.py:218` | worker | `capture_array` copy, then a second full-frame channel-reversal copy (`:113-124`). |
| 5 | Detection | IDLE tick | `_idle_tick` `:346` → `DocumentDetector.detect` `document_detector.py:380` → `detect_document_geometry` | **event loop (blocking)** | needs `detect_frames_required=3` consecutive confident frames. |
| 6 | Lock | DOCUMENT_DETECTED | `_document_detected_tick` `:518` → `lock_document` | loop | uses cached geometry; falls back to a fixed 15% margin quad when none (`document_detector.py:164-190`). |
| 7 | Stability | WAITING_FOR_STABILITY | `_stability_tick` `:533` | loop | 8 consecutive low-motion frames (`stable_frames_required`). Cheap gray diff, not the geometry detector. |
| 8 | Autofocus | FOCUSING | `_focusing_tick` `:552` → `AutofocusController.run_autofocus` `autofocus.py:137` | mixed | see §6. |
| 9 | Pre-capture check | CAPTURING | `_check_motion_abort` `:325` → `structural_motion_in_lock` | **loop (blocking)** | runs the full geometry detector again. |
| 10 | Capture | CAPTURING | `CaptureService.capture` `capture.py:28` → `capture_still_to_file` `manager.py:232` | worker, 45 s timeout | mode switch, JPEG encode, atomic rename. |
| 11 | Validate | IMAGE_VALIDATION | `_validation_tick` `:637` → `ImageValidator.validate` `quality.py:40` | **loop (blocking)** | sha256 + 3 decodes + stats. |
| 12 | Upload | UPLOADING | `_uploading_tick` `:675` → `ApiClient.upload_document` `api_client.py:56` | loop (async I/O) | single POST. |
| 13 | Response | WAITING_FOR_SERVER → PROCESSING_RESPONSE | `:713`, `:718` | loop | WAITING_FOR_SERVER does nothing but transition. |
| 14 | Cleanup | success or any failure | `temp_store.cleanup(...)` at `:704,710,734,750,672,627`; `_cleanup_transaction_files` `:948` | loop | file is unlinked in all paths. |
| 15 | Removal | SUCCESS (2 s blocking sleep `:760`) → WAITING_FOR_REMOVAL | `_removal_tick` `:765` → `is_scene_empty` (geometry detector again) | loop | needs 10 empty frames. |

Avoidable work found: every state transition costs one extra loop iteration (sleep `1/preview_fps` = 200 ms, plus watchdog and a `statvfs` per iteration); the DOCUMENT_DETECTED and WAITING_FOR_SERVER states do no useful work; the success state blocks the loop for 2 s; the geometry detector runs once per tick in four different places; the preview is captured again at the start of each of DOCUMENT_DETECTED, FOCUSING, CAPTURING and again twice inside autofocus.

---

## 6. Camera Initialization and Autofocus Analysis

**API:** Picamera2/libcamera. Preview config `create_preview_configuration(main 1920×1080 RGB888, buffer_count=4)`; still config `create_still_configuration(main 9152×6944 RGB888, buffer_count=2)` (`manager.py:164-174`). RGB888 at 64MP is 190.7 MB per buffer, so the still config reserves about 381 MB of camera buffer memory (computed; CMA behaviour *needs verification*). Lens/camera selection: `Picamera2()` with no index — first camera only; no lens/zoom/torch/stabilisation control anywhere.

**AF mode:** Continuous (Fast) is requested at start and after every capture (`manager.py:187,268,490-515`), falling back to Auto+Trigger if the camera does not advertise Continuous. During FOCUSING:

1. `apply_document_focus_window` maps the locked document's inner ROI to sensor coordinates via live `ScalerCrop` and sets `AfMetering=Windows`, `AfWindows=[roi]`, `AfRange=Full` (`manager.py:325-362`). AF windows are therefore only on the document **after** detection+stability, not during the earlier preview.
2. If the preview Tenengrad score already passes, AF is skipped ("already_sharp").
3. Otherwise `trigger_autofocus` sets `AfMode=Auto`, `AfSpeed=Fast`, `AfRange=Full`, `AfTrigger=Start` (`:364-380`) and `wait_for_lens_settle` polls `LensPosition` every 0.1 s: fixed 0.25 s sleep + 3 consecutive readings within 0.05 → ≥ ~0.55 s, capped at `min(af_timeout_sec, 3.5)` (`:382-411`, `autofocus.py:409`).
4. Accept if Tenengrad improves ≥ 15% and ≥ 1.5, or stays ≥ floor 18 and ≥ 92% of before (`sharpness.py:85-103`); then `pause_autofocus` (`manager.py:419`).
5. At capture the lens position is read, the sensor is reconfigured to still mode, `AfMode=Manual` + that `LensPosition` is applied (`:241-256`).

**Focus state:** never read. `probe_capabilities` tests `"AfState" in camera.get_controls()` (`autofocus.py:117`), but `get_controls()` returns `camera_controls` (settable controls). `AfState` is per-frame *metadata* in libcamera, so this flag is probably always False and the code deliberately falls back to lens-position settling (comments at `manager.py:370-372`, `:382`). *Likely; verify on device by printing `capture_metadata()["AfState"]`.*

### Findings

**CAM-012 — "already_sharp" test is tautological, so AF is skipped whenever the preview score ≥ 18**
Severity Medium · Confidence Confirmed (logic), effect Needs Verification · `autofocus.py:177-190`, `sharpness.py:85-103`
Evidence: the call passes `before=max(before*0.9, 1.0), after=before`. The ratio is always 1.11 (<1.15), so acceptance falls to the second rule `after ≥ floor and after ≥ before_arg*0.92`, which is true for any `before ≥ 18` (`QualityConfig.preview_min_sharpness`, `schema.py:104`). The stage therefore reports success without any evidence the document region is focused if the preview merely contains texture.
Impact: a slightly defocused document can skip the AF cycle and fail only after a 64MP capture + validation (≥ 10 s lost, user re-presents the document).
Fix: compare the document-ROI score against a calibrated per-device absolute minimum and/or a short local-maximum search (see recommendation R2). Remove the synthetic `before*0.9`.
Verification: unit test with a defocused synthetic text frame (score between 18 and the calibrated floor) must trigger AF.

**CAM-013 — FOCUS_VERIFICATION is unreachable; no focus check immediately before the still**
Severity Low-Medium · Confirmed · `orchestrator.py:581` (FOCUSING → CAPTURING), `autofocus.py:253` (`verify_focus` never called outside tests), `state_machine.py:197-198`
Impact: lens is committed once, then a ~7 s (measured on audit host) geometry run (`_check_motion_abort`) and a mode switch happen before the exposure; nothing re-measures sharpness in between. `docs/ARCHITECTURE.md` still shows the state.
Fix: either delete the state or call a cheap (≤30 ms) ROI Tenengrad check right before `capture_still_to_file`.

### Prioritised autofocus recommendations

| Pri | Recommendation | Existing limitation | Files/functions | Proposal | Risk / regression test |
|---|---|---|---|---|---|
| R1 | Fix geometry blur (CAM-001) so focus/capture is not starved | one detector call ≈ 7.6 s on audit host | `document_geometry.py:143-146` | compute `bg` on a 1/8-scale copy (code in §14) | verify bounding boxes unchanged on the existing `test_document_geometry.py` set |
| R2 | Use real AF state | `AfState` unused, lens-settle heuristic costs ≥0.55 s and can mistake a stalled lens for "focused" | `manager.py:382-417`, `autofocus.py:117` | after setting Auto+Trigger, poll `capture_metadata()["AfState"]` (0 idle, 1 scanning, 2 focused, 3 failed) with the existing lens-settle path as fallback when the key is absent; or call Picamera2's `autofocus_cycle()` | must keep fallback for cameras that omit AfState; hardware test on 3 units |
| R3 | Avoid mode flip hunting | Continuous AF runs on the preview with default windows, then is replaced by one-shot Auto | `manager.py:490-515`, `autofocus.py:403` | apply the document `AfWindows` as soon as the lock exists (before stability wait) so Continuous AF converges while the 8 stability frames elapse; then `AfPause`/hold | measure time-to-focus before/after |
| R4 | Real focus gate | CAM-012/013 | `autofocus.py:177`, `orchestrator.py:581` | calibrated document-ROI floor + pre-capture re-check | unit + hardware |
| R5 | Low light | no exposure/gain/torch handling; brightness range only checked after capture (30–225) | `manager.py`, `quality.py:127-130` | read `ExposureTime`/`AnalogueGain` from metadata; if exposure is long, wait for a stable AE frame or raise `FrameDurationLimits`; do not recommend torch/zoom unless hardware exists | needs device testing; hardware-specific |
| R6 | Reduce refocus on motion | motion ≥ 2.5× threshold aborts the whole transaction (`orchestrator.py:333-343`) | `_check_motion_abort` | on motion, return to WAITING_FOR_STABILITY and re-run AF only | keep failure LED only for repeated failures |

Front-facing vs rear: there is a single CSI camera; no front/rear logic exists. Manual focus control: only the existing `LensPosition` hold is used; nothing unsupported is proposed.

---

## 7. Scanning and Image Capture Performance

**Model:** live preview stream (1080p) for detection/stability/focus; one still capture for the deliverable. Not every frame is analysed: loop is paced by `asyncio.sleep(1/preview_fps)` after each tick, so the effective rate is *below* 5 fps (frame grab + processing + 200 ms sleep).

**Detection method:** classical OpenCV: gray normalisation + CLAHE, ~19 segmentation variants (12 adaptive thresholds with 15×15 morphology, 2 background-subtract Otsu, 2 Otsu, morph-gradient, 2 Canny), contour scoring (`document_geometry.py:110-168,397-441`). No ML, no OCR, no barcode/MRZ reading anywhere.

**Threading:** `capture_array` and the still run in worker threads via `to_thread`, but detection (`orchestrator.py:366`), motion-in-lock (`:332`), removal check (`:778`) and validation (`:648`) run directly on the event-loop thread.

### Findings

**CAM-001 — Geometry detector dominated by a σ≈192 Gaussian blur**
Severity High · Confirmed (measured) · `document_geometry.py:143-146` (`sigma_bg = max(w, h) / 10.0`, `cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_bg, …)`)
Evidence: cProfile of one `detect_document_geometry` call on a synthetic 1920×1080 frame (OpenCV 5.0.0, 8 threads): total 7.59 s, `GaussianBlur` 6.95 s (2 calls; the small one is ~ms), `adaptiveThreshold` 0.17 s, `morphologyEx` 0.11 s. A drop-in replacement (INTER_AREA resize to 1/8, σ/8, linear resize back) took 21 ms with mean absolute difference 0.24 and max 2 grey levels against the original.
Impact: with the detector in the idle loop, the real cadence on the audit host would be seconds per frame, so "detect_frames_required=3" alone takes tens of seconds. **Pi 5 cost is unmeasured**; the kernel size is host-independent, so a large relative gain is expected but the absolute number must be measured on device.
Fix: §14 R1. Verification: run `tests/unit/test_document_geometry.py`; assert bbox IoU ≥ 0.98 between old and new on the synthetic fixtures; time on device.

**CAM-002 — CPU-bound vision and validation run on the asyncio event loop**
Severity High · Confirmed · `orchestrator.py:332,366,648,778`
Impact: while they run, the watchdog (`:224`), state timeouts (`:233`), SIGTERM handling (`:156`, which only sets an Event) and status updates stall. State timeouts such as `image_validation: 10` cannot interrupt a blocking call, so they only fire after the work completes.
Fix: wrap each call in `await asyncio.to_thread(...)` (the detector is not thread-safe across overlapping ticks, but `_state_tick_running` at `:285-289` already serialises ticks).
Verification: add a test that a 2 s blocking validator does not delay a concurrent watchdog tick.

**CAM-011 — Two sensor mode switches per capture under the camera lock; timeout cannot cancel the worker**
Severity Medium · Confirmed (code) · `manager.py:232-273`, `capture.py:34-37`
Evidence: `stop → configure(still) → start → set manual lens → capture_metadata → capture_file → finally stop → configure(preview) → start → re-enable AF`, all inside `with self._lock`. On `asyncio.TimeoutError` the coroutine is abandoned but `to_thread` keeps running and keeps the RLock; the orchestrator moves to failure handling while the thread may still be writing the `.partial` file that `cleanup` just unlinked (`capture.py:41-43`).
Impact: preview is blind during the whole switch; recovery paths (`camera.restart`) will block on the lock.
Fix: set a worker "cancel" flag and have the next camera operation wait on the thread future; consider a single camera configuration that streams `main` 1080p and `raw`/`lores` for the stills (Picamera2 `switch_mode_and_capture_file` also exists and removes one manual restart). *Hardware-dependent; needs measurement of the 64MP mode-switch time (unknown).*

**CAM-014 — Fixed scheduling latency**
Severity Low · Confirmed · `orchestrator.py:219-238,760`
Estimated minimum waits (deterministic from config): detection 3 frames + stability 8 frames at 5 fps ≈ 2.2 s + about 8 state transitions × 200 ms ≈ 1.6 s + 2 s blocking success sleep, before any real processing time. These are lower bounds from configuration, not measurements.
Fix: run tick handlers back-to-back when a handler transitions (skip the sleep), drop WAITING_FOR_SERVER/DOCUMENT_DETECTED as separate sleeps, and raise preview fps to 10–15 once the detector is cheap.

---

## 8. Image Processing and Compression

Transformations applied to the uploaded file: **none.** Grep for `warpPerspective|getPerspective|resize|crop|rotate|imencode|save(` finds no transformation of the capture. The file written by `capture_file(path, format="jpeg")` is renamed and uploaded as is.

| Property | Value | Evidence |
|---|---|---|
| Format | JPEG (encoder inferred from suffix; `.partial.jpg` is used so the suffix stays valid) | `manager.py:261`, `temp_files.py:39-45` |
| Dimensions | configured 9152×6944 | `schema.py` CameraConfig, `manager.py:168-174` |
| Pixel format at encode | RGB888 array → JPEG | `manager.py:170` |
| Quality | **not set by idCam.** `jpeg_quality=95` is only used by the mock (`manager.py:591`) and diagnostics; the real path leaves Picamera2's default (documented as 90; *verify on device*) | grep: no `options["quality"]` |
| EXIF / orientation | none added or corrected; no `transform` in the configs. Orientation of the final image depends on how the sensor is mounted — *Needs verification* | — |
| Colour profile / alpha | none / none (JPEG) | — |
| Thumbnails/variants | none | — |
| Original vs processed | single file; no second copy | — |

`tmp-test-capture.jpg` in the repo root is 9152×6944, 1.1 MB, JFIF only (no EXIF), 23 distinct grey values, mean ≈ 218. That matches the mock generator (`manager.py:577-591`), so it is **not** evidence of real camera output size or EXIF content; I did not render or open it for content.

**CAM-006 — `jpeg_quality` setting has no effect on real captures**
Severity Medium · Confirmed · `manager.py:261`, `config/schema.py:65`, `config.example.yaml`
Impact: operators believe quality is 95; file size, legibility margin and upload time are controlled by an undocumented library default.
Fix: before `start()` for the still config set `self._picam2.options["quality"] = self.config.jpeg_quality`, or pass the Picamera2-supported option for your installed version; log the effective value.
Verification: capture on device at 85/90/95 and compare size and ROI Laplacian.

**CAM-007 — Entire uncropped frame is uploaded**
Severity Medium (privacy/size) · Confirmed · no crop code; document quad is computed (`document_lock.py`) but used only for focus and sharpness ROIs
Impact: background, desk, other people's papers and any second document in the field of view are sent; every upload carries 64MP of mostly non-document pixels. Real sizes are unmeasured (file-size limits in config are 50 MB).
Fix: see recommended pipeline below.

**CAM-020 — BGR/RGB mismatch in still metrics**
Severity Low · Confirmed · `quality.py:74` (`cv2.imread` → BGR) fed to `sharpness.py:44,148` (`COLOR_RGB2GRAY`)
Impact: red and blue luminance weights are swapped for the brightness and sharpness thresholds. Minor numerical drift only.
Fix: decode directly as grayscale (see CAM-003).

### Recommended pipeline for identity documents

1. Keep capturing a full-quality JPEG (do not shrink below legibility); set quality explicitly (start at 90–92 and validate OCR/barcode readability on real samples).
2. Validate on a **reduced-resolution** decode (`PIL.Image.draft('L', (w//4, h//4))` or `cv2.IMREAD_REDUCED_GRAYSCALE_4`) for global sharpness/brightness; only the document ROI is decoded at full resolution.
3. Crop to the locked document quad with a margin and deskew once (single warp from the original), re-encode once. This removes background (data minimisation) and typically reduces size substantially — **not benchmarked here; measure on real captures** before committing to a target.
4. Preserve or strip metadata deliberately (currently none is written).

---

## 9. Upload Forensics: Exactly What Is Sent

All outbound runtime network code is in `network/api_client.py` (httpx). There are two request types and nothing else.

### 9.1 Upload paths

| Item | Detail | Status |
|---|---|---|
| Trigger | state UPLOADING after validation (`orchestrator.py:675`); also reachable from the debug viewer `POST /preview/capture` which only queues a capture through the same state machine | Confirmed |
| Call chain | `_uploading_tick` `:675` → `ApiClient.upload_document` `api_client.py:56` → `httpx.AsyncClient.post` `:91` | Confirmed |
| Destination | `str(server.url).rstrip("/") + server.path` (`api_client.py:69`); value comes from `/etc/idcam/config.yaml` or env `IDCAM_SERVER_URL` (`loader.py:107-122`); default path `/api/document`; repository only holds `https://unconfigured.invalid` / `https://api.example.com` | host: **Unknown without the deployed config** |
| Protocol | HTTP/1.1 via httpx POST; HTTPS not enforced by code (CAM-008) | Confirmed |
| Authentication | `Authorization: Bearer <token>` only if `auth.type=bearer` **and** the token is non-empty (`config/schema.py:46-52`, `api_client.py:75-77`); token from YAML or `IDCAM_AUTH_TOKEN` | Confirmed |
| Request headers | `X-Device-ID`, `X-Capture-ID`, `Idempotency-Key` (= capture id), optional `Authorization`; httpx supplies `Content-Type: multipart/form-data; boundary=…`, `Content-Length`/chunked, `Host`, `User-Agent: python-httpx/<ver>`, `Accept`, `Accept-Encoding`, `Connection` | idCam headers Confirmed; others library defaults |
| Body | one multipart part: name `image`, filename = temp file name `capture-<uuid4 hex>.jpg`, content type literal `image/jpeg` (`api_client.py:90`) | Confirmed |
| File source | the validated, finalised temp JPEG (`self._capture_path`), i.e. the original capture, not a copy | Confirmed |
| Images per request | exactly 1 | Confirmed |
| Redirects | `follow_redirects=False` (`:37`); a 3xx is parsed as non-2xx without exception and ends in an `upload_failed` after the body was already sent to the first URL | Confirmed |
| Progress / cancel | none; no progress callbacks, no cancellation except the state-machine failure path | Confirmed |
| Retries | designed: one connect-error retry; actual: none (CAM-004). No retry after the body is sent; no offline queue | Confirmed |
| Response handling | JSON object required to contain `success`; `capture_id` mismatch → ProtocolError; ≥400 raises; 2xx with malformed JSON or non-JSON body is treated as success (CAM-010) | Confirmed |
| After upload | local JPEG is unlinked on success (`orchestrator.py:734`) and on every failure branch | Confirmed |
| Other recipients | no SDKs, no webhooks, no additional services | Confirmed (grep) |

Second request: `HEAD {server.url}` at startup with 5 s timeout and **no custom headers and no auth** (`api_client.py:45-54`, called from `orchestrator.py:175`). It carries no image or identifier beyond httpx defaults.

### 9.2 Required data-flow table

| Data or Field | Source File or Function | Transformation | Destination | Transmission Condition | Evidence |
|---|---|---|---|---|---|
| Image bytes (JPEG, full frame, ~9152×6944) | `manager.py:261` `capture_file` → `temp_files.py:47` finalise | none after capture (rename + chmod only) | `server.url + server.path`, part `image` | after validation passes | Confirmed |
| Multipart filename `capture-<32hex>.jpg` | `temp_files.py:35-37` `uuid.uuid4().hex` | none | same | always | Confirmed |
| Part content type `image/jpeg` | `api_client.py:90` | literal | same | always | Confirmed |
| `X-Device-ID` | `config.device_id` via `UploadRequest.device_id` (`orchestrator.py:693`) | none | header | always | Confirmed |
| `X-Capture-ID` and `Idempotency-Key` | `ActiveCaptureTransaction.generate_id()` `transaction.py:61` (`cap-` + uuid4 hex) | none | headers | always | Confirmed; runtime-generated |
| `Authorization: Bearer …` | `AuthConfig.resolved_token()` | none | header | only if type=bearer and token non-empty | Confirmed |
| `UploadRequest.created_at` | `orchestrator.py:695` | UTC string | **not transmitted** — never read by `ApiClient` | — | Confirmed |
| SHA-256, width, height, sharpness, brightness | `quality.py` → transaction | — | **not transmitted** (local logs only, sha truncated to 16 chars in logs) | — | Confirmed |
| Document ROI/corners, OCR, barcodes, MRZ, selfie, video, thumbnails | not implemented | — | not transmitted | — | Confirmed absent |
| Location, device model, telemetry, OS info | not collected | — | not transmitted (except httpx `User-Agent`) | — | Confirmed absent |
| EXIF | none written by idCam; library behaviour for Picamera2 JPEG EXIF *Needs verification on device* | — | would travel inside the JPEG | — | Unknown |

### 9.3 Request reconstruction (synthetic values)

```
POST /api/document HTTP/1.1
Host: <server host from config>
Authorization: Bearer <redacted>              # only if configured
X-Device-ID: pi5-doc-capture-001
X-Capture-ID: cap-0f3c5d1a8b7e4c2d9a6b1e0f7c3d2a41
Idempotency-Key: cap-0f3c5d1a8b7e4c2d9a6b1e0f7c3d2a41
User-Agent: python-httpx/0.27.x
Content-Type: multipart/form-data; boundary=<generated>

--<boundary>
Content-Disposition: form-data; name="image"; filename="capture-9b2e...f1.jpg"
Content-Type: image/jpeg

<binary JPEG bytes>
--<boundary>--
```

Expected response (documented in `docs/API_PROTOCOL.md`, parsed by `protocol.py:39-99`):

```json
{ "success": true, "capture_id": "cap-0f3c…", "result": "…", "message": "…" }
```

No Base64, no JSON request body, no signed URLs, no two-step storage flow exists.

**Verification limit:** the above is the request as constructed by source. I made no network capture. What the backend stores, OCRs, or retains is **Unknown** — the server is not in this repository. Recommended controlled test: run `idcam --mock-camera` against a local mock HTTP server (synthetic image) and dump the raw request; `respx` is already a dev dependency and `tests/unit/test_api_client.py` shows the pattern.

---

## 10. File Paths, Local Storage, and Cleanup

| File Type | Creation Location | Read/Write Operations | Upload Usage | Deletion Policy | Risks |
|---|---|---|---|---|---|
| Capture partial `capture-<hex>.partial.jpg` | `TempFileStore.create_partial_path` `temp_files.py:39`, dir `temp_dir` default `/var/lib/idcam/tmp` | written by Picamera2 worker; fsync; `os.replace` | no | unlinked on capture error/timeout (`capture.py:39-54`); orphan swept only at next startup if older than `stale_temp_max_age_sec` | worker may still be writing after timeout cleanup (CAM-011) |
| Final capture `capture-<hex>.jpg` | `temp_files.py:47-55`, chmod 0640 (`:64-68`); temp dir chmod 0770 (`:28-33`) | read by validator (3 decodes) and by httpx | yes — the same file | unlink after success and after every failure (`orchestrator.py:627,672,704,710,734,750,948`) | plain `unlink` on SD flash storage is not secure erase; crash/power loss leaves the file until restart (CAM-015) |
| `status.json` + `.tmp` | `ui/status.py:641` in `/run/idcam` (tmpfs) | rewritten on every transition | no | removed with `RuntimeDirectory` on stop | contains `capture_id`, health, and `last_error` which can include **server-supplied text** |
| Logs `/var/log/idcam/idcam.log` | `logging/setup.py` rotating 10 MB×5 | append | no | rotation only | log lines contain image path (uuid name), 16-char SHA prefix, capture ids, device ids; no token, no image data |
| Debug JPEG frames | in memory only (`live_viewer.py`) | served over HTTP when enabled | no | — | exposes preview frames (CAM-005) |

Filenames are generated server-side by uuid, so there is no user-controlled path component and no traversal risk. Concurrency: exactly one transaction at a time (`_begin_transaction` raises otherwise, `orchestrator.py:409-411`); `_capture_path` is cleared in `_clear_transaction`. Backups: the temp directory is under `/var/lib/idcam`; whether a system backup picks it up is deployment-specific (*Unknown*).

**CAM-015 — Residual identity images on crash/power loss; stale sweep only at boot**
Severity Medium · Confirmed (code) · `orchestrator.py:167`, `temp_files.py:82-101`
Impact: an interrupted capture (SIGKILL, power cut, Pi hang) leaves a full identity image on the SD card until the next successful start and age > 1 h; deletion is not secure erase.
Fix: run `cleanup_stale` periodically (e.g. every 5 min in the main loop), shorten age to ~5 min, consider a tmpfs `temp_dir` (≈ 400 MB headroom needed; verify RAM) which also removes SD-card wear (the 64MP JPEG is written, fsynced and rewritten for every scan), and encrypt the data partition.
Verification: kill -9 test during CAPTURING and confirm the sweeper removes the file within the new age.

---

## 11. API Endpoints and External Integrations

| Endpoint | Purpose | Method | Base URL source | Auth | Timeouts | Retry | Called by reachable code? | Server code in repo? |
|---|---|---|---|---|---|---|---|---|
| `{server.url}{server.path}` (default `/api/document`) | upload document image | POST multipart | `server.url` in `/etc/idcam/config.yaml`, overridable by `IDCAM_SERVER_URL` | Bearer (optional) | connect 10 s, read/write 60 s, pool 10 s (`api_client.py:28-33`); `total_timeout_sec` (90) is **never used** (`schema.py:22`) | none in practice (CAM-004) | yes | **No** |
| `{server.url}` | startup connectivity | HEAD | same | none | 5 s | none | yes (`orchestrator.py:175`) | No |
| `http://<host>:8088/preview/live`, `/preview/live.mjpg`, `/preview/status`, `POST /preview/capture` | local debug UI | GET/POST (inbound server) | `debug.host/port` | none | n/a | n/a | only if `debug.enabled` | n/a |
| `https://github.com/ArduCAM/…/install_pivariety_pkgs.sh` | install-time driver script | wget | hard-coded in `scripts/install.sh:389` | none | — | — | installer only | n/a |

Hard-coded URLs in runtime code: none. Dev/mock endpoints: `--mock-camera`/`IDCAM_MOCK_CAMERA` select a mock camera (still uploads to the configured server). `admin_socket` is configured (`schema.py:175`) but no code binds it (`admin.py` is file-based).

What is *not* known: whether the backend stores the file, runs OCR/verification, deduplicates `Idempotency-Key`, or deletes it — backend absent.

---

## 12. Security and Privacy Assessment

**CAM-005 — Debug viewer: unauthenticated live frames and remote capture trigger**
Severity Medium by default (feature is off), High if enabled on a shared network · Confirmed · `live_viewer.py:410-433,444-483,577-593`, `schema.py:184-188`
Evidence: no authentication, no origin check; default `host: "0.0.0.0"`; `GET /preview/live.mjpg` streams the preview; `POST /preview/capture` calls `request_manual_capture`, which queues a real capture → validation → upload with the device's credentials. A browser page on the same LAN can issue the POST (simple POST, no preflight needed for a cross-site form).
Impact: anyone who can reach port 8088 can watch the capture area (including documents) and cause uploads; the manual path skips the stability wait (`orchestrator.py:527-530`).
Fix: bind `127.0.0.1` by default, require a random token (header or query) generated at startup, add `Host`/`Origin` checks, reject POST without a custom header, and refuse `0.0.0.0` unless an explicit auth token is set.
Verification: unit test that a request without the token gets 401 and a cross-origin POST is refused.

**CAM-008 — Transport and credential enforcement is weaker than documented**
Severity Medium · Confirmed · `schema.py:18,23` (`HttpUrl` accepts `http://`; `verify_tls: bool`), `loader.py:128-139` (production guard only rejects example hosts), `schema.py:46` (`type` defaults to `none`), `docs/SECURITY.md` ("never disabled")
Impact: a mis-edited config can upload identity images in cleartext or without certificate validation; `auth.type: bearer` with an empty token silently sends **no** Authorization header (`resolved_token()` returns `None`), so a freshly installed device can upload unauthenticated without any warning.
Fix: in production mode require `https`, require `verify_tls=True`, and fail startup when `type=bearer` and the token is empty. Log a warning (not the token) when auth is disabled.
Verification: config tests for each rejection.

**CAM-004 — Retry policy never retries; idempotency key does not protect re-scans**
Severity Medium · Confirmed (executed) · `retry_policy.py:17-21`, `api_client.py:24,151-157`, `docs/SECURITY.md`, `docs/API_PROTOCOL.md`
Evidence: `RetryPolicy().should_retry_connection(1)` returned `False` when executed; `ApiClient(config)` is built with the default policy (`orchestrator.py:83`). `tests/unit/test_retry_and_protocol.py:12-15` passes only because it constructs `max_connection_retries=2`. Additionally every re-scan gets a new `capture_id`, so the server-side dedupe the docs rely on only helps same-capture retries, which never happen.
Impact: a transient connect failure fails the scan; operators believe a retry occurs.
Fix: define `max_connection_attempts` (attempts, not retries) and test the default; for stronger delivery guarantees keep the validated JPEG until the HTTP result is known and re-send with the same id.
Verification: respx test where the first connect fails and the second succeeds with the default policy.

**CAM-010 — Success inferred from incomplete server responses**
Severity Medium · Confirmed · `api_client.py:109-110,115-116`
Impact: a 2xx with truncated JSON, an HTML error page behind a proxy, or a captive portal returning 200 is recorded as `success=True`; the local image is deleted and the user is told it worked.
Fix: require the documented `success` field (or an explicit header) on 2xx; treat anything else as `SENT_UNKNOWN`.

**CAM-009 — Upload duration is not bounded by `total_timeout_sec` or the state timeout**
Severity Medium · Confirmed · `schema.py:22` unused (grep), `orchestrator.py:237,698`
Evidence: the main loop awaits the whole tick; the upload runs inside the tick, so `_handle_state_timeout`/watchdog cannot run until httpx returns. httpx timeouts are per operation (idle write/read), so a server that accepts bytes slowly or trickles a response can hold the device far past 60 s.
Fix: wrap in `asyncio.wait_for(..., total_timeout_sec)` and honour `state_timeouts.uploading`; collapse WAITING_FOR_SERVER.

**CAM-021 — Installer executes an unpinned remote script as root from a predictable `/tmp` path**
Severity Medium (install-time) · Confirmed · `scripts/install.sh:389-398`
Impact: supply-chain/TOCTOU exposure (no checksum, `/tmp/install_pivariety_pkgs.sh` fixed name). Fix: pin the release by commit/tag + SHA-256, download into a `mktemp -d` directory, verify before execution.

**CAM-017 — Untrusted server text is written to logs and `status.json`**
Severity Low · Confirmed · `protocol.py:86`, `orchestrator.py:748-755`, `ui/status.py`
Fix: truncate (e.g. 200 chars) and strip control characters before storing.

Other checks:
- Secrets: none hard-coded in the repository; token comes from `IDCAM_AUTH_TOKEN`/YAML; `admin config` redacts the token (`admin.py:710`); installer sets `/etc/idcam/environment` to 0640 root:idcam. Tokens are not logged (grep of log calls: `Authorization`/token never interpolated).
- File permissions: capture 0640, temp dir 0770 owner+group; `RuntimeDirectoryMode=0755` makes `/run/idcam` listable by all while `UMask=0027` makes files 0640.
- Service hardening is good (`NoNewPrivileges`, `ProtectSystem=strict`, empty capability set, restricted address families).
- Dependencies: no lockfile and lower-bound-only pins, so supply-chain drift is possible; I did not run a vulnerability scanner (needs network).
- Excessive retention: no retention beyond transaction scope; no thumbnails/history.
- Debug flags: `debug.enabled` and `--mock-camera` default off; `IDCAM_ENV=production` set by the unit.

---

## 13. Bugs and Reliability Issues

Also listed under earlier sections: CAM-001, 002, 004, 005, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 017, 020, 021. Additional:

**CAM-003 — Validation decodes the image three times and `np.std` allocates ~1.5 GB**
Severity High · Confirmed (measured on audit host with the repo sample) · `quality.py:62-74,93-94,132`
Evidence (sample 9152×6944, a low-entropy synthetic — real photographic JPEG decode will be slower): `PIL verify` 10 ms, `PIL load` 176 ms, `cv2.imread` 228 ms, `Laplacian(CV_64F)` on the 70% ROI 572 ms, `mean(gray)` 128 ms, `np.std(frame)` over the whole 190.7 MB array **4,792 ms**. `np.std` on a uint8 array accumulates in float64, so a ~1.5 GB temporary is expected (computed from 190.65 M elements × 8 B; peak RSS not measured). The Laplacian also converts the ROI to float64 (~240 MB), done twice (global and document ROI). Everything is synchronous on the event loop (CAM-002), against a 10 s `image_validation` timeout.
Impact: seconds of dead time per scan, memory pressure and OOM risk on a 4 GB Pi (Needs Verification); a failure here discards the 64MP capture.
Fix: single grayscale decode (`cv2.IMREAD_GRAYSCALE`, or reduced decode for global checks), compute uniformity via `cv2.meanStdDev` on that gray image, use `CV_32F`/`CV_16S` for Laplacian, drop `Image.verify()` + `Image.load()` duplication, run in `to_thread`.
Verification: assert identical pass/fail on a corpus of sharp/blurry/dark/truncated JPEGs; measure time and peak RSS on the Pi.

**CAM-016 — Config/doc drift**
Severity Low · Confirmed. Schema defaults differ from shipped templates (`af_max_retries` 1 vs 3, `af_timeout_sec` 6 vs 10, `require_expected_dimensions` True vs false); `docs/ARCHITECTURE.md` state timeouts (DOCUMENT_DETECTED 30 s, FOCUSING 30 s, CAPTURING 60 s) differ from the config (5/10/45); `watchdog.check_interval_sec` and `state_timeout_sec` are never read (`watchdog.py` only stores `config`); `ui.admin_socket` unused; duplicate import `QualityConfig, QualityConfig` (`autofocus.py:10`); `ServerResponse.from_http` branch for 3xx yields SERVER_ERROR without an exception. Fix: single source of truth + a test that loads the template.

Other items examined and found acceptable: state-transition table is consistent, single-transaction guard works, manual-capture queueing uses a lock and event, SIGTERM path reaches `_shutdown`, camera is shut down in `finally`, LED thread is stopped. Items I could not evaluate without hardware: memory leaks in Picamera2 buffers, cross-device focus differences, EXIF/orientation of real JPEGs, recovery loop behaviour after a real camera fault.

### Findings ranked by severity

| ID | Sev | Conf | Title |
|---|---|---|---|
| CAM-001 | High | Confirmed (measured) | σ≈192 Gaussian blur dominates detector |
| CAM-002 | High | Confirmed | CPU-bound vision/validation on event loop |
| CAM-003 | High | Confirmed (measured) | Triple decode + whole-frame `np.std` in validator |
| CAM-005 | Medium (High if enabled) | Confirmed | Debug viewer unauthenticated, remote capture |
| CAM-004 | Medium | Confirmed (executed) | Retry never occurs |
| CAM-006 | Medium | Confirmed | `jpeg_quality` not applied |
| CAM-007 | Medium | Confirmed | Full uncropped frame uploaded |
| CAM-008 | Medium | Confirmed | Transport/auth enforcement gaps |
| CAM-009 | Medium | Confirmed | Upload not bounded by total timeout |
| CAM-010 | Medium | Confirmed | Success inferred from malformed 2xx |
| CAM-011 | Medium | Confirmed (code) | Mode switches + uncancellable capture thread |
| CAM-012 | Medium | Confirmed (logic) | "already_sharp" tautology |
| CAM-015 | Medium | Confirmed | Residual images after crash |
| CAM-021 | Medium | Confirmed | Unpinned root-executed installer download |
| CAM-013 | Low-Med | Confirmed | Dead FOCUS_VERIFICATION |
| CAM-014 | Low | Confirmed | Fixed scheduling latency |
| CAM-016 | Low | Confirmed | Config/doc drift, unused settings |
| CAM-017 | Low | Confirmed | Server text in logs/status |
| CAM-020 | Low | Confirmed | BGR/RGB swap in still metrics |

---

## 14. Performance Bottlenecks and Optimization Plan

| Stage | Current cost | Source | Fix | Expected effect | Effort / Risk |
|---|---|---|---|---|---|
| Detection (per tick) | 7.6 s measured (audit host, synthetic) | CAM-001 | downscaled background blur | blur 6.95 s → ~21 ms in isolation; remaining ~0.6 s of the call is other segmentation | Low / Low |
| Detector call count | up to 4 call sites per transaction | `orchestrator.py:332,366,778` | cache one geometry per frame; downscale whole detection to ~640 px width and scale corners back | further large reduction (unmeasured) | Medium / Medium |
| Event-loop blocking | all CPU work | CAM-002 | `to_thread` | keeps watchdog/timeouts live | Low / Low |
| Validation | ≈5.7 s + (real decode time) on audit host | CAM-003 | one gray decode, `meanStdDev`, reduced decode | seconds saved + ~1.5 GB temporary removed | Low / Low |
| Capture | two mode switches, 64MP RGB888 encode | CAM-011 | measure; consider `switch_mode_and_capture_file`, or capture only when the quad fills the frame | unknown until timed on Pi | Medium / Medium |
| Scheduling | ≥ ~5.8 s fixed waits (computed from config) | CAM-014 | tick chaining, fewer states, 2 s success sleep → async task | ≈ 3–4 s | Low / Low |
| AF | ≥0.55 s settle + tautological skip | CAM-012 | AfState, calibrated floor | variable | Medium / Medium |
| Upload size | full 64MP frame | CAM-007 | crop+deskew to quad, set quality explicitly | likely large; unmeasured | Medium / Medium |
| Upload latency | no total bound | CAM-009 | `wait_for` total timeout; stream file in a thread | bounded worst case | Low / Low |
| Disk | full JPEG written+fsynced+read 4× | CAM-015 | tmpfs temp dir, skip separate sha256 pass (hash only if the server needs it) | less SD wear | Low / Low |

Reference patch for R1 (do not apply blindly; tests required):

```python
# document_geometry.py — replace the large-sigma blur
sigma_bg = max(w, h) / 10.0
small = cv2.resize(gray, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma_bg / 8, sigmaY=sigma_bg / 8)
bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
```

Reference patch for CAM-003:

```python
gray = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)  # one decode
mean, std = cv2.meanStdDev(gray)
lap = cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_32F).var()
```

Risks to regression-test: threshold values (`min_sharpness=80`, `min_document_sharpness=40`) were tuned on float64 Laplacian over a BGR→gray conversion; changing the numeric type or colour conversion shifts scores slightly, so re-baseline them on real captures.

---

## 15. Testing and Benchmarking Results

### Executed

| Check | Command (no files written) | Result |
|---|---|---|
| Unit suite | `PYTHONDONTWRITEBYTECODE=1 python -I -m pytest tests/unit -q -p no:cacheprovider` | 277 passed. 280 tests are collected in total; the other 3 are in `tests/integration` (hardware-marked) and I did not run them |
| Retry default | `RetryPolicy().should_retry_connection(1)` | `False` |
| Validator step timings | in-memory timing on `tmp-test-capture.jpg` | sha256 4 ms; PIL verify 10 ms; PIL load 176 ms; `cv2.imread` 228 ms; Laplacian f64 ROI 572 ms; mean gray 128 ms; `np.std` whole frame 4,792 ms; PIL draft decode (1/8) 113 ms |
| Geometry detector | synthetic 1920×1080, 10 runs | median 7,708 ms (min 7,564, max 8,438); found=True, confidence 0.99; profile: GaussianBlur 6.95 s of 7.59 s |
| Alternative blur | same frame | 21 ms, mean abs diff 0.24, max 2 |

**Caveats:** audit host is Windows x86 with OpenCV 5.0.0 / 8 threads, not the Pi; the sample JPEG is a low-entropy synthetic (likely the mock), so real decode and encode times will differ; the geometry frame is synthetic. These numbers show where the time goes, not the Pi's absolute latency. No autofocus, capture or upload latency was measured.

### Existing test coverage vs gaps

Covered: state machine, config, API client with respx, protocol parsing, temp-file lifecycle, autofocus logic with fakes, document geometry on synthetic frames, LED, viewer, install script text.
Not covered: default retry behaviour (test builds a custom policy), 64MP validation time/memory, event-loop blocking, `jpeg_quality` on the real encoder, cancellation of a timed-out capture, debug-viewer authentication, TLS/http rejection, empty bearer token, malformed-2xx handling, crash-time cleanup.

### Benchmark plan (to run on the Pi 5 + Hawkeye)

Instrument with `time.perf_counter_ns()` at each existing `event=pipeline_stage` log line (already emitted) and record: camera startup→preview-ready; document-enters→detected (target after R1: report, don't assume); stability; AF trigger→lens settled and AfState; stable→capture start; `capture_still_to_file` split into stop/configure/start/encode/restore; validation time and peak RSS (`/usr/bin/time -v`, `tracemalloc`); encoded size at quality 85/90/95; upload duration under `tc netem` delay/loss/bandwidth caps against a local mock server; scenarios: bright/dim light, moving vs static, 3+ units, repeated 100 scans, cancel mid-upload, low memory (`systemd-run -p MemoryMax`). Keep measured baselines and targets in separate columns. Hardware tests exist under `tests/integration/test_hardware.py` (`IDC_RUN_HARDWARE_TESTS=1`).

---

## 16. Setup and Execution Instructions

Taken from `README.md`, `scripts/install.sh`, `deploy/idcam.service`:

- Target: Raspberry Pi 5, Arducam 64MP Hawkeye, Raspberry Pi OS Bookworm/Trixie 64-bit.
- Production: `sudo bash ./scripts/install.sh` (creates `idcam` user, venv at `/opt/idcam/venv` with `--system-site-packages`, config at `/etc/idcam/config.yaml`, env file `/etc/idcam/environment`, systemd unit). The service is intentionally not started until `device_id` (not `CHANGE_ME`), `server.url` (real HTTPS URL) and a token are set. Then `sudo systemctl enable --now idcam`; logs via `journalctl -u idcam -f`; status `cat /run/idcam/status.json`.
- Config keys: see `config.example.yaml`; env overrides `IDCAM_CONFIG`, `IDCAM_AUTH_TOKEN`, `IDCAM_DEVICE_ID`, `IDCAM_SERVER_URL`, `IDCAM_ENV`, `IDCAM_MOCK_CAMERA`.
- Dev (off-device): `pip install -e ".[dev]"` then `IDCAM_MOCK_CAMERA=1 idcam --config config.example.yaml`. The example config points at `https://api.example.com` and a `/var/...` temp dir, so a local `temp_dir`, `log_dir`, `ui.status_file` and a mock server URL are needed on a PC (I did not run the app).
- Tests: `pytest tests/ -v`; hardware: `IDC_RUN_HARDWARE_TESTS=1 pytest tests/ -v -m hardware`.
- Known blockers: placeholders rejected at startup by design; no lockfile; no backend in the repo; Picamera2/libcamera come from apt, not pip.

---

## 17. Current Implementation Status

| Feature | Status |
|---|---|
| State machine, single-transaction guard, shutdown | Implemented and unit-tested |
| Document detection (classical CV) | Implemented, unit-tested on synthetic frames; real-world speed **unverified**, very slow on audit host (CAM-001) |
| Stability gating | Implemented |
| Autofocus | Implemented via lens-settle heuristic; AfState path unused; acceptance logic has gaps (CAM-012) |
| FOCUS_VERIFICATION state | Defined but unreachable (CAM-013) |
| 64MP still capture + atomic save | Implemented; real timing unverified |
| JPEG quality setting | Declared, not applied (CAM-006) |
| Cropping / deskew / resize / OCR / barcode | Not implemented |
| Upload (multipart) | Implemented |
| Connection retry | Declared, non-functional (CAM-004) |
| Offline queue | Not implemented (by design) |
| Admin socket | Config only, not implemented |
| Debug viewer, LED strip | Implemented, optional; viewer unauthenticated |
| Backend | Absent from repo |

---

## 18. Prioritised Remediation Roadmap

| Order | Action | Benefit | Effort | Risk |
|---|---|---|---|---|
| 1 | CAM-001 downscaled background blur | removes ~90% of detector time (measured on host) | S | Low |
| 2 | CAM-002 move detector/validator to `to_thread` | liveness, timeouts, shutdown | S | Low |
| 3 | CAM-003 single gray decode + `meanStdDev` | seconds + ~1.5 GB temp | S | Low (re-baseline thresholds) |
| 4 | CAM-004/009/010 fix retry, total timeout, strict success | correctness of upload semantics | S | Low |
| 5 | CAM-008 enforce HTTPS/verify/token in production | privacy | S | Low |
| 6 | CAM-005 secure or remove debug viewer POST | closes remote trigger | S | Low |
| 7 | CAM-006 set JPEG quality explicitly | predictable size/quality | S | Low (verify on device) |
| 8 | CAM-012/013, R2/R3 AF improvements | focus reliability, time | M | Med (hardware) |
| 9 | CAM-007 crop+deskew before upload | data minimisation, size | M | Med (legibility tests) |
| 10 | CAM-011/014 capture switching and scheduling | latency | M | Med |
| 11 | CAM-015 periodic sweep / tmpfs | retention | S | Low |
| 12 | CAM-021 pin installer download | supply chain | S | Low |

---

## 19. Final Conclusions — direct answers

1. **What does ID Cam do?** A Raspberry Pi 5 kiosk service: detect a document, wait for stillness, autofocus, capture a 64MP JPEG, validate, upload one image to a configured HTTPS API, delete it, wait for removal.
2. **Flow from camera open to result:** §5 — startup → camera init (preview + AF) → idle detection (3 frames) → lock → stability (8 frames) → autofocus → motion check → mode switch and still capture → validation → single POST → response parse → delete file → SUCCESS → removal wait → IDLE.
3. **What is uploaded:** one multipart part `image` (the whole captured JPEG, filename `capture-<hex>.jpg`, `image/jpeg`) plus headers `X-Device-ID`, `X-Capture-ID`, `Idempotency-Key`, optional Bearer token. Nothing else per source (`created_at`, hash, size, ROI are not sent). Exact bytes at the server: not verified.
4. **Who receives it:** the host in `server.url` of the deployed `/etc/idcam/config.yaml` (not in the repo) at `server.path`; a HEAD goes to the same base URL at startup. No other service in code.
5. **Where images live:** `/var/lib/idcam/tmp/capture-<hex>.partial.jpg` then `capture-<hex>.jpg` (0640). No separate processed copy exists.
6. **Retention after upload:** the file is unlinked after success and on every failure path; residue is possible after crashes until the next-boot sweep; backend retention is unknown.
7. **Autofocus bottleneck:** weak focus decision logic (CAM-012) combined with a ≥0.55 s lens-settle heuristic instead of AfState, and AF windows applied only after detection and stability; the 7.6 s detection cost (CAM-001) delays everything before it on the measured host.
8. **Scanning/capture bottleneck:** the geometry detector's large-σ blur (CAM-001, measured), then validation's triple decode and whole-frame `np.std` (CAM-003, measured); on the camera side, two sensor mode switches (CAM-011, unmeasured).
9. **Upload bottleneck:** sending the full uncropped 64MP frame (CAM-007) with no total timeout or retry (CAM-004/009). Actual upload time depends on file size and link; not measured.
10. **Top five bugs/risks:** CAM-003, CAM-001/002, CAM-004, CAM-005/008, CAM-012.
11. **Top five performance gains:** CAM-001 blur fix; CAM-003 validation rewrite; offload to threads (CAM-002); crop+deskew+explicit quality (CAM-006/007); removing fixed scheduling waits and the redundant detector calls (CAM-014).
12. **Unknown without runtime/backend:** Pi 5 timings for every stage; real JPEG size and EXIF; Picamera2 default JPEG quality on the installed version; whether AfState is present in metadata on the deployed camera; memory peaks; mode-switch duration; the deployed server host, its storage/OCR/retention behaviour and idempotency handling; whether the sensor orientation needs a transform.
