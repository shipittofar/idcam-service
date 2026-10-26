# idCam — System Architecture

Production document-capture application for Raspberry Pi 5 + Arducam 64MP Hawkeye.

## A. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ApplicationOrchestrator                      │
│              (single authority, asyncio event loop)                │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ StateMachine │  │  Watchdog    │  │  StatusPublisher (UI)  │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
└───────────┬─────────────────┬──────────────────┬────────────────┘
            │                 │                  │
    ┌───────▼───────┐ ┌───────▼───────┐  ┌──────▼──────┐
    │ CameraManager │ │ VisionPipeline│  │  ApiClient  │
    │  (Picamera2)  │ │ detect/stab/  │  │   (httpx)   │
    │               │ │   sharpness   │  │             │
    └───────────────┘ └───────────────┘  └─────────────┘
            │                 │                  │
    ┌───────▼───────┐ ┌───────▼───────┐  ┌──────▼──────┐
    │ Autofocus     │ │ ImageValidator│  │ HealthMonitor│
    │ Capture       │ │ TempFileStore │  │ Config       │
    └───────────────┘ └───────────────┘  └─────────────┘
```

**Principles:**
- Modular monolith on a single Pi process
- One orchestrator owns state, one active capture transaction
- Camera accessed only through `CameraManager` (thread-safe via lock)
- Network I/O is async; camera I/O runs in a dedicated executor
- No job queue, no background workers beyond the main loop

## B. State Machine

```
BOOT
  → INITIALIZING (load config, init logging)
  → CAMERA_INITIALIZING
  → CAMERA_READY (camera verified, preview running)
  → IDLE ("Ready — Place document")

Document workflow (single transaction):
  IDLE
  → DOCUMENT_DETECTED
  → WAITING_FOR_STABILITY
  → FOCUSING
  → FOCUS_VERIFICATION
  → CAPTURING
  → IMAGE_VALIDATION
  → UPLOADING
  → WAITING_FOR_SERVER
  → PROCESSING_RESPONSE
  → SUCCESS
  → WAITING_FOR_REMOVAL ("Please remove document")
  → IDLE

Failure paths:
  Any operational state → RECOVERY (bounded attempts) → CAMERA_READY or ERROR
  ERROR → RECOVERY (if recoverable) or SHUTDOWN
  SHUTDOWN (SIGTERM/SIGINT)
```

**Per-state contract:**

| State | Timeout | Success | Failure |
|-------|---------|---------|---------|
| INITIALIZING | 60s | CAMERA_INITIALIZING | ERROR |
| CAMERA_INITIALIZING | 120s | CAMERA_READY | RECOVERY |
| IDLE | ∞ | DOCUMENT_DETECTED | — |
| DOCUMENT_DETECTED | 30s | WAITING_FOR_STABILITY | IDLE |
| WAITING_FOR_STABILITY | 15s | FOCUSING | IDLE |
| FOCUSING | 30s | FOCUS_VERIFICATION | RECOVERY |
| FOCUS_VERIFICATION | 10s | CAPTURING | FOCUSING (retry) / RECOVERY |
| CAPTURING | 60s | IMAGE_VALIDATION | RECOVERY |
| IMAGE_VALIDATION | 10s | UPLOADING | RECOVERY |
| UPLOADING | configurable | WAITING_FOR_SERVER | FAIL transaction |
| WAITING_FOR_SERVER | configurable | PROCESSING_RESPONSE | FAIL transaction |
| PROCESSING_RESPONSE | 10s | SUCCESS | FAIL transaction |
| SUCCESS | 5s | WAITING_FOR_REMOVAL | — |
| WAITING_FOR_REMOVAL | 120s | IDLE | IDLE (timeout) |
| RECOVERY | 180s | CAMERA_READY | ERROR |
| ERROR | — | RECOVERY (manual/auto) | SHUTDOWN |

## C. Component Responsibilities

| Module | Responsibility |
|--------|----------------|
| `app/orchestrator.py` | Main loop, state transitions, transaction lifecycle |
| `app/state_machine.py` | State enum, transition rules, timeout tracking |
| `camera/manager.py` | Picamera2 lifecycle, preview, capability probe |
| `camera/autofocus.py` | AF trigger, state polling, lens-position fallback |
| `camera/capture.py` | Full-res still capture to temp file |
| `camera/quality.py` | Post-capture validation (size, decode, dims) |
| `vision/document_detector.py` | Scene change / document presence in ROI |
| `vision/stability_detector.py` | Motion metric over consecutive frames |
| `preview/live_viewer.py` | Optional debug MJPEG observer of the latest preview frame |
| `vision/sharpness.py` | Laplacian variance on document region |
| `network/api_client.py` | HTTPS upload, timeouts, error classification |
| `network/protocol.py` | Request/response models, idempotency |
| `network/retry_policy.py` | Safe retry rules (connection-only, bounded) |
| `config/schema.py` | Pydantic validation |
| `config/loader.py` | Load YAML, merge defaults |
| `ui/status.py` | User messages, status file, admin diagnostics |
| `storage/temp_files.py` | Atomic write, cleanup |
| `system/health.py` | Component health aggregation |
| `system/watchdog.py` | Stuck-state detection |
| `logging/setup.py` | Structured JSON logs, rotation |

## D. Data Flow

1. **Preview loop** (IDLE): 5–10 FPS downscaled frame → document detector (optional debug viewer observes this same frame; it never captures independently)
2. **Detection**: frame diff + ROI edge density exceeds threshold → DOCUMENT_DETECTED
3. **Stability**: N consecutive frames with motion < threshold → FOCUSING
4. **Focus**: trigger `autofocus_cycle()` or AfTrigger; poll AfState; verify Laplacian variance
5. **Capture**: switch still config (9152×6944 or configured), JPEG to `.tmp`, atomic rename
6. **Validate**: file size, PIL decode, dimensions, min sharpness, brightness range
7. **Upload**: multipart POST with `X-Capture-ID`, `Idempotency-Key`, `X-Device-ID`
8. **Response**: parse JSON, update health, show user message
9. **Removal wait**: scene must return to baseline before IDLE

## E. Failure / Recovery Strategy

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Camera init fail | Exception / timeout | Stop camera, wait 5s, reinit (max 3) |
| Camera lost mid-op | Capture/AF exception | RECOVERY → reinit |
| AF fail | AfState timeout / low sharpness | Retry AF (max 3), then fail transaction |
| Blur capture | Sharpness below threshold | Fail transaction, no upload |
| Upload network error | ConnectionError, timeout | Fail transaction (no queue); user notified |
| HTTP 5xx | Status code | Fail transaction; safe to retry manually |
| HTTP 4xx | Status code | Fail transaction; log, no retry |
| Ambiguous upload | Connection dropped after send | Idempotency key; server must dedupe |
| Stuck state | Watchdog timeout | Force RECOVERY |
| Disk full | Write error | ERROR, admin alert |
| Invalid config | Startup validation | ERROR, refuse to start |

**Pi reboot**: last resort only after 3 consecutive camera recovery failures.

## F. Camera Integration Strategy

**Stack:** Arducam driver + libcamera + Picamera2 (Python).

**Assumptions (verified at runtime):**
- `dtoverlay=arducam-64mp` in `/boot/firmware/config.txt`
- Picamera2 available after Arducam package install
- AF controls (`AfMode`, `AfTrigger`, `AfState`, `LensPosition`) may or may not be advertised depending on driver version — **probed at startup**

**Preview:** 1280×720 or 1920×1080 RGB, buffer_count=4, low CPU.

**Capture:** Full sensor resolution JPEG via `capture_file()` still configuration.

**Autofocus workflow:**
1. Set `AfMode` to Auto (1) if advertised
2. Call `autofocus_cycle()` or `AfTrigger: Start`
3. Poll `AfState` until `Focused` (2) or timeout
4. Verify sharpness on preview ROI (Laplacian variance ≥ threshold)
5. If AF controls unavailable: log warning, use sharpness-only verification after capture at fixed lens position

**PDAF note:** Set `LIBCAMERA_NOTPARSED_EMBEDDED_DATA=1` in systemd unit if PDAF causes issues (documented in install).

**Memory:** Never hold full 64MP frame in RAM; stream to disk.

## G. Server Protocol

```
POST {server.url}{server.path}
Headers:
  Authorization: Bearer {token}          # optional
  Content-Type: multipart/form-data
  X-Device-ID: {device.id}
  X-Capture-ID: {capture_id}
  Idempotency-Key: {capture_id}
Body:
  image: JPEG file

Response 200:
{
  "success": true,
  "capture_id": "...",
  "result": "...",
  "message": "..."
}

Response 4xx/5xx:
{
  "success": false,
  "error": "...",
  "message": "..."
}
```

**Semantics (documented, not magic):**
- **At-most-once** from client perspective on ambiguous failure (no auto-retry after body sent)
- **Effectively-once** when server honors `Idempotency-Key`
- Client generates `capture_id` before upload; same ID on any safe connection retry

**Upload outcome classification:**
- `NEVER_SENT` — connection failed before request completed
- `SENT_UNKNOWN` — sent but response lost (treat as failure locally; server may have processed)
- `REJECTED` — 4xx
- `ACCEPTED` — 2xx with valid JSON
- `SERVER_ERROR` — 5xx

## H. Configuration Design

YAML at `/etc/idcam/config.yaml` (override via `IDCAM_CONFIG` env).

Validated with Pydantic on startup. Invalid config → process exit code 2 (systemd does not restart-loop).

See `config.example.yaml` (development) and `deploy/config.yaml.template` (first production install).

Secrets: `IDCAM_AUTH_TOKEN` in `/etc/idcam/environment` (root:idcam 0640), or `auth.token` in config.yaml with the same mode. Production systemd sets `IDCAM_ENV=production` so `example.com` endpoints are rejected.

## I. Directory Structure

```
idCam/
├── app/
│   ├── main.py
│   ├── orchestrator.py
│   └── state_machine.py
├── camera/
│   ├── manager.py
│   ├── autofocus.py
│   ├── capture.py
│   └── quality.py
├── vision/
│   ├── document_detector.py
│   ├── stability_detector.py
│   └── sharpness.py
├── preview/
│   └── live_viewer.py
├── network/
│   ├── api_client.py
│   ├── protocol.py
│   └── retry_policy.py
├── config/
│   ├── schema.py
│   └── loader.py
├── ui/
│   ├── status.py
│   └── messages.py
├── storage/
│   ├── temp_files.py
│   └── paths.py
├── system/
│   ├── health.py
│   └── watchdog.py
├── logging/
│   └── setup.py
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   ├── install.sh
│   └── uninstall.sh
├── deploy/
│   ├── idcam.service
│   ├── config.yaml.template
│   ├── environment
│   └── install.sh
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TROUBLESHOOTING.md
│   └── SECURITY.md
├── config.example.yaml
├── pyproject.toml
└── README.md
```

## J. Testing Strategy

**Unit tests (no hardware):**
- State machine transitions and timeouts
- Config validation (valid, missing, invalid)
- Document/stability detectors on synthetic frames
- Sharpness calculator
- Image validator with fixture JPEGs
- API client with `httpx` mock transport
- Retry policy boundaries
- Temp file atomic writes

**Integration tests (Pi hardware, marked `@pytest.mark.hardware`):**
- Camera init and preview
- AF cycle completion
- Full-res capture
- End-to-end with mock server

**Failure scenario coverage:** See test modules mapping to requirements 1–30 in `tests/unit/test_failure_scenarios.py`.
