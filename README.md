# idCam

Production document capture for **Raspberry Pi 5** + **Arducam 64MP Hawkeye** autofocus camera.

Continuously monitors the capture area, detects documents, stabilizes, autofocuses, captures, validates, and uploads to a remote server. Designed for 24/7 unattended operation.

## Features

- Explicit deterministic state machine (no callback spaghetti)
- Single active capture transaction (no queue)
- Arducam 64MP / libcamera / Picamera2 integration
- Autofocus with sharpness verification
- Document detection and stability gating
- Atomic capture files with validation
- HTTPS upload with idempotency key
- Structured JSON logging with rotation
- systemd autostart with recovery
- Mock camera mode for development/testing

## Quick start

Try it on any machine (Python 3.11+) with the mock camera. No hardware or server is needed:

```bash
git clone https://github.com/shipittofar/idcam-service.git
cd idcam-service
pip install -e ".[dev]"
pytest tests/unit
IDCAM_MOCK_CAMERA=1 idcam --config config.example.yaml
```

Run from a plain folder on the Pi instead of the systemd service (one command; it
installs dependencies, builds `.venv`, copies settings from `/etc/idcam` if present):

```bash
git clone https://github.com/shipittofar/idcam-service.git && cd idcam-service && bash scripts/setup_local.sh --stop-service && ./run.sh
```

On a Raspberry Pi with the camera, run `sudo bash scripts/install.sh` and follow
[Production installation](#production-installation-on-raspberry-pi).

## Requirements

- Raspberry Pi 5
- Arducam 64MP Hawkeye (CSI, CAM1 port on Pi 5)
- Raspberry Pi OS Bookworm or Trixie (64-bit)
- Arducam driver stack (see Installation)

## Production installation on Raspberry Pi

On a fresh Raspberry Pi OS (**Bookworm or Trixie, 64-bit / arm64**) the intended
first-install flow is:

```bash
git clone https://github.com/shipittofar/idCamV3.git
cd idCamV3
sudo bash ./scripts/install.sh
```

The installer is the bootstrap. It installs OS packages, the `idcam` system user,
application venv, production config template, and systemd unit. **Do not**
pre-create `/etc/idcam`, `/var/lib/idcam`, `/var/log/idcam`, or `/run/idcam`, and
**do not** install Python/camera packages by hand to work around the installer.

The installer walks eight steps:

1. Checking supported OS (Raspberry Pi OS 64-bit, Bookworm/Trixie)
2. Checking package manager
3. Installing system dependencies (`python3-picamera2` / `python3-libcamera` from apt;
   numpy and OpenCV from PyPI wheels — not `libatlas-base-dev`, which Trixie removed)
4. Creating the `idcam` service user and directories
5. Installing the Python application into `/opt/idcam/venv` (`--system-site-packages`
   so Picamera2 comes from Raspberry Pi OS, not pip)
6. Installing configuration and the systemd unit
7. Validating camera/application environment
8. Final installation validation

It creates the `idcam` user/group (not root), writes a first-boot config **only if**
`/etc/idcam/config.yaml` does not already exist, installs
`/etc/systemd/system/idcam.service`, reloads systemd, and enables the unit. It does
**not** start the service until `device_id` and `server.url` are real values (the
shipped template uses `CHANGE_ME` / `https://unconfigured.invalid` and will fail
validation on purpose). Duplicate APT-source warnings on the Pi are ignored; the
installer does not rewrite `/etc/apt`.

If a **required** package has no installation candidate, the installer stops with
that package name and does not continue as if install succeeded.

Site-specific values the installer cannot guess — set these after the first run:

```bash
sudo nano /etc/idcam/config.yaml
```

Set at least:

- `device_id` — unique id for this Pi (not `CHANGE_ME`)
- `server.url` — real HTTPS API base URL (not `example.com` / `unconfigured.invalid`)
- `server.path` — upload path (default `/api/document`)

Set the auth token in the environment file (preferred; never commit secrets):

```bash
sudo nano /etc/idcam/environment
```

```bash
IDCAM_AUTH_TOKEN=your-token-here
```

Optional overrides in the same file: `IDCAM_DEVICE_ID`, `IDCAM_SERVER_URL`.

Then enable and start:

```bash
sudo systemctl enable --now idcam
```

Verify:

```bash
sudo systemctl status idcam
journalctl -u idcam -f
cat /run/idcam/status.json
```

### Upgrade

Existing config, token, captured data, and logs are preserved.

```bash
cd idCamV3
git pull
sudo bash ./scripts/install.sh
sudo systemctl restart idcam   # if the service was not already running
```

### Uninstall

```bash
sudo ./scripts/uninstall.sh              # keeps /etc/idcam, /var/lib/idcam, /var/log/idcam
sudo ./scripts/uninstall.sh --purge      # type DELETE to also remove config/data/logs
```

## Development (off-device)

```bash
pip install -e ".[dev]"
IDCAM_MOCK_CAMERA=1 idcam --config config.example.yaml
```

## Configuration

See `config.example.yaml`. Key settings:

| Setting | Description |
|---------|-------------|
| `device_id` | Unique device identifier |
| `server.url` | HTTPS API base URL |
| `server.path` | Upload endpoint path |
| `auth.token` | Bearer token (prefer `IDCAM_AUTH_TOKEN` env) |
| `camera.capture_width/height` | Full-res capture (default 9152×6944) |
| `quality.min_sharpness` | Laplacian variance threshold |

Environment overrides: `IDCAM_CONFIG`, `IDCAM_AUTH_TOKEN`, `IDCAM_DEVICE_ID`, `IDCAM_SERVER_URL`, `IDCAM_ENV`, `IDCAM_MOCK_CAMERA`.

## Pages held in a hand

A page held in the fingers is found with the hand cut out: when a hand-sized amount of skin colour is in view, extra segmentation variants without the skin pixels are tried, and a candidate that swallowed the hand ranks below one that did not (its own confidence is unchanged, and a hand-free candidate needs `detection.hand_confidence_relax` less). While skin is next to the page its stability is judged by how far the corners move between frames (`detection.hand_corner_motion_px`, default 10 px), because the picture-difference measure cannot tell a tremor from a real move. Limits: dark red and brown covers are close to skin colour (a luma floor keeps burgundy out, tan/orange pages are not separated from a hand by colour), and a hand covering two sides of a page lowers its confidence below the threshold.

`detection.geometry_touch_penalty_vertical` (default 0.8) eases the penalty for a page touching the top or bottom edge of the frame; with `background_check` a value up to 0.9 is safe.

## Pages in corners, at the edges, or held in a hand

With `detection.background_check: true` the detector also segments by *what is new against the learned empty desk* (`bgdiff` in the log): it needs no edges and no local contrast, so a page that is blurred, dim or in a vignetted corner is found as plainly as one in the middle, and a held page is found by cutting the warm pixels of the hand out of that mask (`bgdiff_noskin`). Candidates that are only the learned background are dropped inside the candidate search, so a static object that scores higher can no longer hide the page. The static-scene gate counts the few thumbnail pixels a dim page changes instead of averaging them away. The background is refreshed only while the picture is still close to it (a page or hand lying there undetected is never learned); a picture that stays different for ~100 s is taken for the new normal (the light changed). Without `background_check` these paths are off.

## False detections (something always in view)

`detection.background_check: true` makes the detector learn what the empty scene looks like (the first ~2 s after a start, once the picture is calm) and ignore a "document" whose area still looks like that background: furniture, a mat, cables, shadows that are always there. The idle log line shows `reason=static_background` for those. It keeps refreshing the background while nothing is detected (follows slow light changes). Limits: a document already lying there when the service starts counts as background until it is taken away and put back; a very large always-present object can hide a small document placed on it; if the picture never gets calm the check is skipped (`background_check_skipped`). Together with `detect_min_iou` (the same outline in consecutive frames) and `min_geometry_confidence`. Off by default.

## Exposure and white balance

Everything is automatic by default. Under `camera:` you can set `exposure_value` (brightness compensation in stops), `ae_exposure_mode: short` (shorter exposures, less motion blur), `awb_mode`, or fix the exposure with `exposure_time_us` (+ `analogue_gain`), which switches auto exposure off for the preview and the still. A fixed exposure only makes sense under constant light: first read `exposure_us` / `gain` in the `focus_sample` and `still_capture_metadata` log lines. Controls the camera does not advertise are dropped with a warning (`image_controls_unsupported`).

## Live Debug Viewer

Optional development viewer. **Disabled by default** (`debug.enabled: false`). It is not required for capture.

When enabled in `/etc/idcam/config.yaml`, the existing preview pipeline publishes the latest frame to a debug HTTP observer (no second camera):

```yaml
debug:
  enabled: true
  host: 0.0.0.0
  port: 8088
  fps: 5.0
  jpeg_quality: 70
  max_width: 1280
  overlay: true
```

This exposes a live camera image over HTTP **without authentication**. Do not enable on untrusted networks. Restart the service after changing config:

```bash
sudo systemctl restart idcam
hostname -I
```

Then open `http://<PI-IP>:8088/preview/live` (MJPEG: `/preview/live.mjpg`, JSON: `/preview/status`). Auth tokens are never shown.

### Manual focus mode

Set `camera.focus_mode: manual` (and `debug.enabled: true`) to switch autofocus off. The panel then shows a **Manual focus** card: move the lens slider (0-32 dioptres) while watching the live image and the sharpness value, then press **Confirm & save**. **Find best focus** runs the automatic search once on the document in view and moves the slider to the sharpest position (it does not save; check the preview, then confirm). The value is written to `manual_focus.json` (default: next to `temp_dir`, or `camera.manual_focus_file`) and used for every later capture and restart. Until a value is saved, `camera.manual_lens_position` is used. The panel endpoint (`/preview/focus`) is unauthenticated like the rest of the viewer, only works in manual mode, accepts JSON only and is refused while a capture is running. A soft image in manual mode is logged (`manual_focus_soft`) but never blocks the capture.

## Status LED strip (optional)

A WS2812B strip on a Raspberry Pi 5 GPIO can show the device state. It is off
by default and never blocks capture: if the libraries or hardware are missing,
idCam logs `event=led_unavailable` and carries on.

1. Wire the strip data line to GPIO18 (physical pin 12), with a common ground.
2. Install with the LED libraries: `sudo ./scripts/install.sh --with-led`
   (adds the `gpio` group to the service user so it can open `/dev/pio0`).
3. Check the strip with the service stopped:
   `sudo systemctl stop idcam && /opt/idcam/venv/bin/python scripts/ws2812_test.py`
4. Set `led.enabled: true` in `/etc/idcam/config.yaml` and restart the service.

| Colour | Meaning (what to do) |
|--------|----------------------|
| Off | Starting up, ready for a document, and while it is detected, focused and captured |
| Blue, blinking | Captured; validating and uploading — wait |
| Green | Uploaded (a moment) |
| Green, blinking | Done — remove the document |
| White, blinking | Failed again and again — remove the document, then place it again |
| Red for 2 s | Capture or upload failed, then back to the current state's colour |
| Yellow, blinking | Recovering |
| Red, blinking | Error (e.g. low storage) |
| Off | Shut down |

## Start on boot

- **Production install** (`sudo ./scripts/install.sh`): installs and enables `idcam.service`, so idCam
  starts at every boot and restarts if it stops.
- **Folder install** (`bash scripts/setup_local.sh`): installs and starts `idcam-local.service`
  (runs `./run.sh` as your user from this folder) and enables it for every boot. It also disables
  the old `idcam.service` because both need the camera. Options: `--no-autostart`,
  `--remove-autostart`.

```bash
sudo systemctl status idcam-local       # is it running?
journalctl -u idcam-local -f            # live logs
sudo systemctl restart idcam-local      # after editing config.yaml or .env
sudo systemctl stop idcam-local && ./run.sh   # run by hand for debugging
```

**Changing the pin or colours:** edit the `led:` block in `config.yaml` (no code change). The data
pin is `led.gpio_pin` (BCM number, default 18 = physical pin 12), `led.count` is the number of
LEDs, and `led.colors` overrides the colour of any status as `[R, G, B]`
(`starting`, `ready`, `working`, `processing`, `success`, `remove`, `failure`, `recovering`,
`error`). Test a pin before changing the config:
`python scripts/ws2812_test.py --pin 12 --count 60` (or `--config config.yaml`).

`scripts/setup_local.sh` installs the LED libraries, adds your user to the `gpio` group and turns
the strip on in `config.yaml`; pass `--no-led` to skip it.

The strip is steady white while no document is detected and goes dark as soon as one is
detected, so it never changes the light on the document the camera focuses and exposes; it
only lights up again afterwards (blue while uploading, green when done). Override any colour
with `led.colors`, e.g. `ready: [0, 0, 0]` to keep the idle strip dark.

## User Status

The application writes real-time status to `/run/idcam/status.json`:

```json
{
  "state": "IDLE",
  "message": "Ready — Place document",
  "camera_healthy": true,
  "network_healthy": true
}
```

## Server API

See [API Protocol](docs/API_PROTOCOL.md) for full idempotency and timeout semantics.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
IDC_RUN_HARDWARE_TESTS=1 pytest tests/ -v -m hardware  # on Pi with camera
```

Hardware tests are skipped off-device. Run the full suite after installer changes.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Security](docs/SECURITY.md)
- [Performance and timing](docs/PERFORMANCE.md): where the time goes, focus speed settings, how to measure (`scripts/timing_report.py`)
- [Codebase audit report](ID_CAM_AUDIT_REPORT.md): architecture, upload contents, performance bottlenecks and a prioritised fix list

## License

Proprietary
