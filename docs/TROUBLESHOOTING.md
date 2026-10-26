# Troubleshooting Guide

## Installer

The supported first install is `sudo bash ./scripts/install.sh`. It installs OS
packages, the camera Python stack (`python3-picamera2`, `python3-libcamera`), and
the application. Do not work around installer failures by running `apt install`
for those packages by hand.

### Required package has no installation candidate

The installer lists the exact package and stops. Common causes:

- Not Raspberry Pi OS 64-bit (vanilla Debian does not ship `python3-picamera2`)
- Mixing Ubuntu package names (for example `libatlas-base-dev` / `libopenblas-base`)
- Broken or unreachable Raspberry Pi apt repositories

`libatlas-base-dev` is **not** required (Debian Trixie removed ATLAS; numpy/opencv
are PyPI wheels). Re-run the installer after the OS/repository is corrected.

### Duplicate APT sources (`configured multiple times`)

Raspberry Pi OS may warn that the same Debian suite is listed in
`/etc/apt/sources.list` and `/etc/apt/sources.list.d/debian.sources`. That is an
environment issue. The installer ignores those warnings and **does not** rewrite
APT sources. Real `E:` apt errors still fail the install.

### Camera packages installed but sensor not detected

OS Picamera2/libcamera packages can be present before the Arducam 64MP overlay
is active. See [Camera](#camera). After the overlay is in
`/boot/firmware/config.txt`, reboot, then re-run the installer if needed.

## Camera

### Camera not detected

```bash
rpicam-hello --list-cameras
libcamera-hello --list-cameras
```

- Verify CSI cable on **CAM1** port (Pi 5)
- Check `dtoverlay=arducam-64mp` in `/boot/firmware/config.txt`
- Reinstall driver: `./install_pivariety_pkgs.sh -p 64mp_pi_hawk_eye_kernel_driver`
- Reboot after driver install

### Autofocus controls not advertised

Some libcamera versions do not expose `AfMode`/`AfTrigger`. idCam probes at startup and falls back to `autofocus_cycle()` or sharpness-only verification.

Try:
- Update Arducam libcamera packages to latest
- Set `LIBCAMERA_NOTPARSED_EMBEDDED_DATA=1` in systemd unit
- Downgrade to libcamera 0.0.10 if AF broken on newer version (see Arducam forum)

### Slow capture (>10s)

64MP JPEG capture is inherently slow. Expected 10–20s at full resolution. Preview runs at lower resolution.

### Capture fails / camera stuck

```bash
sudo systemctl restart idcam
journalctl -u idcam --since "10 min ago"
```

idCam auto-recovers up to `camera.recovery_max_attempts` times.

### Live Debug Viewer

The viewer is **off** unless `debug.enabled: true`. It does not create a second camera; it observes the existing 1920×1080 preview. After enabling and restarting:

```bash
sudo systemctl restart idcam
sudo systemctl status idcam --no-pager
hostname -I
```

Open `http://<PI-IP>:8088/preview/live`. Direct stream: `/preview/live.mjpg`. Diagnostics JSON: `/preview/status`.

If the page says `WAITING FOR CAMERA...`, the service is still initializing. `CAMERA ERROR` means the preview path failed; check `journalctl -u idcam -n 100 --no-pager` and `/run/idcam/status.json`. Enabling this exposes the live image on the LAN with no login.

## Network

### Upload failures

```bash
curl -v https://your-api.example.com/
journalctl -u idcam | grep upload
```

- Check Wi-Fi: `nmcli dev wifi list`
- Verify TLS cert is valid (do not disable `verify_tls`)
- Check `IDCAM_AUTH_TOKEN` is set

### Ambiguous upload (connection dropped)

If upload may have reached server but response was lost, server must deduplicate via `Idempotency-Key`. Client does not auto-retry POST after send.

## Application

### Stuck state

Watchdog triggers recovery after `watchdog.state_timeout_sec` (default 300s).

### Status file

```bash
cat /run/idcam/status.json | python3 -m json.tool
```

### Logs

```bash
tail -f /var/log/idcam/idcam.log
journalctl -u idcam -f
```

### Invalid configuration

Application exits with code 1. Fix `/etc/idcam/config.yaml` and restart.

### Disk full

```bash
df -h
du -sh /var/log/idcam /var/lib/idcam/tmp
```

Temp files auto-cleaned after 1 hour. Logs rotate at 10MB × 5 files.

## Repeated captures of same document

After success, system waits in `WAITING_FOR_REMOVAL` until scene returns to baseline. If triggered repeatedly, increase `detection.removal_frames_required` or `motion_threshold`.

## Service won't start

```bash
systemctl status idcam
journalctl -u idcam -e
/opt/idcam/venv/bin/idcam --config /etc/idcam/config.yaml  # run manually (as user idcam)
```

Check StartLimit: `systemctl reset-failed idcam`

Configuration errors exit with code **2** and systemd will not restart-loop them.
Edit `/etc/idcam/config.yaml` and `/etc/idcam/environment`, then `sudo systemctl start idcam`.

### Permission denied on /var/log/idcam, /var/lib/idcam, or /run/idcam

Do not `mkdir`/`chmod` those paths by hand and do not run the app with sudo.

```bash
sudo bash ./scripts/install.sh
sudo systemctl restart idcam
```

`/run/idcam` is created by systemd `RuntimeDirectory=idcam` when the unit starts.
If you ran `idcam` from a login shell as a normal user, that is expected to fail against production paths — use the systemd service.

### First install still using CHANGE_ME / unconfigured.invalid

The service is supposed to refuse to start until `device_id` and `server.url` are real.
That is not a permissions bug.
