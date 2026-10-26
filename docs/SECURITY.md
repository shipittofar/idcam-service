# Security Considerations

## Transport

- All server communication uses HTTPS (`server.verify_tls: true` by default)
- TLS certificate verification is **never** disabled in production
- Connection timeouts prevent hung connections

## Credentials

- Auth tokens belong in `/etc/idcam/environment` as `IDCAM_AUTH_TOKEN` (root:idcam, mode 0640)
- Tokens may also be set in `/etc/idcam/config.yaml` (`auth.token`); that file is root:idcam 0640 so the service can read it but not write it
- Tokens are **never** logged and must not be committed to git
- Service runs as unprivileged `idcam` user (never root)
- systemd unit uses `ProtectSystem=strict`, `NoNewPrivileges=true`, `RuntimeDirectory=idcam`

## Input Validation

- Configuration validated with Pydantic on startup
- Server responses parsed as JSON with malformed-response handling
- Image file size bounded by `server.max_upload_bytes`
- File paths use `pathlib`; no shell interpolation

## Upload Semantics

- Unique `capture_id` and `Idempotency-Key` per transaction
- No blind POST retries after body sent (prevents duplicate processing)
- One safe connection retry on pre-send failures only

## Filesystem

- Captures written to temp files, atomic rename
- Temp directory owned by service user
- Log rotation prevents disk exhaustion

## Attack Surface

- No admin HTTP server exposed by default
- The optional Live Debug Viewer (`debug.enabled`, default **false**) serves a live camera image over HTTP with **no authentication**. Leave it disabled in production. Do not expose `IDCAM_AUTH_TOKEN` or other credentials on that page (the viewer does not include them).
- Status file is local JSON (readable by service user group)
- No arbitrary command execution
- No shell=True in subprocess calls

## Recommendations

1. Use short-lived API tokens, rotate regularly
2. Restrict `/etc/idcam/` to root:idcam 0750; config and environment files 0640. Never make `/etc/idcam`, `/var/lib/idcam`, `/var/log/idcam`, or `/run/idcam` world-writable.
3. Keep Raspberry Pi OS and Arducam drivers updated
4. Use firewall rules to limit outbound connections to API host only
5. Enable full-disk encryption if documents are sensitive
