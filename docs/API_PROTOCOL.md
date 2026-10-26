# Server API Protocol

## Endpoint

```
POST {server.url}{server.path}
```

Default path: `/api/document`

## Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | If configured | `Bearer {token}` |
| `Content-Type` | Yes | `multipart/form-data` |
| `X-Device-ID` | Yes | Device identifier from config |
| `X-Capture-ID` | Yes | Unique capture transaction ID (`cap-{uuid}`) |
| `Idempotency-Key` | Yes | Same value as `X-Capture-ID` |

## Request Body

| Field | Type | Description |
|-------|------|-------------|
| `image` | file | JPEG document image |

## Response (200 OK)

```json
{
  "success": true,
  "capture_id": "cap-abc123...",
  "result": "...",
  "message": "..."
}
```

## Response (4xx/5xx)

```json
{
  "success": false,
  "error": "...",
  "message": "..."
}
```

## Idempotency Semantics

**Server requirement:** When the server receives a request with the same `Idempotency-Key` and equivalent body, it MUST return the same result without reprocessing the document.

| Scenario | Client behavior | Server responsibility |
|----------|-----------------|----------------------|
| Connection fails before send | Retry (up to `server.max_attempts`, exponential backoff) | N/A |
| Connection drops / read timeout after send | Retry with the **same** `Idempotency-Key` when `server.retry_after_send` is true (default); otherwise fail as ambiguous | Dedupe by `Idempotency-Key` |
| 408 / 429 / 5xx | Retry like above (honours `Retry-After`, max 30 s); after the last attempt fail the transaction | Idempotent if already processed |
| Other 4xx | Fail transaction; never retried | Reject invalid requests |
| 2xx with an HTML body | Protocol error (captive portal / proxy page, not the API) | N/A |

All attempts for one document stop at `server.total_timeout_sec`.

## Delivery Guarantees

- **Client:** up to `server.max_attempts` attempts per document, all with the same `Idempotency-Key`, inside `server.total_timeout_sec`
- **With server idempotency:** effectively-once processing
- **NOT claimed:** exactly-once without server-side deduplication. If your server does not dedupe,
  set `server.retry_after_send: false` so only never-sent attempts are retried.

## Response Validation

The client validates:

1. HTTP status code
2. JSON structure (when content-type is JSON)
3. `capture_id` in response matches request (if present)
4. `success` field presence

Mismatch of `capture_id` is treated as a **protocol error** and fails the transaction.

## Timeouts

| Timeout | Default | Config key |
|---------|---------|------------|
| Connect | 10s | `server.connect_timeout_sec` |
| Read | 60s | `server.read_timeout_sec` |
| Total upload (all attempts) | 90s | `server.total_timeout_sec` |
| Upload state | 120s | `state_timeouts.uploading` |
| Server wait | 60s | `state_timeouts.waiting_for_server` |

## Ambiguous Network Failure

If the client cannot determine whether the server received the request:

- Transaction is marked failed locally
- User is instructed to remove document and retry
- Server must handle duplicate `Idempotency-Key` safely
- Client does **not** queue the image for later upload
