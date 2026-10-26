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
| Connection fails before send | Safe to retry once (client does) | N/A |
| Connection drops after send | **No auto-retry**; mark ambiguous locally | Dedupe by `Idempotency-Key` |
| Server returns 5xx | Fail transaction; no auto-retry | Idempotent if already processed |
| Server returns 4xx | Fail transaction; no retry | Reject invalid requests |

## Delivery Guarantees

- **Client:** at-most-one automatic retry (connection failure only, before request completes)
- **With server idempotency:** effectively-once processing
- **NOT claimed:** exactly-once without server-side deduplication

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
| Upload state | 60s | `state_timeouts.uploading` |
| Server wait | 60s | `state_timeouts.waiting_for_server` |

## Ambiguous Network Failure

If the client cannot determine whether the server received the request:

- Transaction is marked failed locally
- User is instructed to remove document and retry
- Server must handle duplicate `Idempotency-Key` safely
- Client does **not** queue the image for later upload
