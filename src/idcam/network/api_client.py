"""HTTP API client for document upload."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx

from idcam.config.schema import AppConfig
from idcam.exceptions import NetworkError, ProtocolError, ServerError, ServerRejectedError
from idcam.logging.setup import get_logger
from idcam.network.protocol import ServerResponse, UploadOutcome, UploadRequest
from idcam.network.retry_policy import RetryPolicy

logger = get_logger("network.api_client")


class _RetryableHttp(Exception):
    """Internal: the server answered with a status that may succeed on retry."""

    def __init__(self, response: httpx.Response, message: str) -> None:
        super().__init__(message)
        self.response = response
        self.message = message


class ApiClient:
    """Dedicated HTTP client for document upload."""

    def __init__(self, config: AppConfig, retry_policy: RetryPolicy | None = None) -> None:
        self.config = config
        server = config.server
        self.retry_policy = retry_policy or RetryPolicy(
            max_connection_retries=server.max_attempts,
            retry_delay_sec=server.retry_backoff_sec,
            retry_after_send=server.retry_after_send,
        )
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        timeout = httpx.Timeout(
            connect=self.config.server.connect_timeout_sec,
            read=self.config.server.read_timeout_sec,
            write=self.config.server.read_timeout_sec,
            pool=self.config.server.connect_timeout_sec,
        )
        self._client = httpx.AsyncClient(
            timeout=timeout,
            verify=self.config.server.verify_tls,
            follow_redirects=False,
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def check_connectivity(self) -> bool:
        if not self._client:
            await self.start()
        assert self._client is not None
        base = str(self.config.server.url).rstrip("/")
        try:
            response = await self._client.head(base, timeout=5.0)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    @staticmethod
    def _snippet(response: httpx.Response, limit: int = 200) -> str:
        """Short single-line response body for diagnostics (never request data or headers)."""
        try:
            text = response.text
        except Exception:
            return ""
        return " ".join(text.split())[:limit]

    @staticmethod
    def _retry_after_sec(response: httpx.Response, default: float) -> float:
        raw = response.headers.get("retry-after", "")
        try:
            return min(max(float(raw), 0.0), 30.0)
        except ValueError:
            return default

    @staticmethod
    def _server_message(response: httpx.Response, default: str) -> str:
        try:
            data = response.json()
        except Exception:
            data = None
        if isinstance(data, dict):
            msg = data.get("message") or data.get("error")
            if msg:
                return f"HTTP {response.status_code}: {msg}"
        snippet = " ".join((response.text or "").split())[:120] if response.content else ""
        return f"HTTP {response.status_code}: {snippet or default}"

    async def upload_document(self, request: UploadRequest) -> ServerResponse:
        """Upload with bounded retries inside ``server.total_timeout_sec``.

        Every attempt carries the same Idempotency-Key. Connection failures are retried; failures
        after the request may have been sent (read timeout, dropped connection, 5xx, 429) only when
        ``server.retry_after_send`` is on, because the server must then dedupe the document.
        """
        if not self._client:
            await self.start()
        assert self._client is not None

        image_path = Path(request.image_path)
        if not image_path.exists():
            raise NetworkError(f"Image not found for upload: {image_path}")

        file_size = image_path.stat().st_size
        if file_size > self.config.server.max_upload_bytes:
            raise NetworkError(f"Image exceeds max upload size: {file_size}")

        url = str(self.config.server.url).rstrip("/") + self.config.server.path
        headers = {
            "X-Device-ID": request.device_id,
            "X-Capture-ID": request.capture_id,
            "Idempotency-Key": request.capture_id,
        }
        token = self.config.auth.resolved_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        policy = self.retry_policy
        max_attempts = max(1, policy.max_connection_retries)
        deadline = time.monotonic() + float(self.config.server.total_timeout_sec)
        attempt = 0

        while True:
            attempt += 1
            started = time.monotonic()
            remaining = deadline - started
            logger.info(
                "event=upload_started capture_id=%s attempt=%d/%d size=%d url=%s",
                request.capture_id,
                attempt,
                max_attempts,
                file_size,
                url,
            )
            failure: ServerResponse | None = None
            retryable = False
            retry_delay = policy.delay_before_retry(attempt)
            try:
                if remaining <= 0:
                    raise httpx.ReadTimeout("total upload timeout exceeded")
                with image_path.open("rb") as image_file:
                    files = {"image": (image_path.name, image_file, "image/jpeg")}
                    response = await asyncio.wait_for(
                        self._client.post(url, headers=headers, files=files), timeout=remaining
                    )
                result = self._interpret(response, request)
                logger.info(
                    "event=upload_completed capture_id=%s status=%d outcome=%s attempt=%d duration_ms=%d",
                    request.capture_id,
                    response.status_code,
                    result.outcome.value,
                    attempt,
                    int((time.monotonic() - started) * 1000),
                )
                return result
            except _RetryableHttp as exc:
                logger.warning(
                    "event=upload_retryable_status capture_id=%s status=%d attempt=%d body=%r",
                    request.capture_id,
                    exc.response.status_code,
                    attempt,
                    self._snippet(exc.response),
                )
                if not policy.should_retry_post_after_send() or attempt >= max_attempts:
                    raise ServerError(exc.message) from None
                retryable = True
                retry_delay = self._retry_after_sec(exc.response, retry_delay)
                failure = ServerResponse(
                    success=False,
                    capture_id=request.capture_id,
                    result=None,
                    message=exc.message,
                    http_status=exc.response.status_code,
                    outcome=UploadOutcome.SERVER_ERROR,
                )
            except (ProtocolError, ServerRejectedError):
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                logger.warning(
                    "event=upload_connection_error capture_id=%s attempt=%d: %s", request.capture_id, attempt, exc
                )
                failure = ServerResponse(
                    success=False,
                    capture_id=request.capture_id,
                    result=None,
                    message=str(exc),
                    http_status=0,
                    outcome=UploadOutcome.NEVER_SENT,
                )
                retryable = policy.should_retry_connection(attempt)
            except (asyncio.TimeoutError, httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                logger.warning(
                    "event=upload_interrupted capture_id=%s attempt=%d: %s",
                    request.capture_id,
                    attempt,
                    str(exc) or type(exc).__name__,
                )
                failure = ServerResponse(
                    success=False,
                    capture_id=request.capture_id,
                    result=None,
                    message=str(exc) or type(exc).__name__,
                    http_status=0,
                    outcome=UploadOutcome.SENT_UNKNOWN,
                )
                retryable = policy.should_retry_post_after_send() and attempt < max_attempts
            except httpx.HTTPError as exc:
                logger.error("event=upload_http_error capture_id=%s: %s", request.capture_id, exc)
                return ServerResponse(
                    success=False,
                    capture_id=request.capture_id,
                    result=None,
                    message=str(exc),
                    http_status=0,
                    outcome=UploadOutcome.SENT_UNKNOWN,
                )

            if not retryable or time.monotonic() + retry_delay >= deadline:
                assert failure is not None
                return failure
            await asyncio.sleep(retry_delay)

    def _interpret(self, response: httpx.Response, request: UploadRequest) -> ServerResponse:
        """Turn an HTTP response into a result, an exception, or a retry request."""
        status = response.status_code
        content_type = response.headers.get("content-type", "").lower()

        if status in (408, 429) or status >= 500:
            raise _RetryableHttp(response, self._server_message(response, "Server error"))
        if 400 <= status < 500:
            logger.error(
                "event=upload_rejected capture_id=%s status=%d body=%r",
                request.capture_id,
                status,
                self._snippet(response),
            )
            raise ServerRejectedError(self._server_message(response, "Server rejected request"))

        # 2xx. An HTML page answered with 200 is a captive portal / proxy login, not our API:
        # calling it "accepted" would silently lose the document.
        if response.content and "text/html" in content_type:
            raise ProtocolError(
                f"Unexpected HTML response (status {status}); a proxy or captive portal answered, not the API"
            )

        body: dict | None = None
        if response.content:
            if "application/json" in content_type or response.content[:1] in (b"{", b"["):
                try:
                    parsed = response.json()
                except json.JSONDecodeError as exc:
                    logger.warning("Malformed JSON response capture_id=%s: %s", request.capture_id, exc)
                    body = {"success": True, "capture_id": request.capture_id}
                else:
                    if not isinstance(parsed, dict):
                        raise ProtocolError("Response JSON is not an object")
                    body = parsed
            else:
                body = {"success": True, "capture_id": request.capture_id}

        if body is not None:
            ServerResponse.validate_json_body(body, request.capture_id)
        return ServerResponse.from_http(status, body, request.capture_id)
