"""HTTP API client for document upload."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from idcam.config.schema import AppConfig
from idcam.exceptions import NetworkError, ProtocolError, ServerError, ServerRejectedError
from idcam.logging.setup import get_logger
from idcam.network.protocol import ServerResponse, UploadOutcome, UploadRequest
from idcam.network.retry_policy import RetryPolicy

logger = get_logger("network.api_client")


class ApiClient:
    """Dedicated HTTP client for document upload."""

    def __init__(self, config: AppConfig, retry_policy: RetryPolicy | None = None) -> None:
        self.config = config
        self.retry_policy = retry_policy or RetryPolicy()
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

    async def upload_document(self, request: UploadRequest) -> ServerResponse:
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

        attempt = 0
        while True:
            attempt += 1
            logger.info(
                "event=upload_started capture_id=%s attempt=%d size=%d",
                request.capture_id,
                attempt,
                file_size,
            )
            try:
                with image_path.open("rb") as image_file:
                    files = {"image": (image_path.name, image_file, "image/jpeg")}
                    response = await self._client.post(url, headers=headers, files=files)

                body: dict | None = None
                content_type = response.headers.get("content-type", "")
                if response.content:
                    if "application/json" in content_type or response.content[:1] in (b"{", b"["):
                        try:
                            parsed = response.json()
                            if isinstance(parsed, dict):
                                body = parsed
                            else:
                                raise ProtocolError("Response JSON is not an object")
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                "Malformed JSON response capture_id=%s: %s",
                                request.capture_id,
                                exc,
                            )
                            if 200 <= response.status_code < 300:
                                body = {"success": True, "capture_id": request.capture_id}
                            elif response.status_code >= 500:
                                raise ServerError(f"Malformed JSON response: {exc}") from exc
                            else:
                                raise ProtocolError(f"Malformed JSON response: {exc}") from exc
                    elif 200 <= response.status_code < 300:
                        body = {"success": True, "capture_id": request.capture_id}

                if body is not None and isinstance(body, dict):
                    ServerResponse.validate_json_body(body, request.capture_id)

                result = ServerResponse.from_http(response.status_code, body, request.capture_id)

                if 400 <= response.status_code < 500:
                    logger.error(
                        "event=upload_rejected capture_id=%s status=%d",
                        request.capture_id,
                        response.status_code,
                    )
                    raise ServerRejectedError(result.message or "Server rejected request")

                if response.status_code >= 500:
                    logger.error(
                        "event=upload_server_error capture_id=%s status=%d",
                        request.capture_id,
                        response.status_code,
                    )
                    raise ServerError(result.message or "Server error")

                logger.info(
                    "event=upload_completed capture_id=%s status=%d outcome=%s",
                    request.capture_id,
                    response.status_code,
                    result.outcome.value,
                )
                return result

            except ProtocolError:
                raise
            except (ServerRejectedError, ServerError):
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                logger.warning("event=upload_connection_error capture_id=%s: %s", request.capture_id, exc)
                if self.retry_policy.should_retry_connection(attempt):
                    import asyncio

                    await asyncio.sleep(self.retry_policy.retry_delay_sec)
                    continue
                return ServerResponse(
                    success=False,
                    capture_id=request.capture_id,
                    result=None,
                    message=str(exc),
                    http_status=0,
                    outcome=UploadOutcome.NEVER_SENT,
                )
            except httpx.ReadTimeout as exc:
                logger.warning("event=upload_read_timeout capture_id=%s: %s", request.capture_id, exc)
                return ServerResponse(
                    success=False,
                    capture_id=request.capture_id,
                    result=None,
                    message=str(exc),
                    http_status=0,
                    outcome=UploadOutcome.SENT_UNKNOWN,
                )
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
