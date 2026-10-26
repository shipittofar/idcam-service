"""API client tests."""

from pathlib import Path

import httpx
import pytest
import respx

from idcam.config.schema import AppConfig, AuthConfig, ServerConfig
from idcam.exceptions import ServerRejectedError
from idcam.network.api_client import ApiClient
from idcam.network.protocol import UploadOutcome, UploadRequest


@pytest.fixture
def app_config():
    return AppConfig(
        device_id="test-pi",
        server=ServerConfig(url="https://api.example.com", retry_backoff_sec=0.01),
        auth=AuthConfig(type="bearer", token="test-token"),
    )


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "doc.jpg"
    path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10000)
    return path


@pytest.fixture
def transaction(sample_image):
    return UploadRequest(
        capture_id="cap-20260824120000abc123",
        device_id="test-pi",
        image_path=str(sample_image),
        created_at="2026-08-24T12:00:00Z",
    )


class TestApiClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_upload(self, app_config, transaction):
        route = respx.post("https://api.example.com/api/document").mock(
            return_value=httpx.Response(
                200,
                json={"success": True, "capture_id": "cap-20260824120000abc123", "result": "ok"},
            )
        )
        client = ApiClient(app_config)
        await client.start()
        response = await client.upload_document(transaction)
        await client.stop()
        assert response.success
        assert response.outcome == UploadOutcome.ACCEPTED
        assert route.called
        request = route.calls[0].request
        assert request.headers["X-Capture-ID"] == transaction.capture_id
        assert request.headers["Idempotency-Key"] == transaction.capture_id
        assert "Bearer test-token" in request.headers["Authorization"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_server_400_rejected(self, app_config, transaction):
        respx.post("https://api.example.com/api/document").mock(
            return_value=httpx.Response(400, json={"success": False, "error": "bad request"})
        )
        client = ApiClient(app_config)
        await client.start()
        with pytest.raises(ServerRejectedError):
            await client.upload_document(transaction)
        await client.stop()

    @pytest.mark.asyncio
    @respx.mock
    async def test_server_500_error(self, app_config, transaction):
        respx.post("https://api.example.com/api/document").mock(
            return_value=httpx.Response(500, json={"success": False})
        )
        client = ApiClient(app_config)
        await client.start()
        from idcam.exceptions import ServerError

        with pytest.raises(ServerError):
            await client.upload_document(transaction)
        await client.stop()

    @pytest.mark.asyncio
    @respx.mock
    async def test_connection_failure(self, app_config, transaction):
        respx.post("https://api.example.com/api/document").mock(side_effect=httpx.ConnectError("refused"))
        client = ApiClient(app_config)
        await client.start()
        response = await client.upload_document(transaction)
        await client.stop()
        assert not response.success
        assert response.outcome == UploadOutcome.NEVER_SENT

    @pytest.mark.asyncio
    @respx.mock
    async def test_malformed_json_success_status(self, app_config, transaction):
        respx.post("https://api.example.com/api/document").mock(
            return_value=httpx.Response(200, content=b"not json")
        )
        client = ApiClient(app_config)
        await client.start()
        response = await client.upload_document(transaction)
        await client.stop()
        assert response.success

    @pytest.mark.asyncio
    async def test_missing_image(self, app_config, tmp_path):
        transaction = UploadRequest(
            capture_id="cap-test",
            device_id="test-pi",
            image_path=str(tmp_path / "missing.jpg"),
            created_at="2026-08-24T12:00:00Z",
        )
        client = ApiClient(app_config)
        await client.start()
        from idcam.exceptions import NetworkError

        with pytest.raises(NetworkError):
            await client.upload_document(transaction)
        await client.stop()


OK_BODY = {"success": True, "capture_id": "cap-20260824120000abc123", "result": "ok"}
URL = "https://api.example.com/api/document"


async def _upload(config, transaction):
    client = ApiClient(config)
    await client.start()
    try:
        return await client.upload_document(transaction)
    finally:
        await client.stop()


class TestUploadRetries:
    @pytest.mark.asyncio
    @respx.mock
    async def test_503_then_success_is_retried_with_same_idempotency_key(self, app_config, transaction):
        route = respx.post(URL).mock(
            side_effect=[httpx.Response(503, text="busy"), httpx.Response(200, json=OK_BODY)]
        )
        response = await _upload(app_config, transaction)
        assert response.success
        assert route.call_count == 2
        keys = {call.request.headers["Idempotency-Key"] for call in route.calls}
        assert keys == {transaction.capture_id}

    @pytest.mark.asyncio
    @respx.mock
    async def test_429_honours_retry_after_and_succeeds(self, app_config, transaction):
        route = respx.post(URL).mock(
            side_effect=[httpx.Response(429, headers={"Retry-After": "0"}), httpx.Response(200, json=OK_BODY)]
        )
        assert (await _upload(app_config, transaction)).success
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_read_timeout_is_retried_when_server_dedupes(self, app_config, transaction):
        route = respx.post(URL).mock(side_effect=[httpx.ReadTimeout("slow"), httpx.Response(200, json=OK_BODY)])
        assert (await _upload(app_config, transaction)).success
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_read_timeout_not_retried_when_after_send_retry_disabled(self, transaction):
        config = AppConfig(
            device_id="test-pi",
            server=ServerConfig(url="https://api.example.com", retry_after_send=False, retry_backoff_sec=0.01),
        )
        route = respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
        response = await _upload(config, transaction)
        assert response.outcome == UploadOutcome.SENT_UNKNOWN
        assert route.call_count == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_connection_error_retried_up_to_max_attempts(self, app_config, transaction):
        route = respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
        response = await _upload(app_config, transaction)
        assert response.outcome == UploadOutcome.NEVER_SENT
        assert route.call_count == app_config.server.max_attempts

    @pytest.mark.asyncio
    @respx.mock
    async def test_4xx_is_never_retried(self, app_config, transaction):
        route = respx.post(URL).mock(return_value=httpx.Response(401, json={"success": False, "error": "bad token"}))
        with pytest.raises(ServerRejectedError, match="bad token"):
            await _upload(app_config, transaction)
        assert route.call_count == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_persistent_503_raises_after_max_attempts(self, app_config, transaction):
        from idcam.exceptions import ServerError

        route = respx.post(URL).mock(return_value=httpx.Response(503, text="down"))
        with pytest.raises(ServerError, match="503"):
            await _upload(app_config, transaction)
        assert route.call_count == app_config.server.max_attempts

    @pytest.mark.asyncio
    @respx.mock
    async def test_html_200_from_captive_portal_is_not_success(self, app_config, transaction):
        from idcam.exceptions import ProtocolError

        respx.post(URL).mock(
            return_value=httpx.Response(200, text="<html>Login</html>", headers={"content-type": "text/html"})
        )
        with pytest.raises(ProtocolError, match="HTML"):
            await _upload(app_config, transaction)

    @pytest.mark.asyncio
    @respx.mock
    async def test_total_timeout_bounds_retries(self, transaction):
        config = AppConfig(
            device_id="test-pi",
            server=ServerConfig(url="https://api.example.com", total_timeout_sec=0.05, max_attempts=10),
        )
        route = respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
        response = await _upload(config, transaction)
        assert not response.success
        assert route.call_count < 10  # the deadline, not max_attempts, ended the retries
