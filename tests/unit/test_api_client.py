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
        server=ServerConfig(url="https://api.example.com"),
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
