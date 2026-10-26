"""Failure scenario tests mapping to requirements."""

import time

import httpx
import pytest
import respx

from idcam.app.state_machine import AppState, StateMachine
from idcam.camera.quality import ImageValidator
from idcam.config.schema import AppConfig, QualityConfig, ServerConfig
from idcam.exceptions import ConfigurationError, ImageValidationError
from idcam.network.api_client import ApiClient
from idcam.network.protocol import UploadRequest, UploadOutcome
from idcam.system.watchdog import Watchdog
from idcam.config.schema import StateTimeoutsConfig, WatchdogConfig


class TestFailureScenarios:
    """Maps to requirement #28 failure scenarios."""

  # 22. Invalid configuration
    def test_invalid_config_rejected(self):
        with pytest.raises(Exception):
            AppConfig.model_validate({"device_id": ""})

    # 23. Missing configuration
    def test_missing_config_file(self, tmp_path):
        from idcam.config.loader import load_config

        with pytest.raises(ConfigurationError):
            load_config(tmp_path / "nonexistent.yaml")

    # 10. Corrupted image
    def test_corrupted_image_rejected(self, tmp_path):
        path = tmp_path / "bad.jpg"
        path.write_bytes(b"CORRUPT")
        validator = ImageValidator(QualityConfig(min_file_bytes=10))
        with pytest.raises(ImageValidationError):
            validator.validate(path)

    # 5-8. Server errors
    @pytest.mark.asyncio
    @respx.mock
    async def test_server_500(self, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        config = AppConfig(device_id="d", server=ServerConfig(url="https://x.com"))
        respx.post("https://x.com/api/document").mock(return_value=httpx.Response(500))
        client = ApiClient(config)
        await client.start()
        from idcam.exceptions import ServerError

        with pytest.raises(ServerError):
            await client.upload_document(
                UploadRequest("cap-id", "d", str(img), "now")
            )
        await client.stop()

    @pytest.mark.asyncio
    @respx.mock
    async def test_server_400(self, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        config = AppConfig(device_id="d", server=ServerConfig(url="https://x.com"))
        respx.post("https://x.com/api/document").mock(return_value=httpx.Response(400))
        client = ApiClient(config)
        await client.start()
        from idcam.exceptions import ServerRejectedError

        with pytest.raises(ServerRejectedError):
            await client.upload_document(
                UploadRequest("cap-id", "d", str(img), "now")
            )
        await client.stop()

    # 4. Network disconnect
    @pytest.mark.asyncio
    @respx.mock
    async def test_network_disconnect(self, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        config = AppConfig(device_id="d", server=ServerConfig(url="https://x.com"))
        respx.post("https://x.com/api/document").mock(side_effect=httpx.ConnectError("down"))
        client = ApiClient(config)
        await client.start()
        resp = await client.upload_document(UploadRequest("cap-id", "d", str(img), "now"))
        assert resp.outcome == UploadOutcome.NEVER_SENT
        await client.stop()

    # 25. DNS fails (connection error)
    @pytest.mark.asyncio
    @respx.mock
    async def test_dns_failure(self, tmp_path):
        img = tmp_path / "i.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 5000)
        config = AppConfig(device_id="d", server=ServerConfig(url="https://x.com"))
        respx.post("https://x.com/api/document").mock(
            side_effect=httpx.ConnectError("[Errno -2] Name or service not known")
        )
        client = ApiClient(config)
        await client.start()
        resp = await client.upload_document(UploadRequest("cap-id", "d", str(img), "now"))
        assert resp.outcome == UploadOutcome.NEVER_SENT
        await client.stop()

    # 14. Same document - state machine prevents re-capture during transaction
    def test_transaction_lock(self):
        sm = StateMachine(state=AppState.CAPTURING)
        assert sm.in_transaction()
        assert not sm.can_transition(AppState.DOCUMENT_DETECTED)

    # 18. Stuck state detection
    def test_watchdog_stuck_capturing(self):
        timeouts = StateTimeoutsConfig(capturing=10.0)
        wd = Watchdog(WatchdogConfig(), timeouts)
        sm = StateMachine(state=AppState.CAPTURING)
        sm.set_timeouts(timeouts)
        sm.entered_at = time.monotonic() - 20
        sm.last_progress_at = time.monotonic() - 20
        alert = wd.check(sm)
        assert alert is not None

    # 15. Processing blocks new capture
    def test_no_parallel_capture(self):
        sm = StateMachine(state=AppState.UPLOADING)
        assert sm.in_transaction()
