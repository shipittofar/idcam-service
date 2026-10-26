"""Capture file lifecycle and .partial path regression tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from idcam.camera.capture import CaptureService
from idcam.config.schema import CameraConfig
from idcam.exceptions import CameraError, CaptureError
from idcam.storage.temp_files import TempFileStore, PARTIAL_SUFFIX


class TestPartialCapturePath:
    def test_partial_path_keeps_jpg_extension(self, tmp_path):
        store = TempFileStore(tmp_path)
        final = store.create_temp_path()
        partial = store.create_partial_path(final)
        assert partial.name.endswith(".partial.jpg")
        assert partial.suffix == ".jpg"
        assert PARTIAL_SUFFIX in partial.name
        assert not partial.name.endswith(".jpg.partial")

    def test_finalize_partial_jpg_to_final(self, tmp_path):
        store = TempFileStore(tmp_path)
        final = store.create_temp_path()
        partial = store.create_partial_path(final)
        partial.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        store.finalize_capture(partial, final)
        assert final.exists()
        assert not partial.exists()

    @pytest.mark.asyncio
    async def test_capture_service_uses_partial_jpg(self, tmp_path):
        store = TempFileStore(tmp_path)
        config = CameraConfig(capture_timeout_sec=5.0)
        service = CaptureService(config, store)
        output = store.create_temp_path()
        partial_written: Path | None = None

        class Cam:
            def capture_still_to_file(self, path: Path) -> None:
                nonlocal partial_written
                partial_written = path
                path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 200)

        result = await service.capture(Cam(), output)
        assert partial_written is not None
        assert partial_written.name.endswith(".partial.jpg")
        assert result == output
        assert output.exists()

    @pytest.mark.asyncio
    async def test_capture_error_not_camera_error(self, tmp_path):
        store = TempFileStore(tmp_path)
        service = CaptureService(CameraConfig(), store)
        output = store.create_temp_path()

        class Cam:
            def capture_still_to_file(self, path: Path) -> None:
                raise RuntimeError("unknown file extension: .partial")

        with pytest.raises(CaptureError, match="unknown file extension"):
            await service.capture(Cam(), output)

    @pytest.mark.asyncio
    async def test_camera_error_propagates(self, tmp_path):
        store = TempFileStore(tmp_path)
        service = CaptureService(CameraConfig(), store)
        output = store.create_temp_path()

        class Cam:
            def capture_still_to_file(self, path: Path) -> None:
                raise CameraError("Failed to restore preview after capture")

        with pytest.raises(CameraError):
            await service.capture(Cam(), output)
