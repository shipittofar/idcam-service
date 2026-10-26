"""Temp file storage tests."""

from pathlib import Path

from idcam.storage.temp_files import TempFileStore


class TestTempFileStore:
    def test_create_temp_path(self, tmp_path):
        store = TempFileStore(tmp_path)
        path = store.create_temp_path()
        assert path.parent == tmp_path
        assert path.suffix == ".jpg"

    def test_atomic_write_finalize(self, tmp_path):
        store = TempFileStore(tmp_path)
        temp = store.create_temp_path()
        temp.write_bytes(b"test data")
        final = tmp_path / "final.jpg"
        store.atomic_write_finalize(temp, final)
        assert final.exists()
        assert not temp.exists()
        assert final.read_bytes() == b"test data"

    def test_cleanup(self, tmp_path):
        store = TempFileStore(tmp_path)
        path = store.create_temp_path()
        path.write_bytes(b"x")
        store.cleanup(path)
        assert not path.exists()

    def test_cleanup_stale(self, tmp_path):
        store = TempFileStore(tmp_path)
        old = store.create_temp_path()
        old.write_bytes(b"old")
        import os
        import time

        os.utime(old, (time.time() - 7200, time.time() - 7200))
        removed = store.cleanup_stale(max_age_sec=3600)
        assert removed == 1
        assert not old.exists()
