"""Test configuration."""

import sys
from pathlib import Path

# Add src to path for tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import pytest


@pytest.fixture(autouse=True)
def _fast_upload_backoff(monkeypatch):
    """Upload retries use exponential backoff (2 s, 4 s ...); never wait for real in unit tests."""
    from idcam.network.retry_policy import RetryPolicy

    monkeypatch.setattr(RetryPolicy, "delay_before_retry", lambda self, attempt: 0.005)
