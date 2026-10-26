"""Safe retry policy for network operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """
    Retry policy for uploads.

    POST retries are limited to connection failures BEFORE the request is fully sent.
    Once body is sent, no automatic retry (idempotency cannot guarantee safety without server).
    """

    max_connection_retries: int = 1
    retry_delay_sec: float = 2.0

    def should_retry_connection(self, attempt: int) -> bool:
        return attempt < self.max_connection_retries

    def should_retry_post_after_send(self) -> bool:
        return False
