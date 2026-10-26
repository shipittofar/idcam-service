"""Retry policy for uploads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """
    Retry policy for uploads.

    Connection failures (request never sent) are always safe to retry. Retrying after the body
    may have been sent is only safe when the server dedupes by ``Idempotency-Key``, so it is
    opt-in (``retry_after_send``) and the upload client enables it from ``server.retry_after_send``.
    """

    max_connection_retries: int = 1
    retry_delay_sec: float = 2.0
    retry_after_send: bool = False

    def should_retry_connection(self, attempt: int) -> bool:
        """``attempt`` is the number of attempts made so far."""
        return attempt < self.max_connection_retries

    def should_retry_post_after_send(self) -> bool:
        return self.retry_after_send

    def delay_before_retry(self, attempt: int) -> float:
        """Exponential backoff: delay, 2x, 4x, ... after attempt 1, 2, 3, ..."""
        return self.retry_delay_sec * (2 ** max(0, attempt - 1))
