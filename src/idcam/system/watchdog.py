"""Stuck-state watchdog."""

from __future__ import annotations

import time
from dataclasses import dataclass

from idcam.app.state_machine import AppState, StateMachine
from idcam.config.schema import StateTimeoutsConfig, WatchdogConfig
from idcam.logging.setup import get_logger

logger = get_logger("system.watchdog")

WATCHED_STATES = {
    AppState.CAPTURING,
    AppState.IMAGE_VALIDATION,
    AppState.UPLOADING,
    AppState.WAITING_FOR_SERVER,
    AppState.PROCESSING_RESPONSE,
    AppState.RECOVERY,
    AppState.CAMERA_INITIALIZING,
}


@dataclass
class WatchdogAlert:
    state: AppState
    duration_sec: float
    timeout_sec: float
    capture_id: str | None
    reason: str


class Watchdog:
    """Detects stuck states using per-state deadlines."""

    def __init__(self, config: WatchdogConfig, state_timeouts: StateTimeoutsConfig) -> None:
        self.config = config
        self.state_timeouts = state_timeouts

    def check(
        self,
        sm: StateMachine,
        *,
        capture_id: str | None = None,
        last_progress_at: float | None = None,
    ) -> WatchdogAlert | None:
        if sm.state not in WATCHED_STATES:
            return None

        timeout = sm.timeout_for_state()
        if timeout is None:
            return None

        # Use progress idle for long-but-valid operations (e.g. 64MP capture)
        progress_ref = last_progress_at if last_progress_at is not None else sm.last_progress_at
        stall_duration = time.monotonic() - progress_ref
        state_duration = sm.elapsed()

        if stall_duration > timeout:
            alert = WatchdogAlert(
                state=sm.state,
                duration_sec=state_duration,
                timeout_sec=timeout,
                capture_id=capture_id,
                reason=f"no_progress_for_{stall_duration:.1f}s",
            )
            logger.error(
                "event=watchdog_triggered state=%s duration=%.1fs timeout=%.1fs capture_id=%s reason=%s",
                alert.state.value,
                alert.duration_sec,
                alert.timeout_sec,
                alert.capture_id,
                alert.reason,
            )
            return alert
        return None
