"""Server API protocol models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from idcam.exceptions import ProtocolError


class UploadOutcome(str, Enum):
    NEVER_SENT = "never_sent"
    SENT_UNKNOWN = "sent_unknown"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    SERVER_ERROR = "server_error"


@dataclass(frozen=True)
class UploadRequest:
    """HTTP upload payload — created only from an active capture transaction."""

    capture_id: str
    device_id: str
    image_path: str
    created_at: str


@dataclass
class ServerResponse:
    success: bool
    capture_id: str | None
    result: str | None
    message: str | None
    http_status: int
    outcome: UploadOutcome

    @classmethod
    def from_http(cls, status: int, body: dict | None, expected_capture_id: str) -> "ServerResponse":
        if body is None:
            if 200 <= status < 300:
                return cls(
                    success=True,
                    capture_id=expected_capture_id,
                    result=None,
                    message=None,
                    http_status=status,
                    outcome=UploadOutcome.ACCEPTED,
                )
            if 400 <= status < 500:
                return cls(
                    success=False,
                    capture_id=expected_capture_id,
                    result=None,
                    message="Server rejected request",
                    http_status=status,
                    outcome=UploadOutcome.REJECTED,
                )
            return cls(
                success=False,
                capture_id=expected_capture_id,
                result=None,
                message="Server error",
                http_status=status,
                outcome=UploadOutcome.SERVER_ERROR,
            )

        response_capture_id = body.get("capture_id")
        if response_capture_id and response_capture_id != expected_capture_id:
            raise ProtocolError(
                f"Response capture_id mismatch: expected={expected_capture_id} got={response_capture_id}"
            )

        success = bool(body.get("success", 200 <= status < 300))
        if 200 <= status < 300 and "success" not in body:
            success = True

        outcome = UploadOutcome.ACCEPTED if success and 200 <= status < 300 else UploadOutcome.REJECTED
        if status >= 500:
            outcome = UploadOutcome.SERVER_ERROR

        return cls(
            success=success and 200 <= status < 300,
            capture_id=response_capture_id or expected_capture_id,
            result=body.get("result"),
            message=body.get("message") or body.get("error"),
            http_status=status,
            outcome=outcome,
        )

    @classmethod
    def validate_json_body(cls, body: dict, expected_capture_id: str) -> None:
        """Validate business response structure."""
        if "success" not in body:
            raise ProtocolError("Response missing required field: success")
        if body.get("capture_id") and body["capture_id"] != expected_capture_id:
            raise ProtocolError(
                f"Response capture_id mismatch: expected={expected_capture_id} got={body['capture_id']}"
            )


# Backward compatibility alias
CaptureTransaction = UploadRequest
