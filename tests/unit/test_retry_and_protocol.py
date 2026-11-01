"""Retry policy and protocol tests."""

import pytest

from idcam.app.transaction import ActiveCaptureTransaction
from idcam.exceptions import ProtocolError
from idcam.network.protocol import ServerResponse, UploadOutcome, UploadRequest
from idcam.network.retry_policy import RetryPolicy


class TestRetryPolicy:
    def test_connection_retry_allowed(self):
        policy = RetryPolicy(max_connection_retries=2)
        assert policy.should_retry_connection(1)
        assert not policy.should_retry_connection(2)

    def test_no_post_retry_after_send(self):
        policy = RetryPolicy()
        assert not policy.should_retry_post_after_send()


class TestCaptureTransaction:
    def test_generate_unique_ids(self):
        ids = {ActiveCaptureTransaction.generate_id() for _ in range(100)}
        assert len(ids) == 100
        assert all(i.startswith("cap-") for i in ids)

    def test_transaction_lifecycle(self):
        txn = ActiveCaptureTransaction(capture_id=ActiveCaptureTransaction.generate_id())
        txn.mark_document_detected()
        txn.mark_focus_started()
        txn.mark_focus_completed(100.0, 80.0)
        assert txn.sharpness_score == 100.0
        assert not txn.is_terminal
        txn.mark_failed("test")
        assert txn.is_terminal


class TestServerResponse:
    def test_from_http_success(self):
        resp = ServerResponse.from_http(200, {"success": True, "capture_id": "abc"}, "abc")
        assert resp.success
        assert resp.outcome == UploadOutcome.ACCEPTED

    def test_from_http_4xx(self):
        resp = ServerResponse.from_http(400, {"success": False}, "abc")
        assert not resp.success
        assert resp.outcome == UploadOutcome.REJECTED

    def test_from_http_5xx(self):
        resp = ServerResponse.from_http(503, None, "abc")
        assert not resp.success
        assert resp.outcome == UploadOutcome.SERVER_ERROR

    def test_capture_id_mismatch_raises(self):
        with pytest.raises(ProtocolError):
            ServerResponse.from_http(200, {"success": True, "capture_id": "wrong"}, "expected")

    def test_validate_json_body_mismatch(self):
        with pytest.raises(ProtocolError):
            ServerResponse.validate_json_body({"success": True, "capture_id": "x"}, "y")


class TestServerStatusField:
    """The OCR server answers {"status": "success" | "fail", "error": ...} with HTTP 200."""

    def test_status_success_is_accepted(self):
        resp = ServerResponse.from_http(200, {"status": "success", "doc_type": "id_card"}, "c1")
        assert resp.success and resp.outcome == UploadOutcome.ACCEPTED

    def test_status_fail_with_http_200_is_not_accepted_and_carries_the_servers_reason(self):
        body = {"status": "fail", "error": "No trained model for 'boarding_pass'. Train first."}
        resp = ServerResponse.from_http(200, body, "c1")
        assert not resp.success and resp.outcome == UploadOutcome.REJECTED
        assert "No trained model" in resp.message

    def test_a_body_with_status_passes_validation(self):
        ServerResponse.validate_json_body({"status": "fail"}, "c1")

    def test_a_body_with_neither_status_nor_success_is_still_a_protocol_error(self):
        with pytest.raises(ProtocolError):
            ServerResponse.validate_json_body({"file": {"saved": "x.jpg"}}, "c1")

    def test_success_key_still_wins(self):
        assert ServerResponse.from_http(200, {"success": False, "status": "success"}, "c1").success is False
