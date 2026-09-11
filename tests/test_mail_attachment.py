"""Tests for email attachment support — MIME construction, validation, error handling."""

import base64
import json
import os
import tempfile
from email import message_from_bytes
from unittest.mock import patch, MagicMock

import pytest

from agent_modules.mail import _do_send_email_with_attachment


class TestMimeConstruction:
    def test_basic_attachment(self, tmp_path):
        """Verify MIME message is constructed correctly with attachment."""
        test_file = tmp_path / "report.txt"
        test_file.write_text("Hello, this is a test report.")

        with patch("agent_modules.mail.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"id": "msg123"}',
                stderr="",
            )

            result = _do_send_email_with_attachment(
                to="bob@example.com",
                subject="Test Report",
                body="Please review the attached report.",
                file_path=str(test_file),
            )

        assert "sent" in result.lower()
        assert "report.txt" in result
        assert "msg123" in result

        # Verify the command was called with correct structure
        call_args = mock_run.call_args
        cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
        assert cmd[0] == "gws"
        assert "send" in cmd

        # Find the --json argument and decode the MIME message
        json_idx = cmd.index("--json") + 1
        payload = json.loads(cmd[json_idx])
        raw_bytes = base64.urlsafe_b64decode(payload["raw"] + "==")
        mime_msg = message_from_bytes(raw_bytes)

        assert mime_msg["To"] == "bob@example.com"
        assert mime_msg["Subject"] == "Test Report"
        assert mime_msg.is_multipart()

        parts = list(mime_msg.walk())
        # Part 0 is the multipart container, 1 is text body, 2 is attachment
        assert len(parts) == 3
        assert parts[1].get_content_type() == "text/plain"
        assert "report" in parts[1].get_payload(decode=True).decode().lower()
        assert parts[2].get_filename() == "report.txt"

    def test_signature_added(self, tmp_path):
        """Verify Jarvis signature is added to the body."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b,c\n1,2,3")

        with patch("agent_modules.mail.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout='{"id": "x"}', stderr="",
            )

            _do_send_email_with_attachment(
                to="alice@example.com",
                subject="Data",
                body="Here's the data.",
                file_path=str(test_file),
            )

        cmd = mock_run.call_args[0][0]
        json_idx = cmd.index("--json") + 1
        payload = json.loads(cmd[json_idx])
        raw_bytes = base64.urlsafe_b64decode(payload["raw"] + "==")
        mime_msg = message_from_bytes(raw_bytes)
        body_text = list(mime_msg.walk())[1].get_payload(decode=True).decode()
        # Closing comes from config defaults ("Best regards,\nJarvis4")
        assert "Jarvis4" in body_text or "Jarvis" in body_text

    def test_cc_bcc(self, tmp_path):
        """Verify CC and BCC headers are set."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        with patch("agent_modules.mail.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout='{"id": "x"}', stderr="",
            )

            _do_send_email_with_attachment(
                to="bob@example.com",
                subject="FYI",
                body="See attached.",
                file_path=str(test_file),
                cc="alice@example.com",
                bcc="admin@example.com",
            )

        cmd = mock_run.call_args[0][0]
        json_idx = cmd.index("--json") + 1
        payload = json.loads(cmd[json_idx])
        raw_bytes = base64.urlsafe_b64decode(payload["raw"] + "==")
        mime_msg = message_from_bytes(raw_bytes)
        assert mime_msg["Cc"] == "alice@example.com"
        assert mime_msg["Bcc"] == "admin@example.com"


class TestValidation:
    def test_missing_file(self):
        result = _do_send_email_with_attachment(
            to="bob@example.com",
            subject="Test",
            body="Body",
            file_path="/nonexistent/file.txt",
        )
        assert "Error" in result
        assert "not found" in result.lower()

    def test_invalid_email(self):
        result = _do_send_email_with_attachment(
            to="Bob Williams",
            subject="Test",
            body="Body",
            file_path="/tmp/whatever.txt",
        )
        assert "Error" in result
        assert "not a valid email" in result.lower()


class TestErrorHandling:
    def test_subprocess_failure(self, tmp_path):
        test_file = tmp_path / "file.txt"
        test_file.write_text("data")

        with patch("agent_modules.mail.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="auth failed",
            )

            result = _do_send_email_with_attachment(
                to="bob@example.com",
                subject="Test",
                body="Body",
                file_path=str(test_file),
            )

        assert "Error" in result
        assert "auth failed" in result

    def test_timeout(self, tmp_path):
        import subprocess

        test_file = tmp_path / "file.txt"
        test_file.write_text("data")

        with patch("agent_modules.mail.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("gws", 60)

            result = _do_send_email_with_attachment(
                to="bob@example.com",
                subject="Test",
                body="Body",
                file_path=str(test_file),
            )

        assert "Error" in result
        assert "timed out" in result.lower()


class TestAgentRegistration:
    def test_attachment_tool_registered(self):
        from agent_modules.mail import mail_agent
        tool_names = [t.name for t in mail_agent.tools]
        assert "send_email_with_attachment" in tool_names
        assert "send_email" in tool_names
        assert "lookup_contact" in tool_names
