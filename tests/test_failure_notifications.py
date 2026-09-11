"""Tests for the failure notification pipeline — bus callback, turn limit detection, alert queue."""

import asyncio
import queue
import threading
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from core.message_bus import MessageBus, Message, MessageStatus


# ---------------------------------------------------------------------------
# Task 2: Failure callback on message bus
# ---------------------------------------------------------------------------

class TestFailureCallback:
    def test_callback_fires_on_failure(self):
        bus = MessageBus()
        received = []
        bus.set_failure_callback(lambda notification: received.append(notification))

        msg = Message(to="mail_agent", from_agent="jarvis", body="send email")
        bus.mark_failed(msg, "connection refused")

        assert len(received) == 1
        assert "mail_agent" in received[0].body
        assert "connection refused" in received[0].body

    def test_callback_prevents_double_delivery(self):
        bus = MessageBus()
        bus.set_failure_callback(lambda n: None)  # succeeds silently

        msg = Message(to="mail_agent", from_agent="jarvis", body="send email")
        bus.mark_failed(msg, "timeout")

        # Pending queue should be empty since callback succeeded
        notifications = bus.get_pending_notifications()
        assert len(notifications) == 0

    def test_callback_failure_falls_back_to_queue(self):
        bus = MessageBus()

        def bad_callback(n):
            raise RuntimeError("callback broken")

        bus.set_failure_callback(bad_callback)

        msg = Message(to="mail_agent", from_agent="jarvis", body="send email")
        bus.mark_failed(msg, "timeout")

        # Should fall back to pending queue
        notifications = bus.get_pending_notifications()
        assert len(notifications) == 1
        assert "mail_agent" in notifications[0].body

    def test_no_callback_uses_queue(self):
        bus = MessageBus()
        # No callback set

        msg = Message(to="mail_agent", from_agent="jarvis", body="send email")
        bus.mark_failed(msg, "error")

        notifications = bus.get_pending_notifications()
        assert len(notifications) == 1

    def test_callback_receives_formatted_message(self):
        bus = MessageBus()
        received = []
        bus.set_failure_callback(lambda n: received.append(n))

        msg = Message(to="search_agent", from_agent="jarvis", body="search for weather in London")
        bus.mark_failed(msg, "API key invalid")

        notification = received[0]
        assert "Background task failed" in notification.body
        assert "search_agent" in notification.body
        assert "API key invalid" in notification.body


# ---------------------------------------------------------------------------
# Task 1: Turn limit detection
# ---------------------------------------------------------------------------

class TestTurnLimitDetection:
    def test_max_turns_detected_as_failure(self):
        from agents.exceptions import MaxTurnsExceeded
        from core.bus_integration import create_agent_executor

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()

        try:
            mock_agent = MagicMock()
            mock_agent.name = "test_agent"
            registry = {"test_agent": mock_agent}

            executor = create_agent_executor(registry, loop)

            # Mock _run_agent as an async function that raises MaxTurnsExceeded
            async def raise_max_turns(agent, instruction):
                raise MaxTurnsExceeded("exceeded 10 turns")

            with patch("core.bus_integration._run_agent", side_effect=raise_max_turns):
                success, result = executor("test_agent", "send a complex email", False)

            assert success is False
            assert "exhausted" in result.lower() or "turn limit" in result.lower()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=2)

    def test_generic_exception_still_caught(self):
        from core.bus_integration import create_agent_executor

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()

        try:
            mock_agent = MagicMock()
            mock_agent.name = "test_agent"
            registry = {"test_agent": mock_agent}

            executor = create_agent_executor(registry, loop)

            async def raise_value_error(agent, instruction):
                raise ValueError("some other error")

            with patch("core.bus_integration._run_agent", side_effect=raise_value_error):
                success, result = executor("test_agent", "do something", False)

            assert success is False
            assert "error" in result.lower()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Task 3: Alert queue wiring
# ---------------------------------------------------------------------------

class TestAlertQueueWiring:
    def test_alert_queue_receives_failure(self):
        """Verify that OrchestratorClient pushes failures to the alert queue."""
        alert_q = queue.Queue()

        # Create a bus and wire the callback manually (simulating what OrchestratorClient does)
        bus = MessageBus()

        def on_failure(notification):
            alert_q.put({
                "action": "task_failure",
                "text": notification.body,
            })

        bus.set_failure_callback(on_failure)

        # Trigger a failure
        msg = Message(to="mail_agent", from_agent="jarvis", body="send email to Bob")
        bus.mark_failed(msg, "Gmail API error")

        # Check the alert queue
        assert not alert_q.empty()
        alert = alert_q.get_nowait()
        assert alert["action"] == "task_failure"
        assert "mail_agent" in alert["text"]
        assert "Gmail API error" in alert["text"]

    def test_no_alert_queue_falls_back(self):
        """Without an alert queue, failures go to pending notifications."""
        bus = MessageBus()
        # Callback that raises (simulating no alert queue)
        bus.set_failure_callback(lambda n: (_ for _ in ()).throw(RuntimeError("no queue")))

        msg = Message(to="comms_agent", from_agent="jarvis", body="send slack message")
        bus.mark_failed(msg, "token expired")

        # Should be in pending queue
        notifications = bus.get_pending_notifications()
        assert len(notifications) == 1
