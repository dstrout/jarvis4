#!/usr/bin/env python3
"""
Message Bus + Dispatcher test suite.

Tests the fire-and-forget dispatch pattern with stub agents.
Uses realistic delays to validate ordering, scheduling, failure
handling, and cross-agent task groups.

Usage:
    python -m tests.test_message_bus          # all tests
    python -m tests.test_message_bus basic     # just basic dispatch
    python -m tests.test_message_bus schedule   # scheduling tests
    python -m tests.test_message_bus group      # task group tests
    python -m tests.test_message_bus stress     # concurrent load
"""

import logging
import sys
import time
import random
from datetime import datetime, timedelta

from core.message_bus import (
    MessageBus,
    Dispatcher,
    Message,
    MessageStatus,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test")


# ---------------------------------------------------------------------------
# Stub agent executor — simulates real agent behavior with delays
# ---------------------------------------------------------------------------

# Simulated execution times per agent (seconds)
AGENT_DELAYS = {
    "mail_agent": {"read": 1.5, "write": 2.5},
    "calendar_tasks_agent": {"read": 0.8, "write": 1.5},
    "comms_agent": {"read": 1.0, "write": 1.2},
    "home_control_agent": {"read": 0.5, "write": 0.7},
    "search_agent": {"read": 2.0, "write": 0.0},
}

# Keywords that indicate write operations
WRITE_KEYWORDS = [
    "send", "create", "schedule", "post", "turn on", "turn off",
    "set", "lock", "unlock", "open", "close", "forward", "reply",
    "delete", "archive", "label", "reschedule", "notify",
]

# Track execution log for assertions
_execution_log: list[dict] = []


def stub_executor(agent_name: str, instruction: str, dry_run: bool) -> tuple[bool, str]:
    """
    Simulates agent execution with realistic delays.

    Classifies the instruction as read/write based on keywords,
    applies appropriate delay, and returns a simulated result.
    """
    is_write = any(kw in instruction.lower() for kw in WRITE_KEYWORDS)
    op_type = "write" if is_write else "read"

    delays = AGENT_DELAYS.get(agent_name, {"read": 1.0, "write": 1.5})
    delay = delays[op_type]

    # Add ±20% jitter to simulate real-world variance
    delay *= random.uniform(0.8, 1.2)

    prefix = "[DRY RUN] " if dry_run else ""
    log.info(
        "%sExecuting %s on %s (%.1fs delay): %s",
        prefix, op_type, agent_name, delay, instruction[:60]
    )

    time.sleep(delay)

    entry = {
        "agent": agent_name,
        "instruction": instruction,
        "op_type": op_type,
        "dry_run": dry_run,
        "timestamp": datetime.now(),
    }
    _execution_log.append(entry)

    if dry_run:
        return True, f"{prefix}{agent_name} would execute: {instruction[:100]}"

    # Simulate configurable failure for specific test scenarios
    if "FAIL_TEST" in instruction:
        return False, f"Simulated failure: {agent_name} could not process: {instruction[:50]}"

    return True, f"{agent_name} completed: {instruction[:100]}"


def failing_executor(agent_name: str, instruction: str, dry_run: bool) -> tuple[bool, str]:
    """Always-fail executor for testing failure handling."""
    time.sleep(0.5)
    _execution_log.append({
        "agent": agent_name,
        "instruction": instruction,
        "op_type": "write",
        "dry_run": dry_run,
        "timestamp": datetime.now(),
    })
    return False, f"Error: {agent_name} service unavailable"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def reset():
    """Reset test state."""
    global _execution_log
    _execution_log = []


def wait_for_idle(bus: MessageBus, dispatcher: Dispatcher, timeout: float = 30.0) -> bool:
    """Wait until the bus is empty AND dispatcher is idle (no in-flight message)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        status = bus.get_queue_status()
        if status["total_queued"] == 0 and dispatcher.is_idle:
            return True
        time.sleep(0.2)
    return False


def assert_true(condition: bool, msg: str):
    if not condition:
        log.error("ASSERTION FAILED: %s", msg)
        raise AssertionError(msg)
    log.info("  PASS: %s", msg)


def print_header(name: str):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")


def print_execution_log():
    """Print what was executed and in what order."""
    print("\n  Execution log:")
    for i, entry in enumerate(_execution_log):
        ts = entry["timestamp"].strftime("%H:%M:%S.%f")[:-3]
        dry = " [DRY]" if entry["dry_run"] else ""
        print(f"    {i+1}. [{ts}] {entry['agent']}{dry}: {entry['instruction'][:60]}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_basic_dispatch():
    """Test: Single fire-and-forget message dispatches and completes."""
    print_header("Basic dispatch — single fire-and-forget message")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)
    dispatcher.start()

    try:
        msg = Message(
            to="mail_agent",
            from_agent="jarvis",
            body="Send an email to bill@example.com with subject 'Hello' and body 'Test message'",
        )
        msg_id = bus.put(msg)

        assert_true(msg_id == msg.id, f"put() returns message ID: {msg_id}")

        # Wait for dispatch
        completed = wait_for_idle(bus, dispatcher, timeout=10)
        assert_true(completed, "Message was dispatched and bus is empty")

        # Check execution log
        assert_true(len(_execution_log) == 1, f"Exactly 1 execution (got {len(_execution_log)})")
        assert_true(_execution_log[0]["agent"] == "mail_agent", "Correct agent executed")
        assert_true(_execution_log[0]["op_type"] == "write", "Classified as write op")

        # Check completed list
        recent = bus.get_recent_completed()
        assert_true(len(recent) == 1, "1 completed message")
        assert_true(recent[0].status == MessageStatus.DONE, "Status is DONE")

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_lifo_ordering():
    """Test: Messages execute in LIFO order (most recent first)."""
    print_header("LIFO ordering — newer messages execute first")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    # Queue 3 messages BEFORE starting dispatcher
    msg1 = Message(to="home_control_agent", from_agent="jarvis", body="Turn on office lights (first)")
    msg2 = Message(to="comms_agent", from_agent="jarvis", body="Send slack to Mike: heads up (second)")
    msg3 = Message(to="mail_agent", from_agent="jarvis", body="Send email to bill: test (third)")

    bus.put(msg1)
    time.sleep(0.01)  # Ensure different timestamps
    bus.put(msg2)
    time.sleep(0.01)
    bus.put(msg3)

    # Now start dispatcher — should process in LIFO order
    dispatcher.start()

    try:
        completed = wait_for_idle(bus, dispatcher, timeout=20)
        assert_true(completed, "All messages dispatched")
        assert_true(len(_execution_log) == 3, f"3 executions (got {len(_execution_log)})")

        # LIFO: msg3 (last in) should execute first
        assert_true(
            _execution_log[0]["agent"] == "mail_agent",
            f"First exec was mail_agent (LIFO): got {_execution_log[0]['agent']}"
        )
        assert_true(
            _execution_log[1]["agent"] == "comms_agent",
            f"Second exec was comms_agent: got {_execution_log[1]['agent']}"
        )
        assert_true(
            _execution_log[2]["agent"] == "home_control_agent",
            f"Third exec was home_control_agent: got {_execution_log[2]['agent']}"
        )

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_new_command_preempts():
    """Test: New user command goes to top of stack, ahead of pending items."""
    print_header("Preemption — new command executes before older queued items")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    # Queue a slow task
    slow = Message(
        to="mail_agent",
        from_agent="jarvis",
        body="Send a long detailed email to the team (slow, queued first)",
    )
    bus.put(slow)

    # Queue a fast task on top
    fast = Message(
        to="home_control_agent",
        from_agent="jarvis",
        body="Turn on office lights (fast, queued second)",
    )
    bus.put(fast)

    dispatcher.start()

    try:
        completed = wait_for_idle(bus, dispatcher, timeout=15)
        assert_true(completed, "All messages dispatched")

        # Fast task should have been picked up first (LIFO)
        assert_true(
            _execution_log[0]["agent"] == "home_control_agent",
            "Fast task executed first (preemption works)"
        )

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_failure_notification():
    """Test: Failed task creates notification for orchestrator."""
    print_header("Failure handling — failed task notifies orchestrator")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)
    dispatcher.start()

    try:
        msg = Message(
            to="comms_agent",
            from_agent="jarvis",
            body="Send slack message to Mike: FAIL_TEST this should fail",
        )
        bus.put(msg)

        wait_for_idle(bus, dispatcher, timeout=10)

        # Check failure notification was created
        notifications = bus.get_pending_notifications()
        assert_true(len(notifications) == 1, f"1 notification (got {len(notifications)})")
        assert_true("failed" in notifications[0].body.lower(), "Notification mentions failure")
        assert_true(notifications[0].to == "jarvis", "Notification addressed to orchestrator")

        # Notifications are drained on read
        notifications2 = bus.get_pending_notifications()
        assert_true(len(notifications2) == 0, "Notifications drained after read")

        # Check completed status
        recent = bus.get_recent_completed()
        failed_msgs = [m for m in recent if m.status == MessageStatus.FAILED]
        assert_true(len(failed_msgs) == 1, "1 failed message in history")

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_task_group_sequential():
    """Test: Task group executes steps in sequence order."""
    print_header("Task group — sequential execution with shared task_id")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    # Multi-step task: schedule meeting, then notify on Slack
    task_id = "group_test_001"
    messages = [
        Message(
            to="calendar_tasks_agent",
            from_agent="jarvis",
            body="Create event: Team sync at 3pm tomorrow",
            task_id=task_id,
        ),
        Message(
            to="comms_agent",
            from_agent="jarvis",
            body="Send slack to engineering: Team sync scheduled for 3pm tomorrow",
            task_id=task_id,
        ),
    ]

    bus.put_group(messages)
    dispatcher.start()

    try:
        completed = wait_for_idle(bus, dispatcher, timeout=15)
        assert_true(completed, "All group steps dispatched")
        assert_true(len(_execution_log) == 2, f"2 executions (got {len(_execution_log)})")

        # Sequence 0 should execute before sequence 1
        assert_true(
            _execution_log[0]["agent"] == "calendar_tasks_agent",
            "Calendar step executed first"
        )
        assert_true(
            _execution_log[1]["agent"] == "comms_agent",
            "Comms step executed second"
        )

        # Both should be in completed
        recent = bus.get_recent_completed()
        done = [m for m in recent if m.status == MessageStatus.DONE]
        assert_true(len(done) == 2, "Both steps completed successfully")

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_task_group_abort_on_failure():
    """Test: If step 1 of a group fails, remaining steps are cancelled."""
    print_header("Task group abort — failure cancels remaining steps")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    # Multi-step: step 1 will fail, step 2 should be cancelled
    task_id = "group_fail_001"
    messages = [
        Message(
            to="calendar_tasks_agent",
            from_agent="jarvis",
            body="FAIL_TEST: Create event that will fail",
            task_id=task_id,
        ),
        Message(
            to="comms_agent",
            from_agent="jarvis",
            body="Send slack notification about the event (should be cancelled)",
            task_id=task_id,
        ),
    ]

    bus.put_group(messages)
    dispatcher.start()

    try:
        # Wait for step 1 to fail and step 2 to be cancelled
        time.sleep(5)

        # Step 1 should have executed (and failed)
        assert_true(len(_execution_log) == 1, f"Only 1 execution (got {len(_execution_log)})")
        assert_true(
            _execution_log[0]["agent"] == "calendar_tasks_agent",
            "Only the first step was attempted"
        )

        # Step 2 should be cancelled
        recent = bus.get_recent_completed()
        cancelled = [m for m in recent if m.status == MessageStatus.CANCELLED]
        assert_true(len(cancelled) == 1, f"1 cancelled step (got {len(cancelled)})")
        assert_true("comms_agent" == cancelled[0].to, "Comms step was cancelled")

        # Failure notification should exist
        notifications = bus.get_pending_notifications()
        assert_true(len(notifications) == 1, "Failure notification created")
        assert_true("cancelled" in notifications[0].body.lower(), "Notification mentions cancellation")

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_dry_run():
    """Test: Dry run flag prevents actual execution but still processes."""
    print_header("Dry run — simulates without executing")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)
    dispatcher.start()

    try:
        msg = Message(
            to="home_control_agent",
            from_agent="jarvis",
            body="Turn on all lights in the house",
            dry_run=True,
        )
        bus.put(msg)

        wait_for_idle(bus, dispatcher, timeout=10)

        assert_true(len(_execution_log) == 1, "1 execution")
        assert_true(_execution_log[0]["dry_run"] is True, "Dry run flag passed to executor")

        recent = bus.get_recent_completed()
        assert_true(recent[0].status == MessageStatus.DONE, "Dry run completes successfully")
        assert_true("would execute" in recent[0].result.lower(), "Result indicates dry run")

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_scheduled_task():
    """Test: Message with future run_at is skipped until its time."""
    print_header("Scheduled task — deferred execution")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    # Schedule a task 3 seconds from now
    scheduled = Message(
        to="home_control_agent",
        from_agent="jarvis",
        body="Turn on office lights (scheduled)",
        run_at=datetime.now() + timedelta(seconds=3),
    )
    bus.put(scheduled)

    # Also queue an immediate task
    immediate = Message(
        to="comms_agent",
        from_agent="jarvis",
        body="Send notification: test (immediate)",
    )
    bus.put(immediate)

    dispatcher.start()

    try:
        # After 2 seconds, only the immediate task should have run
        time.sleep(2.5)
        assert_true(
            len(_execution_log) == 1,
            f"Only immediate task ran after 2s (got {len(_execution_log)})"
        )
        assert_true(
            _execution_log[0]["agent"] == "comms_agent",
            "Immediate task ran first"
        )

        # After 4 more seconds, the scheduled task should have run too
        time.sleep(4)
        assert_true(
            len(_execution_log) == 2,
            f"Scheduled task ran after delay (got {len(_execution_log)})"
        )
        assert_true(
            _execution_log[1]["agent"] == "home_control_agent",
            "Scheduled task ran second"
        )

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_recurring_task():
    """Test: Recurring task re-queues after execution with updated run_at."""
    print_header("Recurring task — repeat execution with interval")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    # Recurring task: run 3 times, every 2 seconds
    recurring = Message(
        to="home_control_agent",
        from_agent="jarvis",
        body="Check device status (recurring)",
        repeat=2,  # Will run 3 total times (initial + 2 repeats)
        interval=timedelta(seconds=2),
    )
    bus.put(recurring)

    dispatcher.start()

    try:
        # Wait for all 3 executions (initial + 2 repeats)
        # Each takes ~0.5s execution + 2s interval = ~8s total
        time.sleep(12)

        assert_true(
            len(_execution_log) == 3,
            f"3 total executions (got {len(_execution_log)})"
        )

        # All should be the same agent
        for entry in _execution_log:
            assert_true(
                entry["agent"] == "home_control_agent",
                "All executions were home_control_agent"
            )

        # Verify timing: ~2s apart
        for i in range(1, len(_execution_log)):
            delta = (_execution_log[i]["timestamp"] - _execution_log[i-1]["timestamp"]).total_seconds()
            assert_true(
                1.5 < delta < 4.0,
                f"Execution {i} was {delta:.1f}s after previous (expected ~2-3s)"
            )

        # Bus should be empty (repeat count exhausted)
        status = bus.get_queue_status()
        assert_true(status["total_queued"] == 0, "No more recurring items in queue")

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_forever_recurring():
    """Test: repeat=-1 runs indefinitely until stopped."""
    print_header("Forever recurring — repeat=-1 (capped at 4 runs for test)")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    forever = Message(
        to="home_control_agent",
        from_agent="jarvis",
        body="Check device status (forever)",
        repeat=-1,
        interval=timedelta(seconds=2),
    )
    bus.put(forever)

    dispatcher.start()

    try:
        # Let it run for ~9 seconds — should get 3-4 executions
        time.sleep(9)

        assert_true(
            len(_execution_log) >= 3,
            f"At least 3 executions (got {len(_execution_log)})"
        )

        # Bus should still have a scheduled item (it's forever)
        status = bus.get_queue_status()
        assert_true(
            status["scheduled"] >= 1,
            "Still has a scheduled recurring item"
        )

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_morning_briefing_scenario():
    """
    Test: Real-world scenario — scheduled morning briefing.

    "Generate morning briefing at 5am, then when the user wakes up,
    Jarvis reads it aloud."

    Simulated: Schedule a search task, then a notification.
    """
    print_header("Scenario: Morning briefing (scheduled multi-step)")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    # Schedule briefing generation 2s from now (simulating 5am)
    task_id = "morning_001"
    messages = [
        Message(
            to="search_agent",
            from_agent="jarvis",
            body="Generate morning briefing: weather, top news, calendar summary",
            task_id=task_id,
            run_at=datetime.now() + timedelta(seconds=2),
        ),
        Message(
            to="comms_agent",
            from_agent="jarvis",
            body="Send notification: Morning briefing is ready",
            task_id=task_id,
            run_at=datetime.now() + timedelta(seconds=2),
        ),
    ]
    bus.put_group(messages)

    dispatcher.start()

    try:
        # Nothing should execute yet
        time.sleep(1.5)
        assert_true(len(_execution_log) == 0, "Nothing executed before schedule time")

        # Wait for both to complete
        time.sleep(10)
        assert_true(len(_execution_log) == 2, f"Both steps executed (got {len(_execution_log)})")
        assert_true(
            _execution_log[0]["agent"] == "search_agent",
            "Search ran first (briefing generation)"
        )
        assert_true(
            _execution_log[1]["agent"] == "comms_agent",
            "Notification ran second"
        )

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_complex_scenario():
    """
    Test: Real-world scenario — "Turn on office lights and read me the morning briefing."

    Two independent fire-and-forget tasks (but sequential because single-lane).
    Home control is fast, search/briefing is slow.
    """
    print_header("Scenario: Lights + briefing (multi-agent)")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    task_id = "complex_001"
    messages = [
        Message(
            to="home_control_agent",
            from_agent="jarvis",
            body="Turn on office lights",
            task_id=task_id,
        ),
        Message(
            to="search_agent",
            from_agent="jarvis",
            body="Read the morning briefing (pre-generated)",
            task_id=task_id,
        ),
    ]
    bus.put_group(messages)

    dispatcher.start()

    try:
        completed = wait_for_idle(bus, dispatcher, timeout=15)
        assert_true(completed, "Both tasks completed")
        assert_true(len(_execution_log) == 2, f"2 executions (got {len(_execution_log)})")

        # Lights should be first (sequence 0)
        assert_true(
            _execution_log[0]["agent"] == "home_control_agent",
            "Lights turned on first"
        )
        assert_true(
            _execution_log[1]["agent"] == "search_agent",
            "Briefing read second"
        )

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_cross_agent_email_slack():
    """
    Test: "Download attachments to Drive and forward the email to Bill,
    then Slack Mike to let him know."

    3-step group across mail and comms agents.
    """
    print_header("Scenario: Email + Drive + Slack (3-step cross-agent)")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    task_id = "cross_001"
    messages = [
        Message(
            to="mail_agent",
            from_agent="jarvis",
            body="Download attachments from message ID msg_123 to Google Drive",
            task_id=task_id,
        ),
        Message(
            to="mail_agent",
            from_agent="jarvis",
            body="Forward message msg_123 to bill@example.com",
            task_id=task_id,
        ),
        Message(
            to="comms_agent",
            from_agent="jarvis",
            body="Send slack DM to Mike: I forwarded you those attachments from the Q1 report",
            task_id=task_id,
        ),
    ]
    bus.put_group(messages)

    dispatcher.start()

    try:
        completed = wait_for_idle(bus, dispatcher, timeout=20)
        assert_true(completed, "All 3 steps completed")
        assert_true(len(_execution_log) == 3, f"3 executions (got {len(_execution_log)})")

        agents = [e["agent"] for e in _execution_log]
        assert_true(agents == ["mail_agent", "mail_agent", "comms_agent"], "Correct execution order")

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_cross_agent_abort():
    """
    Test: "Schedule a meeting and Slack Mike to let him know."

    If scheduling fails, don't Slack Mike.
    """
    print_header("Scenario: Schedule + Slack with abort (meeting fails)")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    task_id = "cross_abort_001"
    messages = [
        Message(
            to="calendar_tasks_agent",
            from_agent="jarvis",
            body="FAIL_TEST: Schedule meeting with team at conflicting time",
            task_id=task_id,
        ),
        Message(
            to="comms_agent",
            from_agent="jarvis",
            body="Send slack to Mike: Meeting scheduled for 3pm",
            task_id=task_id,
        ),
    ]
    bus.put_group(messages)

    dispatcher.start()

    try:
        time.sleep(5)

        assert_true(len(_execution_log) == 1, "Only step 1 executed")
        assert_true(
            _execution_log[0]["agent"] == "calendar_tasks_agent",
            "Calendar step was attempted"
        )

        # Slack step should be cancelled
        recent = bus.get_recent_completed()
        cancelled = [m for m in recent if m.status == MessageStatus.CANCELLED]
        assert_true(len(cancelled) == 1, "Slack step was cancelled")

        # Notification exists
        notifications = bus.get_pending_notifications()
        assert_true(len(notifications) == 1, "Failure notification created")
        log.info("  Notification: %s", notifications[0].body[:100])

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_interleaved_user_commands():
    """
    Test: User fires multiple commands in quick succession.

    Simulates: "Send email to Bill" then immediately "Turn on the lights"
    Lights should execute first (LIFO) since they were queued last.
    """
    print_header("Interleaved commands — rapid user input")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    # Simulate rapid commands
    cmd1 = Message(
        to="mail_agent",
        from_agent="jarvis",
        body="Send email to bill@example.com about the meeting",
    )
    cmd2 = Message(
        to="home_control_agent",
        from_agent="jarvis",
        body="Turn on the office lights",
    )
    cmd3 = Message(
        to="comms_agent",
        from_agent="jarvis",
        body="Send notification: don't forget standup",
    )

    bus.put(cmd1)
    time.sleep(0.05)
    bus.put(cmd2)
    time.sleep(0.05)
    bus.put(cmd3)

    dispatcher.start()

    try:
        completed = wait_for_idle(bus, dispatcher, timeout=20)
        assert_true(completed, "All commands completed")

        # LIFO: cmd3 first, cmd2 second, cmd1 last
        agents = [e["agent"] for e in _execution_log]
        assert_true(
            agents == ["comms_agent", "home_control_agent", "mail_agent"],
            f"LIFO order: {agents}"
        )

        print_execution_log()
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_queue_status():
    """Test: Queue status reporting is accurate."""
    print_header("Queue status — monitoring and debugging")
    reset()

    bus = MessageBus()

    # Queue some messages
    bus.put(Message(to="mail_agent", from_agent="jarvis", body="Send email"))
    bus.put(Message(
        to="home_control_agent", from_agent="jarvis", body="Check status",
        run_at=datetime.now() + timedelta(hours=1),
    ))
    bus.put(Message(to="comms_agent", from_agent="jarvis", body="Send slack"))

    status = bus.get_queue_status()
    assert_true(status["total_queued"] == 3, f"3 queued (got {status['total_queued']})")
    assert_true(status["ready"] == 2, f"2 ready (got {status['ready']})")
    assert_true(status["scheduled"] == 1, f"1 scheduled (got {status['scheduled']})")

    print(f"  Status: {status}")
    print("  RESULT: PASS")


def test_dispatcher_stats():
    """Test: Dispatcher tracks execution statistics."""
    print_header("Dispatcher stats — operational metrics")
    reset()

    bus = MessageBus()
    dispatcher = Dispatcher(bus, stub_executor, heartbeat_interval=1.0)

    bus.put(Message(to="home_control_agent", from_agent="jarvis", body="Turn on lights"))
    bus.put(Message(to="comms_agent", from_agent="jarvis", body="FAIL_TEST: this will fail"))

    dispatcher.start()

    try:
        time.sleep(8)

        stats = dispatcher.stats
        assert_true(stats["dispatched"] == 2, f"2 dispatched (got {stats['dispatched']})")
        assert_true(stats["succeeded"] == 1, f"1 succeeded (got {stats['succeeded']})")
        assert_true(stats["failed"] == 1, f"1 failed (got {stats['failed']})")
        assert_true(stats["heartbeats"] > 0, f"Heartbeats counted: {stats['heartbeats']}")

        print(f"  Stats: {stats}")
        print("  RESULT: PASS")
    finally:
        dispatcher.stop()


def test_mixed_sync_async():
    """
    Test: Synchronous (wait=True) messages are not put on the bus.

    In real use, wait=True bypasses the bus entirely and executes
    inline. Here we verify the bus ignores wait=True messages
    (the orchestrator client handles them directly).
    """
    print_header("Mixed sync/async — wait flag behavior")
    reset()

    bus = MessageBus()

    # In the real system, wait=True messages would bypass the bus entirely.
    # This test verifies that if someone mistakenly puts one on the bus,
    # it still executes (wait is just a hint for the caller).
    async_msg = Message(
        to="mail_agent", from_agent="jarvis",
        body="Send email (fire and forget)", wait=False,
    )
    sync_msg = Message(
        to="search_agent", from_agent="jarvis",
        body="Search for weather (synchronous)", wait=True,
    )

    bus.put(async_msg)
    bus.put(sync_msg)

    # Both should be in the queue (bus doesn't filter on wait)
    status = bus.get_queue_status()
    assert_true(status["total_queued"] == 2, "Both messages queued")

    print("  Note: In production, wait=True bypasses the bus at the caller level.")
    print("  RESULT: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_suite(suite: str = "all"):
    """Run the specified test suite."""
    all_tests = {
        "basic": [
            test_basic_dispatch,
            test_lifo_ordering,
            test_new_command_preempts,
            test_dry_run,
            test_queue_status,
            test_dispatcher_stats,
            test_mixed_sync_async,
        ],
        "schedule": [
            test_scheduled_task,
            test_recurring_task,
            test_forever_recurring,
        ],
        "group": [
            test_task_group_sequential,
            test_task_group_abort_on_failure,
        ],
        "scenario": [
            test_morning_briefing_scenario,
            test_complex_scenario,
            test_cross_agent_email_slack,
            test_cross_agent_abort,
            test_interleaved_user_commands,
        ],
        "failure": [
            test_failure_notification,
        ],
    }

    if suite == "all":
        tests = []
        for group in all_tests.values():
            tests.extend(group)
    elif suite in all_tests:
        tests = all_tests[suite]
    else:
        print(f"Unknown suite: {suite}. Available: {', '.join(all_tests.keys())}, all")
        sys.exit(1)

    print(f"\n{'#'*70}")
    print(f"# Message Bus Test Suite: {suite}")
    print(f"# {len(tests)} test(s)")
    print(f"{'#'*70}")

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except (AssertionError, Exception) as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            log.error("FAILED: %s — %s", test_fn.__name__, e)

    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*70}")

    return failed == 0


if __name__ == "__main__":
    suite = sys.argv[1] if len(sys.argv) > 1 else "all"
    success = run_suite(suite)
    sys.exit(0 if success else 1)
