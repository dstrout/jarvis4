#!/usr/bin/env python3
"""
Integration test for the message bus with the orchestrator.

Tests that:
1. Bus-aware tools are correctly wired to the orchestrator
2. Write operations dispatch to the bus (fire-and-forget)
3. Read operations run synchronously
4. Failure notifications are injected into conversation context
5. The orchestrator handles "Task accepted" responses naturally

Uses stub agents with realistic delays. Does NOT require MCP servers
or external services.

Usage:
    python -m tests.test_bus_integration
    python -m tests.test_bus_integration quick    # Just basic tests
"""

import asyncio
import logging
import sys
import time

import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test.integration")

# Initialize agents framework (config first — initialize() reads it)
from core import config
from core.agents_init import initialize

_CFG = config.load()
initialize()

from agents import Runner
from core.orchestrator import build_orchestrator, get_agent_registry
from core.message_bus import MessageBus, Dispatcher, Message
from core.bus_integration import (
    create_agent_executor,
    make_bus_tool,
    is_write_operation,
)

# Built once — discovery is the expensive part and these tests only read it.
orchestrator = build_orchestrator(_CFG)

# Agent discovery is config-gated: home_control needs integrations.homeassistant,
# mail needs integrations.google. Tests that drive those agents need a real
# config.json with credentials, so they skip rather than fail on a clean clone.
_REGISTRY = get_agent_registry(orchestrator)
_NEEDED = {"home_control_agent", "mail_agent"}
_MISSING = sorted(_NEEDED - set(_REGISTRY))

requires_integrations = pytest.mark.skipif(
    bool(_MISSING),
    reason=(
        "needs integrations enabled in config.json "
        f"(missing agents: {', '.join(_MISSING)})"
    ),
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def print_header(name: str):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")


def assert_true(condition: bool, msg: str):
    if not condition:
        log.error("ASSERTION FAILED: %s", msg)
        raise AssertionError(msg)
    log.info("  PASS: %s", msg)


def test_write_classification():
    """Test that read/write classification works correctly."""
    print_header("Write classification — keyword matching")

    # Mail agent
    assert_true(
        is_write_operation("mail_agent", "Send an email to Bill about the meeting"),
        "send email → write"
    )
    assert_true(
        not is_write_operation("mail_agent", "Check my inbox"),
        "check inbox → read"
    )
    assert_true(
        is_write_operation("mail_agent", "Forward that message to Sarah"),
        "forward → write"
    )
    assert_true(
        not is_write_operation("mail_agent", "What's my most recent email?"),
        "recent email → read"
    )
    assert_true(
        is_write_operation("mail_agent", "Reply to that with 'sounds good'"),
        "reply → write"
    )

    # Calendar agent
    assert_true(
        is_write_operation("calendar_tasks_agent", "Schedule a meeting at 3pm"),
        "schedule → write"
    )
    assert_true(
        not is_write_operation("calendar_tasks_agent", "What's on my agenda today?"),
        "agenda → read"
    )
    assert_true(
        is_write_operation("calendar_tasks_agent", "Create a task to review the budget"),
        "create task → write"
    )

    # Home control
    assert_true(
        is_write_operation("home_control_agent", "Turn on the office lights"),
        "turn on → write"
    )
    assert_true(
        not is_write_operation("home_control_agent", "What's the thermostat at?"),
        "thermostat status → read"
    )
    assert_true(
        is_write_operation("home_control_agent", "Lock the front door"),
        "lock → write"
    )

    # Comms
    assert_true(
        is_write_operation("comms_agent", "Send a Slack message to Mike"),
        "send slack → write"
    )
    assert_true(
        not is_write_operation("comms_agent", "What's the latest in #engineering?"),
        "read channel → read"
    )

    print("  RESULT: PASS")


def test_bus_tool_creation():
    """Test that bus-aware tools can be created for all agents."""
    print_header("Bus tool creation — all agents")

    bus = MessageBus()
    registry = get_agent_registry(orchestrator)

    for agent_name, agent in registry.items():
        tool = make_bus_tool(
            agent,
            tool_name=f"test_{agent_name}",
            tool_description=f"Test tool for {agent_name}",
            bus=bus,
        )
        assert_true(tool.name == f"test_{agent_name}", f"Tool name correct for {agent_name}")
        assert_true(
            "input" in tool.params_json_schema.get("properties", {}),
            f"Tool has 'input' parameter for {agent_name}"
        )

    print(f"  Created bus tools for {len(registry)} agents")
    print("  RESULT: PASS")


@requires_integrations
def test_bus_tool_write_dispatch():
    """Test that write operations dispatch to the bus."""
    print_header("Bus tool write dispatch — fire-and-forget")

    bus = MessageBus()
    registry = get_agent_registry(orchestrator)

    # Create a bus tool for home control (fast agent)
    home_agent = registry["home_control_agent"]
    tool = make_bus_tool(
        home_agent,
        tool_name="home",
        tool_description="Control home devices",
        bus=bus,
    )

    # Invoke the tool with a write request
    loop = asyncio.new_event_loop()
    try:
        import json
        args = json.dumps({"input": "Turn on the office lights"})

        result = loop.run_until_complete(tool.on_invoke_tool(None, args))

        assert_true(
            "task accepted" in result.lower() or "background" in result.lower(),
            f"Write returns 'task accepted': {result}"
        )

        # Check that message is on the bus
        status = bus.get_queue_status()
        assert_true(status["total_queued"] == 1, "1 message on bus")

        # Check message contents
        msg = bus.get_next_ready()
        assert_true(msg is not None, "Message is ready")
        assert_true(msg.to == "home_control_agent", "Correct target agent")
        assert_true("lights" in msg.body.lower(), "Body contains the instruction")

        print("  RESULT: PASS")
    finally:
        loop.close()


@requires_integrations
def test_bus_tool_read_sync():
    """Test that read operations still run synchronously."""
    print_header("Bus tool read sync — blocking execution")

    bus = MessageBus()
    registry = get_agent_registry(orchestrator)

    # Create a bus tool for home control
    home_agent = registry["home_control_agent"]
    tool = make_bus_tool(
        home_agent,
        tool_name="home",
        tool_description="Control home devices",
        bus=bus,
    )

    # Invoke with a read request
    loop = asyncio.new_event_loop()
    try:
        import json
        args = json.dumps({"input": "What's the status of all devices?"})

        t0 = time.perf_counter()
        result = loop.run_until_complete(tool.on_invoke_tool(None, args))
        elapsed = time.perf_counter() - t0

        assert_true(
            "task accepted" not in result.lower(),
            "Read does NOT return 'task accepted'"
        )
        assert_true(
            len(result) > 20,
            f"Read returns actual data ({len(result)} chars)"
        )
        # Bus should be empty — reads don't go through bus
        status = bus.get_queue_status()
        assert_true(status["total_queued"] == 0, "No messages on bus (read was sync)")

        log.info("  Read result preview: %s", result[:100])
        log.info("  Read took %.2fs (expected: blocking)", elapsed)

        print("  RESULT: PASS")
    finally:
        loop.close()


@requires_integrations
def test_executor_creation():
    """Test that the agent executor works with the async event loop."""
    print_header("Agent executor — sync-to-async bridge")

    loop = asyncio.new_event_loop()
    import threading as _threading
    thread = _threading.Thread(
        target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()),
        daemon=True,
    )
    thread.start()
    time.sleep(0.1)  # Let loop start

    try:
        registry = get_agent_registry(orchestrator)
        executor = create_agent_executor(registry, loop)

        # Test with home control agent (fast, stub)
        success, result = executor("home_control_agent", "What's the thermostat status?", False)
        assert_true(success, f"Executor succeeded: {result[:80]}")
        assert_true(len(result) > 0, "Got non-empty result")

        # Test dry run
        success, result = executor("home_control_agent", "Turn on lights", True)
        assert_true(success, "Dry run succeeded")
        assert_true("dry run" in result.lower(), f"Dry run indicated: {result[:80]}")

        # Test unknown agent
        success, result = executor("nonexistent_agent", "Do something", False)
        assert_true(not success, "Unknown agent fails gracefully")

        print("  RESULT: PASS")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=3)


def test_notification_injection():
    """Test that failure notifications are injected into conversation context."""
    print_header("Notification injection — failure reporting")

    bus = MessageBus()

    # Simulate a failed background task
    msg = Message(
        to="mail_agent",
        from_agent="jarvis",
        body="Send email to bill@example.com about the meeting",
    )
    bus.mark_failed(msg, "SMTP connection refused")

    # Check notifications
    notifications = bus.get_pending_notifications()
    assert_true(len(notifications) == 1, "1 notification pending")
    assert_true("failed" in notifications[0].body.lower(), "Notification mentions failure")
    assert_true("mail_agent" in notifications[0].body, "Notification mentions the agent")

    # Format for injection
    from core.bus_integration import format_notifications
    text = format_notifications(notifications)
    assert_true(text is not None, "Formatted text is not None")
    assert_true("background task" in text.lower(), f"Formatted: {text[:100]}")

    # Verify drain
    text2 = format_notifications(bus.get_pending_notifications())
    assert_true(text2 is None, "Notifications drained after read")

    print("  RESULT: PASS")


@requires_integrations
def test_orchestrator_tools_replaced():
    """Building with a bus swaps the orchestrator's tools for bus-aware ones."""
    print_header("Tool replacement — build_orchestrator(bus=...) wiring")

    bus = MessageBus()

    # Tools on a bus-less orchestrator, for comparison
    original_tools = list(orchestrator.tools)
    original_count = len(original_tools)
    log.info("  Original tools: %d (%s)", original_count, [t.name for t in original_tools])

    # A bus-wired orchestrator gets bus-aware tools instead
    bus_orchestrator = build_orchestrator(_CFG, bus=bus)

    new_tools = bus_orchestrator.tools
    log.info("  Bus tools: %d (%s)", len(new_tools), [t.name for t in new_tools])

    assert_true(len(new_tools) >= 2, f"At least 2 bus tools (got {len(new_tools)})")

    # Check that expected tools exist
    tool_names = {t.name for t in new_tools}
    assert_true("search" in tool_names, "search tool exists")
    assert_true("email" in tool_names, "email tool exists")

    # Additional stub agents should be present
    if "calendar" in tool_names:
        log.info("  Calendar tool: present")
    if "comms" in tool_names:
        log.info("  Comms tool: present")
    if "home" in tool_names:
        log.info("  Home tool: present")

    print("  RESULT: PASS")


@requires_integrations
def test_full_bus_dispatch_cycle():
    """
    End-to-end test: write goes through bus, dispatcher executes it,
    result is available.
    """
    print_header("Full dispatch cycle — bus + dispatcher + executor")

    bus = MessageBus()

    # Set up event loop for executor
    loop = asyncio.new_event_loop()
    import threading as _threading
    thread = _threading.Thread(
        target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()),
        daemon=True,
    )
    thread.start()
    time.sleep(0.1)

    try:
        registry = get_agent_registry(orchestrator)
        executor = create_agent_executor(registry, loop)
        dispatcher = Dispatcher(bus, executor, heartbeat_interval=1.0)
        dispatcher.start()

        # Dispatch a write via bus tool
        home_agent = registry["home_control_agent"]
        tool = make_bus_tool(home_agent, "home", "Home control", bus)

        import json
        args = json.dumps({"input": "Turn on the office lights"})

        # Invoke tool from a different thread (simulating orchestrator call)
        t0 = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(
            tool.on_invoke_tool(None, args), loop
        )
        result = future.result(timeout=10)
        tool_elapsed = time.perf_counter() - t0

        assert_true(
            "task accepted" in result.lower(),
            f"Tool returned immediately: {result}"
        )
        log.info("  Tool call took %.3fs (should be near-instant)", tool_elapsed)

        # Wait for dispatcher to process
        time.sleep(5)

        # Check that it was executed
        recent = bus.get_recent_completed()
        assert_true(len(recent) >= 1, f"At least 1 completed ({len(recent)})")

        from core.message_bus import MessageStatus
        done = [m for m in recent if m.status == MessageStatus.DONE]
        assert_true(len(done) >= 1, "At least 1 successfully completed")
        assert_true("lights" in done[0].body.lower(), "Correct task was executed")
        log.info("  Completed result: %s", done[0].result[:100] if done[0].result else "None")

        print("  RESULT: PASS")
    finally:
        dispatcher.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_suite(suite: str = "all"):
    all_tests = {
        "quick": [
            test_write_classification,
            test_bus_tool_creation,
            test_notification_injection,
        ],
        "dispatch": [
            test_bus_tool_write_dispatch,
            test_bus_tool_read_sync,
            test_executor_creation,
        ],
        "integration": [
            test_orchestrator_tools_replaced,
            test_full_bus_dispatch_cycle,
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
    print(f"# Bus Integration Test Suite: {suite}")
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
            log.error("FAILED: %s — %s", test_fn.__name__, e, exc_info=True)

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
