"""
Message Bus — async task dispatch and scheduling for Jarvis agents.

Decouples the orchestrator from sub-agent execution. Write operations
(send email, post to Slack, toggle lights) are fire-and-forget: the
orchestrator puts a message on the bus and returns immediately. Read
operations (check inbox, get weather) bypass the bus and execute
synchronously as before.

Messages have a run_at timestamp for deferred/scheduled execution and
a repeat count for recurring tasks (-1 = forever, 0 = delete after run,
N = run N more times).

The dispatcher runs on a 1-second heartbeat, pulling the top message
from the stack and executing it if its run_at time has passed.

Architecture:
- LIFO for new commands (user commands always on top)
- Scheduled items sit at the bottom with future timestamps
- After execution, recurring items get updated run_at and re-inserted
- Failures produce a new message addressed to the orchestrator
- Task groups (shared task_id) allow sequential multi-step workflows
  with abort-on-failure semantics
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

log = logging.getLogger("jarvis.bus")


class MessageStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Message:
    """A task message on the bus."""

    # Routing
    to: str                          # Target agent name
    from_agent: str                  # Originating agent name
    body: str                        # The instruction text (same as sync tool call)

    # Execution control
    wait: bool = False               # If True, bypass queue, execute synchronously
    dry_run: bool = False            # If True, agent simulates but doesn't execute

    # Scheduling
    run_at: datetime = field(default_factory=datetime.now)
    repeat: int = 0                  # -1=forever, 0=one-shot, N=run N more times
    interval: Optional[timedelta] = None  # For recurring: time between runs

    # Task grouping
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    sequence: int = 0                # Ordering within a task group

    # State
    status: MessageStatus = MessageStatus.QUEUED
    result: Optional[str] = None     # Agent response after execution
    error: Optional[str] = None      # Error message if failed
    created_at: datetime = field(default_factory=datetime.now)

    # Internal
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __repr__(self):
        sched = ""
        if self.repeat != 0:
            sched = f" repeat={self.repeat}"
            if self.interval:
                sched += f" every {self.interval}"
        return (
            f"Message(id={self.id}, to={self.to}, "
            f"status={self.status.value}{sched}, "
            f"body={self.body[:50]}...)"
        )


# Type alias for agent executor functions
# Takes (agent_name, instruction_text, dry_run) -> (success, response_text)
AgentExecutor = Callable[[str, str, bool], tuple[bool, str]]


class MessageBus:
    """
    LIFO message stack with scheduling support.

    New messages go on top. The dispatcher pops from the top,
    skipping items whose run_at is in the future.
    """

    def __init__(self):
        self._stack: list[Message] = []
        self._lock = threading.Lock()
        self._completed: list[Message] = []  # Recent completions for inspection
        self._max_completed = 50

        # Pending results for the orchestrator to pick up
        self._pending_notifications: list[Message] = []
        self._notify_lock = threading.Lock()

        # Immediate failure callback (e.g., push to voice pipeline alert queue)
        self._failure_callback: Optional[Callable[[Message], None]] = None

    def set_failure_callback(self, callback: Callable[[Message], None]):
        """Register a callback invoked immediately when a task fails.

        If the callback succeeds, the notification is NOT added to
        _pending_notifications (avoids double-delivery). If it raises,
        we fall back to the pending queue.
        """
        self._failure_callback = callback

    def put(self, msg: Message) -> str:
        """
        Push a message onto the bus. Returns the message ID.

        New messages go on top of the stack (LIFO), so user commands
        always execute before scheduled items.
        """
        with self._lock:
            self._stack.append(msg)
            log.info("Queued: %s", msg)
        return msg.id

    def put_group(self, messages: list[Message]) -> str:
        """
        Push a group of sequential messages sharing the same task_id.

        They are inserted in sequence order so the first step is on top.
        Returns the shared task_id.
        """
        if not messages:
            return ""

        task_id = messages[0].task_id
        # Ensure all share the same task_id and have correct sequence
        for i, msg in enumerate(messages):
            msg.task_id = task_id
            msg.sequence = i

        with self._lock:
            # Insert in reverse so sequence 0 is on top
            for msg in reversed(messages):
                self._stack.append(msg)
                log.info("Queued (group %s, seq %d): %s", task_id, msg.sequence, msg)

        return task_id

    def get_next_ready(self) -> Optional[Message]:
        """
        Pop the highest (most recent) message whose run_at has passed.

        Scans from top of stack down. Skips messages with future run_at.
        Returns None if no message is ready.
        """
        now = datetime.now()

        with self._lock:
            # Scan from top (end of list) down
            for i in range(len(self._stack) - 1, -1, -1):
                msg = self._stack[i]
                if msg.status == MessageStatus.QUEUED and msg.run_at <= now:
                    msg.status = MessageStatus.RUNNING
                    self._stack.pop(i)
                    return msg

        return None

    def cancel_group(self, task_id: str, reason: str = "prior step failed"):
        """
        Cancel all remaining queued messages in a task group.

        Called when one step in a sequential group fails —
        don't execute the remaining steps.
        """
        cancelled = []
        with self._lock:
            for msg in self._stack:
                if msg.task_id == task_id and msg.status == MessageStatus.QUEUED:
                    msg.status = MessageStatus.CANCELLED
                    msg.error = reason
                    cancelled.append(msg)

            # Remove cancelled from stack
            self._stack = [
                m for m in self._stack
                if not (m.task_id == task_id and m.status == MessageStatus.CANCELLED)
            ]

        for msg in cancelled:
            log.info("Cancelled (group %s): %s", task_id, msg)
            self._record_completed(msg)

        return len(cancelled)

    def mark_done(self, msg: Message, result: str):
        """Mark a message as successfully completed."""
        msg.status = MessageStatus.DONE
        msg.result = result
        self._record_completed(msg)

        # Handle recurring: re-queue with updated run_at
        if msg.repeat != 0 and msg.interval:
            next_msg = Message(
                to=msg.to,
                from_agent=msg.from_agent,
                body=msg.body,
                dry_run=msg.dry_run,
                run_at=datetime.now() + msg.interval,
                repeat=msg.repeat - 1 if msg.repeat > 0 else -1,
                interval=msg.interval,
                task_id=msg.task_id,
                sequence=msg.sequence,
            )
            # Scheduled items go at the bottom (beginning of list)
            with self._lock:
                self._stack.insert(0, next_msg)
            log.info("Re-scheduled (next: %s): %s", next_msg.run_at, next_msg)

    def mark_failed(self, msg: Message, error: str):
        """Mark a message as failed and notify via callback or pending queue."""
        msg.status = MessageStatus.FAILED
        msg.error = error
        self._record_completed(msg)

        # Cancel remaining group steps
        cancelled_count = self.cancel_group(msg.task_id, f"step {msg.sequence} failed: {error}")

        # Create notification
        notification = Message(
            to="jarvis",
            from_agent="dispatcher",
            body=self._format_failure_notification(msg, cancelled_count),
        )

        # Try immediate callback first (proactive alert to voice pipeline)
        delivered = False
        if self._failure_callback:
            try:
                self._failure_callback(notification)
                delivered = True
                log.warning("Failure alert delivered via callback: %s", notification.body[:100])
            except Exception as e:
                log.error("Failure callback error, falling back to queue: %s", e)

        # Fall back to pending queue if no callback or callback failed
        if not delivered:
            with self._notify_lock:
                self._pending_notifications.append(notification)
            log.warning("Failure notification queued: %s", notification.body[:100])

    def get_pending_notifications(self) -> list[Message]:
        """
        Drain pending notifications for the orchestrator.

        Called at the start of each user interaction to inject
        failure reports from background tasks.
        """
        with self._notify_lock:
            notifications = list(self._pending_notifications)
            self._pending_notifications.clear()
        return notifications

    def get_queue_status(self) -> dict:
        """Return a snapshot of the queue state for debugging."""
        with self._lock:
            queued = [m for m in self._stack if m.status == MessageStatus.QUEUED]
            scheduled = [m for m in queued if m.run_at > datetime.now()]
            ready = [m for m in queued if m.run_at <= datetime.now()]

        return {
            "total_queued": len(queued),
            "ready": len(ready),
            "scheduled": len(scheduled),
            "pending_notifications": len(self._pending_notifications),
            "recent_completed": len(self._completed),
        }

    def get_recent_completed(self) -> list[Message]:
        """Return recently completed messages for debugging."""
        return list(self._completed)

    def _record_completed(self, msg: Message):
        """Track completed messages (capped)."""
        self._completed.append(msg)
        if len(self._completed) > self._max_completed:
            self._completed = self._completed[-self._max_completed:]

    def _format_failure_notification(self, msg: Message, cancelled_count: int) -> str:
        """Format a human-readable failure notification."""
        parts = [f"Background task failed: {msg.to} could not complete: {msg.body[:200]}"]
        if msg.error:
            parts.append(f"Error: {msg.error}")
        if cancelled_count > 0:
            parts.append(f"{cancelled_count} follow-up step(s) were cancelled.")
        return " ".join(parts)


class Dispatcher:
    """
    Polls the message bus and executes tasks via agent executors.

    Runs on a 1-second heartbeat in a background thread.
    Blocks on each agent call (single-lane execution).
    """

    def __init__(
        self,
        bus: MessageBus,
        executor: AgentExecutor,
        heartbeat_interval: float = 1.0,
    ):
        self.bus = bus
        self.executor = executor
        self.heartbeat_interval = heartbeat_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._in_flight: Optional[Message] = None  # Currently executing message
        self._in_flight_lock = threading.Lock()
        self._stats = {
            "dispatched": 0,
            "succeeded": 0,
            "failed": 0,
            "heartbeats": 0,
        }

    def start(self):
        """Start the dispatcher background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="dispatcher"
        )
        self._thread.start()
        log.info("Dispatcher started (heartbeat=%.1fs)", self.heartbeat_interval)

    def stop(self, timeout: float = 5.0):
        """Stop the dispatcher and wait for it to finish."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        log.info("Dispatcher stopped. Stats: %s", self._stats)

    def _run_loop(self):
        """Main dispatcher loop — heartbeat polling."""
        while self._running:
            self._stats["heartbeats"] += 1

            msg = self.bus.get_next_ready()
            if msg:
                self._dispatch(msg)

            time.sleep(self.heartbeat_interval)

    @property
    def is_idle(self) -> bool:
        """True if no message is currently being executed."""
        with self._in_flight_lock:
            return self._in_flight is None

    def _dispatch(self, msg: Message):
        """Execute a single message via the appropriate agent."""
        with self._in_flight_lock:
            self._in_flight = msg
        self._stats["dispatched"] += 1
        log.info("Dispatching: %s -> %s: %s", msg.from_agent, msg.to, msg.body[:80])

        try:
            success, result = self.executor(msg.to, msg.body, msg.dry_run)

            if success:
                self.bus.mark_done(msg, result)
                self._stats["succeeded"] += 1
                log.info(
                    "Completed: %s -> %s (result: %s)",
                    msg.from_agent, msg.to, result[:100]
                )
            else:
                self.bus.mark_failed(msg, result)
                self._stats["failed"] += 1
                log.warning(
                    "Failed: %s -> %s (error: %s)",
                    msg.from_agent, msg.to, result[:200]
                )

        except Exception as e:
            self.bus.mark_failed(msg, str(e))
            self._stats["failed"] += 1
            log.error("Dispatch error for %s: %s", msg, e, exc_info=True)
        finally:
            with self._in_flight_lock:
                self._in_flight = None

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def is_running(self) -> bool:
        return self._running
