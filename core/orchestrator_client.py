"""
Orchestrator Client — bridges the sync voice pipeline to the async multi-agent system.

Drop-in replacement for CerebrasClient / LLMClient. Exposes the same
.chat(message) -> str interface but routes through the full orchestrator
with sub-agent delegation (search, email, etc.).

Conversation history is maintained via the Agents SDK's input_list pattern.

Supports two modes:
- Default (use_bus=False): All tool calls are synchronous (blocking).
- Bus mode (use_bus=True): Write operations are fire-and-forget via
  the message bus. Read operations remain synchronous. Failed background
  tasks generate notifications injected into the next conversation turn.
"""

import asyncio
import logging
import queue
import re
import threading
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.mcp_manager import MCPManager

log = logging.getLogger("jarvis.orchestrator_client")

# Pattern matching reasoning junk that leaks into content.
# Cerebras gpt-oss models sometimes emit chain-of-thought as content
# instead of in the reasoning field — hundreds of '…', '...', stray
# punctuation tokens, then eventually the real answer.
_JUNK_LINE = re.compile(
    r"^[\s.…?!—\-*\u2026\xa0]*$"      # only whitespace / ellipsis / punctuation / nbsp
    r"|^\{[\s\S]*?\}$"                  # bare JSON object (tool call leak)
    r"|^\"input\":\s"                   # tool call fragment
)

# A "real sentence" has at least 3 words and ends with sentence-end punct
# or is a short confident phrase like "Done." or "On it."
_REAL_SENTENCE = re.compile(r"(\w+\s+){2,}|^\w[\w\s,']{4,}[.!?]?$")


def _clean_response(text: str) -> str:
    """Strip reasoning / thinking tokens that leak into model output.

    Cerebras gpt-oss models sometimes emit their chain-of-thought directly
    in the content field: hundreds of ellipsis tokens, stray words,
    internal monologue ("We have a user request..."), and JSON tool-call
    fragments, followed by the actual spoken response at the end.

    Strategy: scan backwards from the end, collecting lines until we hit
    a run of junk.  Then also strip any internal-monologue lines from
    what we collected (they contain telltale phrases).
    """
    original_len = len(text)

    # Fast path: short clean responses don't need processing
    if original_len < 200 and "…" not in text and "..." not in text[:50]:
        return text.strip()

    lines = text.split("\n")

    # Walk backwards to find trailing real content
    clean = []
    for line in reversed(lines):
        if _JUNK_LINE.match(line):
            if clean:
                break
        else:
            clean.append(line)

    if not clean:
        return text.strip()

    clean.reverse()
    joined = "\n".join(clean).strip()

    # Second pass: strip internal-monologue / reasoning sentences.
    # These look like "We have a user request...", "Likely they want...",
    # "Let's interpret as..." — full English sentences the model was
    # thinking to itself.  The real spoken answer is after them.
    monologue_patterns = [
        r"(?:We have|We need to|We don't|The user|User says|Likely|"
        r"Let's|Could be|Probably|So respond|Already did|"
        r"Let me|I should|I need to|I'll|The previous|The last message|"
        r"Use \w+ tool|interpret as|ambiguous|maybe we can|"
        r"just repeat|Need to)",
    ]
    monologue_re = re.compile("|".join(monologue_patterns), re.IGNORECASE)

    # Split on sentence-ish boundaries and keep only the tail after monologue ends.
    # First try line-based splitting (multi-line responses).
    final_lines = joined.split("\n")
    last_monologue_idx = -1
    for i, line in enumerate(final_lines):
        if monologue_re.search(line):
            last_monologue_idx = i

    if last_monologue_idx >= 0 and last_monologue_idx < len(final_lines) - 1:
        final_lines = final_lines[last_monologue_idx + 1:]
    elif last_monologue_idx >= 0:
        # Monologue is on the same (possibly only) line as the real answer.
        # Split by sentence boundaries and find where monologue ends.
        text_block = final_lines[last_monologue_idx]
        # Split on sentence boundaries: period/!/? followed by optional
        # space and an uppercase letter. Handles "respond.Done" too.
        sentences = re.split(r'(?<=[.!?])\s*(?=[A-Z])', text_block)
        last_mono_sent = -1
        for i, sent in enumerate(sentences):
            if monologue_re.search(sent):
                last_mono_sent = i
        if last_mono_sent >= 0 and last_mono_sent < len(sentences) - 1:
            real_answer = " ".join(sentences[last_mono_sent + 1:])
            final_lines = [real_answer] + final_lines[last_monologue_idx + 1:]

    # Strip remaining JSON fragments and clean up lines that start with }
    cleaned_final = []
    for l in final_lines:
        stripped = l.strip()
        if stripped.startswith("{") or stripped.startswith('"input"'):
            continue
        # Remove leading } from lines like "}Consider it handled."
        if stripped.startswith("}"):
            stripped = stripped[1:].strip()
            if not stripped:
                continue
            l = stripped
        cleaned_final.append(l)
    final_lines = cleaned_final

    result = "\n".join(final_lines).strip()

    # If we stripped everything, fall back to the pre-monologue-cleaned version
    if not result:
        result = joined

    if len(result) < original_len * 0.5:
        log.warning(
            "Cleaned reasoning junk from response: %d→%d chars",
            original_len, len(result),
        )

    return result


class OrchestratorClient:
    """Sync wrapper around the async multi-agent orchestrator."""

    def __init__(
        self,
        mcp_manager: Optional["MCPManager"] = None,
        use_bus: bool = False,
    ):
        # Initialize the agents framework (must happen before imports)
        from core.agents_init import initialize
        initialize()

        # Wire MCP manager into search agent
        if mcp_manager:
            from agent_modules.search import set_mcp_manager
            set_mcp_manager(mcp_manager)

        # Build orchestrator via auto-discovery
        from core import config as cfg
        from core.orchestrator import build_orchestrator

        full_config = cfg._CONFIG or {}

        # Message bus for fire-and-forget writes
        self._bus = None
        self._dispatcher = None
        self._alert_queue: Optional[queue.Queue] = None

        # Dedicated event loop running in a background thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="orchestrator-loop"
        )
        self._thread.start()

        if use_bus:
            self._init_bus_early()
            self._orchestrator = build_orchestrator(full_config, bus=self._bus)
            self._init_bus_late()
        else:
            self._orchestrator = build_orchestrator(full_config)

        # Conversation history for multi-turn (Agents SDK input_list format)
        self._input_list: list = []

        # Max history turns to prevent unbounded growth
        self._max_history_items = 100

        log.info(
            "OrchestratorClient initialized (bus=%s, tools=%d)",
            "enabled" if use_bus else "disabled",
            len(self._orchestrator.tools),
        )

    def _init_bus_early(self):
        """Create the message bus (before orchestrator build)."""
        from core.message_bus import MessageBus
        self._bus = MessageBus()

    def _init_bus_late(self):
        """Start the dispatcher and wire up failure callbacks (after orchestrator build)."""
        from core.message_bus import Dispatcher
        from core.orchestrator import get_agent_registry
        from core.bus_integration import create_agent_executor

        # Create executor that bridges dispatcher thread -> async event loop
        agent_registry = get_agent_registry(self._orchestrator)
        executor = create_agent_executor(agent_registry, self._loop)

        self._dispatcher = Dispatcher(
            self._bus,
            executor,
            heartbeat_interval=1.0,
        )
        self._dispatcher.start()

        # Register failure callback for proactive alerts
        self._bus.set_failure_callback(self._on_task_failure)

    def _init_bus(self):
        """Initialize the message bus and dispatcher (legacy path)."""
        self._init_bus_early()
        from core import config as cfg
        from core.orchestrator import build_orchestrator
        full_config = cfg._CONFIG or {}
        self._orchestrator = build_orchestrator(full_config, bus=self._bus)
        self._init_bus_late()

    def set_alert_queue(self, q: queue.Queue):
        """Connect the voice pipeline's state-machine queue for proactive alerts.

        When a background task fails, the failure notification is pushed
        directly to this queue, bypassing the pending-notifications path
        so the user hears about it immediately.
        """
        self._alert_queue = q
        log.info("Alert queue connected for proactive failure notifications")

    def _on_task_failure(self, notification):
        """Called by the message bus when a background task fails."""
        if self._alert_queue is not None:
            self._alert_queue.put({
                "action": "task_failure",
                "text": notification.body,
            })
            log.info("Failure alert pushed to voice pipeline")
        else:
            raise RuntimeError("No alert queue — fall back to pending notifications")

    def _run_loop(self):
        """Run the asyncio event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def chat(self, user_message: str) -> Optional[str]:
        """
        Send a message through the orchestrator and return the response.

        Handles delegation to sub-agents transparently. Maintains
        conversation history across calls for multi-turn context.

        In bus mode, checks for pending notifications from background
        tasks and injects them into the conversation context.
        """
        if not user_message or not user_message.strip():
            return None

        user_message = user_message.strip()
        log.info("User: %s", user_message[:100])

        t0 = time.perf_counter()

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._run(user_message), self._loop
            )
            # Timeout: 120s should be enough for even complex multi-agent chains
            result = future.result(timeout=120)
        except asyncio.TimeoutError:
            log.error("Orchestrator timed out after 120s")
            return "I'm sorry, that request took too long. Could you try again?"
        except Exception as e:
            log.error("Orchestrator error: %s", e, exc_info=True)
            return None

        elapsed = time.perf_counter() - t0

        # Clean up reasoning junk that leaks into content
        if result:
            result = _clean_response(result)

        log.info("Response in %.2fs: %s", elapsed, (result or "")[:100])

        return result

    async def _run(self, user_message: str) -> Optional[str]:
        """Async orchestrator invocation with history management."""
        from agents import Runner

        # Inject pending notifications from background tasks
        self._inject_notifications()

        # Append user message to history
        self._input_list.append({"role": "user", "content": user_message})

        try:
            result = await Runner.run(self._orchestrator, self._input_list)

            # Update history with the full conversation (including tool calls)
            self._input_list = result.to_input_list()

            # Trim history if it gets too long
            if len(self._input_list) > self._max_history_items:
                self._input_list = self._trim_history(self._input_list)

            return result.final_output

        except Exception as e:
            log.error("Runner.run failed: %s", e, exc_info=True)

            # If history is corrupted (e.g., orphaned tool call IDs from trimming),
            # clear it entirely so the next request can succeed.
            err_str = str(e)
            if "422" in err_str or "not found in the messages" in err_str:
                log.warning("Corrupted history detected — clearing conversation state")
                self._input_list = []
            else:
                # Remove the user message we just added since it failed
                if self._input_list and self._input_list[-1].get("content") == user_message:
                    self._input_list.pop()
            return None

    def _trim_history(self, items: list) -> list:
        """Trim history while preserving tool call/result pairs.

        Naive slicing can orphan a tool_result whose tool_call was trimmed,
        causing Cerebras to reject the request with a 422. We find a safe
        cut point that doesn't split a pair.
        """
        target = self._max_history_items
        if len(items) <= target:
            return items

        # Start from the desired cut point and walk forward until we find
        # a "user" role message — that's always a safe boundary.
        cut = len(items) - target
        while cut < len(items):
            item = items[cut]
            role = item.get("role") if isinstance(item, dict) else None
            if role == "user":
                break
            cut += 1

        trimmed = items[cut:]
        if len(trimmed) < len(items):
            log.info("History trimmed: %d → %d items", len(items), len(trimmed))
        return trimmed

    def _inject_notifications(self):
        """
        Check for pending notifications from background tasks and inject
        them into the conversation context.

        Notifications appear as a system-like message before the user's
        message so the orchestrator can report them naturally.
        """
        if self._bus is None:
            return

        from core.bus_integration import format_notifications
        notifications = self._bus.get_pending_notifications()
        notice_text = format_notifications(notifications)

        if notice_text:
            log.info("Injecting notification: %s", notice_text[:100])
            self._input_list.append({
                "role": "user",
                "content": (
                    f"[SYSTEM NOTICE — not from the user, report this naturally] "
                    f"{notice_text}"
                ),
            })

    def test_connection(self) -> bool:
        """Test if the orchestrator can respond."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._test(), self._loop
            )
            return future.result(timeout=30)
        except Exception:
            return False

    async def _test(self) -> bool:
        """Quick test that the model responds."""
        from agents import Runner
        try:
            result = await Runner.run(self._orchestrator, "hi")
            return bool(result.final_output)
        except Exception as e:
            log.error("Connection test failed: %s", e)
            return False

    def clear_history(self):
        """Clear conversation history."""
        self._input_list = []

    def get_history_length(self) -> int:
        """Return number of items in history."""
        return len(self._input_list)

    @property
    def bus(self) -> Optional["MessageBus"]:
        """Access the message bus (None if bus mode disabled)."""
        return self._bus

    @property
    def dispatcher(self) -> Optional["Dispatcher"]:
        """Access the dispatcher (None if bus mode disabled)."""
        return self._dispatcher

    def get_bus_status(self) -> Optional[dict]:
        """Get bus queue status (None if bus mode disabled)."""
        if self._bus is None:
            return None
        status = self._bus.get_queue_status()
        if self._dispatcher:
            status["dispatcher"] = self._dispatcher.stats
        return status

    def close(self):
        """Shut down the background event loop and dispatcher."""
        if self._dispatcher:
            self._dispatcher.stop()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        log.info("OrchestratorClient closed")
