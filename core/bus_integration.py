"""
Bus Integration — wires the message bus into the orchestrator's tool calls.

Replaces agent.as_tool() with bus-aware tools that:
- Classify requests as read or write using keyword matching
- Reads: run the sub-agent synchronously (same as before)
- Writes: put a message on the bus and return immediately ("fire-and-forget")

The orchestrator LLM doesn't know about the bus — it just sees that write
tool calls return fast with "Task accepted" while read calls return data.
"""

import asyncio
import json
import logging
from typing import Any

from agents import Agent, Runner
from agents.exceptions import MaxTurnsExceeded
from agents.tool import FunctionTool

from core.message_bus import MessageBus, Dispatcher, Message

log = logging.getLogger("jarvis.bus_integration")


# Keywords that indicate a write operation, per agent
WRITE_KEYWORDS = {
    "mail_agent": [
        "send", "forward", "reply", "delete", "archive", "label",
        "draft", "compose", "move", "mark", "star",
    ],
    "calendar_tasks_agent": [
        "create", "schedule", "reschedule", "cancel", "delete",
        "add", "update", "move", "remove", "set",
    ],
    "comms_agent": [
        "send", "post", "notify", "dm ", "announce",
    ],
    "home_control_agent": [
        "turn on", "turn off", "set", "lock", "unlock",
        "open", "close", "toggle", "dim", "brighten",
    ],
}

# Default keywords for agents not explicitly listed
DEFAULT_WRITE_KEYWORDS = [
    "send", "create", "schedule", "delete", "forward", "reply",
    "post", "set", "turn", "lock", "unlock", "open", "close",
    "compose", "draft", "archive", "move", "remove", "update",
    "notify", "toggle",
]


def is_write_operation(agent_name: str, request: str) -> bool:
    """Classify a request as read or write based on keywords."""
    keywords = WRITE_KEYWORDS.get(agent_name, DEFAULT_WRITE_KEYWORDS)
    request_lower = request.lower()
    return any(kw in request_lower for kw in keywords)


def make_bus_tool(
    agent: Agent,
    tool_name: str,
    tool_description: str,
    bus: MessageBus,
    write_keywords: list[str] | None = None,
) -> FunctionTool:
    """
    Create a bus-aware tool that replaces agent.as_tool().

    Read requests run the sub-agent synchronously (blocking).
    Write requests dispatch to the message bus (fire-and-forget).
    """
    agent_name = agent.name
    keywords = write_keywords or WRITE_KEYWORDS.get(agent_name, DEFAULT_WRITE_KEYWORDS)

    async def on_invoke(ctx: Any, arguments_json: str) -> str:
        args = json.loads(arguments_json)
        request = args.get("input", "")

        request_lower = request.lower()
        is_write = any(kw in request_lower for kw in keywords)

        if is_write:
            msg = Message(
                to=agent_name,
                from_agent="jarvis",
                body=request,
            )
            bus.put(msg)
            log.info("Fire-and-forget: %s -> %s: %s", "jarvis", agent_name, request[:80])
            return "Task accepted. I'm handling that in the background."
        else:
            log.info("Sync read: %s -> %s: %s", "jarvis", agent_name, request[:80])
            try:
                result = await Runner.run(agent, request)
                return result.final_output or "No result."
            except Exception as e:
                log.error("Sub-agent %s failed: %s", agent_name, e, exc_info=True)
                return f"Error: {e}"

    return FunctionTool(
        name=tool_name,
        description=tool_description,
        params_json_schema={
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The request to process. Include all relevant details.",
                }
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        on_invoke_tool=on_invoke,
        strict_json_schema=True,
    )


def make_bus_tool_group(
    agent: Agent,
    tool_name: str,
    tool_description: str,
    bus: MessageBus,
) -> FunctionTool:
    """
    Create a bus-aware tool that supports multi-step task groups.

    Same as make_bus_tool but accepts an optional task_id for grouping
    sequential cross-agent tasks.
    """
    agent_name = agent.name
    keywords = WRITE_KEYWORDS.get(agent_name, DEFAULT_WRITE_KEYWORDS)

    async def on_invoke(ctx: Any, arguments_json: str) -> str:
        args = json.loads(arguments_json)
        request = args.get("input", "")
        task_id = args.get("task_id", "")

        request_lower = request.lower()
        is_write = any(kw in request_lower for kw in keywords)

        if is_write:
            msg = Message(
                to=agent_name,
                from_agent="jarvis",
                body=request,
            )
            if task_id:
                msg.task_id = task_id
            bus.put(msg)
            log.info("Fire-and-forget: %s -> %s: %s", "jarvis", agent_name, request[:80])
            return "Task accepted. I'm handling that in the background."
        else:
            log.info("Sync read: %s -> %s: %s", "jarvis", agent_name, request[:80])
            try:
                result = await Runner.run(agent, request)
                return result.final_output or "No result."
            except Exception as e:
                log.error("Sub-agent %s failed: %s", agent_name, e, exc_info=True)
                return f"Error: {e}"

    return FunctionTool(
        name=tool_name,
        description=tool_description,
        params_json_schema={
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The request to process. Include all relevant details.",
                },
                "task_id": {
                    "type": "string",
                    "description": (
                        "Optional task group ID. Use this when multiple tool calls "
                        "are part of the same user request and should be executed "
                        "sequentially. Use the same task_id for all steps."
                    ),
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        on_invoke_tool=on_invoke,
        strict_json_schema=True,
    )


def create_agent_executor(
    agent_registry: dict[str, Agent],
    loop: asyncio.AbstractEventLoop,
) -> callable:
    """
    Create an executor function for the Dispatcher.

    Bridges the sync dispatcher thread to the async event loop where
    agents run. Uses the same event loop as the orchestrator.
    """

    def executor(agent_name: str, instruction: str, dry_run: bool) -> tuple[bool, str]:
        agent = agent_registry.get(agent_name)
        if agent is None:
            return False, f"Unknown agent: {agent_name}"

        if dry_run:
            return True, f"[DRY RUN] {agent_name} would process: {instruction[:200]}"

        try:
            future = asyncio.run_coroutine_threadsafe(
                _run_agent(agent, instruction), loop
            )
            result = future.result(timeout=120)
            if result:
                return True, result
            else:
                return False, f"{agent_name} returned no output"
        except TimeoutError:
            return False, f"{agent_name} timed out after 120s"
        except MaxTurnsExceeded:
            return False, (
                f"{agent_name} exhausted its turn limit trying to complete: "
                f"{instruction[:100]}"
            )
        except Exception as e:
            return False, f"{agent_name} error: {e}"

    return executor


async def _run_agent(agent: Agent, instruction: str) -> str | None:
    """Run a sub-agent with the given instruction."""
    result = await Runner.run(agent, instruction)
    return result.final_output


def format_notifications(notifications: list[Message]) -> str | None:
    """
    Format pending notifications for injection into the orchestrator context.

    Returns None if no notifications pending.
    """
    if not notifications:
        return None

    parts = []
    for n in notifications:
        parts.append(n.body)

    return "Background task update: " + " | ".join(parts)
