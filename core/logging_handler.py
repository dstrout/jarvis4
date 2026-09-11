"""
Interaction logger for Jarvis agents.

Logs every LLM call with full request/response details including reasoning,
tool calls, and timing. Output is structured JSONL for easy analysis and
prompt tuning.

Log files:
  logs/agents_YYYYMMDD.jsonl  — one JSON object per LLM interaction
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import Agent, AgentHooks, RunContextWrapper, Tool
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.items import TResponseInputItem, TResponseOutputItem

log = logging.getLogger("jarvis.log")

LOG_DIR = Path(__file__).parent.parent / "logs"


class InteractionLogger:
    """Logs all agent interactions to structured JSONL files."""

    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_date: str = ""
        self._file = None

    def _get_file(self):
        today = datetime.now().strftime("%Y%m%d")
        if today != self._current_date:
            if self._file:
                self._file.close()
            self._current_date = today
            path = self.log_dir / f"agents_{today}.jsonl"
            self._file = open(path, "a", buffering=1)  # line-buffered
        return self._file

    def log_event(self, event: dict[str, Any]):
        """Write a single event as a JSON line."""
        event["timestamp"] = datetime.now().isoformat()
        try:
            line = json.dumps(event, default=str, ensure_ascii=False)
            self._get_file().write(line + "\n")
        except Exception as e:
            log.error("Failed to write log: %s", e)

    def log_llm_call(
        self,
        agent_name: str,
        model: str,
        messages: list[dict],
        response_content: str | None,
        reasoning: str | None,
        tool_calls: list[dict] | None,
        elapsed_ms: float,
        usage: dict | None = None,
    ):
        """Log a complete LLM request/response cycle."""
        self.log_event({
            "type": "llm_call",
            "agent": agent_name,
            "model": model,
            "elapsed_ms": round(elapsed_ms, 1),
            "messages": _truncate_messages(messages),
            "response": {
                "content": response_content,
                "reasoning": reasoning,
                "tool_calls": tool_calls,
            },
            "usage": usage,
        })

    def log_tool_execution(
        self,
        agent_name: str,
        tool_name: str,
        arguments: dict,
        result: str,
        elapsed_ms: float,
    ):
        """Log a tool execution."""
        self.log_event({
            "type": "tool_call",
            "agent": agent_name,
            "tool": tool_name,
            "arguments": arguments,
            "result": result[:2000] if len(result) > 2000 else result,
            "elapsed_ms": round(elapsed_ms, 1),
        })

    def log_agent_delegation(
        self,
        parent_agent: str,
        child_agent: str,
        input_text: str,
        output_text: str | None,
        elapsed_ms: float,
    ):
        """Log an agent-to-agent delegation (agent-as-tool call)."""
        self.log_event({
            "type": "delegation",
            "parent": parent_agent,
            "child": child_agent,
            "input": input_text[:1000],
            "output": output_text[:2000] if output_text else None,
            "elapsed_ms": round(elapsed_ms, 1),
        })

    def log_user_interaction(
        self,
        user_input: str,
        final_output: str | None,
        agents_involved: list[str],
        total_elapsed_ms: float,
    ):
        """Log the complete user-facing interaction."""
        self.log_event({
            "type": "interaction",
            "input": user_input,
            "output": final_output,
            "agents": agents_involved,
            "total_elapsed_ms": round(total_elapsed_ms, 1),
        })

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


def _truncate_messages(messages: list[dict], max_content: int = 500) -> list[dict]:
    """Truncate message content for logging (keep structure, trim text)."""
    result = []
    for msg in messages:
        entry = {"role": msg.get("role", "?")}
        content = msg.get("content")
        if isinstance(content, str) and len(content) > max_content:
            entry["content"] = content[:max_content] + f"... ({len(content)} chars)"
        elif content is not None:
            entry["content"] = content
        if msg.get("tool_calls"):
            entry["tool_calls"] = msg["tool_calls"]
        if msg.get("reasoning"):
            r = msg["reasoning"]
            entry["reasoning"] = r[:max_content] + "..." if len(r) > max_content else r
        if msg.get("tool_call_id"):
            entry["tool_call_id"] = msg["tool_call_id"]
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Hooks that plug into the Agents SDK lifecycle
# ---------------------------------------------------------------------------

class LoggingHooks(AgentHooks):
    """Agent lifecycle hooks that log interactions."""

    def __init__(self, logger: InteractionLogger):
        self.logger = logger
        self._call_start: float = 0

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self._call_start = time.perf_counter()
        log.debug("Agent '%s' started", agent.name)

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        elapsed = (time.perf_counter() - self._call_start) * 1000
        log.debug("Agent '%s' completed in %.0fms", agent.name, elapsed)

    async def on_tool_start(
        self, context: RunContextWrapper, agent: Agent, tool: Tool
    ) -> None:
        log.debug("Agent '%s' calling tool '%s'", agent.name, tool.name)

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        log.debug(
            "Agent '%s' tool '%s' returned %d chars",
            agent.name, tool.name, len(result) if result else 0,
        )


# Singleton logger instance
_logger: InteractionLogger | None = None


def get_logger() -> InteractionLogger:
    """Get or create the singleton interaction logger."""
    global _logger
    if _logger is None:
        _logger = InteractionLogger()
    return _logger


def create_logging_hooks() -> LoggingHooks:
    """Create agent hooks wired to the singleton logger."""
    return LoggingHooks(get_logger())
