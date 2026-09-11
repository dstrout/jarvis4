"""
System Tools Agent — direct file I/O and shell access.

Provides run_bash, read_file, write_file as Agent SDK tools.
Disabled by default — requires security.enable_system_tools = true.

These tools are added directly to the orchestrator's tool list
(not as a sub-agent), giving the orchestrator direct system access.
"""

import logging
import os
import subprocess

from agents import Agent, function_tool
from core.agent_base import make_agent

log = logging.getLogger("jarvis.system_tools")


# ---------------------------------------------------------------------------
# Tools — direct system access for the orchestrator
# ---------------------------------------------------------------------------

@function_tool
def run_bash(command: str) -> str:
    """Execute a shell command and return its output.
    Use for system tasks, file operations, or anything not covered by a specialist tool.
    """
    log.info("bash: %s", command)
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"Error: {err}" if err else f"Error: exit code {result.returncode}"
        if not output:
            return "Success (no output)"
        if len(output) > 4000:
            return output[:4000] + "\n... (truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30s"
    except Exception as e:
        return f"Error: {e}"


@function_tool
def read_file(path: str, max_lines: int = 200) -> str:
    """Read a file and return its contents.
    Args:
        path: Absolute or relative file path.
        max_lines: Maximum lines to return (default 200). Use 0 for unlimited.
    """
    path = os.path.expanduser(path)
    try:
        with open(path, "r") as f:
            if max_lines > 0:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... (truncated at {max_lines} lines)")
                        break
                    lines.append(line)
                return "".join(lines).rstrip()
            else:
                content = f.read()
                if len(content) > 50000:
                    return content[:50000] + "\n... (truncated at 50KB)"
                return content.rstrip()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error: {e}"


@function_tool
def write_file(path: str, content: str, append: bool = False) -> str:
    """Write content to a file. Creates parent directories if needed.
    Args:
        path: Absolute or relative file path.
        content: Text content to write.
        append: If true, append to existing file instead of overwriting.
    """
    path = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode) as f:
            f.write(content)
        action = "Appended to" if append else "Wrote"
        return f"{action} {path} ({len(content)} bytes)"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error: {e}"


# Auto-discovery metadata (underscore file — dangerous, not auto-loaded)
AGENT_CONFIG = {
    "tool_name": "_system_tools",
    "tool_description": (
        "Direct system tools: run shell commands, read files, write files. "
        "Use for system tasks not covered by specialist agents."
    ),
    "write_keywords": [
        "write", "create", "delete", "remove", "move", "rename",
    ],
    # These tools are added directly to the orchestrator, not as a sub-agent
    "direct_tools": [run_bash, read_file, write_file],
}
REQUIRES_SECURITY = "enable_system_tools"

# No agent instance — tools are added directly to the orchestrator
agent = None
