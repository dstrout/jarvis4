"""
Template Agent — copy this file to create a new agent.

Rename to <name>.py (no underscore prefix) for auto-discovery.
Files starting with _ are only loaded when explicitly enabled via
REQUIRES_SECURITY or REQUIRES_INTEGRATION.

Built on agent_base: run_command + lookup_skill + find_skills.
"""

import logging

from agents import Agent, RunContextWrapper, function_tool
from core.agent_base import make_agent

log = logging.getLogger("jarvis.template")


# ---------------------------------------------------------------------------
# Typed tools — add your tools here
# ---------------------------------------------------------------------------

@function_tool
def example_tool(query: str) -> str:
    """Example tool description. Replace with your actual tool."""
    return f"You asked: {query}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_instructions(context: RunContextWrapper, agent: Agent) -> str:
    return """\
You are a specialist agent. Replace this with your actual instructions.

## Your tools
- example_tool(query): Describe what it does.
- run_command: Execute shell commands (from agent_base).
- lookup_skill / find_skills: Access skill documentation (from agent_base).

## Key rules
- Report concisely. No markdown. Plain text only.
"""


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

template_agent = make_agent(
    name="template_agent",
    instructions=_build_instructions,
    extra_tools=[example_tool],
)

# Auto-discovery metadata
AGENT_CONFIG = {
    "tool_name": "template",
    "tool_description": "Replace with a description the orchestrator uses to decide when to call this agent.",
    "write_keywords": [],  # Keywords that indicate write operations (fire-and-forget in bus mode)
}

# Uncomment if this agent requires an integration to be enabled:
# REQUIRES_INTEGRATION = "google"  # or "slack", "homeassistant", etc.

# Uncomment if this agent requires a security setting:
# REQUIRES_SECURITY = "enable_system_tools"

# Optional: called during discovery with the full config dict
# def on_load(config: dict):
#     pass

# Export the agent instance for discovery
agent = template_agent
