"""
Orchestrator Agent — the voice and mind of Jarvis.

This is the only agent the user interacts with. It handles direct conversation,
classifies intent, delegates to specialist agents, and synthesizes all responses
into a consistent, conversational voice.

Design principles:
- The orchestrator ALWAYS owns the conversation. Sub-agents are tools, not peers.
- Simple conversation (greetings, follow-ups, opinions) is handled directly — no delegation.
- Complex tasks are delegated to specialists via the agents-as-tools pattern.
- The orchestrator synthesizes sub-agent results into natural speech.
- Personality is defined HERE, not in sub-agents.

Auto-discovery:
- Agent modules in agent_modules/ are discovered via convention (AGENT_CONFIG export).
- No manual imports or wiring needed for new agents.
- Underscore-prefixed agents (_claude_code, _system_tools) require security gates.
- Call build_orchestrator(config) to create the orchestrator with discovered agents.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents import Agent, RunContextWrapper
from core.agents_init import (
    MODEL_ORCHESTRATOR,
    ORCHESTRATOR_MODEL_SETTINGS,
    make_model,
)
from core.logging_handler import create_logging_hooks
from core.agent_base import (
    lookup_skill,
    find_skills,
    discover_agents,
    discover_underscore_agents,
)

log = logging.getLogger("jarvis.orchestrator")

_AGENTS_DIR = Path(__file__).parent.parent / "agent_modules"


def _build_instructions(discovered: list[dict]):
    """Create the dynamic system prompt factory with discovered agent info."""

    # Build the tool listing for the system prompt
    tool_lines = []
    for entry in discovered:
        cfg = entry["config"]
        tool_name = cfg["tool_name"]
        # Skip direct tools (system_tools) — they're listed separately
        if cfg.get("direct_tools"):
            continue
        if entry["agent"] is None:
            continue
        desc = cfg["tool_description"].split(".")[0]  # First sentence
        tool_lines.append(f'  - "{tool_name}" for {desc.lower()}')

    tool_listing = "\n".join(tool_lines)

    def build(context: RunContextWrapper, agent: Agent) -> str:
        now = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")

        personality_file = Path(__file__).parent.parent / "personality.md"
        personality = ""
        if personality_file.exists():
            personality = personality_file.read_text().strip()

        if not personality:
            personality = (
                "You are a sharp, resourceful, occasionally dry, always competent assistant. "
                "Think the presence of a trusted chief of staff who happens to know everything."
            )

        # Check if system tools are available
        has_system_tools = any(
            e["config"].get("direct_tools") for e in discovered
        )
        system_tools_block = ""
        if has_system_tools:
            system_tools_block = """\
- You also have direct system tools:
  - "run_bash" to execute shell commands for system tasks, file ops, or anything \
not covered by a specialist
  - "read_file" to read file contents
  - "write_file" to create or update files"""

        return f"""\
Current date and time: {time_str}

{personality}

## Operational rules
- For simple conversation, greetings, opinions, or follow-ups: respond directly. \
Do not delegate. Do not use tools. Just talk.
- For anything requiring external information or action: delegate to the \
appropriate specialist. You have specialist tools available:
{tool_listing}
{system_tools_block}
- You also have direct tools:
  - "lookup_skill" to load full instructions for a skill or routine by name
  - "find_skills" to search available skills by keyword
- When you get results from a specialist, synthesize them into natural speech. \
Don't just parrot data back — interpret it, contextualize it, make it useful.
- For ALL email operations — including drafting, composing, and sending — you \
MUST call the email tool. NEVER compose email text yourself. NEVER show the \
user an email body you wrote. The email tool creates real Gmail drafts and \
sends real messages. If you write the email yourself, nothing actually happens \
in Gmail. Always delegate, even if you think you could write the email. \
For send and forward: confirm intent with the user first, then call the tool.
- For calendar and time information: be brief. Name and time only unless asked \
for details.
- If a request is ambiguous, make your best judgment call. Don't ask for \
clarification unless genuinely necessary. You're trusted to use judgment.
- When multiple things need to happen, handle them in sequence. Don't explain \
your plan — just execute it.
- When a tool reports "Task accepted" or similar, it means the action is being \
handled in the background. Acknowledge naturally — "Done" or "On it" or \
"Consider it handled" — and move on. Don't wait for confirmation.
- When a tool returns DATA (messages, status, information), you MUST relay \
that information to the user in your response. Summarize and speak it naturally. \
Never just say "Done" when a tool gives you data the user asked for.
"""

    return build


def build_orchestrator(
    config: dict,
    bus: Optional["MessageBus"] = None,
) -> Agent:
    """Build the orchestrator agent with auto-discovered sub-agents.

    Args:
        config: Full config dict (from core.config.load()).
        bus: Optional MessageBus for fire-and-forget write operations.

    Returns:
        The orchestrator Agent, ready to use.
    """
    # Discover standard agents (non-underscore)
    discovered = discover_agents(_AGENTS_DIR, config)

    # Discover underscore agents that pass their gates
    underscore = discover_underscore_agents(_AGENTS_DIR, config)
    discovered.extend(underscore)

    log.info(
        "Discovered %d agents: %s",
        len(discovered),
        ", ".join(e["config"]["tool_name"] for e in discovered),
    )

    # Build tools from discovered agents
    tools = []

    for entry in discovered:
        cfg = entry["config"]
        agent_instance = entry["agent"]

        # Handle direct tools (system_tools pattern)
        direct_tools = cfg.get("direct_tools")
        if direct_tools:
            tools.extend(direct_tools)
            continue

        # Skip entries with no agent instance
        if agent_instance is None:
            continue

        if bus is not None:
            from core.bus_integration import make_bus_tool
            tool = make_bus_tool(
                agent_instance,
                tool_name=cfg["tool_name"],
                tool_description=cfg["tool_description"],
                bus=bus,
                write_keywords=cfg.get("write_keywords") or None,
            )
        else:
            tool = agent_instance.as_tool(
                tool_name=cfg["tool_name"],
                tool_description=cfg["tool_description"],
            )

        tools.append(tool)

    # Always include lookup_skill and find_skills as direct tools
    tools.extend([lookup_skill, find_skills])

    orchestrator = Agent(
        name="jarvis",
        instructions=_build_instructions(discovered),
        model=make_model(MODEL_ORCHESTRATOR, agent_name="jarvis"),
        model_settings=ORCHESTRATOR_MODEL_SETTINGS,
        hooks=create_logging_hooks(),
        tools=tools,
    )

    # Stash discovered agents for get_agent_registry
    orchestrator._discovered = discovered

    return orchestrator


def get_agent_registry(orchestrator: Agent) -> dict[str, Agent]:
    """Return a mapping of agent names to agent instances from a built orchestrator."""
    discovered = getattr(orchestrator, "_discovered", [])
    registry = {}
    for entry in discovered:
        agent_instance = entry["agent"]
        if agent_instance is not None:
            registry[agent_instance.name] = agent_instance
    return registry
