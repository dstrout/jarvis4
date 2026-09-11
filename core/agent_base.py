"""
Base tools and agent factory for all Jarvis sub-agents.

Every sub-agent gets the same core tool set:
- run_command: bash execution
- lookup_skill: load full SKILL.md instructions
- find_skills: search skill catalog by keyword

Agents are differentiated by their system prompt and skill catalog filter.
Additional shortcut tools (e.g., MCP-based web_search) can be added per agent
for operations that are faster or not reachable via bash.

Also provides discover_agents() for convention-based auto-registration.
"""

import importlib
import logging
import subprocess
from pathlib import Path

from agents import Agent, ModelSettings, function_tool, RunContextWrapper
from core.agents_init import MODEL_AGENT, AGENT_MODEL_SETTINGS, make_model
from core.logging_handler import create_logging_hooks
from core.skill_loader import get_skill_loader

log = logging.getLogger("jarvis.agent")


# ---------------------------------------------------------------------------
# Core tools — shared by all agents
# ---------------------------------------------------------------------------

@function_tool
def run_command(command: str) -> str:
    """Execute a shell command and return its output. Use this to run CLI
    commands (gws, curl, jq, etc.) after looking up the correct syntax
    via lookup_skill. The command is run as-is in a bash shell.
    """
    log.info("exec: %s", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            error = result.stderr.strip()
            if error:
                return f"Error: {error}"
            return f"Error: command exited with code {result.returncode}"
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
def lookup_skill(skill_name: str) -> str:
    """Look up full instructions for a skill or recipe by name.
    Use this to learn the correct CLI syntax before running commands.
    """
    loader = get_skill_loader()
    instructions = loader.get_instructions(skill_name)
    if instructions:
        return instructions
    return f"Skill '{skill_name}' not found. Use find_skills to search."


@function_tool
def find_skills(query: str) -> str:
    """Search available skills by keyword. Returns matching skill names
    and descriptions. Use when you're not sure which skill applies.
    """
    loader = get_skill_loader()
    results = loader.search(query, limit=5)
    if not results:
        return "No matching skills found."
    lines = []
    for r in results:
        lines.append(f"- {r['name']}: {r['description']}")
    return "\n".join(lines)


# The base tool set every agent gets
BASE_TOOLS = [run_command, lookup_skill, find_skills]


def make_agent(
    name: str,
    instructions,
    extra_tools: list | None = None,
    model_id: str | None = None,
    model_settings: ModelSettings | None = None,
) -> Agent:
    """Create a sub-agent with the standard tool set.

    Args:
        name: Agent name (used in logs and delegation)
        instructions: Static string or dynamic callable(context, agent) -> str
        extra_tools: Additional tools beyond the base set (e.g., MCP shortcuts)
        model_id: Override model (defaults to MODEL_AGENT)
        model_settings: Override settings (defaults to AGENT_MODEL_SETTINGS)
    """
    tools = list(BASE_TOOLS)
    if extra_tools:
        tools.extend(extra_tools)

    return Agent(
        name=name,
        instructions=instructions,
        model=make_model(model_id or MODEL_AGENT, agent_name=name),
        model_settings=model_settings or AGENT_MODEL_SETTINGS,
        hooks=create_logging_hooks(),
        tools=tools,
        tool_use_behavior="run_llm_again",
    )


def discover_agents(agents_dir: Path, config: dict) -> list[dict]:
    """Scan agent_modules/ for auto-registrable agents.

    Returns list of {"agent": Agent | None, "config": AGENT_CONFIG, "module": module} dicts.

    Convention:
    - Files starting with '_' are skipped (dangerous/template agents).
    - Files must export AGENT_CONFIG dict and `agent` instance.
    - If REQUIRES_INTEGRATION is set, the named integration must be
      enabled in config["integrations"][name]["enabled"].
    - If REQUIRES_SECURITY is set, the named security setting must be
      true in config["security"][name].
    - If the module exports on_load(config), it is called after import.
    """
    results = []
    agents_dir = Path(agents_dir)

    for py_file in sorted(agents_dir.glob("*.py")):
        name = py_file.stem

        # Skip __init__, __pycache__, and underscore-prefixed files
        if name.startswith("_"):
            log.debug("Skipping underscore file: %s", name)
            continue

        try:
            module = importlib.import_module(f"agent_modules.{name}")
        except Exception as e:
            log.warning("Failed to import agent_modules.%s: %s", name, e)
            continue

        agent_config = getattr(module, "AGENT_CONFIG", None)
        if agent_config is None:
            log.debug("No AGENT_CONFIG in agent_modules.%s — skipping", name)
            continue

        # Check integration requirement
        req_integration = getattr(module, "REQUIRES_INTEGRATION", None)
        if req_integration:
            integrations = config.get("integrations", {})
            integration_config = integrations.get(req_integration, {})
            if not integration_config.get("enabled", False):
                log.info(
                    "Skipping agent_modules.%s — integration '%s' not enabled",
                    name, req_integration,
                )
                continue

        # Check security requirement
        req_security = getattr(module, "REQUIRES_SECURITY", None)
        if req_security:
            security = config.get("security", {})
            if not security.get(req_security, False):
                log.info(
                    "Skipping agent_modules.%s — security '%s' not enabled",
                    name, req_security,
                )
                continue

        # Call on_load if exported
        on_load = getattr(module, "on_load", None)
        if callable(on_load):
            try:
                on_load(config)
            except Exception as e:
                log.warning("on_load failed for agent_modules.%s: %s", name, e)

        agent_instance = getattr(module, "agent", None)

        results.append({
            "agent": agent_instance,
            "config": agent_config,
            "module": module,
        })
        log.info("Discovered agent: %s (tool_name=%s)", name, agent_config.get("tool_name"))

    return results


def discover_underscore_agents(agents_dir: Path, config: dict) -> list[dict]:
    """Scan agent_modules/ for underscore-prefixed agents that pass their gates.

    Same logic as discover_agents but only loads _*.py files.
    Used for dangerous/privileged agents like _claude_code and _system_tools.
    """
    results = []
    agents_dir = Path(agents_dir)

    for py_file in sorted(agents_dir.glob("_*.py")):
        name = py_file.stem

        # Skip __init__ and __pycache__
        if name.startswith("__"):
            continue
        # Skip template
        if name == "_template":
            continue

        try:
            module = importlib.import_module(f"agent_modules.{name}")
        except Exception as e:
            log.warning("Failed to import agent_modules.%s: %s", name, e)
            continue

        agent_config = getattr(module, "AGENT_CONFIG", None)
        if agent_config is None:
            continue

        # Check integration requirement
        req_integration = getattr(module, "REQUIRES_INTEGRATION", None)
        if req_integration:
            integrations = config.get("integrations", {})
            integration_config = integrations.get(req_integration, {})
            if not integration_config.get("enabled", False):
                log.info("Skipping %s — integration '%s' not enabled", name, req_integration)
                continue

        # Check security requirement
        req_security = getattr(module, "REQUIRES_SECURITY", None)
        if req_security:
            security = config.get("security", {})
            if not security.get(req_security, False):
                log.info("Skipping %s — security '%s' not enabled", name, req_security)
                continue

        # Call on_load if exported
        on_load = getattr(module, "on_load", None)
        if callable(on_load):
            try:
                on_load(config)
            except Exception as e:
                log.warning("on_load failed for %s: %s", name, e)

        agent_instance = getattr(module, "agent", None)

        results.append({
            "agent": agent_instance,
            "config": agent_config,
            "module": module,
        })
        log.info("Discovered underscore agent: %s (tool_name=%s)", name, agent_config.get("tool_name"))

    return results
