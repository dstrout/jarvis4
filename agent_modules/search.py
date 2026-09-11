"""
Search Agent — information retrieval specialist.

Handles web search, weather queries, and general factual lookups.
Has access to Brave Search and OpenWeatherMap via MCP servers.

Built on agent_base with MCP shortcut tools for web_search, get_weather,
get_forecast — faster than CLI for the common case.

Supports skills via progressive disclosure:
- System prompt includes a lightweight skill catalog (names + descriptions)
- Agent can look up full skill instructions on demand via lookup_skill tool
- Recipes provide multi-step research workflows (sport updates, morning briefings, etc.)
"""

import os

from agents import Agent, RunContextWrapper, function_tool
from core.agent_base import make_agent, BASE_TOOLS
from core.agents_init import (
    MODEL_AGENT,
    MODEL_ORCHESTRATOR,
    ORCHESTRATOR_MODEL_SETTINGS,
)
from core.skill_loader import get_skill_loader

from typing import Any


# ---------------------------------------------------------------------------
# MCP shortcut tools — faster than CLI for common operations
# ---------------------------------------------------------------------------

@function_tool
def web_search(query: str) -> str:
    """Search the web for current information, news, facts, or anything the user asks about.
    Use this for any question about current events, people, places, or topics you're not certain about.
    """
    return _call_mcp_tool("brave-search.brave_web_search", {"query": query})


@function_tool
def get_weather(city: str, country: str = "US", state: str = "", units: str = "imperial") -> str:
    """Get the current weather for a location. Use imperial units (Fahrenheit) by default for US locations."""
    location = {"city": city, "country": country}
    if state:
        location["state"] = state
    return _call_mcp_tool(
        "weather.get_current_weather",
        {"location": location, "units": units},
    )


@function_tool
def get_forecast(city: str, country: str = "US", state: str = "", units: str = "imperial", days: int = 3) -> str:
    """Get a multi-day weather forecast for a location."""
    location = {"city": city, "country": country}
    if state:
        location["state"] = state
    return _call_mcp_tool(
        "weather.get_forecast",
        {"location": location, "units": units, "days": days},
    )


# ---------------------------------------------------------------------------
# MCP bridge — connects shortcut tools to the shared MCPManager
# ---------------------------------------------------------------------------

_mcp_manager = None


def set_mcp_manager(manager: Any):
    """Inject the shared MCPManager instance. Called during startup."""
    global _mcp_manager
    _mcp_manager = manager


def _call_mcp_tool(full_name: str, arguments: dict) -> str:
    """Execute an MCP tool and return the text result."""
    if _mcp_manager is None:
        return "Error: MCP manager not initialized"

    result = _mcp_manager.call_tool(full_name, arguments)

    if result is None:
        return f"Error: Tool {full_name} returned no result"

    if isinstance(result, dict):
        if "error" in result:
            return f"Error: {result['error']}"
        content = result.get("content", [])
        if content and isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            if texts:
                return "\n".join(texts)
        import json
        return json.dumps(result)

    return str(result)


# ---------------------------------------------------------------------------
# Build system prompt with skill catalog
# ---------------------------------------------------------------------------

def _build_search_instructions(context: RunContextWrapper, agent: Agent) -> str:
    """Dynamic system prompt with embedded skill catalog."""
    loader = get_skill_loader()
    recipe_catalog = loader.catalog_for_prompt(
        categories=["recipe"],
        name_filter=[
            "sport", "morning-briefing", "deep-dive", "topic",
            "news", "weather", "research",
        ],
    )

    return f"""\
You are an information retrieval specialist. Your job is to find accurate, \
current information and return it in a structured, concise format.

## Core tools
- web_search: Search the web for any information
- get_weather: Current weather for a location
- get_forecast: Multi-day weather forecast
- run_command: Execute shell commands (for CLI tools, data processing, etc.)

## Recipes — YOUR FIRST STEP FOR EVERY REQUEST
You have pre-defined research recipes. Your FIRST action for every request \
must be to check if a recipe below matches. If it does, call lookup_skill \
with the exact recipe name to load full instructions, then follow those steps.

DO NOT skip recipes and go straight to web_search. A recipe always produces \
a better answer than a single search.

### Your recipes:
{recipe_catalog}

### Recipe matching rules:
- "What's going on in [sport/league]" → recipe-sport-update
- "Morning briefing" or "catch me up" → recipe-morning-briefing
- Any broad question needing multiple perspectives → recipe-topic-deep-dive
- If unsure, call find_skills with keywords from the query

If NO recipe matches at all, then use web_search directly.

## Guidelines
- Formulate precise search queries — don't just repeat the user's words verbatim
- If the first search doesn't answer the question, try a refined query
- For weather: extract the city name and use get_weather or get_forecast
- For simple factual questions: use web_search directly
- For broad or complex queries: use a recipe

Your response will be read by another agent who will speak it to the user, \
so return the key facts clearly and concisely. Do not add conversational fluff. \
Do not use markdown formatting. Just state the facts.

For weather, report: conditions, temperature, humidity, and wind. \
For search results, summarize the most relevant findings in 2-3 sentences.
"""


# ---------------------------------------------------------------------------
# Agent definition — built on agent_base
# ---------------------------------------------------------------------------

# Allow switching search agent model via env: SEARCH_AGENT_MODEL=gpt-oss-120b
_search_model = os.environ.get("SEARCH_AGENT_MODEL", MODEL_AGENT)
_search_settings = (
    ORCHESTRATOR_MODEL_SETTINGS if _search_model == MODEL_ORCHESTRATOR
    else None  # make_agent uses AGENT_MODEL_SETTINGS by default
)

search_agent = make_agent(
    name="search_agent",
    instructions=_build_search_instructions,
    extra_tools=[web_search, get_weather, get_forecast],
    model_id=_search_model,
    model_settings=_search_settings,
)

# Auto-discovery metadata
AGENT_CONFIG = {
    "tool_name": "search",
    "tool_description": (
        "Search for information: web search, weather, news, facts, "
        "current events, or any question requiring up-to-date data. "
        "Pass the user's question or a clear description of what to look up."
    ),
    "write_keywords": [],
}
agent = search_agent
