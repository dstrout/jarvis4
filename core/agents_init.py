"""
Initialize the Agents SDK — called once at startup.

Configures the Agents SDK to use the LLM provider specified in config.json.
Applies Cerebras-specific patches when using Cerebras as provider.
"""

import logging

from openai import AsyncOpenAI
from agents import (
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
    ModelSettings,
)

from core import config
from core.model import Jarvis4Model as CerebrasModel

log = logging.getLogger("jarvis4.agents_init")

# Shared client instance (set during initialize())
_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Get the shared AsyncOpenAI client. Must call initialize() first."""
    if _client is None:
        raise RuntimeError("Call initialize() before get_client()")
    return _client


def make_model(model_id: str, agent_name: str) -> CerebrasModel:
    """Create a model instance with logging for a specific agent."""
    return CerebrasModel(
        model=model_id,
        openai_client=get_client(),
        agent_name=agent_name,
    )


def initialize():
    """Initialize the Agents SDK for the configured LLM provider.

    Must be called once before any agent operations.
    Reads provider, API key, base URL from core.config.
    """
    global _client

    api_key = config.get("llm", "api_key") or "not-configured"
    base_url = config.get("llm", "base_url")
    provider = config.get("llm", "provider")

    _client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    set_default_openai_client(_client)

    # Cerebras implements OpenAI chat completions, not the Responses API
    set_default_openai_api("chat_completions")

    # Tracing sends data to OpenAI — disable it
    set_tracing_disabled(True)

    # Apply Cerebras-specific patches if using Cerebras
    if provider == "cerebras":
        from core.patches.cerebras_reasoning import apply_patches
        apply_patches()

    _apply_reasoning_settings(provider)

    log.info(
        "Initialized: provider=%s, orchestrator=%s, agents=%s, reasoning=%s",
        provider,
        config.get("llm", "orchestrator_model"),
        config.get("llm", "agent_model"),
        bool(ORCHESTRATOR_MODEL_SETTINGS.extra_body),
    )


# Settings objects that carry Cerebras-only request extensions, paired with
# the extras to restore when the provider does support them.
_REASONING_REGISTRY: list[tuple[ModelSettings, dict]] = []

# Provider resolved by initialize(); None until then.
_provider: str | None = None


def register_reasoning_settings(
    settings: ModelSettings, cerebras_extra: dict
) -> ModelSettings:
    """Register a ModelSettings whose extra_body is Cerebras-specific.

    The `reasoning_effort` / `reasoning_format` fields are Cerebras extensions.
    A generic OpenAI-compatible backend (llama.cpp, vLLM, Ollama) rejects
    requests carrying them, so they are stripped unless the provider is
    Cerebras and `llm.reasoning` is left enabled.

    Agent modules call this at import time; initialize() then resolves every
    registered object once the provider is known. Registering after
    initialize() applies the current setting immediately, so import order does
    not matter.

    Returns the same settings object, for use as an assignment expression.
    """
    _REASONING_REGISTRY.append((settings, dict(cerebras_extra)))
    if _provider is not None:
        _resolve_one(settings, cerebras_extra, _provider)
    return settings


def _resolve_one(settings: ModelSettings, cerebras_extra: dict, provider: str) -> None:
    """Apply or strip one settings object's Cerebras extras, in place."""
    use_reasoning = provider == "cerebras" and config.get(
        "llm", "reasoning", default=True
    )
    settings.extra_body = dict(cerebras_extra) if use_reasoning else {}


def _apply_reasoning_settings(provider: str) -> None:
    """Resolve every registered settings object for the configured provider.

    Mutates in place so that modules which imported the settings at import
    time observe the change.
    """
    global _provider
    _provider = provider
    for settings, cerebras_extra in _REASONING_REGISTRY:
        _resolve_one(settings, cerebras_extra, provider)


# --- Model IDs (read from config at import time would fail before load) ---
# Use these functions instead of constants to read from loaded config.

def get_orchestrator_model() -> str:
    return config.get("llm", "orchestrator_model", default="gpt-oss-120b")

def get_agent_model() -> str:
    return config.get("llm", "agent_model", default="qwen-3-235b-a22b-instruct-2507")


# Backward-compatible constants — these are the defaults.
# At runtime, use get_orchestrator_model() / get_agent_model() to read from config.
MODEL_ORCHESTRATOR = "gpt-oss-120b"
MODEL_AGENT = "qwen-3-235b-a22b-instruct-2507"


# --- Model settings ---

# Cerebras-only request extensions. initialize() strips these for other
# providers via _apply_reasoning_settings().
_CEREBRAS_ORCHESTRATOR_EXTRA = {
    "reasoning_effort": "medium",
    "reasoning_format": "parsed",
}
_CEREBRAS_AGENT_EXTRA = {
    "reasoning_format": "hidden",
}

# Orchestrator: reasoning enabled for complex routing decisions
ORCHESTRATOR_MODEL_SETTINGS = ModelSettings(
    temperature=0.7,
    top_p=0.9,
    extra_body=dict(_CEREBRAS_ORCHESTRATOR_EXTRA),
)

# Sub-agents: more deterministic, reasoning enabled for tool planning
AGENT_MODEL_SETTINGS = ModelSettings(
    temperature=0.3,
    top_p=0.9,
    extra_body=dict(_CEREBRAS_AGENT_EXTRA),
)

register_reasoning_settings(ORCHESTRATOR_MODEL_SETTINGS, _CEREBRAS_ORCHESTRATOR_EXTRA)
register_reasoning_settings(AGENT_MODEL_SETTINGS, _CEREBRAS_AGENT_EXTRA)
