"""
Jarvis4 configuration — single source of truth.

Loads config.json from project root, applies environment variable overrides.
Resolution order (last wins): defaults -> config.json -> env vars.

Env var format: JARVIS4_<SECTION>_<KEY> (e.g., JARVIS4_LLM_API_KEY)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("jarvis4.config")

_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG: Optional[dict] = None
_DEFAULTS = {
    "llm": {
        "provider": "cerebras",
        "api_key": "",
        "orchestrator_model": "gpt-oss-120b",
        "agent_model": "qwen-3-235b-a22b-instruct-2507",
        "base_url": "https://api.cerebras.ai/v1",
        # Cerebras-only reasoning fields. Ignored (and stripped) for other
        # providers, which reject requests carrying them.
        "reasoning": True,
    },
    "asr": {
        "provider": "whisper-live",
        "host": "localhost",
        "port": 9090,
        "model": "small",
    },
    "tts": {
        "provider": "kokoro-wyoming",
        "host": "localhost",
        "port": 10200,
        "speed": 1.5,
    },
    "voice": {
        "wake_word": "hey_jarvis",
        "wake_threshold": 0.5,
        "silence_timeout": 1.2,
        "followup_timeout": 3.0,
        "vad_threshold": None,
        "chime": "sounds/chime_correct.wav",
        "audio_driver": "generic",
    },
    "integrations": {
        "homeassistant": {"enabled": False, "url": "", "token": ""},
        "slack": {"enabled": False, "bot_token": "", "app_token": ""},
        "google": {"enabled": False, "oauth_credentials": ""},
    },
    "email": {
        "greeting": "",
        "closing": "Best regards,\nJarvis4",
    },
    "contacts": {},
    "devices": {},
    "http": {"port": 8787},
    "mcp_servers": {},
    "security": {"enable_system_tools": False},
    "logging": {"level": "INFO"},
}

# Env var mapping: JARVIS4_SECTION_KEY -> config path
_ENV_OVERRIDES = {
    "JARVIS4_LLM_API_KEY": ("llm", "api_key"),
    "JARVIS4_LLM_PROVIDER": ("llm", "provider"),
    "JARVIS4_LLM_BASE_URL": ("llm", "base_url"),
    "JARVIS4_LLM_REASONING": ("llm", "reasoning"),
    "JARVIS4_LLM_ORCHESTRATOR_MODEL": ("llm", "orchestrator_model"),
    "JARVIS4_LLM_AGENT_MODEL": ("llm", "agent_model"),
    "JARVIS4_ASR_HOST": ("asr", "host"),
    "JARVIS4_ASR_PORT": ("asr", "port"),
    "JARVIS4_TTS_HOST": ("tts", "host"),
    "JARVIS4_TTS_PORT": ("tts", "port"),
    "JARVIS4_HTTP_PORT": ("http", "port"),
    "JARVIS4_HA_TOKEN": ("integrations", "homeassistant", "token"),
    "JARVIS4_HA_URL": ("integrations", "homeassistant", "url"),
    "JARVIS4_SLACK_BOT_TOKEN": ("integrations", "slack", "bot_token"),
    "JARVIS4_SLACK_APP_TOKEN": ("integrations", "slack", "app_token"),
    # Legacy compatibility
    "CEREBRAS_API_KEY": ("llm", "api_key"),
    "HA_TOKEN": ("integrations", "homeassistant", "token"),
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursing into nested dicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides to config."""
    for env_var, path in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is None:
            continue

        d = config
        for key in path[:-1]:
            d = d.setdefault(key, {})

        existing = d.get(path[-1])
        if isinstance(existing, int):
            value = int(value)
        elif isinstance(existing, float):
            value = float(value)
        elif isinstance(existing, bool):
            value = value.lower() in ("true", "1", "yes")

        d[path[-1]] = value
        log.debug("Env override: %s -> %s", env_var, ".".join(path))

    return config


def load(config_path: Optional[str] = None) -> dict:
    """Load config from file with env var overrides. Call once at startup."""
    global _CONFIG

    if config_path is None:
        config_path = _PROJECT_ROOT / "config.json"
    else:
        config_path = Path(config_path)

    config = json.loads(json.dumps(_DEFAULTS))

    if config_path.exists():
        with open(config_path) as f:
            file_config = json.load(f)
        config = _deep_merge(config, file_config)
        log.info("Loaded config from %s", config_path)
    else:
        log.warning("No config file at %s — using defaults", config_path)

    config = _apply_env_overrides(config)

    _CONFIG = config
    return config


def get(*path: str, default: Any = None) -> Any:
    """Get a config value by dotted path. e.g., get("llm", "api_key")"""
    if _CONFIG is None:
        raise RuntimeError("Config not loaded. Call config.load() first.")

    d = _CONFIG
    for key in path:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


def get_section(section: str) -> dict:
    """Get an entire config section as a dict."""
    return get(section, default={})


def project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT


def is_loaded() -> bool:
    """Check if config has been loaded."""
    return _CONFIG is not None
