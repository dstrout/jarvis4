"""Tests for core/config.py — loading, env overrides, defaults."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load, get, get_section, _deep_merge, _apply_env_overrides


class TestDeepMerge:
    def test_flat_merge(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested(self):
        base = {"llm": {"provider": "cerebras", "api_key": ""}}
        override = {"llm": {"api_key": "sk-123"}}
        result = _deep_merge(base, override)
        assert result["llm"]["provider"] == "cerebras"
        assert result["llm"]["api_key"] == "sk-123"

    def test_nested_new_key(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": 1, "c": 2}}


class TestLoad:
    def test_defaults_when_no_file(self, tmp_path):
        config = load(str(tmp_path / "nonexistent.json"))
        assert config["llm"]["provider"] == "cerebras"
        assert config["http"]["port"] == 8787
        assert config["voice"]["audio_driver"] == "generic"

    def test_file_overrides_defaults(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"http": {"port": 9999}}))
        config = load(str(config_file))
        assert config["http"]["port"] == 9999
        assert config["llm"]["provider"] == "cerebras"

    def test_env_overrides_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"llm": {"api_key": "from-file"}}))
        with patch.dict(os.environ, {"JARVIS4_LLM_API_KEY": "from-env"}):
            config = load(str(config_file))
        assert config["llm"]["api_key"] == "from-env"

    def test_legacy_env_var(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        with patch.dict(os.environ, {"CEREBRAS_API_KEY": "legacy-key"}):
            config = load(str(config_file))
        assert config["llm"]["api_key"] == "legacy-key"

    def test_int_coercion(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        with patch.dict(os.environ, {"JARVIS4_HTTP_PORT": "9090"}):
            config = load(str(config_file))
        assert config["http"]["port"] == 9090
        assert isinstance(config["http"]["port"], int)


class TestGet:
    def test_get_nested(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"llm": {"api_key": "test"}}))
        load(str(config_file))
        assert get("llm", "api_key") == "test"

    def test_get_default(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        load(str(config_file))
        assert get("nonexistent", "key", default="fallback") == "fallback"

    def test_get_section(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"http": {"port": 8787}}))
        load(str(config_file))
        section = get_section("http")
        assert section == {"port": 8787}

    def test_get_before_load_raises(self):
        import core.config as cfg
        old = cfg._CONFIG
        cfg._CONFIG = None
        try:
            with pytest.raises(RuntimeError, match="Config not loaded"):
                get("llm", "api_key")
        finally:
            cfg._CONFIG = old
