"""Tests for agent auto-discovery — discover_agents() in core/agent_base.py."""

import importlib
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent_base import discover_agents, discover_underscore_agents


@pytest.fixture
def agents_dir(tmp_path):
    """Create a temporary agent_modules directory with test agents."""
    agents = tmp_path / "agent_modules"
    agents.mkdir()
    (agents / "__init__.py").write_text("")
    return agents


def _make_agent_file(agents_dir, name, content):
    """Write a Python file into the temp agent_modules dir."""
    (agents_dir / f"{name}.py").write_text(content)


def _make_fake_module(
    agent_config=None,
    requires_integration=None,
    requires_security=None,
    on_load=None,
    agent_instance="fake_agent",
):
    """Create a fake module object with the expected attributes."""
    mod = types.ModuleType("fake_agent_module")
    if agent_config is not None:
        mod.AGENT_CONFIG = agent_config
    if requires_integration is not None:
        mod.REQUIRES_INTEGRATION = requires_integration
    if requires_security is not None:
        mod.REQUIRES_SECURITY = requires_security
    if on_load is not None:
        mod.on_load = on_load
    mod.agent = agent_instance
    return mod


class TestDiscoverSkipsUnderscoreFiles:
    """discover_agents() should skip files starting with underscore."""

    def test_skips_underscore_prefixed(self, agents_dir):
        """Files like _claude_code.py should not be loaded by discover_agents."""
        # Create a normal agent module
        normal_mod = _make_fake_module(
            agent_config={
                "tool_name": "search",
                "tool_description": "Search stuff",
                "write_keywords": [],
            }
        )
        # Create an underscore agent module
        underscore_mod = _make_fake_module(
            agent_config={
                "tool_name": "build",
                "tool_description": "Build stuff",
                "write_keywords": [],
            }
        )

        _make_agent_file(agents_dir, "search", "# placeholder")
        _make_agent_file(agents_dir, "_dangerous", "# placeholder")

        with patch("core.agent_base.importlib.import_module") as mock_import:
            mock_import.return_value = normal_mod
            config = {"integrations": {}, "security": {}}
            results = discover_agents(agents_dir, config)

        # Only the non-underscore file should be discovered
        assert len(results) == 1
        assert results[0]["config"]["tool_name"] == "search"

        # import_module should only be called for 'search', not '_dangerous'
        import_calls = [c[0][0] for c in mock_import.call_args_list]
        assert not any("_dangerous" in c for c in import_calls)

    def test_skips_dunder_files(self, agents_dir):
        """Files like __init__.py should be skipped."""
        _make_agent_file(agents_dir, "__init__", "")
        _make_agent_file(agents_dir, "__pycache__", "")

        config = {"integrations": {}, "security": {}}
        # These all start with _ so discover_agents skips them
        with patch("core.agent_base.importlib.import_module") as mock_import:
            results = discover_agents(agents_dir, config)

        assert len(results) == 0
        mock_import.assert_not_called()


class TestDiscoverSkipsDisabledIntegration:
    """discover_agents() should skip agents whose integration is disabled."""

    def test_skips_when_integration_disabled(self, agents_dir):
        """Agent with REQUIRES_INTEGRATION='google' skipped when google not enabled."""
        mod = _make_fake_module(
            agent_config={
                "tool_name": "email",
                "tool_description": "Email stuff",
                "write_keywords": ["send"],
            },
            requires_integration="google",
        )

        _make_agent_file(agents_dir, "mail", "# placeholder")

        config = {
            "integrations": {"google": {"enabled": False}},
            "security": {},
        }

        with patch("core.agent_base.importlib.import_module", return_value=mod):
            results = discover_agents(agents_dir, config)

        assert len(results) == 0

    def test_loads_when_integration_enabled(self, agents_dir):
        """Agent with REQUIRES_INTEGRATION='google' loaded when google is enabled."""
        mod = _make_fake_module(
            agent_config={
                "tool_name": "email",
                "tool_description": "Email stuff",
                "write_keywords": ["send"],
            },
            requires_integration="google",
        )

        _make_agent_file(agents_dir, "mail", "# placeholder")

        config = {
            "integrations": {"google": {"enabled": True}},
            "security": {},
        }

        with patch("core.agent_base.importlib.import_module", return_value=mod):
            results = discover_agents(agents_dir, config)

        assert len(results) == 1
        assert results[0]["config"]["tool_name"] == "email"

    def test_skips_when_integration_missing(self, agents_dir):
        """Agent skipped when required integration not in config at all."""
        mod = _make_fake_module(
            agent_config={
                "tool_name": "home",
                "tool_description": "Home control",
                "write_keywords": [],
            },
            requires_integration="homeassistant",
        )

        _make_agent_file(agents_dir, "home_control", "# placeholder")

        config = {"integrations": {}, "security": {}}

        with patch("core.agent_base.importlib.import_module", return_value=mod):
            results = discover_agents(agents_dir, config)

        assert len(results) == 0


class TestDiscoverCallsOnLoad:
    """discover_agents() should call on_load(config) if exported."""

    def test_calls_on_load(self, agents_dir):
        """on_load(config) is called with the full config dict."""
        on_load_mock = MagicMock()
        mod = _make_fake_module(
            agent_config={
                "tool_name": "test",
                "tool_description": "Test agent",
                "write_keywords": [],
            },
            on_load=on_load_mock,
        )

        _make_agent_file(agents_dir, "test_agent", "# placeholder")

        config = {"integrations": {}, "security": {}, "custom": "value"}

        with patch("core.agent_base.importlib.import_module", return_value=mod):
            results = discover_agents(agents_dir, config)

        assert len(results) == 1
        on_load_mock.assert_called_once_with(config)

    def test_on_load_exception_doesnt_crash(self, agents_dir):
        """If on_load raises, the agent is still discovered."""
        def bad_on_load(cfg):
            raise ValueError("boom")

        mod = _make_fake_module(
            agent_config={
                "tool_name": "test",
                "tool_description": "Test agent",
                "write_keywords": [],
            },
            on_load=bad_on_load,
        )

        _make_agent_file(agents_dir, "test_agent", "# placeholder")
        config = {"integrations": {}, "security": {}}

        with patch("core.agent_base.importlib.import_module", return_value=mod):
            results = discover_agents(agents_dir, config)

        # Agent is still loaded despite on_load failure
        assert len(results) == 1


class TestDiscoverLoadsEnabledAgents:
    """discover_agents() loads agents with no requirements or satisfied requirements."""

    def test_loads_agent_with_no_requirements(self, agents_dir):
        """Agent with no REQUIRES_* is always loaded."""
        mod = _make_fake_module(
            agent_config={
                "tool_name": "search",
                "tool_description": "Search the web",
                "write_keywords": [],
            },
        )

        _make_agent_file(agents_dir, "search", "# placeholder")
        config = {"integrations": {}, "security": {}}

        with patch("core.agent_base.importlib.import_module", return_value=mod):
            results = discover_agents(agents_dir, config)

        assert len(results) == 1
        assert results[0]["config"]["tool_name"] == "search"
        assert results[0]["agent"] == "fake_agent"

    def test_loads_multiple_agents(self, agents_dir):
        """Multiple qualifying agents are all discovered."""
        mod_search = _make_fake_module(
            agent_config={
                "tool_name": "search",
                "tool_description": "Search",
                "write_keywords": [],
            },
        )
        mod_mail = _make_fake_module(
            agent_config={
                "tool_name": "email",
                "tool_description": "Email",
                "write_keywords": ["send"],
            },
            requires_integration="google",
        )

        _make_agent_file(agents_dir, "search", "# placeholder")
        _make_agent_file(agents_dir, "mail", "# placeholder")

        config = {
            "integrations": {"google": {"enabled": True}},
            "security": {},
        }

        def fake_import(name):
            if "search" in name:
                return mod_search
            return mod_mail

        with patch("core.agent_base.importlib.import_module", side_effect=fake_import):
            results = discover_agents(agents_dir, config)

        assert len(results) == 2
        tool_names = {r["config"]["tool_name"] for r in results}
        assert tool_names == {"search", "email"}

    def test_skips_module_without_agent_config(self, agents_dir):
        """Modules without AGENT_CONFIG are silently skipped."""
        mod = types.ModuleType("bare_module")
        # No AGENT_CONFIG attribute

        _make_agent_file(agents_dir, "bare", "# no config")
        config = {"integrations": {}, "security": {}}

        with patch("core.agent_base.importlib.import_module", return_value=mod):
            results = discover_agents(agents_dir, config)

        assert len(results) == 0


class TestDiscoverUnderscoreAgents:
    """discover_underscore_agents() loads _*.py files that pass their gates."""

    def test_loads_underscore_with_security_enabled(self, agents_dir):
        """_system_tools.py loaded when enable_system_tools is true."""
        mod = _make_fake_module(
            agent_config={
                "tool_name": "_system_tools",
                "tool_description": "System tools",
                "write_keywords": [],
            },
            requires_security="enable_system_tools",
        )

        _make_agent_file(agents_dir, "_system_tools", "# placeholder")
        config = {
            "integrations": {},
            "security": {"enable_system_tools": True},
        }

        with patch("core.agent_base.importlib.import_module", return_value=mod):
            results = discover_underscore_agents(agents_dir, config)

        assert len(results) == 1

    def test_skips_underscore_with_security_disabled(self, agents_dir):
        """_system_tools.py skipped when enable_system_tools is false."""
        mod = _make_fake_module(
            agent_config={
                "tool_name": "_system_tools",
                "tool_description": "System tools",
                "write_keywords": [],
            },
            requires_security="enable_system_tools",
        )

        _make_agent_file(agents_dir, "_system_tools", "# placeholder")
        config = {
            "integrations": {},
            "security": {"enable_system_tools": False},
        }

        with patch("core.agent_base.importlib.import_module", return_value=mod):
            results = discover_underscore_agents(agents_dir, config)

        assert len(results) == 0

    def test_skips_template(self, agents_dir):
        """_template.py is always skipped."""
        _make_agent_file(agents_dir, "_template", "# template")
        config = {"integrations": {}, "security": {}}

        with patch("core.agent_base.importlib.import_module") as mock_import:
            results = discover_underscore_agents(agents_dir, config)

        assert len(results) == 0
        mock_import.assert_not_called()
