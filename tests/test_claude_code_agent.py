"""Tests for the Claude Code agent — spec writing, status checking, integration validation."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent_modules._claude_code import (
    _slugify,
    _make_build_dir,
    _do_write_build_spec,
    _do_check_build_status,
    _do_run_tests,
    _do_integrate_module,
    _do_restart_service,
    _BUILDS_DIR,
    _PROJECT_ROOT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_basic(self):
        assert _slugify("RSS Feed Reader") == "rss-feed-reader"

    def test_special_chars(self):
        assert _slugify("Build an RSS reader for foo.com!") == "build-an-rss-reader-for-foo-com"

    def test_truncation(self):
        long = "a" * 100
        assert len(_slugify(long)) <= 60

    def test_empty(self):
        assert _slugify("") == "module"
        assert _slugify("!!!") == "module"


class TestMakeBuildDir:
    def test_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent_modules._claude_code._BUILDS_DIR", tmp_path / "builds")
        build_dir = _make_build_dir("test-module")
        assert build_dir.exists()
        assert "test-module" in build_dir.name


# ---------------------------------------------------------------------------
# Tool: write_build_spec
# ---------------------------------------------------------------------------

class TestWriteBuildSpec:
    def test_agent_spec(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent_modules._claude_code._BUILDS_DIR", tmp_path)
        result = _do_write_build_spec(
            task_description="Build an RSS feed reader",
            module_type="agent",
        )
        assert "Spec written to" in result

        spec_dirs = list(tmp_path.iterdir())
        assert len(spec_dirs) == 1
        spec_file = spec_dirs[0] / "ASSIGNMENT.md"
        assert spec_file.exists()

        content = spec_file.read_text()
        assert "RSS feed reader" in content
        assert "make_agent()" in content
        assert "tests/test_" in content

    def test_skill_spec(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent_modules._claude_code._BUILDS_DIR", tmp_path)
        result = _do_write_build_spec(
            task_description="Create a Jira integration skill",
            module_type="skill",
        )
        assert "Spec written to" in result
        spec_dirs = list(tmp_path.iterdir())
        spec_file = spec_dirs[0] / "ASSIGNMENT.md"
        content = spec_file.read_text()
        assert "SKILL.md" in content
        assert "skills/" in content

    def test_standalone_spec(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent_modules._claude_code._BUILDS_DIR", tmp_path)
        result = _do_write_build_spec(
            task_description="Build a utility module",
            module_type="standalone",
        )
        assert "Spec written to" in result

    def test_invalid_module_type(self):
        result = _do_write_build_spec(
            task_description="Build something",
            module_type="invalid",
        )
        assert "Error" in result


# ---------------------------------------------------------------------------
# Tool: check_build_status
# ---------------------------------------------------------------------------

class TestCheckBuildStatus:
    def test_missing_dir(self):
        result = _do_check_build_status(build_dir="/nonexistent/path")
        assert "Error" in result

    def test_missing_metadata(self, tmp_path):
        result = _do_check_build_status(build_dir=str(tmp_path))
        assert "Error" in result
        assert "metadata" in result.lower()

    def test_with_metadata(self, tmp_path):
        meta = {
            "build_branch": "build/test-module",
            "starting_branch": "master",
            "exit_code": 0,
        }
        (tmp_path / "metadata.json").write_text(json.dumps(meta))

        with patch("agent_modules._claude_code._git") as mock_git:
            mock_git.return_value = (0, "file1.py\nfile2.py")
            result = _do_check_build_status(build_dir=str(tmp_path))

        assert "Build status" in result
        assert "build/test-module" in result


# ---------------------------------------------------------------------------
# Tool: integrate_module
# ---------------------------------------------------------------------------

class TestIntegrateModule:
    def test_requires_tests_passed(self, tmp_path):
        meta = {
            "build_branch": "build/test",
            "starting_branch": "master",
            "tests_passed": False,
        }
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        result = _do_integrate_module(build_dir=str(tmp_path), module_type="agent")
        assert "tests must pass" in result.lower()

    def test_missing_metadata(self, tmp_path):
        result = _do_integrate_module(build_dir=str(tmp_path))
        assert "Error" in result


# ---------------------------------------------------------------------------
# Tool: run_tests
# ---------------------------------------------------------------------------

class TestRunTests:
    def test_missing_metadata(self, tmp_path):
        result = _do_run_tests(build_dir=str(tmp_path))
        assert "Error" in result

    def test_missing_branch(self, tmp_path):
        meta = {"starting_branch": "master"}
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        result = _do_run_tests(build_dir=str(tmp_path))
        assert "Error" in result


# ---------------------------------------------------------------------------
# Tool: restart_service
# ---------------------------------------------------------------------------

class TestRestartService:
    @patch("agent_modules._claude_code.subprocess.run")
    @patch("agent_modules._claude_code.time.sleep")
    def test_success(self, mock_sleep, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="active\n", stderr=""),
        ]
        result = _do_restart_service()
        assert "running" in result.lower()

    @patch("agent_modules._claude_code.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Permission denied")
        result = _do_restart_service()
        assert "Error" in result


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

class TestAgentDefinition:
    def test_agent_exists(self):
        from agent_modules._claude_code import claude_code_agent
        assert claude_code_agent.name == "claude_code_agent"

    def test_agent_has_tools(self):
        from agent_modules._claude_code import claude_code_agent
        tool_names = [t.name for t in claude_code_agent.tools]
        assert "write_build_spec" in tool_names
        assert "invoke_claude_code" in tool_names
        assert "check_build_status" in tool_names
        assert "run_tests" in tool_names
        assert "integrate_module" in tool_names
        assert "restart_service" in tool_names
        # Also has base tools
        assert "run_command" in tool_names
        assert "lookup_skill" in tool_names

    def test_registered_in_orchestrator(self):
        from core.orchestrator import build_orchestrator
        config = {
            "integrations": {},
            "security": {"enable_system_tools": True},
        }
        orch = build_orchestrator(config)
        tool_names = [t.name for t in orch.tools]
        assert "build" in tool_names
