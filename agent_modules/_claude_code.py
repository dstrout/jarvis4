"""
Claude Code Agent — self-building specialist.

Delegates software engineering tasks to the `claude` CLI tool, enabling
Jarvis to build new modules, agents, and capabilities for itself.

Lifecycle: spec → git branch → claude build → test → integrate → restart.

Uses --print mode for non-interactive execution. Builds happen on isolated
git branches (build/<slug>) that are left for manual review and merge.
"""

import json
import logging
import os
import re
import signal
import subprocess
import textwrap
import time
from datetime import datetime
from pathlib import Path

from agents import Agent, RunContextWrapper, function_tool
from core.agent_base import make_agent

log = logging.getLogger("jarvis.claude_code")

_PROJECT_ROOT = Path(__file__).parent.parent
_BUILDS_DIR = _PROJECT_ROOT / "builds"

# Limits
_DEFAULT_TIMEOUT_MIN = 10
_MAX_TIMEOUT_MIN = 30
_MAX_TURNS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert text to a filesystem/branch-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    slug = slug.strip("-")[:60]
    return slug or "module"


def _make_build_dir(slug: str) -> Path:
    """Create a timestamped build directory."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    build_dir = _BUILDS_DIR / f"{timestamp}-{slug}"
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


def _git(*args: str, cwd: Path | None = None) -> tuple[int, str]:
    """Run a git command and return (returncode, output)."""
    result = subprocess.run(
        ["git"] + list(args),
        cwd=cwd or _PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        err = result.stderr.strip()
        output = err if err else output
    return result.returncode, output


def _current_branch() -> str:
    """Get current git branch name."""
    rc, out = _git("branch", "--show-current")
    return out if rc == 0 else "unknown"


# ---------------------------------------------------------------------------
# Tool implementations — plain functions, testable without SDK
# ---------------------------------------------------------------------------

def _do_write_build_spec(task_description: str, module_type: str = "agent") -> str:
    if module_type not in ("agent", "skill", "standalone"):
        return f"Error: module_type must be agent, skill, or standalone — got '{module_type}'"

    slug = _slugify(task_description[:40])
    build_dir = _make_build_dir(slug)

    if module_type == "agent":
        integration_context = textwrap.dedent("""\
        ## Integration contract — AGENT module
        - Create a new file: `agent_modules/<name>.py`
        - Use `make_agent()` from `core.agent_base` (see existing agents for pattern)
        - Export the agent instance at module level (e.g., `rss_agent = make_agent(...)`)
        - Use `@function_tool` from `agents` for all tool definitions
        - Tools must be typed Python functions — never construct shell commands from LLM output
        - Follow the exact pattern in `agent_modules/search.py` or `agent_modules/home_control.py`
        - The agent's system prompt should be a dynamic function `_build_<name>_instructions(context, agent) -> str`
        - Tool return values must be plain text (no markdown, no emojis) — results are spoken aloud

        ## Registration (do this too)
        - In `core/orchestrator.py`:
          1. Import the agent at the top of `get_agent_registry()` (lazy import pattern)
          2. Add it to the registry dict
          3. Add it to `_default_tools` via `.as_tool(tool_name=..., tool_description=...)`
          4. Add a matching `make_bus_tool()` entry in `set_bus()`
          5. Add a line to the orchestrator's system prompt listing the new tool
        """)
    elif module_type == "skill":
        integration_context = textwrap.dedent("""\
        ## Integration contract — SKILL module
        - Create directory: `skills/<skill-name>/`
        - Create `skills/<skill-name>/SKILL.md` with proper agent-skills-sdk frontmatter:
          ```
          ---
          name: <skill-name>
          description: <one-line description>
          metadata:
            openclaw:
              category: productivity
          ---
          ```
        - The skill_loader auto-discovers from skills/ — no orchestrator changes needed
        - Skill instructions should include CLI examples and step-by-step usage
        """)
    else:
        integration_context = textwrap.dedent("""\
        ## Integration contract — STANDALONE module
        - Create the module as a Python file in the project root
        - Must be importable with no side effects on import
        - Include a clear docstring explaining the module's purpose and interface
        - Document how to wire it into the system (but don't auto-wire)
        """)

    spec = textwrap.dedent(f"""\
    # Build Assignment

    ## Task
    {task_description}

    ## Module type
    {module_type}

    {integration_context}

    ## Codebase context
    - Project root: {_PROJECT_ROOT}
    - Python 3.12, match existing code style (no type stubs, minimal comments)
    - Voice-first assistant — all tool outputs are spoken aloud, so keep them concise plain text
    - See `personality.md` for voice/tone guidelines
    - See `core/agent_base.py` for the base agent factory
    - See `core/agents_init.py` for model configuration
    - See existing agents in `agent_modules/` for patterns to follow
    - Dependencies: if you need new packages, add them to `requirements.txt`

    ## Testing requirements
    - Create `tests/test_<module_name>.py`
    - Tests must be runnable with `python -m pytest tests/test_<module_name>.py -v`
    - Test the tool functions directly (mock external APIs/services)
    - At minimum: test happy path, test error handling, test edge cases

    ## Rules
    - Do NOT modify unrelated files
    - Do NOT add unnecessary abstractions or over-engineer
    - Do NOT add docstrings/comments to code you didn't write
    - Keep it simple — match the style and complexity of existing agents
    - If you need credentials/tokens, follow the existing pattern (env var with file fallback)
    """)

    spec_path = build_dir / "ASSIGNMENT.md"
    spec_path.write_text(spec)

    log.info("Wrote build spec: %s", spec_path)
    return f"Spec written to {spec_path}"


def _do_invoke_claude_code(spec_path: str, timeout_minutes: int = 10) -> str:
    spec_path = Path(spec_path)
    if not spec_path.exists():
        return f"Error: spec not found at {spec_path}"

    build_dir = spec_path.parent
    timeout_minutes = min(max(timeout_minutes, 1), _MAX_TIMEOUT_MIN)
    timeout_seconds = timeout_minutes * 60

    spec_content = spec_path.read_text()

    dir_name = build_dir.name
    slug = "-".join(dir_name.split("-")[2:]) or "module"
    branch_name = f"build/{slug}"

    starting_branch = _current_branch()
    (build_dir / "metadata.json").write_text(json.dumps({
        "starting_branch": starting_branch,
        "build_branch": branch_name,
        "started_at": datetime.now().isoformat(),
        "spec_path": str(spec_path),
        "timeout_minutes": timeout_minutes,
    }, indent=2))

    rc, out = _git("checkout", "-b", branch_name)
    if rc != 0:
        rc, out = _git("checkout", branch_name)
        if rc != 0:
            return f"Error creating branch {branch_name}: {out}"

    log.info("Building on branch %s (timeout %dm)", branch_name, timeout_minutes)

    log_file = build_dir / "build.log"
    prompt = (
        f"Read and execute this build assignment. Work in {_PROJECT_ROOT}. "
        f"Create all files, write tests, and verify they pass.\n\n"
        f"{spec_content}"
    )

    cmd = [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--max-turns", str(_MAX_TURNS),
        "--model", "opus",
        "-p", prompt,
    ]

    try:
        with open(log_file, "w") as lf:
            lf.write(f"=== Claude Code Build ===\n")
            lf.write(f"Branch: {branch_name}\n")
            lf.write(f"Started: {datetime.now().isoformat()}\n")
            lf.write(f"Timeout: {timeout_minutes}m\n")
            lf.write(f"Command: {' '.join(cmd[:6])} ...\n")
            lf.write(f"{'=' * 40}\n\n")
            lf.flush()

            process = subprocess.Popen(
                cmd,
                cwd=_PROJECT_ROOT,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,
            )

            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(2)
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                lf.write(f"\n\n=== TIMED OUT after {timeout_minutes}m ===\n")
                log.warning("Claude Code timed out after %dm", timeout_minutes)

            exit_code = process.returncode

        log_content = log_file.read_text()
        log_tail = log_content[-3000:] if len(log_content) > 3000 else log_content

        rc, diff_stat = _git("diff", "--stat", f"{starting_branch}...{branch_name}")
        if rc != 0:
            diff_stat = "(could not determine changes)"

        meta_path = build_dir / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["finished_at"] = datetime.now().isoformat()
        meta["exit_code"] = exit_code
        meta_path.write_text(json.dumps(meta, indent=2))

        status = "completed" if exit_code == 0 else f"exited with code {exit_code}"
        summary = (
            f"Build {status} on branch '{branch_name}'.\n\n"
            f"Files changed:\n{diff_stat}\n\n"
            f"Build log tail:\n{log_tail[-1500:]}"
        )

        log.info("Build %s (exit code %s)", status, exit_code)
        return summary

    except FileNotFoundError:
        return "Error: 'claude' CLI not found. Is it installed and on PATH?"
    except Exception as e:
        log.exception("Build failed")
        return f"Error during build: {e}"
    finally:
        _git("checkout", starting_branch)


def _do_check_build_status(build_dir: str) -> str:
    build_dir = Path(build_dir)
    if not build_dir.exists():
        return f"Error: build directory not found: {build_dir}"

    meta_path = build_dir / "metadata.json"
    if not meta_path.exists():
        return "Error: no metadata.json — build may not have started"

    meta = json.loads(meta_path.read_text())
    branch = meta.get("build_branch", "unknown")
    starting = meta.get("starting_branch", "master")
    exit_code = meta.get("exit_code", "unknown")

    rc, diff_stat = _git("diff", "--stat", f"{starting}...{branch}")
    if rc != 0:
        diff_stat = "(branch may not exist)"

    rc, new_files = _git("diff", "--name-only", "--diff-filter=A", f"{starting}...{branch}")
    rc, mod_files = _git("diff", "--name-only", "--diff-filter=M", f"{starting}...{branch}")

    test_files = [f for f in (new_files + "\n" + mod_files).split("\n")
                  if f.startswith("tests/") and f.endswith(".py")]

    log_file = build_dir / "build.log"
    if log_file.exists():
        content = log_file.read_text()

    status = "success" if exit_code == 0 else "failed"
    if exit_code == 0 and not test_files:
        status = "partial (no tests found)"

    report = [
        f"Build status: {status}",
        f"Branch: {branch}",
        f"Exit code: {exit_code}",
        f"",
        f"New files:\n{new_files or '  (none)'}",
        f"",
        f"Modified files:\n{mod_files or '  (none)'}",
        f"",
        f"Test files: {', '.join(test_files) if test_files else '(none)'}",
        f"",
        f"Diff summary:\n{diff_stat}",
    ]

    return "\n".join(report)


def _do_run_tests(build_dir: str, test_path: str = "") -> str:
    build_dir = Path(build_dir)
    meta_path = build_dir / "metadata.json"
    if not meta_path.exists():
        return "Error: no metadata.json found"

    meta = json.loads(meta_path.read_text())
    branch = meta.get("build_branch", "")
    starting = meta.get("starting_branch", _current_branch())

    if not branch:
        return "Error: no build branch in metadata"

    rc, out = _git("checkout", branch)
    if rc != 0:
        return f"Error checking out {branch}: {out}"

    try:
        if not test_path:
            rc, new_files = _git("diff", "--name-only", "--diff-filter=A",
                                 f"{starting}...{branch}")
            test_files = [f for f in new_files.split("\n")
                          if f.startswith("tests/") and f.endswith(".py")]
            if not test_files:
                return "Error: no test files found in the build"
            test_path = " ".join(test_files)

        result = subprocess.run(
            f"python -m pytest {test_path} -v --tb=short 2>&1",
            shell=True,
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=120,
        )

        output = result.stdout + result.stderr
        if len(output) > 4000:
            output = output[-4000:]

        (build_dir / "test_results.txt").write_text(output)

        passed = result.returncode == 0
        meta["tests_passed"] = passed
        meta["test_output_tail"] = output[-500:]
        meta_path.write_text(json.dumps(meta, indent=2))

        status = "PASSED" if passed else "FAILED"
        return f"Tests {status}:\n{output}"

    except subprocess.TimeoutExpired:
        return "Error: tests timed out after 120s"
    except Exception as e:
        return f"Error running tests: {e}"
    finally:
        _git("checkout", starting)


def _do_integrate_module(build_dir: str, module_type: str = "agent") -> str:
    build_dir = Path(build_dir)
    meta_path = build_dir / "metadata.json"
    if not meta_path.exists():
        return "Error: no metadata.json found"

    meta = json.loads(meta_path.read_text())
    branch = meta.get("build_branch", "")
    starting = meta.get("starting_branch", _current_branch())

    if not meta.get("tests_passed"):
        return "Error: tests must pass before integration. Run run_tests first."

    rc, _ = _git("checkout", branch)
    if rc != 0:
        return f"Error checking out {branch}"

    issues = []
    info = []

    try:
        if module_type == "agent":
            rc, new_files = _git("diff", "--name-only", "--diff-filter=A",
                                 f"{starting}...{branch}")
            agent_files = [f for f in new_files.split("\n")
                           if f.startswith("agent_modules/") and f.endswith(".py")]

            if not agent_files:
                issues.append("No new agent files found in agent_modules/")
            else:
                for af in agent_files:
                    content = (_PROJECT_ROOT / af).read_text()
                    if "make_agent(" not in content:
                        issues.append(f"{af}: does not use make_agent()")
                    if "@function_tool" not in content:
                        issues.append(f"{af}: no @function_tool definitions")
                    info.append(f"Agent file: {af}")

            rc, mod_files = _git("diff", "--name-only", "--diff-filter=M",
                                 f"{starting}...{branch}")
            if "core/orchestrator.py" not in mod_files:
                issues.append("orchestrator.py was not modified — agent not registered")
            else:
                info.append("Orchestrator updated with new agent registration")

        elif module_type == "skill":
            rc, new_files = _git("diff", "--name-only", "--diff-filter=A",
                                 f"{starting}...{branch}")
            skill_files = [f for f in new_files.split("\n")
                           if f.startswith("skills/") and "SKILL.md" in f]
            if not skill_files:
                issues.append("No SKILL.md found in skills/")
            else:
                info.append(f"Skill definition: {', '.join(skill_files)}")

        elif module_type == "standalone":
            rc, new_files = _git("diff", "--name-only", "--diff-filter=A",
                                 f"{starting}...{branch}")
            py_files = [f for f in new_files.split("\n")
                        if f.endswith(".py") and "/" not in f]
            if py_files:
                info.append(f"Standalone modules: {', '.join(py_files)}")
            else:
                info.append("No new root-level Python files (may be in subdirectory)")

        rc, diff = _git("diff", f"{starting}...{branch}", "--", "requirements.txt")
        if diff:
            added_deps = [line[1:].strip() for line in diff.split("\n")
                          if line.startswith("+") and not line.startswith("+++")]
            if added_deps:
                info.append(f"New dependencies: {', '.join(added_deps)}")

        meta["integrated"] = len(issues) == 0
        meta["integration_issues"] = issues
        meta_path.write_text(json.dumps(meta, indent=2))

        if issues:
            return "Integration issues:\n" + "\n".join(f"  - {i}" for i in issues) + \
                   "\n\nInfo:\n" + "\n".join(f"  - {i}" for i in info)
        else:
            return "Integration validated.\n" + "\n".join(f"  - {i}" for i in info) + \
                   f"\n\nBuild branch '{branch}' is ready for review and merge."

    finally:
        _git("checkout", starting)


def _do_restart_service() -> str:
    log.info("Restarting jarvis.service")
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", "jarvis.service"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"Error restarting service: {err}"

        time.sleep(2)
        status = subprocess.run(
            ["sudo", "systemctl", "is-active", "jarvis.service"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        state = status.stdout.strip()

        if state == "active":
            return "Jarvis service restarted and running."
        else:
            return f"Service restarted but status is: {state}. Check journalctl for details."

    except subprocess.TimeoutExpired:
        return "Error: restart command timed out"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# @function_tool wrappers — thin delegates to _do_ functions
# ---------------------------------------------------------------------------

@function_tool
def write_build_spec(task_description: str, module_type: str = "agent") -> str:
    """Write a structured build specification for Claude Code.

    Args:
        task_description: What to build, in detail. Include the user's
            original request plus any context about how it should integrate.
        module_type: One of "agent", "skill", or "standalone".
            - agent: A new sub-agent in agent_modules/ using make_agent()
            - skill: A new skill directory in skills/ with SKILL.md
            - standalone: A module in the project root, manual wiring
    """
    return _do_write_build_spec(task_description, module_type)


@function_tool
def invoke_claude_code(spec_path: str, timeout_minutes: int = 10) -> str:
    """Invoke the claude CLI to execute a build assignment.

    Creates a git branch, runs claude in --print mode, and captures output.
    The build happens on a 'build/<slug>' branch — main is never touched.

    Args:
        spec_path: Path to the ASSIGNMENT.md file (from write_build_spec)
        timeout_minutes: Max time for the build (default 10, max 30)
    """
    return _do_invoke_claude_code(spec_path, timeout_minutes)


@function_tool
def check_build_status(build_dir: str) -> str:
    """Check the status of a completed build.

    Reads the build log, git diff, and metadata to assess results.

    Args:
        build_dir: Path to the build directory (e.g., builds/20260315-143022-rss-reader)
    """
    return _do_check_build_status(build_dir)


@function_tool
def run_tests(build_dir: str, test_path: str = "") -> str:
    """Run tests for a build.

    Checks out the build branch, runs pytest, then switches back.

    Args:
        build_dir: Path to the build directory
        test_path: Specific test file to run (auto-detected if empty)
    """
    return _do_run_tests(build_dir, test_path)


@function_tool
def integrate_module(build_dir: str, module_type: str = "agent") -> str:
    """Validate that a build is ready for integration.

    Checks that the build produced the expected files and follows the right
    patterns. For agents, verifies make_agent() usage and module-level export.
    For skills, verifies SKILL.md exists with proper frontmatter.

    Note: The actual integration (orchestrator registration, imports) should
    have been done by Claude Code during the build. This tool verifies it.

    Args:
        build_dir: Path to the build directory
        module_type: "agent", "skill", or "standalone"
    """
    return _do_integrate_module(build_dir, module_type)


@function_tool
def restart_service() -> str:
    """Restart the Jarvis systemd service to pick up new code.

    Only call this after a successful build, test, and integration.
    Runs: sudo systemctl restart jarvis.service
    """
    return _do_restart_service()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_claude_code_instructions(context: RunContextWrapper, agent: Agent) -> str:
    build_lines = []
    if _BUILDS_DIR.exists():
        for d in sorted(_BUILDS_DIR.iterdir(), reverse=True)[:5]:
            if d.is_dir():
                meta_path = d / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    status = "passed" if meta.get("tests_passed") else "pending"
                    build_lines.append(f"  - {d.name} [{status}] branch: {meta.get('build_branch', '?')}")

    recent = "\n".join(build_lines) if build_lines else "  (none)"

    return f"""\
You are a software engineering specialist. You build new modules for the \
Jarvis voice assistant by writing specifications and delegating construction \
to Claude Code (the `claude` CLI tool).

## Workflow — follow these steps IN ORDER
1. Analyze the request. Determine module_type: "agent", "skill", or "standalone".
2. Write a detailed spec with write_build_spec. Be thorough — include all \
requirements, API details, and integration points. This is the blueprint \
Claude Code will follow.
3. Invoke Claude Code with invoke_claude_code. Default timeout is 10 minutes. \
Use longer for complex builds.
4. Check results with check_build_status.
5. If the build completed, run tests with run_tests.
6. If tests pass, validate integration with integrate_module.
7. Do NOT restart the service or merge — the build branch is left for review.

## Rules
- ALWAYS create a spec before building. Never invoke claude code with a vague prompt.
- If the build or tests fail, report the failure clearly with specifics from \
the build log. Do not retry automatically.
- For agent modules: the new agent MUST use make_agent() from agent_base and \
follow existing patterns (search.py, home_control.py).
- For skill modules: SKILL.md must follow agent-skills-sdk format.
- Never modify existing agents' core functionality unless explicitly asked.
- Report concisely: what was built, what tests passed, what was integrated. \
Plain text only — your output is spoken aloud.

## Recent builds
{recent}

## Project root
{_PROJECT_ROOT}
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

claude_code_agent = make_agent(
    name="claude_code_agent",
    instructions=_build_claude_code_instructions,
    extra_tools=[
        write_build_spec,
        invoke_claude_code,
        check_build_status,
        run_tests,
        integrate_module,
        restart_service,
    ],
)

# Auto-discovery metadata (underscore file — dangerous agent, not auto-loaded)
AGENT_CONFIG = {
    "tool_name": "build",
    "tool_description": (
        "Build a new module, agent, or capability for Jarvis. "
        "Use when asked to create new functionality, integrations, "
        "or tools. Handles the full lifecycle: spec, build, test, "
        "integrate, and report. Pass the user's request with all details."
    ),
    "write_keywords": [
        "build", "create", "write", "implement", "generate",
    ],
}
REQUIRES_SECURITY = "enable_system_tools"
agent = claude_code_agent
