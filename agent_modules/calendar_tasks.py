"""
Calendar & Tasks Agent — scheduling and task management specialist.

Handles Google Calendar and Google Tasks via gws CLI.
Read ops use run_command (agenda, task list, event details).
Write ops use typed Python tools (create/update/delete events and tasks)
to avoid LLM-constructed JSON payloads.

Built on agent_base: run_command + lookup_skill + find_skills.
Pattern: skill docs (knowledge) + bash tool (capability) + typed tools (writes)
"""

import json
import logging
import subprocess
from datetime import datetime, timedelta

from agents import Agent, RunContextWrapper, function_tool
from core.agent_base import make_agent
from core.skill_loader import get_skill_loader

log = logging.getLogger("jarvis.calendar")

# The user's default task list ID (discovered at module load time)
_default_tasklist_id: str | None = None


def _get_default_tasklist() -> str:
    """Get the default task list ID, caching after first lookup."""
    global _default_tasklist_id
    if _default_tasklist_id is not None:
        return _default_tasklist_id

    raw = _run_gws("gws tasks tasklists list --format json")
    try:
        data = json.loads(raw)
        items = data.get("items", [])
        if items:
            _default_tasklist_id = items[0]["id"]
            return _default_tasklist_id
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    return ""


# ---------------------------------------------------------------------------
# Typed tools — calendar writes
# ---------------------------------------------------------------------------

@function_tool
def create_event(
    summary: str,
    start_time: str,
    end_time: str,
    location: str = "",
    description: str = "",
    attendees: str = "",
) -> str:
    """Create a calendar event.

    Args:
        summary: Event title
        start_time: Start in RFC3339 format (e.g. 2026-03-15T14:00:00-04:00)
        end_time: End in RFC3339 format
        location: Optional location
        description: Optional description/notes
        attendees: Optional comma-separated email addresses
    """
    cmd = [
        "gws", "calendar", "+insert",
        "--summary", summary,
        "--start", start_time,
        "--end", end_time,
        "--format", "json",
    ]
    if location:
        cmd.extend(["--location", location])
    if description:
        cmd.extend(["--description", description])
    if attendees:
        for email in attendees.split(","):
            email = email.strip()
            if email:
                cmd.extend(["--attendee", email])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"Error creating event: {err}" if err else f"Error: exit code {result.returncode}"
        try:
            data = json.loads(output)
            event_id = data.get("id", "")
            html_link = data.get("htmlLink", "")
            return f"Event created: '{summary}'. Event ID: {event_id}"
        except json.JSONDecodeError:
            return output or "Event created (no confirmation ID)."
    except subprocess.TimeoutExpired:
        return "Error: event creation timed out after 30s"
    except Exception as e:
        return f"Error: {e}"


@function_tool
def update_event(
    event_id: str,
    summary: str = "",
    start_time: str = "",
    end_time: str = "",
    location: str = "",
    description: str = "",
) -> str:
    """Update/reschedule an existing calendar event.

    Args:
        event_id: The event ID to update
        summary: New title (leave empty to keep current)
        start_time: New start time in RFC3339 (leave empty to keep current)
        end_time: New end time in RFC3339 (leave empty to keep current)
        location: New location (leave empty to keep current)
        description: New description (leave empty to keep current)
    """
    patch_body = {}
    if summary:
        patch_body["summary"] = summary
    if start_time:
        patch_body["start"] = {"dateTime": start_time, "timeZone": "America/New_York"}
    if end_time:
        patch_body["end"] = {"dateTime": end_time, "timeZone": "America/New_York"}
    if location:
        patch_body["location"] = location
    if description:
        patch_body["description"] = description

    if not patch_body:
        return "Error: nothing to update — provide at least one field."

    return _run_gws(
        f"gws calendar events patch "
        f"--params '{{\"calendarId\": \"primary\", \"eventId\": \"{event_id}\", \"sendUpdates\": \"all\"}}' "
        f"--json '{json.dumps(patch_body)}' "
        f"--format json"
    )


@function_tool
def delete_event(event_id: str) -> str:
    """Delete a calendar event by ID."""
    result = _run_gws(
        f"gws calendar events delete "
        f"--params '{{\"calendarId\": \"primary\", \"eventId\": \"{event_id}\"}}'"
    )
    if "Error" in result:
        return result
    return f"Event {event_id} deleted."


@function_tool
def quick_add_event(text: str) -> str:
    """Create an event from natural language text (Google's quickAdd).

    Examples: 'Lunch with Sarah tomorrow at noon', 'Dentist Friday 3pm'
    Google parses the date/time automatically.
    """
    result = _run_gws(
        f"gws calendar events quickAdd "
        f"--params '{{\"calendarId\": \"primary\", \"text\": \"{_escape_json(text)}\"}}' "
        f"--format json"
    )
    if result.startswith("Error"):
        return result
    try:
        data = json.loads(result)
        event_id = data.get("id", "")
        summary = data.get("summary", text)
        start = data.get("start", {})
        start_str = start.get("dateTime", start.get("date", ""))
        return f"Event created: '{summary}' at {start_str}. Event ID: {event_id}"
    except json.JSONDecodeError:
        return result


# ---------------------------------------------------------------------------
# Typed tools — task writes
# ---------------------------------------------------------------------------

@function_tool
def create_task(
    title: str,
    notes: str = "",
    due_date: str = "",
) -> str:
    """Create a new task in Google Tasks.

    Args:
        title: Task title
        notes: Optional notes/description
        due_date: Optional due date in YYYY-MM-DD or RFC3339 format
    """
    tasklist = _get_default_tasklist()
    if not tasklist:
        return "Error: could not find default task list."

    task_body: dict = {"title": title}
    if notes:
        task_body["notes"] = notes
    if due_date:
        # Ensure RFC3339 format
        if "T" not in due_date:
            due_date = f"{due_date}T00:00:00Z"
        task_body["due"] = due_date

    cmd = [
        "gws", "tasks", "tasks", "insert",
        "--params", json.dumps({"tasklist": tasklist}),
        "--json", json.dumps(task_body),
        "--format", "json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"Error creating task: {err}" if err else f"Error: exit code {result.returncode}"
        try:
            data = json.loads(output)
            task_id = data.get("id", "")
            return f"Task created: '{title}'. Task ID: {task_id}"
        except json.JSONDecodeError:
            return output or "Task created (no confirmation ID)."
    except subprocess.TimeoutExpired:
        return "Error: task creation timed out"
    except Exception as e:
        return f"Error: {e}"


@function_tool
def complete_task(task_id: str) -> str:
    """Mark a task as completed."""
    tasklist = _get_default_tasklist()
    if not tasklist:
        return "Error: could not find default task list."

    return _run_gws(
        f"gws tasks tasks patch "
        f"--params '{{\"tasklist\": \"{tasklist}\", \"task\": \"{task_id}\"}}' "
        f"--json '{{\"status\": \"completed\"}}' "
        f"--format json"
    )


@function_tool
def delete_task(task_id: str) -> str:
    """Delete a task by ID."""
    tasklist = _get_default_tasklist()
    if not tasklist:
        return "Error: could not find default task list."

    result = _run_gws(
        f"gws tasks tasks delete "
        f"--params '{{\"tasklist\": \"{tasklist}\", \"task\": \"{task_id}\"}}'"
    )
    if "Error" in result:
        return result
    return f"Task {task_id} deleted."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_gws(command: str) -> str:
    """Run a gws command and return stdout or an error string."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"Error: {err}" if err else f"Error: exit code {result.returncode}"
        return output or "{}"
    except subprocess.TimeoutExpired:
        return "Error: command timed out"
    except Exception as e:
        return f"Error: {e}"


def _escape_json(s: str) -> str:
    """Escape a string for embedding in a JSON value inside shell quotes."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_calendar_instructions(context: RunContextWrapper, agent: Agent) -> str:
    """Dynamic system prompt with skill catalog and current date."""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    today_rfc = now.strftime("%Y-%m-%dT%H:%M:%S") + "-04:00"
    weekday = now.strftime("%A")

    loader = get_skill_loader()
    catalog = loader.catalog_for_prompt(
        name_filter=[
            "calendar", "agenda", "event", "task",
            "schedule", "reschedule", "recurring", "focus",
            "free-time", "overdue",
        ],
    )

    return f"""\
You are a calendar and task management specialist. You handle Google Calendar
and Google Tasks operations by running gws CLI commands.

Today is {weekday}, {today}. Current time reference: {today_rfc}.
Timezone: America/New_York (Eastern).

## Your typed tools — use these for write operations
- create_event(summary, start_time, end_time, location, description, attendees): \
Create a calendar event. Times must be RFC3339 (e.g. 2026-03-15T14:00:00-04:00).
- update_event(event_id, summary, start_time, end_time, location, description): \
Reschedule or modify an event. Only pass fields you want to change.
- delete_event(event_id): Delete an event.
- quick_add_event(text): Create event from natural language ("Lunch Friday noon"). \
Google parses the date/time.
- create_task(title, notes, due_date): Create a task. Due date is YYYY-MM-DD.
- complete_task(task_id): Mark a task as done.
- delete_task(task_id): Delete a task.

## Read operations — use run_command
- Today's agenda: gws calendar +agenda --today --format json
- This week: gws calendar +agenda --week --format json
- Next N days: gws calendar +agenda --days N --format json
- Specific event: gws calendar events get --params '{{"calendarId": "primary", "eventId": "ID"}}' --format json
- Task list: gws tasks tasks list --params '{{"tasklist": "TASKLIST_ID", "showCompleted": false}}' --format json
- All task lists: gws tasks tasklists list --format json
- Free/busy: gws calendar freebusy query --json '{{...}}'

## Key rules
- For event creation, ALWAYS compute RFC3339 timestamps from the user's natural \
language. Today is {today}. Eastern timezone offset is -04:00 (EDT).
- When showing agenda, summarize concisely: time, title, duration. Skip duplicate \
entries from shared calendars.
- For tasks, the default task list ID is discovered automatically — you don't \
need to look it up.
- Include event/task IDs in your responses so the orchestrator can reference them \
for follow-up operations.
- For anything not listed above, look up the skill first.

## Available skills
{catalog}

## Recipe matching
- "Block focus time" → recipe-block-focus-time
- "Find free time" or "when am I free" → recipe-find-free-time
- "Schedule recurring" → recipe-schedule-recurring-event
- "Reschedule meeting" → recipe-reschedule-meeting
- "Review overdue tasks" → recipe-review-overdue-tasks
- "Plan my week" → recipe-plan-weekly-schedule
- For unfamiliar operations, call find_skills

## Response format
- Summarize concisely — the orchestrator will speak it aloud
- Include event/task IDs when useful for follow-up
- No markdown formatting. Plain text only.
"""


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

calendar_tasks_agent = make_agent(
    name="calendar_tasks_agent",
    instructions=_build_calendar_instructions,
    extra_tools=[
        create_event, update_event, delete_event, quick_add_event,
        create_task, complete_task, delete_task,
    ],
)

# Auto-discovery metadata
AGENT_CONFIG = {
    "tool_name": "calendar",
    "tool_description": (
        "Calendar and task management: check agenda, schedule events, "
        "create tasks, reschedule meetings, manage reminders. "
        "Pass dates, times, attendees, and other details."
    ),
    "write_keywords": [
        "create", "schedule", "reschedule", "cancel", "delete",
        "add", "update", "move", "remove", "set",
    ],
}
REQUIRES_INTEGRATION = "google"
agent = calendar_tasks_agent
