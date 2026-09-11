"""
Communications Agent — Slack messaging and notifications.

Sends and reads Slack messages via the Slack Web API (chat.postMessage,
conversations.history).  Desktop notifications via notify-send.

Auth: Bot token loaded from config.json (integrations.slack.bot_token), or the
JARVIS4_SLACK_BOT_TOKEN environment variable.
Channel/user mapping lives in slack_contacts.json (project root) for easy editing.
"""

import json
import logging
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from agents import Agent, RunContextWrapper, function_tool
from core.agent_base import make_agent

log = logging.getLogger("jarvis.comms")

_PROJECT_ROOT = Path(__file__).parent.parent
_SLACK_API = "https://slack.com/api"
_TIMEOUT = 10

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_bot_token: Optional[str] = None


def _get_bot_token() -> str:
    global _bot_token
    if _bot_token is not None:
        return _bot_token
    from core import config
    tok = config.get("integrations", "slack", "bot_token")
    if tok:
        _bot_token = tok
        return _bot_token
    raise RuntimeError(
        "No Slack bot token. Set integrations.slack.bot_token in config.json "
        "or the JARVIS4_SLACK_BOT_TOKEN environment variable."
    )


def _slack_post(method: str, data: dict) -> dict:
    """Call a Slack Web API method. Returns the parsed JSON response."""
    payload = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{_SLACK_API}/{method}",
        data=payload,
        headers={
            "Authorization": f"Bearer {_get_bot_token()}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                return {"ok": False, "error": result.get("error", "unknown")}
            return result
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"network error: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _slack_get(method: str, params: str = "") -> dict:
    """GET a Slack Web API method."""
    url = f"{_SLACK_API}/{method}"
    if params:
        url += f"?{params}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {_get_bot_token()}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                return {"ok": False, "error": result.get("error", "unknown")}
            return result
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"network error: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Contact / channel registry
# ---------------------------------------------------------------------------

_contacts: Optional[dict] = None


def _load_contacts() -> dict:
    global _contacts
    if _contacts is not None:
        return _contacts

    contacts_file = _PROJECT_ROOT / "slack_contacts.json"
    if not contacts_file.exists():
        log.warning("slack_contacts.json not found — using empty contacts")
        _contacts = {}
        return _contacts

    with open(contacts_file) as f:
        _contacts = json.load(f)
    log.info("Loaded %d contacts, %d channels from slack_contacts.json",
             len(_contacts.get("people", {})),
             len(_contacts.get("channels", {})))
    return _contacts


def _resolve_channel(target: str) -> tuple[str, str]:
    """Resolve a target name to a (channel_id, display_name) tuple.

    Checks people first, then channels, then returns the raw string
    assuming it's already a channel ID.
    """
    contacts = _load_contacts()
    target_lower = target.lower().strip()

    # Check people
    for name, info in contacts.get("people", {}).items():
        if target_lower == name.lower():
            return info["channel_id"], name
        for alias in info.get("aliases", []):
            if target_lower == alias.lower():
                return info["channel_id"], name

    # Check channels
    for name, info in contacts.get("channels", {}).items():
        if target_lower == name.lower() or target_lower == f"#{name.lower()}":
            return info["channel_id"], f"#{name}"
        for alias in info.get("aliases", []):
            if target_lower == alias.lower():
                return info["channel_id"], f"#{name}"

    # Assume it's a raw channel ID
    return target, target


# ---------------------------------------------------------------------------
# Typed tools
# ---------------------------------------------------------------------------

@function_tool
def send_slack_message(to: str, message: str) -> str:
    """Send a Slack message to a person or channel.
    Args:
        to: Person name (e.g. "Alice") or channel name (e.g. "general", "#dev")
        message: The message text to send
    """
    channel_id, display = _resolve_channel(to)
    log.info("Slack send → %s (%s): %s", display, channel_id, message[:80])

    result = _slack_post("chat.postMessage", {
        "channel": channel_id,
        "text": message,
    })

    if not result.get("ok"):
        return f"Error sending to {display}: {result.get('error')}"

    return f"Message sent to {display}."


@function_tool
def read_slack(target: str, count: int = 10) -> str:
    """Read recent messages from a Slack channel or DM.
    Args:
        target: Person name or channel name
        count: Number of recent messages to fetch (default 10, max 50)
    """
    channel_id, display = _resolve_channel(target)
    count = min(count, 50)

    result = _slack_get(
        "conversations.history",
        f"channel={channel_id}&limit={count}",
    )

    if not result.get("ok"):
        return f"Error reading {display}: {result.get('error')}"

    messages = result.get("messages", [])
    if not messages:
        return f"No recent messages in {display}."

    # Format messages (newest first from API, reverse for chronological)
    lines = []
    for msg in reversed(messages):
        user = msg.get("user", "bot")
        text = msg.get("text", "")
        # Truncate long messages
        if len(text) > 200:
            text = text[:200] + "..."
        lines.append(f"  [{user}] {text}")

    return f"Recent messages in {display} ({len(messages)}):\n" + "\n".join(lines)


@function_tool
def list_channels() -> str:
    """List available Slack channels the bot can access."""
    result = _slack_get("conversations.list", "types=public_channel&limit=50")

    if not result.get("ok"):
        return f"Error listing channels: {result.get('error')}"

    channels = result.get("channels", [])
    if not channels:
        return "No channels found."

    lines = []
    for ch in channels:
        name = ch.get("name", "?")
        members = ch.get("num_members", "?")
        lines.append(f"  #{name} ({members} members)")

    return "Channels:\n" + "\n".join(lines)


@function_tool
def send_desktop_notification(title: str, body: str) -> str:
    """Send a desktop notification via notify-send.
    Args:
        title: Notification title
        body: Notification body text
    """
    try:
        subprocess.run(
            ["notify-send", "--app-name=Jarvis", title, body],
            timeout=5, capture_output=True,
        )
        return f"Desktop notification sent: {title}"
    except FileNotFoundError:
        return "Error: notify-send not available on this system."
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_comms_instructions(context: RunContextWrapper, agent: Agent) -> str:
    contacts = _load_contacts()
    people = contacts.get("people", {})
    channels = contacts.get("channels", {})

    people_lines = []
    for name, info in people.items():
        aliases = info.get("aliases", [])
        tag = f" (aliases: {', '.join(aliases)})" if aliases else ""
        people_lines.append(f"  - {name}{tag}")

    channel_lines = []
    for name, info in channels.items():
        channel_lines.append(f"  - #{name}")

    return f"""\
You are a communications specialist. You handle Slack messaging and desktop notifications.

## Your tools
- send_slack_message(to, message): Send a Slack message to a person or channel.
- read_slack(target, count): Read recent messages from a channel or DM.
- list_channels(): List available Slack channels.
- send_desktop_notification(title, body): Send a desktop notification.

## Known contacts
{chr(10).join(people_lines) if people_lines else "  (none configured)"}

## Known channels
{chr(10).join(channel_lines) if channel_lines else "  (none configured)"}

## Key rules
- Use the contact/channel names listed above. The registry handles resolution to Slack IDs.
- When sending a DM, use the person's name as the 'to' parameter.
- When sending to a channel, use the channel name (with or without #).
- Confirm what was sent and to whom.
- No markdown formatting. Plain text only.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

comms_agent = make_agent(
    name="comms_agent",
    instructions=_build_comms_instructions,
    extra_tools=[
        send_slack_message, read_slack, list_channels,
        send_desktop_notification,
    ],
)

# Auto-discovery metadata
AGENT_CONFIG = {
    "tool_name": "comms",
    "tool_description": (
        "Communications: send Slack messages, desktop notifications, SMS. "
        "Read Slack channels. Pass recipient, message, and channel details."
    ),
    "write_keywords": [
        "send", "post", "notify", "dm ", "announce",
    ],
}
REQUIRES_INTEGRATION = "slack"
agent = comms_agent
