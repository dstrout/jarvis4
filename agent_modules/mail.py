"""
Mail Agent — email management specialist.

Handles Gmail operations via the `gws` CLI, guided by skill docs.
Contact lookup is wrapped in a typed tool (lookup_contact) because
the raw API requires a warmup call and nested JSON parsing that
LLMs handle unreliably. All other operations use the gws helpers
(+send, +reply, +forward, +triage) via run_command.

Built on agent_base: run_command + lookup_skill + find_skills.
Pattern: skill docs (knowledge) + bash tool (capability) + typed tools (where needed)
"""

import base64
import json
import logging
import mimetypes
import os
import subprocess
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

from agents import Agent, RunContextWrapper, function_tool
from core.agent_base import make_agent
from core.skill_loader import get_skill_loader

log = logging.getLogger("jarvis.mail")

# ---------------------------------------------------------------------------
# Typed tool — contact lookup
# ---------------------------------------------------------------------------

# Track whether the contacts cache has been warmed this session
_contacts_warmed = False


@function_tool
def lookup_contact(name: str) -> str:
    """Look up a contact's email address by name. Use this whenever the user
    refers to a person by name instead of giving an email address.
    Returns the email address, or an error if not found.
    """
    global _contacts_warmed

    # Google requires a warmup call with empty query before search works
    if not _contacts_warmed:
        _run_gws(
            "gws people people searchContacts "
            "--params '{\"query\": \"\", \"readMask\": \"names,emailAddresses\"}' "
            "--format json"
        )
        _contacts_warmed = True

    raw = _run_gws(
        f"gws people people searchContacts "
        f"--params '{{\"query\": \"{name}\", \"readMask\": \"names,emailAddresses\"}}' "
        f"--format json"
    )

    if raw.startswith("Error:"):
        return raw

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return f"Error: could not parse contacts response"

    results = data.get("results", [])
    if not results:
        return f"No contact found matching '{name}'."

    # Return all matches so the agent can pick the right one
    contacts = []
    for r in results:
        person = r.get("person", {})
        display_name = ""
        names = person.get("names", [])
        if names:
            display_name = names[0].get("displayName", "")

        emails = person.get("emailAddresses", [])
        if emails:
            email = emails[0].get("value", "")
            contacts.append(f"{display_name}: {email}")

    if not contacts:
        return f"Contact '{name}' found but has no email address."

    return "\n".join(contacts)


@function_tool
def send_email(to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> str:
    """Send an email via Gmail. The 'to' field MUST be an email address, not a name.
    Use lookup_contact first if you only have a name. Body can contain newlines.
    Optional: cc and bcc for additional recipients (comma-separated emails).
    """
    if "@" not in to:
        return f"Error: '{to}' is not a valid email address. Use lookup_contact to find the address first."

    # Add greeting/closing from config
    try:
        from core import config
        greeting = config.get("email", "greeting", default="")
        closing = config.get("email", "closing", default="")
    except Exception:
        greeting, closing = "", ""

    if greeting:
        body = f"{greeting}\n\n{body}"
    if closing:
        body = f"{body}\n\n{closing}"

    cmd = ["gws", "gmail", "+send", "--to", to, "--subject", subject, "--body", body]
    if cc:
        cmd.extend(["--cc", cc])
    if bcc:
        cmd.extend(["--bcc", bcc])
    cmd.extend(["--format", "json"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"Error: {err}" if err else f"Error: exit code {result.returncode}"
        # Extract message ID from response
        try:
            data = json.loads(output)
            msg_id = data.get("id", "")
            return f"Email sent successfully. Message ID: {msg_id}"
        except json.JSONDecodeError:
            return output or "Email sent (no confirmation ID)."
    except subprocess.TimeoutExpired:
        return "Error: send timed out after 30s"
    except Exception as e:
        return f"Error: {e}"


def _do_send_email_with_attachment(
    to: str, subject: str, body: str, file_path: str,
    cc: str = "", bcc: str = "",
) -> str:
    """Core logic for send_email_with_attachment — testable without @function_tool."""
    if "@" not in to:
        return f"Error: '{to}' is not a valid email address. Use lookup_contact first."

    if not os.path.exists(file_path):
        return f"Error: file not found: {file_path}"

    # Build MIME message
    msg = MIMEMultipart()
    msg["To"] = to
    msg["Subject"] = subject
    if cc:
        msg["Cc"] = cc
    if bcc:
        msg["Bcc"] = bcc

    # Add greeting/closing from config
    try:
        from core import config
        greeting = config.get("email", "greeting", default="")
        closing = config.get("email", "closing", default="")
    except Exception:
        greeting, closing = "", ""

    if greeting:
        body = f"{greeting}\n\n{body}"
    if closing:
        body += f"\n\n{closing}"

    msg.attach(MIMEText(body, "plain"))

    # Attach file
    filename = os.path.basename(file_path)
    content_type, _ = mimetypes.guess_type(file_path)
    if content_type is None:
        content_type = "application/octet-stream"
    main_type, sub_type = content_type.split("/", 1)

    with open(file_path, "rb") as f:
        attachment = MIMEBase(main_type, sub_type)
        attachment.set_payload(f.read())
    encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", "attachment", filename=filename)
    msg.attach(attachment)

    # Send via Gmail API raw endpoint
    raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    cmd = [
        "gws", "gmail", "users", "messages", "send",
        "--params", json.dumps({"userId": "me"}),
        "--json", json.dumps({"raw": raw_message}),
        "--format", "json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"Error: {err}" if err else f"Error: exit code {result.returncode}"
        try:
            data = json.loads(result.stdout)
            return f"Email with attachment '{filename}' sent to {to}. Message ID: {data.get('id', 'unknown')}"
        except json.JSONDecodeError:
            return result.stdout.strip() or "Email sent (no confirmation)."
    except subprocess.TimeoutExpired:
        return "Error: send timed out after 60s"
    except Exception as e:
        return f"Error: {e}"


@function_tool
def send_email_with_attachment(
    to: str, subject: str, body: str, file_path: str,
    cc: str = "", bcc: str = "",
) -> str:
    """Send an email with a file attachment via Gmail.

    Use this when the user wants to email a file. The file_path must be
    an absolute path to an existing file on this machine.

    Args:
        to: Recipient email address (use lookup_contact first if you have a name)
        subject: Email subject line
        body: Email body text
        file_path: Absolute path to the file to attach
        cc: Optional CC recipients (comma-separated emails)
        bcc: Optional BCC recipients (comma-separated emails)
    """
    return _do_send_email_with_attachment(to, subject, body, file_path, cc, bcc)


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


# ---------------------------------------------------------------------------
# System prompt with skill catalog
# ---------------------------------------------------------------------------

def _get_email_signature_instructions() -> str:
    """Build signature instructions from config."""
    try:
        from core import config
        closing = config.get("email", "closing", default="")
        if closing:
            return f"Always sign emails with:\n{closing}\nNever use [Your Name] or any other placeholder."
    except Exception:
        pass
    return "Sign emails appropriately. Never use [Your Name] or any other placeholder."


def _build_mail_instructions(context: RunContextWrapper, agent: Agent) -> str:
    """Dynamic system prompt with embedded skill catalog and current date."""
    from datetime import datetime, timedelta
    now = datetime.now()
    today = now.strftime("%Y/%m/%d")
    week_ago = (now - timedelta(days=7)).strftime("%Y/%m/%d")

    loader = get_skill_loader()
    mail_catalog = loader.catalog_for_prompt(
        name_filter=[
            "gmail", "email", "forward", "reply",
            "label", "archive", "attachment", "filter",
        ],
    )

    return f"""\
You are an email management specialist. You handle Gmail operations by \
running gws CLI commands.

Today's date: {today}. One week ago: {week_ago}.
Gmail date syntax uses YYYY/MM/DD format: after:{week_ago} before:{today}

## How you work
1. Check if a skill below matches the request
2. Call lookup_skill to load the full CLI syntax
3. Run the command via run_command
4. Interpret the results and respond

## Your typed tools — use these instead of run_command for these operations
- lookup_contact(name): Resolves a person's name to their email address. \
Call this FIRST when the user refers to someone by name.
- send_email(to, subject, body): Sends an email. The 'to' MUST be an email \
address — use lookup_contact first if you only have a name. \
Body can contain newlines naturally. Also accepts optional cc and bcc.
- send_email_with_attachment(to, subject, body, file_path): Sends an email \
with a file attachment. Use this when the user wants to email a file. \
The file_path must be an absolute path. Use lookup_contact for name resolution.

## Read operations — use run_command with gws helpers
- Inbox check: gws gmail +triage --format json
- Search: gws gmail +triage --query 'from:someone' --format json
- Reply: gws gmail +reply --message-id ID --body 'BODY'
- Forward: gws gmail +forward --message-id ID --to EMAIL
- Read message: gws gmail users messages get --params '{{"userId": "me", "id": "MSG_ID"}}' --format json
  (body is base64url in payload.parts[0].body.data — decode with: echo 'DATA' | base64 -d)

## Key rules
- NEVER put a person's name where an email address is expected. \
Always use lookup_contact to resolve names to email addresses first.
- For anything not listed above, look up the skill first.

## Email signature
{_get_email_signature_instructions()}

## Write operations safety
NEVER send, reply, or forward without the orchestrator explicitly confirming \
the user wants this action. When in doubt, create a draft instead. \
Read operations (triage, search, read) are always safe.

## Available skills
{mail_catalog}

## Recipe matching
- "Save attachments to Drive" → recipe-save-email-attachments
- "Draft from a document" → recipe-draft-email-from-doc
- "Label and archive" → recipe-label-and-archive-emails
- "Forward labeled emails" → recipe-forward-labeled-emails
- "Create a Gmail filter" → recipe-create-gmail-filter
- For unfamiliar operations, call find_skills

## Response format
- Summarize email content concisely — the orchestrator will speak it aloud
- Include message IDs when the orchestrator may need them for follow-up
- No markdown formatting. Plain text only.
"""


# ---------------------------------------------------------------------------
# Agent definition — built on agent_base, no extra tools needed
# ---------------------------------------------------------------------------

mail_agent = make_agent(
    name="mail_agent",
    instructions=_build_mail_instructions,
    extra_tools=[lookup_contact, send_email, send_email_with_attachment],
)

# Auto-discovery metadata
AGENT_CONFIG = {
    "tool_name": "email",
    "tool_description": (
        "Handle email operations: check inbox, search messages, read emails, "
        "send, reply, forward, draft, summarize threads, manage labels. "
        "Can look up contacts by name — you don't need an email address, just pass the name. "
        "Pass the user's request with all relevant details (recipient name or email, subject, body, etc)."
    ),
    "write_keywords": [
        "send", "forward", "reply", "delete", "archive", "label",
        "draft", "compose", "move", "mark", "star",
    ],
}
REQUIRES_INTEGRATION = "google"
agent = mail_agent
