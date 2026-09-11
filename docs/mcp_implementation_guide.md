# MCP (Model Context Protocol) Implementation Guide

This guide explains how to integrate MCP servers into your application using the `mcp_manager.py` module and proper tool calling patterns.

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Using mcp_manager.py](#using-mcp_managerpy)
4. [Tool Calling Format](#tool-calling-format)
5. [System Prompt Structure](#system-prompt-structure)
6. [Complete Integration Example](#complete-integration-example)

---

## Overview

MCP (Model Context Protocol) allows LLMs to access external tools through a standardized JSON-RPC 2.0 interface. MCP servers run as subprocesses and communicate via stdin/stdout.

**Architecture:**
```
Your App
    └── LLM Client
        └── MCPManager
            ├── MCPServer: weather
            ├── MCPServer: calendar
            └── MCPServer: search
```

---

## Configuration

### Configuration File Location

Create a JSON configuration file. The default path used by `mcp_manager.py` is:
```
~/.config/jarvis/mcp_servers.json
```

You can customize this path when initializing `MCPManager`.

### Configuration Format

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@package/mcp-server"],
      "cwd": "/optional/working/directory",
      "env": {
        "API_KEY": "your-api-key",
        "OTHER_VAR": "value"
      }
    }
  }
}
```

**Fields:**
- `command` (required): The executable to run (e.g., `npx`, `python`, `node`)
- `args` (optional): List of command-line arguments
- `cwd` (optional): Working directory for the process
- `env` (optional): Environment variables to set (merged with system env)

### Example Configuration

```json
{
  "mcpServers": {
    "weather": {
      "command": "npx",
      "args": ["-y", "@tristau/openweathermap-mcp", "--apikey", "YOUR_API_KEY"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@brave/brave-search-mcp-server"],
      "env": {
        "BRAVE_API_KEY": "YOUR_BRAVE_KEY"
      }
    },
    "google-calendar": {
      "command": "npx",
      "args": ["-y", "@cocal/google-calendar-mcp"],
      "env": {
        "GOOGLE_OAUTH_CREDENTIALS": "/path/to/credentials.json"
      }
    }
  }
}
```

---

## Using mcp_manager.py

### Basic Usage

```python
from mcp_manager import MCPManager

# Initialize manager (uses default config path)
manager = MCPManager()

# Or specify custom config path
from pathlib import Path
manager = MCPManager(config_path=Path("/path/to/config.json"))

# Load configuration
if not manager.load_config():
    print("Failed to load MCP config")
    exit(1)

# Connect to all servers
connected = manager.connect_all()
print(f"Connected to {connected} MCP servers")

# Get all available tools
tools = manager.get_all_tools()
for tool in tools:
    print(f"Tool: {tool['full_name']}")

# Get tool descriptions for LLM prompt
tools_description = manager.get_tools_description()
print(tools_description)

# Call a tool
result = manager.call_tool("weather.get_current_weather", {
    "location": {"city": "Seattle", "country": "US"},
    "units": "imperial"
})
print(result)

# Disconnect when done
manager.disconnect_all()
```

### Key Methods

| Method | Description |
|--------|-------------|
| `load_config()` | Load server configuration from JSON file. Returns `True` on success. |
| `connect_all()` | Connect to all configured servers. Returns number of successful connections. |
| `get_all_tools()` | Get list of all available tools with `full_name` (server.tool_name format). |
| `get_tools_description()` | Get formatted text description of tools for LLM system prompt. |
| `call_tool(full_name, arguments)` | Execute a tool. `full_name` is `server.tool_name` format. |
| `disconnect_all()` | Cleanly disconnect from all servers. |

### Tool Naming Convention

Tools are namespaced by server name:
- Server name: `weather`
- Tool name: `get_current_weather`
- Full name: `weather.get_current_weather`

This allows multiple servers to have tools with the same name without conflicts.

---

## Tool Calling Format

### LLM Output Format

The LLM should output tool calls in this exact format:

```
[TOOL: server.tool_name({"param1": "value1", "param2": "value2"})]
```

**Examples:**
```
[TOOL: weather.get_current_weather({"location": {"city": "Seattle", "state": "WA", "country": "US"}, "units": "imperial"})]

[TOOL: brave-search.brave_web_search({"query": "latest AI news"})]

[TOOL: calendar.list-events({"calendarId": "primary", "timeMin": "2025-01-01T00:00:00", "timeMax": "2025-01-01T23:59:59"})]
```

### Parsing Tool Calls

Use this regex pattern to detect and parse tool calls:

```python
import re

TOOL_CALL_PATTERN = re.compile(
    r'\[TOOL:\s*([a-zA-Z0-9_.-]+)\s*\(\s*(\{.*?\})?\s*\)\]',
    re.DOTALL
)

def parse_tool_call(response: str) -> tuple[str, dict] | None:
    """Parse a tool call from LLM response."""
    if "[TOOL:" not in response:
        return None

    match = TOOL_CALL_PATTERN.search(response)
    if not match:
        return None

    tool_name = match.group(1)
    args_str = match.group(2)

    if not args_str:
        return tool_name, {}

    try:
        arguments = json.loads(args_str)
        return tool_name, arguments
    except json.JSONDecodeError:
        return None
```

### Executing Tool Calls

```python
def execute_tool(mcp_manager, tool_name: str, arguments: dict) -> str:
    """Execute a tool and return result as string."""
    result = mcp_manager.call_tool(tool_name, arguments)

    if result is None:
        return "Error: Tool execution failed"

    # MCP tools return content in a specific format
    if isinstance(result, dict):
        content = result.get("content", [])
        if content and isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            if texts:
                return "\n".join(texts)
        return json.dumps(result)

    return str(result)
```

### Tool Result Format

MCP servers return results in this structure:
```json
{
  "content": [
    {"type": "text", "text": "The actual result text..."}
  ]
}
```

---

## System Prompt Structure

### Required Components

Your system prompt should include:

1. **Base Instructions**: Your assistant's personality and behavior
2. **Current Date/Time**: So the LLM knows when "today" is
3. **Tool Calling Instructions**: Exact format the LLM should use
4. **Tool-Specific Guidance**: How to use specific tools correctly
5. **Available Tools List**: Generated from `get_tools_description()`

### Example System Prompt Template

```python
from datetime import datetime

def build_system_prompt(mcp_manager, base_prompt: str) -> str:
    """Build complete system prompt with tools."""

    # Add current date/time
    now = datetime.now()
    time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
    prompt = f"Current date and time: {time_str}\n\n{base_prompt}"

    # Add tool instructions if tools are available
    if mcp_manager:
        tools_desc = mcp_manager.get_tools_description()
        if tools_desc:
            prompt += """

You have access to external tools. To use a tool, respond ONLY with the tool call in this exact format:
[TOOL: tool_name({"param": "value"})]

Important rules:
- Arguments must be valid JSON
- Only call one tool at a time
- Wait for the tool result before responding to the user
- After receiving a tool result, summarize it conversationally

""" + tools_desc

    return prompt
```

### Tool-Specific Instructions

Add guidance for tools that need special handling:

```python
tool_instructions = """
For weather queries:
- Use {"city": "City Name", "state": "ST", "country": "US"} for locations
- Use "imperial" for Fahrenheit, "metric" for Celsius

For calendar queries:
- Use ISO 8601 format for dates: "2025-01-15T00:00:00"
- timeMin and timeMax define the date range

For web search:
- Use concise, specific search queries
- Good: {"query": "Python 3.12 new features"}
- Bad: {"query": "tell me about python"}
"""
```

### Complete System Prompt Example

```
Current date and time: Monday, January 15, 2025 at 2:30 PM

You are a helpful assistant.

You have access to external tools. To use a tool, respond ONLY with the tool call in this exact format:
[TOOL: tool_name({"param": "value"})]

Important rules:
- Arguments must be valid JSON
- Only call one tool at a time
- Wait for the tool result before responding to the user
- After receiving a tool result, summarize it conversationally

For weather queries, use location format: {"city": "Name", "country": "US"}

Example: [TOOL: weather.get_current_weather({"location": {"city": "Seattle", "country": "US"}, "units": "imperial"})]

Available tools:
- weather.get_current_weather(location*: object, units: string): Get current weather for a location
- brave-search.brave_web_search(query*: string): Search the web using Brave Search
```

---

## Complete Integration Example

Here's how to integrate MCP tools with an LLM:

```python
#!/usr/bin/env python3
"""Example LLM client with MCP tool support."""

import json
import re
from datetime import datetime
from mcp_manager import MCPManager

TOOL_CALL_PATTERN = re.compile(
    r'\[TOOL:\s*([a-zA-Z0-9_.-]+)\s*\(\s*(\{.*?\})?\s*\)\]',
    re.DOTALL
)

class LLMClientWithTools:
    def __init__(self, mcp_manager: MCPManager):
        self.mcp_manager = mcp_manager
        self.base_prompt = "You are a helpful assistant."

    @property
    def system_prompt(self) -> str:
        """Build system prompt with tools."""
        now = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")

        prompt = f"Current date and time: {time_str}\n\n{self.base_prompt}"

        tools_desc = self.mcp_manager.get_tools_description()
        if tools_desc:
            prompt += f"""

You have access to external tools. To use a tool, respond ONLY with the tool call in this exact format:
[TOOL: tool_name({{"param": "value"}})]

Important: Arguments must be valid JSON.
After using a tool, you will receive the result and should summarize it.

{tools_desc}"""

        return prompt

    def _parse_tool_call(self, response: str):
        """Parse tool call from response."""
        if "[TOOL:" not in response:
            return None

        match = TOOL_CALL_PATTERN.search(response)
        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2)

        if not args_str:
            return tool_name, {}

        try:
            return tool_name, json.loads(args_str)
        except json.JSONDecodeError:
            return None

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute tool and return result."""
        result = self.mcp_manager.call_tool(tool_name, arguments)

        if result is None:
            return "Error: Tool execution failed"

        if isinstance(result, dict):
            content = result.get("content", [])
            if content and isinstance(content, list):
                texts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                if texts:
                    return "\n".join(texts)
            return json.dumps(result)

        return str(result)

    def chat(self, user_message: str, call_llm_fn) -> str:
        """
        Process a user message, handling tool calls.

        Args:
            user_message: The user's input
            call_llm_fn: Function that takes messages list and returns response string

        Returns:
            Final response string
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Get initial response
        response = call_llm_fn(messages)

        # Check for tool call
        tool_call = self._parse_tool_call(response)
        if tool_call:
            tool_name, arguments = tool_call

            # Execute the tool
            tool_result = self._execute_tool(tool_name, arguments)

            # Ask LLM to summarize result
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Tool result:\n{tool_result}\n\nPlease summarize this result."
            })

            response = call_llm_fn(messages)

        return response


# Usage example
if __name__ == "__main__":
    # Initialize MCP
    manager = MCPManager()
    manager.load_config()
    manager.connect_all()

    # Create client
    client = LLMClientWithTools(manager)

    # Your LLM call function (implement based on your LLM)
    def call_llm(messages):
        # Replace with your actual LLM API call
        # e.g., OpenAI, Anthropic, local llama.cpp, etc.
        pass

    # Chat
    response = client.chat("What's the weather in Seattle?", call_llm)
    print(response)

    # Cleanup
    manager.disconnect_all()
```

---

## MCP Protocol Details

For reference, here are the JSON-RPC 2.0 protocol details used:

### Initialize Request
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "your-app",
      "version": "1.0.0"
    }
  }
}
```

### List Tools Request
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

### Call Tool Request
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": {"param": "value"}
  }
}
```

### Initialized Notification (sent after initialize)
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized",
  "params": {}
}
```

---

## Troubleshooting

### Server Won't Connect
- Check that the command exists (e.g., `npx` is installed)
- Verify the MCP package name is correct
- Check environment variables are set correctly

### Tool Calls Failing
- Ensure arguments are valid JSON
- Check required parameters are provided (marked with `*` in tool descriptions)
- Look at stderr output from the MCP server for error messages

### LLM Not Using Tools
- Verify the system prompt includes tool instructions
- Check that `get_tools_description()` returns non-empty string
- Make sure the tool calling format examples are clear

### JSON Parse Errors
- Ensure the LLM outputs properly escaped JSON
- Check for trailing commas or unquoted strings
- Verify nested objects are properly structured
