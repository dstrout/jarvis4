#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Manager for Jarvis.

Manages connections to MCP servers, discovers tools, and executes tool calls.
Configuration is read from ~/.config/jarvis4/mcp_servers.json
"""

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional


# Config file location
CONFIG_PATH = Path.home() / ".config" / "jarvis4" / "mcp_servers.json"

# Tool call timeout in seconds
TOOL_TIMEOUT = 15

# Tools to hide from the LLM (still callable, just not advertised)
HIDDEN_TOOLS = {
    "brave-search.brave_video_search",
    "brave-search.brave_image_search",
}


class MCPServer:
    """Manages a single MCP server subprocess."""

    def __init__(self, name: str, command: str, args: list[str],
                 cwd: Optional[str] = None, env: Optional[dict] = None):
        self.name = name
        self.command = command
        self.args = args
        self.cwd = cwd
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self.tools: list[dict] = []
        self._lock = threading.Lock()
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def connect(self) -> bool:
        """Start the MCP server process and discover tools."""
        try:
            # Build environment
            process_env = os.environ.copy()
            process_env.update(self.env)

            # Start process
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=process_env,
                text=True,
                bufsize=1,
            )

            # Initialize the connection
            if not self._initialize():
                self.disconnect()
                return False

            # Discover tools
            self.tools = self._list_tools()
            return True

        except Exception as e:
            print(f"MCP Error: Failed to start {self.name}: {e}")
            return False

    def _send_request(self, method: str, params: Optional[dict] = None) -> Optional[dict]:
        """Send a JSON-RPC request and get response."""
        if not self.process or self.process.poll() is not None:
            return None

        with self._lock:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": method,
            }
            if params:
                request["params"] = params

            try:
                # Send request
                request_line = json.dumps(request) + "\n"
                self.process.stdin.write(request_line)
                self.process.stdin.flush()

                # Read response
                response_line = self.process.stdout.readline()
                if not response_line:
                    return None

                response = json.loads(response_line)
                return response

            except Exception as e:
                print(f"MCP Error: Request failed for {self.name}: {e}")
                return None

    def _initialize(self) -> bool:
        """Send initialize request to MCP server."""
        response = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "jarvis",
                "version": "1.0.0"
            }
        })

        if not response or "result" not in response:
            return False

        # Send initialized notification
        self._send_notification("notifications/initialized", {})
        return True

    def _send_notification(self, method: str, params: Optional[dict] = None):
        """Send a JSON-RPC notification (no response expected)."""
        if not self.process or self.process.poll() is not None:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        try:
            notification_line = json.dumps(notification) + "\n"
            self.process.stdin.write(notification_line)
            self.process.stdin.flush()
        except Exception:
            pass

    def _list_tools(self) -> list[dict]:
        """Get list of available tools from server."""
        response = self._send_request("tools/list", {})

        if not response or "result" not in response:
            return []

        return response["result"].get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> Optional[Any]:
        """Execute a tool and return the result."""
        response = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        if not response:
            return None

        if "error" in response:
            return {"error": response["error"]}

        if "result" in response:
            return response["result"]

        return None

    def disconnect(self):
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        self.tools = []


class MCPManager:
    """Manages all MCP server connections."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self.servers: dict[str, MCPServer] = {}

    def load_config(self) -> bool:
        """Load MCP server configuration from file."""
        if not self.config_path.exists():
            print(f"MCP config not found: {self.config_path}")
            return False

        try:
            with open(self.config_path) as f:
                config = json.load(f)

            mcp_servers = config.get("mcpServers", {})

            for name, server_config in mcp_servers.items():
                command = server_config.get("command")
                args = server_config.get("args", [])
                cwd = server_config.get("cwd")
                env = server_config.get("env", {})

                if not command:
                    print(f"MCP Warning: No command for server {name}")
                    continue

                self.servers[name] = MCPServer(name, command, args, cwd, env)

            return True

        except Exception as e:
            print(f"MCP Error: Failed to load config: {e}")
            return False

    def connect_all(self) -> int:
        """Connect to all configured MCP servers. Returns number of successful connections."""
        connected = 0
        for name, server in self.servers.items():
            print(f"MCP: Connecting to {name}...")
            if server.connect():
                print(f"MCP: {name} connected ({len(server.tools)} tools)")
                connected += 1
            else:
                print(f"MCP: {name} failed to connect")
        return connected

    def get_all_tools(self) -> list[dict]:
        """Get aggregated list of tools from all servers with server prefix."""
        all_tools = []
        for server_name, server in self.servers.items():
            for tool in server.tools:
                full_name = f"{server_name}.{tool['name']}"
                if full_name in HIDDEN_TOOLS:
                    continue
                # Add server name prefix to tool name for routing
                tool_copy = tool.copy()
                tool_copy["_server"] = server_name
                tool_copy["full_name"] = full_name
                all_tools.append(tool_copy)
        return all_tools

    def call_tool(self, full_name: str, arguments: dict) -> Optional[Any]:
        """
        Call a tool by its full name (server.tool_name).

        Args:
            full_name: Tool name in format "server.tool_name"
            arguments: Tool arguments

        Returns:
            Tool result or None on error
        """
        if "." not in full_name:
            print(f"MCP Error: Invalid tool name format: {full_name}")
            return None

        server_name, tool_name = full_name.split(".", 1)

        if server_name not in self.servers:
            print(f"MCP Error: Unknown server: {server_name}")
            return None

        server = self.servers[server_name]
        return server.call_tool(tool_name, arguments)

    def get_tools_description(self, compact: bool = True) -> str:
        """Get a text description of all available tools for the LLM prompt.

        Args:
            compact: If True, only show required params and truncate descriptions.
        """
        tools = self.get_all_tools()
        if not tools:
            return ""

        lines = ["Available tools:"]
        for tool in tools:
            full_name = tool["full_name"]
            description = tool.get("description", "No description")

            # Truncate long descriptions for compact mode
            if compact:
                # Take first sentence only
                desc_end = description.find(". ")
                if desc_end > 0 and desc_end < 150:
                    description = description[:desc_end + 1]
                elif len(description) > 150:
                    description = description[:147] + "..."

            # Format input schema
            input_schema = tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            params = []
            for param_name, param_info in properties.items():
                # In compact mode, only show required params
                if compact and param_name not in required:
                    continue
                param_type = param_info.get("type", "any")
                req_marker = "*" if param_name in required else ""
                params.append(f"{param_name}{req_marker}: {param_type}")

            params_str = ", ".join(params) if params else ""
            lines.append(f"- {full_name}({params_str}): {description}")

        return "\n".join(lines)

    def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for name, server in self.servers.items():
            print(f"MCP: Disconnecting {name}...")
            server.disconnect()


if __name__ == "__main__":
    # Test the MCP manager
    manager = MCPManager()

    print("Loading config...")
    if not manager.load_config():
        print("Failed to load config")
        exit(1)

    print(f"Found {len(manager.servers)} servers")

    print("\nConnecting to servers...")
    connected = manager.connect_all()
    print(f"Connected to {connected} servers")

    print("\nAvailable tools:")
    print(manager.get_tools_description())

    # Test a tool call if weather server is available
    if "weather" in manager.servers and manager.servers["weather"].tools:
        print("\nTesting weather.get_current_weather for London...")
        result = manager.call_tool("weather.get_current_weather", {
            "location": {"city": "London", "country": "GB"},
            "units": "metric"
        })
        print(f"Result: {json.dumps(result, indent=2)}")

    print("\nDisconnecting...")
    manager.disconnect_all()
    print("Done")
