"""
Home Control Agent — home automation specialist.

Controls lights, switches, and devices via the HomeAssistant REST API.
Device registry lives in ha_devices.json (project root) for easy editing.

Auth: HA_TOKEN env var, or ha_token.txt in project root.
API: POST /api/services/<domain>/<service> with entity_id in body.
"""

import json
import logging
import os
from pathlib import Path

import httpx

from agents import Agent, RunContextWrapper, function_tool
from core.agent_base import make_agent

log = logging.getLogger("jarvis.home")

_PROJECT_ROOT = Path(__file__).parent.parent
def _get_ha_url():
    try:
        from core import config
        url = config.get("integrations", "homeassistant", "url")
        if url:
            return url
    except Exception:
        pass
    return os.environ.get("HA_URL", "http://homeassistant.local:8123")

_HA_BASE = _get_ha_url()
_TIMEOUT = 10.0

# ---------------------------------------------------------------------------
# Auth + HTTP
# ---------------------------------------------------------------------------

def _get_token() -> str:
    # Try config first, then env var, then legacy file
    try:
        from core import config
        tok = config.get("integrations", "homeassistant", "token")
        if tok:
            return tok
    except Exception:
        pass
    tok = os.environ.get("HA_TOKEN")
    if tok:
        return tok
    tok_file = _PROJECT_ROOT / "ha_token.txt"
    if tok_file.exists():
        return tok_file.read_text().strip()
    raise RuntimeError("No HA token. Set in config.json, HA_TOKEN env, or ha_token.txt")


def _ha_headers() -> dict:
    return {"Authorization": f"Bearer {_get_token()}"}


def _ha_post(domain: str, service: str, data: dict) -> dict | str:
    """Call a HA service. Returns the JSON response or error string."""
    url = f"{_HA_BASE}/api/services/{domain}/{service}"
    log.info("HA POST %s/%s entity=%s", domain, service, data.get("entity_id", "?"))
    try:
        resp = httpx.post(url, json=data, headers=_ha_headers(), timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except httpx.HTTPStatusError as e:
        return f"Error: HA returned {e.response.status_code}: {e.response.text[:200]}"
    except httpx.ConnectError:
        return "Error: cannot reach HomeAssistant"
    except Exception as e:
        return f"Error: {e}"


def _ha_get(path: str) -> dict | list | str:
    """GET a HA API endpoint. Returns parsed JSON or error string."""
    url = f"{_HA_BASE}{path}"
    try:
        resp = httpx.get(url, headers=_ha_headers(), timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return f"Error: HA returned {e.response.status_code}"
    except httpx.ConnectError:
        return "Error: cannot reach HomeAssistant"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Device registry
# ---------------------------------------------------------------------------

_registry: dict | None = None


def _load_registry() -> dict:
    global _registry
    if _registry is not None:
        return _registry

    reg_file = _PROJECT_ROOT / "ha_devices.json"
    if not reg_file.exists():
        log.error("ha_devices.json not found at %s", reg_file)
        _registry = {}
        return _registry

    with open(reg_file) as f:
        _registry = json.load(f)
    log.info("Loaded %d devices from ha_devices.json",
             len(_registry.get("devices", {})))
    return _registry


def _find_device(name: str) -> tuple[str, dict] | None:
    """Find a device by name or alias. Returns (canonical_name, device_dict)."""
    reg = _load_registry()
    devices = reg.get("devices", {})
    name_lower = name.lower().strip()

    # Exact key match
    if name_lower in devices:
        return name_lower, devices[name_lower]

    # Alias match
    for key, dev in devices.items():
        if name_lower in [a.lower() for a in dev.get("aliases", [])]:
            return key, dev

    # Partial match — name is substring of key or vice versa
    for key, dev in devices.items():
        if name_lower in key or key in name_lower:
            return key, dev
        for alias in dev.get("aliases", []):
            if name_lower in alias.lower() or alias.lower() in name_lower:
                return key, dev

    return None


def _find_automation(name: str) -> tuple[str, dict] | None:
    reg = _load_registry()
    autos = reg.get("automations", {})
    name_lower = name.lower().strip()

    if name_lower in autos:
        return name_lower, autos[name_lower]

    for key, auto in autos.items():
        if name_lower in [a.lower() for a in auto.get("aliases", [])]:
            return key, auto
        if name_lower in key or key in name_lower:
            return key, auto

    return None


def _find_sensor(name: str) -> tuple[str, dict] | None:
    reg = _load_registry()
    sensors = reg.get("sensors", {})
    name_lower = name.lower().strip()

    if name_lower in sensors:
        return name_lower, sensors[name_lower]

    for key, sensor in sensors.items():
        if name_lower in [a.lower() for a in sensor.get("aliases", [])]:
            return key, sensor
        if name_lower in key or key in name_lower:
            return key, sensor

    return None


# ---------------------------------------------------------------------------
# Typed tools
# ---------------------------------------------------------------------------

@function_tool
def control_device(device: str, action: str = "toggle") -> str:
    """Turn a device on, off, or toggle it.
    Args:
        device: Device name (e.g. "office light", "workbench", "heater")
        action: "on", "off", or "toggle"
    """
    match = _find_device(device)
    if not match:
        return f"Error: device '{device}' not found in registry."

    name, dev = match
    entity_id = dev["entity_id"]
    domain = entity_id.split(".")[0]

    service_map = {"on": "turn_on", "off": "turn_off", "toggle": "toggle"}
    service = service_map.get(action.lower())
    if not service:
        return f"Error: action must be on, off, or toggle — got '{action}'"

    result = _ha_post(domain, service, {"entity_id": entity_id})
    if isinstance(result, str) and result.startswith("Error"):
        return result

    return f"{name.title()} turned {action}."


@function_tool
def set_light(
    device: str,
    brightness: int = -1,
    color: str = "",
) -> str:
    """Set brightness and/or color on a light entity.
    Args:
        device: Light name (e.g. "office light")
        brightness: 0-100 percent (-1 to leave unchanged)
        color: Color name like "red", "blue", "warm_white", or RGB hex "#FF0000"
    """
    match = _find_device(device)
    if not match:
        return f"Error: device '{device}' not found."

    name, dev = match
    entity_id = dev["entity_id"]
    domain = entity_id.split(".")[0]

    if domain != "light":
        return f"Error: '{name}' is a {domain}, not a light. Use control_device for on/off."

    data: dict = {"entity_id": entity_id}

    if brightness >= 0:
        # HA uses 0-255
        data["brightness"] = max(0, min(255, int(brightness * 255 / 100)))

    if color:
        color = color.strip().lower()
        # Named colors → RGB tuples
        named = {
            "red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255],
            "white": [255, 255, 255], "warm_white": [255, 180, 100],
            "cool_white": [200, 220, 255], "yellow": [255, 255, 0],
            "orange": [255, 140, 0], "purple": [128, 0, 255],
            "pink": [255, 105, 180], "cyan": [0, 255, 255],
        }
        if color in named:
            data["rgb_color"] = named[color]
        elif color.startswith("#") and len(color) == 7:
            data["rgb_color"] = [int(color[i:i+2], 16) for i in (1, 3, 5)]
        else:
            return f"Error: unknown color '{color}'. Use a name (red, blue, warm_white...) or hex (#FF0000)."

    result = _ha_post("light", "turn_on", data)
    if isinstance(result, str) and result.startswith("Error"):
        return result

    parts = [f"{name.title()}"]
    if brightness >= 0:
        parts.append(f"brightness {brightness}%")
    if color:
        parts.append(f"color {color}")
    return " — ".join(parts) + "."


@function_tool
def get_device_states(room: str = "", group: str = "") -> str:
    """Get current state of devices. Filter by room or group, or leave blank for all.
    Args:
        room: Filter by room (e.g. "office", "rec room")
        group: Filter by group (e.g. "office_aquarium", "rec_room_aquarium")
    """
    reg = _load_registry()
    devices = reg.get("devices", {})

    targets = {}
    for name, dev in devices.items():
        if room and dev.get("room", "").lower() != room.lower():
            continue
        if group and dev.get("group", "").lower() != group.lower():
            continue
        targets[name] = dev

    if not targets:
        return f"No devices found" + (f" in room '{room}'" if room else "") + (f" in group '{group}'" if group else "") + "."

    # Batch-fetch all states in one call
    all_states = _ha_get("/api/states")
    if isinstance(all_states, str):
        return all_states

    state_map = {s["entity_id"]: s for s in all_states}

    lines = []
    for name, dev in targets.items():
        eid = dev["entity_id"]
        ha_state = state_map.get(eid)
        if not ha_state:
            lines.append(f"  {name}: unknown (entity not found)")
            continue
        state_val = ha_state["state"]
        attrs = ha_state.get("attributes", {})
        extra = ""
        if dev.get("type") == "light" and state_val == "on":
            brightness = attrs.get("brightness")
            if brightness is not None:
                extra = f" ({int(brightness * 100 / 255)}%)"
        lines.append(f"  {name}: {state_val}{extra}")

    return "\n".join(lines)


@function_tool
def get_sensor(sensor: str) -> str:
    """Read a sensor value.
    Args:
        sensor: Sensor name (e.g. "ups battery", "phone battery", "printer ink")
    """
    match = _find_sensor(sensor)
    if not match:
        return f"Error: sensor '{sensor}' not found in registry."

    name, sen = match
    entity_id = sen["entity_id"]

    state = _ha_get(f"/api/states/{entity_id}")
    if isinstance(state, str):
        return state

    value = state.get("state", "unknown")
    unit = sen.get("unit", "")
    friendly = state.get("attributes", {}).get("friendly_name", name)
    return f"{friendly}: {value}{unit}"


@function_tool
def trigger_automation(name: str) -> str:
    """Trigger a HomeAssistant automation by name.
    Args:
        name: Automation name (e.g. "all office lights on", "morning lights")
    """
    match = _find_automation(name)
    if not match:
        return f"Error: automation '{name}' not found."

    auto_name, auto = match
    entity_id = auto["entity_id"]

    result = _ha_post("automation", "trigger", {"entity_id": entity_id})
    if isinstance(result, str) and result.startswith("Error"):
        return result

    return f"Triggered automation: {auto_name}."


@function_tool
def activate_scene(name: str) -> str:
    """Activate a HomeAssistant scene.
    Args:
        name: Scene name (e.g. "office")
    """
    reg = _load_registry()
    scenes = reg.get("scenes", {})
    name_lower = name.lower().strip()

    entity_id = None
    scene_name = None
    for key, sc in scenes.items():
        if name_lower == key or name_lower in [a.lower() for a in sc.get("aliases", [])]:
            entity_id = sc["entity_id"]
            scene_name = key
            break

    if not entity_id:
        return f"Error: scene '{name}' not found."

    result = _ha_post("scene", "turn_on", {"entity_id": entity_id})
    if isinstance(result, str) and result.startswith("Error"):
        return result

    return f"Activated scene: {scene_name}."


@function_tool
def control_group(group: str, action: str = "toggle") -> str:
    """Control all devices in a group at once.
    Args:
        group: Group name (e.g. "office_aquarium", "rec_room_aquarium")
        action: "on", "off", or "toggle"
    """
    reg = _load_registry()
    devices = reg.get("devices", {})

    targets = {k: v for k, v in devices.items()
               if v.get("group", "").lower() == group.lower()}

    if not targets:
        return f"Error: no devices in group '{group}'."

    service_map = {"on": "turn_on", "off": "turn_off", "toggle": "toggle"}
    service = service_map.get(action.lower())
    if not service:
        return f"Error: action must be on, off, or toggle."

    results = []
    for name, dev in targets.items():
        entity_id = dev["entity_id"]
        domain = entity_id.split(".")[0]
        result = _ha_post(domain, service, {"entity_id": entity_id})
        if isinstance(result, str) and result.startswith("Error"):
            results.append(f"  {name}: {result}")
        else:
            results.append(f"  {name}: {action}")

    return f"Group '{group}':\n" + "\n".join(results)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_home_instructions(context: RunContextWrapper, agent: Agent) -> str:
    reg = _load_registry()
    devices = reg.get("devices", {})
    automations = reg.get("automations", {})
    scenes = reg.get("scenes", {})
    sensors = reg.get("sensors", {})

    # Build concise device list for the prompt
    device_lines = []
    for name, dev in devices.items():
        room = dev.get("room", "")
        group = dev.get("group", "")
        dtype = dev.get("type", "")
        extras = []
        if room:
            extras.append(room)
        if group:
            extras.append(f"group:{group}")
        if dtype == "light":
            extras.append("RGB")
        tag = f" ({', '.join(extras)})" if extras else ""
        device_lines.append(f"  - {name}{tag}")

    auto_lines = [f"  - {name}" for name in automations]
    scene_lines = [f"  - {name}" for name in scenes]
    sensor_lines = [f"  - {name} ({s.get('unit', '')})" for name, s in sensors.items()]

    return f"""\
You are a home automation specialist. You control devices via HomeAssistant.

## Your tools
- control_device(device, action): Turn a device on/off/toggle. Works for all switches and lights.
- set_light(device, brightness, color): Set brightness (0-100) and/or color on RGB lights.
- get_device_states(room, group): Get current state of devices. Filter by room or group.
- get_sensor(sensor): Read a sensor value (UPS, battery, ink levels, etc).
- trigger_automation(name): Trigger a HA automation.
- activate_scene(name): Activate a HA scene.
- control_group(group, action): Control all devices in a group at once (on/off/toggle).

## Known devices
{chr(10).join(device_lines)}

## Groups
  - office_aquarium: filter, heater, air pump, light, co2
  - rec_room_aquarium: light, filter, air pump, heater

## Automations
{chr(10).join(auto_lines)}

## Scenes
{chr(10).join(scene_lines)}

## Sensors
{chr(10).join(sensor_lines)}

## Key rules
- Use the exact device names listed above. The registry handles fuzzy matching.
- For "turn on/off the office lights" (plural), trigger the automation "all office lights on/off".
- For individual lights, use control_device or set_light.
- For aquarium operations, prefer control_group to handle all devices at once.
- Report what you did concisely. No markdown. Plain text only.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

home_control_agent = make_agent(
    name="home_control_agent",
    instructions=_build_home_instructions,
    extra_tools=[
        control_device, set_light, get_device_states, get_sensor,
        trigger_automation, activate_scene, control_group,
    ],
)

# Auto-discovery metadata
AGENT_CONFIG = {
    "tool_name": "home",
    "tool_description": (
        "Home automation: control lights, thermostat, locks, garage door. "
        "Check device status. Pass device names and desired actions."
    ),
    "write_keywords": [
        "turn on", "turn off", "set", "lock", "unlock",
        "open", "close", "toggle", "dim", "brighten",
    ],
}
REQUIRES_INTEGRATION = "homeassistant"
agent = home_control_agent
