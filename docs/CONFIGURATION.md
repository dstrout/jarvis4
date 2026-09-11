# Configuration

Everything is driven by a single `config.json` in the project root. Copy the
example and edit it:

```bash
cp config.example.json config.json
```

Resolution order, last wins: built-in defaults → `config.json` → environment
variables.

## Sections

### `llm`

The orchestrator and every specialist agent talk to one OpenAI-compatible
endpoint.

| Key | Meaning |
|-----|---------|
| `provider` | `cerebras`, or any other name for a generic OpenAI-compatible backend |
| `api_key` | API key. Not required by most local servers |
| `base_url` | Endpoint, e.g. `https://api.cerebras.ai/v1` or `http://localhost:8080/v1` |
| `orchestrator_model` | Model for routing and synthesis — benefits from a reasoning model |
| `agent_model` | Model the specialist agents run on |
| `reasoning` | Cerebras-only reasoning fields. Automatically stripped when `provider` is not `cerebras`, since other backends reject them |

To point at a local llama.cpp or vLLM server:

```json
"llm": {
  "provider": "local",
  "base_url": "http://localhost:8080/v1",
  "api_key": "",
  "orchestrator_model": "your-model-id",
  "agent_model": "your-model-id"
}
```

### `asr`

A [WhisperLive](https://github.com/collabora/WhisperLive) server for streaming
speech-to-text. `host` / `port` / `model`.

### `tts`

A Wyoming-protocol TTS server — [Kokoro](https://github.com/remsky/Kokoro-FastAPI)
by default. `host` / `port` / `speed`.

### `voice`

| Key | Meaning |
|-----|---------|
| `wake_word` | An [openWakeWord](https://github.com/dscripka/openWakeWord) model name, e.g. `hey_jarvis` |
| `wake_threshold` | Detection confidence, 0–1. Lower triggers more easily |
| `silence_timeout` | Seconds of silence that end an utterance |
| `followup_timeout` | Seconds to keep listening for a follow-up without the wake word |
| `vad_threshold` | Energy threshold, or `null` for the audio driver's default |
| `audio_driver` | `generic` for any microphone via PyAudio, or `respeaker` for a ReSpeaker XVF3800 |
| `chime` | Path to the acknowledgement sound |

### `integrations`

Each integration is off by default. An agent is only loaded when its
integration is enabled, so a minimal install runs with just search.

| Integration | Enables | Needs |
|-------------|---------|-------|
| `homeassistant` | Home control agent | `url` and a long-lived access token |
| `slack` | Comms agent | A bot token (`xoxb-…`) |
| `google` | Mail and calendar agents | The `gws` CLI, authenticated |

### `security`

`enable_system_tools` exposes shell, file-read and file-write tools to the
orchestrator. Off by default. Turn it on only where you would be comfortable
letting the model run commands.

### Other

`http.port` for the local API, `logging.level`, and `mcp_servers` for inline
MCP server definitions.

## Environment variables

Any value can be overridden without touching the file. Useful for keeping
secrets out of `config.json`:

```
JARVIS4_LLM_API_KEY        JARVIS4_ASR_HOST
JARVIS4_LLM_BASE_URL       JARVIS4_ASR_PORT
JARVIS4_LLM_PROVIDER       JARVIS4_TTS_HOST
JARVIS4_LLM_REASONING      JARVIS4_TTS_PORT
JARVIS4_HA_URL             JARVIS4_HTTP_PORT
JARVIS4_HA_TOKEN           JARVIS4_SLACK_BOT_TOKEN
JARVIS4_SLACK_APP_TOKEN
```

`CEREBRAS_API_KEY` and `HA_TOKEN` are also honoured.

## Credentials

| Credential | Where to get it |
|------------|-----------------|
| Cerebras API key | https://cloud.cerebras.ai |
| Home Assistant token | Profile → Security → Long-lived access tokens |
| Slack bot token | https://api.slack.com/apps → OAuth & Permissions. Scopes: `chat:write`, `channels:history`, `channels:read`, `users:read` |
| Google Workspace | `gws` CLI, authenticated against your own OAuth client |

`config.json` is gitignored. Keep it that way.

## Companion files

Both are gitignored and optional; each has a committed `.example` showing the
format. Without them the relevant agent simply starts with an empty registry.

- **`ha_devices.json`** — maps spoken names to Home Assistant entity IDs, with
  aliases for fuzzy matching. The LLM sees the short names, not raw entity IDs.
- **`slack_contacts.json`** — maps names and channel names to Slack channel IDs.

`personality.md` defines the assistant's character and is committed, since it
contains no personal data. Edit it freely — it is loaded as the orchestrator's
system prompt, and the memory agent appends to its "Evolved notes" section over
time.

## MCP servers

Defined in `~/.config/jarvis4/mcp_servers.json`:

```json
{
  "mcpServers": {
    "weather": {
      "command": "python3",
      "args": ["/path/to/mcp-weather-server/server.py"]
    }
  }
}
```

Their tools are merged into the orchestrator's tool list at startup. Missing or
unreachable servers are logged and skipped.

## Command-line flags

Config covers normal operation; these are for debugging.

```
--config PATH      Use a different config file
--log-level LEVEL  DEBUG, INFO, WARNING, ERROR
--no-asr           Skip transcription
--no-tts           Skip speech output
--no-llm           Echo the transcription instead of answering
--no-http          Skip the HTTP API
--save-audio       Write captured audio to recordings/
```

## HTTP API

Served on `http.port` (default 8787) while Jarvis4 runs.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/status` | Health and current pipeline state |
| `POST` | `/chat` | Send text, get the assistant's reply |
| `POST` | `/speak` | Speak a string through TTS |
| `POST` | `/alert` | Inject a proactive notification |

## Runtime data

Written at runtime, all gitignored: `memory.db` (SQLite memory store),
`logs/`, and `recordings/` when `--save-audio` is set.
