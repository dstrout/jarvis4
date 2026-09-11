# Jarvis4

A self-hosted voice assistant that routes speech to specialist agents, built on
the OpenAI Agents SDK and any OpenAI-compatible LLM.

## Problem

Voice assistants feel slow, and the model usually is not the reason. Inference
got fast; the pipeline around it did not. Three things add the latency you
actually hear:

- **Record, then process.** Most pipelines wait for you to stop talking before
  transcription begins, so every utterance costs its own length before any work
  starts.
- **Blocking on tools.** "Turn on the office lights and tell me tomorrow's
  weather" holds the whole conversation open while a home automation API is
  poked. The assistant goes quiet at exactly the moment it should be answering.
- **Synthesize, then speak.** Waiting for a complete response before producing
  the first audio adds the length of the answer to the wait.

The other half of the problem is reach. A cloud assistant cannot read your
mail, drive your home automation, or be taught a new capability by you.

## Solution

Jarvis4 runs the whole loop on hardware you control, and treats the pipeline —
not the model — as the thing worth optimising.

- **Streaming transcription.** Audio streams to a WhisperLive server as you
  speak, so text is ready close to when you stop.
- **A message bus for writes.** The orchestrator classifies each delegation as
  a read or a write. Reads run synchronously because their answer is the reply.
  Writes dispatch fire-and-forget, return "task accepted" immediately, and
  report back through the conversation if they fail. Turning on a light never
  blocks a sentence.
- **Specialist agents, not one prompt.** A fast orchestrator model routes to
  focused agents — search, email, calendar, home control, comms, memory — each
  with its own tools and a smaller, sharper prompt.
- **Skills loaded on demand.** Capabilities live as Markdown files that agents
  load when relevant, keeping prompts small while the reachable surface stays
  large.
- **Any OpenAI-compatible backend.** Cerebras by default; point `base_url` at
  llama.cpp, vLLM or Ollama and the Cerebras-only request fields are stripped
  automatically.
- **Long-term memory.** A SQLite store with optional semantic search remembers
  preferences and habits across conversations, and evolves the assistant's
  personality file over time.

## Install

Requires Python 3.11+, a microphone, and Linux (systemd for the service; the
rest is portable).

```bash
git clone https://github.com/dstrout/jarvis4.git
cd jarvis4

sudo apt install $(cat docs/system_packages.txt)   # portaudio, libusb, ffmpeg libs
./install.sh                                        # config + deps + systemd service
```

Then add your credentials:

```bash
cp config.example.json config.json   # install.sh does this if absent
$EDITOR config.json                  # set llm.api_key at minimum
```

Two external services do the speech work. Both are optional — without them
Jarvis4 still runs over the HTTP API:

| Service | Purpose | Default |
|---------|---------|---------|
| [WhisperLive](https://github.com/collabora/WhisperLive) | Streaming speech-to-text | `localhost:9090` |
| [Kokoro](https://github.com/remsky/Kokoro-FastAPI) via Wyoming | Text-to-speech | `localhost:10200` |

Optional extras:

```bash
./install.sh --memory    # semantic memory search (pulls in torch, ~400MB)
```

Without it, memory still works — search falls back to keyword matching.

Integrations are off by default. Enable the ones you want in `config.json`;
each unlocks its agent at startup. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Usage

```bash
python3 jarvis4.py
```

Say the wake word, then talk:

> "Hey Jarvis — what's the weather tomorrow?"
> "Hey Jarvis — turn the office lights to 30% and tell me if I have anything this afternoon."

The second one is the interesting case: the light change dispatches to the bus
and the calendar lookup answers in the same breath, rather than one waiting on
the other.

As a service:

```bash
sudo systemctl start jarvis4
journalctl -u jarvis4 -f
```

Over HTTP, with no microphone involved:

```bash
curl -X POST localhost:8787/chat -d '{"message": "what did I miss in email today?"}'
```

Useful flags while developing — `--no-tts` to keep it quiet, `--no-llm` to echo
transcription and check the audio path, `--log-level DEBUG` for routing
decisions. Full list in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Architecture

```
microphone ─→ wake word ─→ streaming ASR ─→ orchestrator ─→ TTS ─→ speaker
              (openWakeWord)  (WhisperLive)       │        (Kokoro)
                                                  │
                          ┌───────────────────────┴─────────────┐
                          │                                     │
                    reads (sync)                        writes (bus)
                          │                                     │
              search · email · calendar              home · comms · build
                          │                                     │
                     answer now                    "task accepted" now,
                                                   failures reported later
```

| Directory | Contents |
|-----------|----------|
| `core/` | Voice pipeline, orchestrator, message bus, config, ASR/TTS clients, MCP, memory store |
| `agent_modules/` | Specialist agents, auto-discovered at startup |
| `audio/` | Microphone abstraction — `generic` (any PyAudio device) or `respeaker` |
| `skills/` | Markdown capability definitions loaded on demand |
| `tests/` | Unit tests, plus integration tests that skip without credentials |

**Adding an agent** means dropping a file in `agent_modules/` that exports
`AGENT_CONFIG` and an `agent`. Discovery picks it up — there is no registry to
edit. Set `REQUIRES_INTEGRATION` or `REQUIRES_SECURITY` and it stays unloaded
until that is enabled in config. `agent_modules/_template.py` is a starting
point.

**Agents that ship:**

| Agent | Tool | Requires |
|-------|------|----------|
| Search | `search` | — |
| Email | `email` | `integrations.google` |
| Calendar and tasks | `calendar` | `integrations.google` |
| Home control | `home` | `integrations.homeassistant` |
| Comms (Slack, notifications) | `comms` | `integrations.slack` |
| Memory | internal | — |
| Build (writes new agents via Claude Code) | `build` | `security.enable_system_tools` |
| System tools (shell, file I/O) | internal | `security.enable_system_tools` |

The last two are off by default and gated behind a security setting, because
they let the model run commands on the host.

## Tests

```bash
python3 -m pytest tests/
```

Tests that need real credentials skip themselves rather than fail, so a fresh
clone runs green.

## Limitations

- Linux-oriented. The service unit and audio setup assume systemd and ALSA.
- Speech quality depends entirely on the ASR and TTS servers you point it at.
- The `build` agent shells out to Claude Code and is genuinely powerful — read
  `agent_modules/_claude_code.py` before enabling `security.enable_system_tools`.
- Memory consolidation is newer than the rest and has less real-world mileage.

## Third-party components

`skills/` contains 92 skill definitions generated by the
[Google Workspace CLI](https://github.com/googleworkspace/cli) (`@googleworkspace/cli`
v0.11.1), vendored under Apache-2.0 so the repository is usable without a
separate generation step. They are not authored by this project. The full
license and details are in [`skills/LICENSE`](skills/LICENSE) and
[`skills/README.md`](skills/README.md); three additional skills in that
directory are original to this project and MIT-licensed with the rest.

The ReSpeaker XVF3800 vendor drivers are **not** redistributed here. If you use
that microphone, clone
[the vendor repository](https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY)
into the project root and set `voice.audio_driver` to `respeaker`.

## License

MIT — see [LICENSE](LICENSE).
