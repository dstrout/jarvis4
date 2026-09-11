#!/usr/bin/env python3
"""
Jarvis4 Voice Assistant — entry point.

Loads config, creates components, runs the voice pipeline.
All settings come from config.json with CLI overrides.
"""

import argparse
import logging
import queue
import signal
import sys

from core import config


def main():
    parser = argparse.ArgumentParser(description='Jarvis4 Voice Assistant')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (default: ./config.json)')
    parser.add_argument('--log-level', type=str, default=None,
                        help='Log level override (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--no-asr', action='store_true',
                        help='Disable ASR transcription')
    parser.add_argument('--no-tts', action='store_true',
                        help='Disable TTS speech output')
    parser.add_argument('--no-llm', action='store_true',
                        help='Disable LLM (echo back transcription)')
    parser.add_argument('--no-http', action='store_true',
                        help='Disable HTTP API server')
    parser.add_argument('--save-audio', action='store_true',
                        help='Save audio recordings to disk')
    args = parser.parse_args()

    # --- Load config (must happen before any other imports that use config) ---
    cfg = config.load(args.config)

    # --- Set up logging ---
    log_level = args.log_level or config.get("logging", "level", default="INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("jarvis4")

    # --- Read settings from config ---
    voice_cfg = config.get_section("voice")
    asr_cfg = config.get_section("asr")
    tts_cfg = config.get_section("tts")
    http_port = config.get("http", "port", default=8787)

    audio_driver_name = voice_cfg.get("audio_driver", "generic")
    wake_word = voice_cfg.get("wake_word", "hey_jarvis")
    wake_threshold = voice_cfg.get("wake_threshold", 0.5)
    silence_timeout = voice_cfg.get("silence_timeout", 1.2)
    followup_timeout = voice_cfg.get("followup_timeout", 3.0)
    vad_threshold = voice_cfg.get("vad_threshold")  # None = driver default

    # --- Audio driver ---
    from audio import create_driver

    log.info("Initializing audio driver: %s", audio_driver_name)
    driver = create_driver(audio_driver_name, sample_rate=16000)
    try:
        driver.open()
        log.info("Audio driver ready (%s)", audio_driver_name)
    except Exception as e:
        log.error("Error opening audio: %s", e)
        sys.exit(1)

    # --- ASR ---
    asr_client = None
    if not args.no_asr:
        from core.asr_client import StreamingASRClient
        asr_host = asr_cfg.get("host", "localhost")
        asr_port = asr_cfg.get("port", 9090)
        asr_model = asr_cfg.get("model", "small")
        asr_client = StreamingASRClient(asr_host, asr_port, asr_model)
        if asr_client.connect():
            log.info("ASR: connected (%s:%d, model=%s)", asr_host, asr_port, asr_model)
        else:
            log.warning("ASR: connection failed, continuing without")
            asr_client = None

    # --- TTS ---
    tts_client = None
    if not args.no_tts:
        from core.tts_client import TTSClient
        tts_host = tts_cfg.get("host", "localhost")
        tts_port = tts_cfg.get("port", 10200)
        tts_speed = tts_cfg.get("speed", 1.5)
        tts_client = TTSClient(host=tts_host, port=tts_port, speed=tts_speed)
        log.info("TTS: %s:%d (speed=%.1f)", tts_host, tts_port, tts_speed)

    # --- MCP servers ---
    mcp_manager = None
    try:
        from core.mcp_manager import MCPManager
        mcp_manager = MCPManager()
        if mcp_manager.load_config():
            mcp_manager.connect_all()
            log.info("MCP: %d servers connected", len(mcp_manager.servers))
        else:
            log.warning("MCP: no config found, continuing without")
            mcp_manager = None
    except Exception as e:
        log.warning("MCP: init failed: %s", e)
        mcp_manager = None

    # --- LLM (orchestrator) ---
    llm_client = None
    if not args.no_llm:
        from core.orchestrator_client import OrchestratorClient
        llm_client = OrchestratorClient(mcp_manager=mcp_manager, use_bus=True)
        log.info("LLM: orchestrator initialized")

    # --- HTTP API ---
    http_queue = queue.Queue()
    if llm_client and not args.no_http:
        from core.http_api import start_http_server
        start_http_server(http_port, llm_client, tts_client, http_queue)
        log.info("HTTP: http://0.0.0.0:%d", http_port)

    # Wire proactive failure alerts
    if llm_client and hasattr(llm_client, 'set_alert_queue'):
        llm_client.set_alert_queue(http_queue)

    # --- Voice pipeline ---
    from core.voice_pipeline import VoicePipeline

    pipeline = VoicePipeline(
        audio_driver=driver,
        asr_client=asr_client,
        tts_client=tts_client,
        llm_client=llm_client,
        http_queue=http_queue,
        wake_word=wake_word,
        wake_threshold=wake_threshold,
        vad_threshold=vad_threshold,
        silence_timeout=silence_timeout,
        followup_timeout=followup_timeout,
        chunk_size=1280,
        save_audio=args.save_audio,
    )

    def signal_handler(sig, frame):
        print("\nShutting down...")
        pipeline.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log.info("Listening for '%s' (threshold: %.2f)", wake_word, wake_threshold)
    print(f"Jarvis4 ready — listening for '{wake_word}'")
    print("Press Ctrl+C to stop.\n")
    pipeline.run()


if __name__ == "__main__":
    main()
