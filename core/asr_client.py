"""
Jarvis4 ASR Client — streaming speech recognition via WhisperLive.

Connects to a WhisperLive server over WebSocket, streams audio chunks,
and provides real-time transcription with completed/in-progress segments.
"""

import json
import logging
import threading
import time
import uuid

import numpy as np
import websockets.sync.client as ws_client

log = logging.getLogger("jarvis4.asr")


class StreamingASRClient:
    """Streaming ASR client for WhisperLive server."""

    def __init__(self, host: str, port: int, model: str = "small"):
        self.host = host
        self.port = port
        self.model = model
        self.uid = str(uuid.uuid4())
        self.ws = None
        self.running = False
        self.server_ready = False
        self.current_transcription = ""
        self._completed_segments = []  # Completed segments from server
        self._receiver_thread = None
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Connect to the WhisperLive server. Returns True on success."""
        try:
            url = f"ws://{self.host}:{self.port}"
            self.ws = ws_client.connect(url, close_timeout=1)
            self.running = True

            # Send config
            config = {
                "uid": self.uid,
                "language": "en",
                "task": "transcribe",
                "model": self.model,
                "use_vad": False,
            }
            self.ws.send(json.dumps(config))

            # Start receiver thread
            self._receiver_thread = threading.Thread(
                target=self._receive_messages, daemon=True
            )
            self._receiver_thread.start()

            # Wait for server ready (with timeout)
            timeout = 5.0
            start = time.time()
            while not self.server_ready and self.running:
                if time.time() - start > timeout:
                    self.close()
                    return False
                time.sleep(0.05)

            return self.server_ready

        except Exception as e:
            log.error("ASR connection error: %s", e)
            self.running = False
            return False

    def _receive_messages(self):
        """Background thread to receive messages from server."""
        while self.running:
            try:
                message = self.ws.recv(timeout=0.5)
                if message:
                    self._handle_message(message)
            except TimeoutError:
                continue
            except Exception:
                if self.running:
                    self.running = False
                break

    def _handle_message(self, message: str):
        """Handle a message from the server."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        if data.get("uid") != self.uid:
            return

        # Handle server ready
        if data.get("message") == "SERVER_READY":
            self.server_ready = True
            return

        # Handle transcription segments
        if "segments" in data and data["segments"]:
            with self._lock:
                # Track completed vs in-progress segments
                completed_texts = []
                in_progress_text = ""

                for seg in data["segments"]:
                    text = seg.get("text", "").strip()
                    if not text:
                        continue

                    if seg.get("completed", False):
                        completed_texts.append(text)
                    else:
                        in_progress_text = text

                # Update completed segments if we have new ones
                if completed_texts:
                    self._completed_segments = completed_texts

                # Build full transcription from completed + in-progress
                all_parts = self._completed_segments.copy()
                if in_progress_text:
                    all_parts.append(in_progress_text)

                if all_parts:
                    self.current_transcription = " ".join(all_parts)

    def send_audio(self, audio_int16: np.ndarray):
        """Send audio chunk to server. Audio should be 16-bit PCM."""
        if not self.running or not self.ws:
            return

        try:
            # Convert to float32 normalized to [-1, 1]
            audio_float = audio_int16.astype(np.float32) / 32768.0
            self.ws.send(audio_float.tobytes())
        except Exception:
            self.running = False

    def get_transcription(self) -> str:
        """Get the current transcription."""
        with self._lock:
            return self.current_transcription

    def reset(self):
        """Reset transcription state for a new utterance by reconnecting.

        WhisperLive server closes the connection when a new config is sent,
        so we must close and reconnect for each new utterance.
        """
        # Close existing connection
        self.running = False
        if self._receiver_thread:
            self._receiver_thread.join(timeout=1.0)
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

        # Clear state
        with self._lock:
            self.current_transcription = ""
            self._completed_segments = []

        # Reconnect with new UID
        self.uid = str(uuid.uuid4())
        self.ws = None
        self.server_ready = False
        self.running = False

        # Reconnect
        try:
            url = f"ws://{self.host}:{self.port}"
            self.ws = ws_client.connect(url, close_timeout=1)
            self.running = True

            # Send config
            config = {
                "uid": self.uid,
                "language": "en",
                "task": "transcribe",
                "model": self.model,
                "use_vad": False,
            }
            self.ws.send(json.dumps(config))

            # Start receiver thread
            self._receiver_thread = threading.Thread(
                target=self._receive_messages, daemon=True
            )
            self._receiver_thread.start()

            # Wait for server ready
            timeout = 2.0
            start = time.time()
            while not self.server_ready and self.running:
                if time.time() - start > timeout:
                    break
                time.sleep(0.05)
        except Exception as e:
            log.error("ASR reconnect error: %s", e)
            self.running = False

    def finish(self):
        """Signal end of audio stream."""
        if self.ws and self.running:
            try:
                self.ws.send("END_OF_AUDIO".encode("utf-8"))
            except Exception:
                pass

    def close(self):
        """Close the connection."""
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.ws = None
        self.server_ready = False
        self.current_transcription = ""
        self._completed_segments = []
