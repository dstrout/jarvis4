"""
Jarvis4 Voice Pipeline — wake word → ASR → LLM → TTS state machine.

States: IDLE → LISTENING → SPEAKING → FOLLOWUP
Handles wake word detection, streaming ASR, LLM queries, TTS playback,
follow-up listening, HTTP queue requests, and proactive failure alerts.
"""

import logging
import os
import queue
import time
import wave
from datetime import datetime
from typing import Optional

import numpy as np
from openwakeword.model import Model

from audio.base import AudioDriver
from core.asr_client import StreamingASRClient

log = logging.getLogger("jarvis4.pipeline")

# State machine states
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_SPEAKING = "speaking"
STATE_FOLLOWUP = "followup"


class VoicePipeline:
    """Voice assistant state machine.

    Takes its dependencies as constructor arguments — no global state.
    Call run() to start the blocking main loop.
    """

    def __init__(
        self,
        audio_driver: AudioDriver,
        asr_client: Optional[StreamingASRClient],
        tts_client,  # TTSClient — optional import
        llm_client,  # OrchestratorClient or any .chat(str)->str
        http_queue: queue.Queue,
        wake_word: str = "hey_jarvis",
        wake_threshold: float = 0.5,
        vad_threshold: float = None,
        silence_timeout: float = 1.2,
        followup_timeout: float = 3.0,
        chunk_size: int = 1280,
        save_audio: bool = False,
        recordings_dir: str = None,
    ):
        self.driver = audio_driver
        self.asr = asr_client
        self.tts = tts_client
        self.llm = llm_client
        self.http_queue = http_queue
        self.wake_word = wake_word
        self.wake_threshold = wake_threshold
        self.silence_timeout = silence_timeout
        self.followup_timeout = followup_timeout
        self.chunk_size = chunk_size
        self.save_audio = save_audio
        self.recordings_dir = recordings_dir or "recordings"

        # Use driver-specific VAD threshold if not overridden
        self.vad_threshold = vad_threshold if vad_threshold is not None else audio_driver.default_vad_threshold

        # Shutdown flag — set from signal handler
        self.running = True

        # State
        self._state = STATE_IDLE
        self._last_speech_time = 0
        self._last_wake_time = 0
        self._wake_cooldown = 2.0
        self._last_displayed_transcription = ""
        self._followup_start_time = 0
        self._audio_buffer = []
        self._recording_counter = 0

        # Wake word model
        self._oww_model = Model()
        if wake_word not in self._oww_model.models:
            raise ValueError(
                f"Wake word '{wake_word}' not found. "
                f"Available: {list(self._oww_model.models.keys())}"
            )
        log.info("Wake word models: %s", list(self._oww_model.models.keys()))

    def stop(self):
        """Signal the pipeline to stop."""
        self.running = False

    def run(self):
        """Main loop — blocks until stop() is called or interrupted."""
        if self.save_audio:
            os.makedirs(self.recordings_dir, exist_ok=True)

        self.driver.set_idle()
        log.info(
            "Pipeline running: wake=%s threshold=%.2f vad=%.0f silence=%.1fs",
            self.wake_word, self.wake_threshold, self.vad_threshold, self.silence_timeout,
        )

        try:
            while self.running:
                self._tick()
        except Exception as e:
            log.error("Pipeline error: %s", e, exc_info=True)
        finally:
            self._cleanup()

    def _tick(self):
        """One iteration of the main loop."""
        try:
            audio_array = self.driver.read_chunk(self.chunk_size)
        except Exception as e:
            log.warning("Audio read error: %s", e)
            return

        current_time = time.time()

        # Check HTTP queue
        http_req = self._poll_http_queue()
        if http_req is not None:
            self._handle_http_request(http_req, current_time)
            return

        if self._state == STATE_IDLE:
            self._tick_idle(audio_array, current_time)
        elif self._state == STATE_LISTENING:
            self._tick_listening(audio_array, current_time)
        elif self._state == STATE_FOLLOWUP:
            self._tick_followup(audio_array, current_time)

    def _poll_http_queue(self):
        try:
            return self.http_queue.get_nowait()
        except queue.Empty:
            return None

    # --- State handlers ---

    def _tick_idle(self, audio_array, current_time):
        prediction = self._oww_model.predict(audio_array)
        score = prediction.get(self.wake_word, 0)

        if score > self.wake_threshold:
            if current_time - self._last_wake_time > self._wake_cooldown:
                self._last_wake_time = current_time
                self._state = STATE_LISTENING
                log.info("Wake word detected (%s: %.3f)", self.wake_word, score)

                self.driver.set_listening()

                if self.asr:
                    self.asr.reset()

                self._last_speech_time = time.time()
                self._last_displayed_transcription = ""

                if self.save_audio:
                    self._audio_buffer = []

    def _tick_listening(self, audio_array, current_time):
        if self.save_audio:
            self._audio_buffer.append(audio_array.copy())

        if self.asr:
            self.asr.send_audio(audio_array)
            transcription = self.asr.get_transcription()
            if transcription and transcription != self._last_displayed_transcription:
                print(f"\r    \033[K>>> {transcription}", end="", flush=True)
                self._last_displayed_transcription = transcription

        # Check VAD
        speech_energy = self.driver.get_speech_energy()
        if speech_energy > self.vad_threshold:
            self._last_speech_time = current_time

        silence_duration = current_time - self._last_speech_time

        if silence_duration > self.silence_timeout:
            self._on_silence_detected()

    def _on_silence_detected(self):
        """User stopped speaking — get transcription, query LLM, speak response."""
        final_transcription = ""
        if self.asr:
            self.asr.finish()
            time.sleep(0.3)
            final_transcription = self.asr.get_transcription()

        if final_transcription:
            print(f"\r    \033[K>>> {final_transcription}")
            log.info("Final transcription: %s", final_transcription)
        else:
            print()

        # Save recording
        if self.save_audio and self._audio_buffer:
            self._save_recording(final_transcription)

        if final_transcription:
            self._process_transcription(final_transcription)
        else:
            self._return_to_idle()

    def _process_transcription(self, text):
        """Send transcription to LLM and speak response."""
        if self.llm:
            self.driver.set_thinking()
            log.info("Querying LLM...")
            response = self.llm.chat(text)
            if response:
                log.info("Response: %s", response[:100])
            else:
                response = "I'm sorry, I couldn't process that request."
                log.warning("LLM returned no response")
        else:
            response = f"You said: {text}"

        if self.tts:
            self._state = STATE_SPEAKING
            self.driver.set_speaking()
            self.tts.synthesize(response)

            self._state = STATE_FOLLOWUP
            self.driver.set_listening()
            self._last_speech_time = time.time()
            self._followup_start_time = time.time()
            self._last_displayed_transcription = ""

            if self.asr:
                self.asr.reset()
        else:
            self._return_to_idle()

    def _tick_followup(self, audio_array, current_time):
        if self.asr:
            self.asr.send_audio(audio_array)
            transcription = self.asr.get_transcription()
            if transcription and transcription != self._last_displayed_transcription:
                print(f"\r    \033[K>>> {transcription}", end="", flush=True)
                self._last_displayed_transcription = transcription

        speech_energy = self.driver.get_speech_energy()
        is_speaking = speech_energy > self.vad_threshold

        if is_speaking:
            self._last_speech_time = current_time
            if self._state == STATE_FOLLOWUP:
                self._state = STATE_LISTENING
                log.info("Follow-up detected")

        if self._state == STATE_FOLLOWUP:
            time_since_tts = current_time - self._followup_start_time
            silence_duration = current_time - self._last_speech_time

            if self._last_displayed_transcription and silence_duration > self.silence_timeout:
                self._state = STATE_LISTENING
            elif time_since_tts > self.followup_timeout and not self._last_displayed_transcription:
                self._return_to_idle()

    def _return_to_idle(self):
        """Return to idle state, reset wake word model."""
        self._state = STATE_IDLE
        self.driver.set_idle()

        # Feed silence to reset wake word detector
        self._oww_model.reset()
        silence_chunk = np.zeros(self.chunk_size, dtype=np.int16)
        for _ in range(24000 // self.chunk_size):
            self._oww_model.predict(silence_chunk)

    # --- HTTP request handling ---

    def _handle_http_request(self, http_req, current_time):
        action = http_req["action"]
        req_text = http_req["text"]
        done_event = http_req.get("done")
        result = http_req.get("result", {})

        log.info("[HTTP/%s] %s", action, req_text[:60])

        if action == "task_failure":
            self._handle_task_failure(req_text, done_event)
            return

        # Standard actions: speak, chat, alert
        self._state = STATE_SPEAKING
        self.driver.set_speaking()

        try:
            if action == "speak":
                if self.tts:
                    self.tts.play_chime()
                    time.sleep(2)
                    self.tts.synthesize(req_text)
                result["status"] = "spoken"
            elif action == "chat":
                self.driver.set_thinking()
                response = self.llm.chat(req_text) if self.llm else None
                result["response"] = response or ""
                if response and self.tts:
                    self.driver.set_speaking()
                    self.tts.play_chime()
                    self.tts.synthesize(response)
            elif action == "alert":
                self.driver.set_thinking()
                response = self.llm.chat(f"ALERT: {req_text}") if self.llm else None
                result["response"] = response or ""
                if response and self.tts:
                    self.driver.set_speaking()
                    self.tts.play_chime()
                    self.tts.synthesize(response)
        except Exception as e:
            log.error("[HTTP/%s] Error: %s", action, e)
            result["error"] = str(e)

        if done_event:
            done_event.set()

        # Transition to FOLLOWUP
        self._state = STATE_FOLLOWUP
        self.driver.set_listening()
        self._last_speech_time = time.time()
        self._followup_start_time = time.time()
        self._last_displayed_transcription = ""

        if self.asr:
            self.asr.reset()

    def _handle_task_failure(self, req_text, done_event):
        """Proactive failure interrupt: chime → 'Excuse me sir' → listen for ack."""
        self._state = STATE_SPEAKING
        self.driver.set_speaking()
        if self.tts:
            self.tts.play_chime()
        time.sleep(2)

        if self.tts:
            self.tts.synthesize("Excuse me sir, a background task requires your attention.")

        # Listen for acknowledgment (5 second window)
        self._state = STATE_FOLLOWUP
        self.driver.set_listening()
        if self.asr:
            self.asr.reset()

        got_ack, got_dismiss = self._listen_for_ack(timeout=5.0)

        if got_ack:
            log.info("Failure acknowledged — speaking details")
            self._state = STATE_SPEAKING
            self.driver.set_speaking()
            if self.tts:
                self.tts.synthesize(req_text)
        else:
            reason = "dismissed" if got_dismiss else "no response"
            log.info("Failure %s — sending to Slack", reason)
            try:
                from agent_modules.comms import send_slack_dm
                send_slack_dm(f"[Task Failure] {req_text}")
            except Exception as e:
                log.error("Slack fallback failed: %s", e)

        self._return_to_idle()
        if done_event:
            done_event.set()

    def _listen_for_ack(self, timeout: float = 5.0) -> tuple[bool, bool]:
        """Listen for acknowledgment phrases. Returns (got_ack, got_dismiss)."""
        ack_phrases = [
            "go ahead", "tell me", "what happened", "yes",
            "what is it", "what's up", "proceed", "okay", "ok",
        ]
        dismiss_phrases = ["wait", "later", "not now", "hold", "no", "busy"]

        start = time.time()
        while time.time() - start < timeout:
            try:
                audio = self.driver.read_chunk(self.chunk_size)
            except Exception:
                continue

            if self.asr:
                self.asr.send_audio(audio)
                transcript = self.asr.get_transcription()
                if transcript:
                    lower = transcript.lower()
                    if any(p in lower for p in ack_phrases):
                        return True, False
                    if any(p in lower for p in dismiss_phrases):
                        return False, True

        return False, False

    # --- Utilities ---

    def _save_recording(self, transcription):
        self._recording_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_text = "".join(
            c if c.isalnum() or c in " -_" else ""
            for c in (transcription or "unknown")[:30]
        ).strip().replace(" ", "_")
        filename = f"{self._recording_counter:04d}_{timestamp}_{safe_text}.wav"
        filepath = os.path.join(self.recordings_dir, filename)

        audio_data = np.concatenate(self._audio_buffer)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.driver.sample_rate)
            wf.writeframes(audio_data.tobytes())
        log.info("Saved recording: %s (%.1fs)", filename, len(audio_data) / self.driver.sample_rate)
        self._audio_buffer = []

    def _cleanup(self):
        """Clean up resources."""
        log.info("Pipeline shutting down")
        if self.asr:
            self.asr.close()
        if self.tts:
            self.tts.close()
        if self.llm and hasattr(self.llm, 'close'):
            self.llm.close()
        self.driver.close()
