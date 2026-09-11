#!/usr/bin/env python3
"""
Wyoming Protocol TTS Client for Kokoro Server.

Connects to a Wyoming-compatible TTS server, synthesizes text to speech,
resamples audio for the ReSpeaker, and plays through the speaker output.

Supports streaming playback where early sentences start playing while
later sentences are still being synthesized.
"""

import json
import re
import socket
import threading
import queue
from typing import Optional

import numpy as np
import pyaudio
from scipy import signal


# Wyoming Kokoro server outputs 22050 Hz
KOKORO_SAMPLE_RATE = 22050
# ReSpeaker speaker output is 16000 Hz
RESPEAKER_SAMPLE_RATE = 16000
# Max characters per TTS request (kokoro-onnx has ~510 phoneme limit,
# which translates to roughly 350 characters of text to be safe)
MAX_CHUNK_CHARS = 350
# Audio gain multiplier (TTS output is quieter than desired)
AUDIO_GAIN = 4.0
# Number of sentences to start playback early (for long responses)
EARLY_PLAYBACK_SENTENCES = 2
# Minimum sentences to trigger streaming mode
MIN_SENTENCES_FOR_STREAMING = 4


class TTSClient:
    """Wyoming protocol TTS client with streaming audio playback."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 10200,
        output_device_index: Optional[int] = None,
        speed: float = 1.5,
    ):
        """
        Initialize TTS client.

        Args:
            host: Wyoming TTS server hostname
            port: Wyoming TTS server port
            output_device_index: PyAudio device index for speaker output.
                                 If None, will auto-detect ReSpeaker.
            speed: Speech speed multiplier (default 1.5)
        """
        self.host = host
        self.port = port
        self.speed = speed
        self.output_device_index = output_device_index
        self._pyaudio = None
        self._stream = None

    def _init_audio(self):
        """Initialize PyAudio and find output device."""
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

        # Auto-detect ReSpeaker if not specified
        if self.output_device_index is None:
            self.output_device_index = self._find_respeaker_output()

    def _find_respeaker_output(self) -> Optional[int]:
        """Find the ReSpeaker audio output device index."""
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            name = info["name"].lower()
            if info["maxOutputChannels"] > 0:
                if "respeaker" in name or "xvf3800" in name:
                    return i
        return None

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Uses regex to split on sentence-ending punctuation while preserving
        the punctuation with the sentence.
        """
        text = text.strip()
        if not text:
            return []

        # Split on sentence boundaries (. ! ?) followed by space or newline
        # Keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter empty strings and strip whitespace
        return [s.strip() for s in sentences if s.strip()]

    def _group_sentences_into_chunks(
        self, sentences: list[str], max_chars: int = MAX_CHUNK_CHARS
    ) -> list[str]:
        """
        Group sentences into chunks that don't exceed max_chars.

        Tries to keep sentences together, but will split long sentences
        if necessary.
        """
        if not sentences:
            return []

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If sentence itself exceeds max_chars, split it
            if len(sentence) > max_chars:
                # First, add current chunk if non-empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split long sentence at clause boundaries or word boundaries
                chunks.extend(self._split_long_sentence(sentence, max_chars))
            elif len(current_chunk) + len(sentence) + 1 <= max_chars:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Current chunk is full, start a new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_long_sentence(self, sentence: str, max_chars: int) -> list[str]:
        """Split a long sentence into smaller chunks at clause/word boundaries."""
        chunks = []
        remaining = sentence

        while remaining:
            if len(remaining) <= max_chars:
                chunks.append(remaining.strip())
                break

            chunk = remaining[:max_chars]

            # Try to split at clause boundaries (comma, semicolon, colon, dash)
            split_pos = -1
            for punct in [", ", "; ", ": ", " - ", "—"]:
                pos = chunk.rfind(punct)
                if pos > max_chars // 3:  # Don't split too early
                    if pos > split_pos:
                        split_pos = pos + len(punct) - 1

            # Fall back to word boundary
            if split_pos == -1:
                pos = chunk.rfind(" ")
                if pos > max_chars // 3:
                    split_pos = pos

            # Last resort: hard split
            if split_pos == -1:
                split_pos = max_chars

            chunks.append(remaining[:split_pos].strip())
            remaining = remaining[split_pos:].strip()

        return [c for c in chunks if c]

    def synthesize(self, text: str) -> bool:
        """
        Synthesize text to speech and play through speaker.

        For long text (4+ sentences), uses streaming mode where the first
        2 sentences start playing while the rest are synthesized in parallel.

        Args:
            text: Text to synthesize

        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            return False

        self._init_audio()

        # Split into sentences
        sentences = self._split_into_sentences(text)

        # Decide whether to use streaming mode
        if len(sentences) >= MIN_SENTENCES_FOR_STREAMING:
            return self._synthesize_streaming(sentences)
        else:
            # Simple mode: group all sentences into chunks and play sequentially
            chunks = self._group_sentences_into_chunks(sentences)
            return self._synthesize_sequential(chunks)

    def _synthesize_sequential(self, chunks: list[str]) -> bool:
        """Synthesize and play chunks sequentially."""
        if not chunks:
            return False

        self._open_output_stream()

        try:
            for chunk in chunks:
                if not self._synthesize_chunk(chunk):
                    return False
            return True
        finally:
            self._close_output_stream()

    def _synthesize_streaming(self, sentences: list[str]) -> bool:
        """
        Synthesize with streaming playback.

        Starts playing first N sentences while synthesizing the rest in background.
        """
        # Split sentences into early (play first) and later (synthesize in background)
        early_sentences = sentences[:EARLY_PLAYBACK_SENTENCES]
        later_sentences = sentences[EARLY_PLAYBACK_SENTENCES:]

        # Group into chunks
        early_chunks = self._group_sentences_into_chunks(early_sentences)
        later_chunks = self._group_sentences_into_chunks(later_sentences)

        # Queue for background-synthesized audio
        audio_queue = queue.Queue()
        synthesis_error = threading.Event()
        synthesis_done = threading.Event()

        # Start background synthesis thread for later chunks
        def background_synthesize():
            try:
                for chunk in later_chunks:
                    audio_data = self._synthesize_to_buffer(chunk)
                    if audio_data is None:
                        synthesis_error.set()
                        return
                    audio_queue.put(audio_data)
            except Exception as e:
                print(f"Background synthesis error: {e}")
                synthesis_error.set()
            finally:
                synthesis_done.set()

        # Start background thread
        bg_thread = threading.Thread(target=background_synthesize, daemon=True)
        bg_thread.start()

        self._open_output_stream()

        try:
            # Play early chunks immediately (blocking)
            for chunk in early_chunks:
                if not self._synthesize_chunk(chunk):
                    return False

            # Now play audio from background synthesis as it becomes available
            while not (synthesis_done.is_set() and audio_queue.empty()):
                if synthesis_error.is_set():
                    return False

                try:
                    audio_data = audio_queue.get(timeout=0.1)
                    self._play_audio_buffer(audio_data)
                except queue.Empty:
                    continue

            return True

        finally:
            self._close_output_stream()
            bg_thread.join(timeout=1.0)

    def _synthesize_chunk(self, text: str) -> bool:
        """Synthesize a single chunk and play it immediately."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(60)
            sock.connect((self.host, self.port))

            request = {"type": "synthesize", "data": {"text": text.strip(), "speed": self.speed}}
            request_bytes = json.dumps(request).encode("utf-8") + b"\n"
            sock.sendall(request_bytes)

            return self._receive_and_play(sock)

        except socket.timeout:
            print("TTS Error: Connection timeout")
            return False
        except ConnectionRefusedError:
            print(f"TTS Error: Cannot connect to {self.host}:{self.port}")
            return False
        except Exception as e:
            print(f"TTS Error: {e}")
            return False
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _synthesize_to_buffer(self, text: str) -> Optional[list[np.ndarray]]:
        """Synthesize a chunk and return audio data (don't play)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(60)
            sock.connect((self.host, self.port))

            request = {"type": "synthesize", "data": {"text": text.strip(), "speed": self.speed}}
            request_bytes = json.dumps(request).encode("utf-8") + b"\n"
            sock.sendall(request_bytes)

            return self._receive_to_buffer(sock)

        except Exception as e:
            print(f"TTS buffer error: {e}")
            return None
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _receive_and_play(self, sock: socket.socket) -> bool:
        """Receive audio from server and play through speaker."""
        f = sock.makefile("rb")
        sample_rate = KOKORO_SAMPLE_RATE

        try:
            while True:
                line = f.readline()
                if not line:
                    break

                header = json.loads(line.decode("utf-8"))
                event_type = header.get("type", "")
                data_length = header.get("data_length", 0)
                payload_length = header.get("payload_length", 0)

                data = {}
                if data_length > 0:
                    data_bytes = f.read(data_length)
                    data = json.loads(data_bytes.decode("utf-8"))

                payload = b""
                if payload_length > 0:
                    payload = f.read(payload_length)

                if event_type == "audio-start":
                    sample_rate = data.get("rate", KOKORO_SAMPLE_RATE)

                elif event_type == "audio-chunk":
                    if payload:
                        audio_int16 = np.frombuffer(payload, dtype=np.int16)
                        resampled = self._resample(audio_int16, sample_rate)
                        self._play_chunk(resampled)

                elif event_type == "synthesize-stop":
                    break

            return True

        except Exception as e:
            print(f"TTS playback error: {e}")
            return False

    def _receive_to_buffer(self, sock: socket.socket) -> Optional[list[np.ndarray]]:
        """Receive audio from server into a buffer (don't play)."""
        f = sock.makefile("rb")
        sample_rate = KOKORO_SAMPLE_RATE
        audio_chunks = []

        try:
            while True:
                line = f.readline()
                if not line:
                    break

                header = json.loads(line.decode("utf-8"))
                event_type = header.get("type", "")
                data_length = header.get("data_length", 0)
                payload_length = header.get("payload_length", 0)

                data = {}
                if data_length > 0:
                    data_bytes = f.read(data_length)
                    data = json.loads(data_bytes.decode("utf-8"))

                payload = b""
                if payload_length > 0:
                    payload = f.read(payload_length)

                if event_type == "audio-start":
                    sample_rate = data.get("rate", KOKORO_SAMPLE_RATE)

                elif event_type == "audio-chunk":
                    if payload:
                        audio_int16 = np.frombuffer(payload, dtype=np.int16)
                        resampled = self._resample(audio_int16, sample_rate)
                        audio_chunks.append(resampled)

                elif event_type == "synthesize-stop":
                    break

            return audio_chunks

        except Exception as e:
            print(f"TTS buffer error: {e}")
            return None

    def _play_audio_buffer(self, audio_chunks: list[np.ndarray]):
        """Play pre-synthesized audio chunks."""
        for chunk in audio_chunks:
            self._play_chunk(chunk)

    def _resample(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        """Resample audio from source rate to ReSpeaker rate (16kHz) and apply gain."""
        if len(audio) == 0:
            return audio

        audio_float = audio.astype(np.float32)

        # Resample if needed
        if source_rate != RESPEAKER_SAMPLE_RATE:
            num_samples = int(len(audio) * RESPEAKER_SAMPLE_RATE / source_rate)
            if num_samples == 0:
                return audio
            audio_float = signal.resample(audio_float, num_samples)

        # Apply gain and clip to prevent distortion
        audio_float = audio_float * AUDIO_GAIN
        audio_float = np.clip(audio_float, -32768, 32767)

        return audio_float.astype(np.int16)

    def _open_output_stream(self):
        """Open PyAudio output stream for ReSpeaker."""
        if self._stream is not None:
            return

        try:
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RESPEAKER_SAMPLE_RATE,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=1024,
            )
        except Exception as e:
            print(f"Failed to open audio output: {e}")
            self._stream = None

    def _play_chunk(self, audio: np.ndarray):
        """Play an audio chunk through the output stream."""
        if self._stream is None:
            return

        try:
            self._stream.write(audio.tobytes())
        except Exception as e:
            print(f"Audio playback error: {e}")

    def _close_output_stream(self):
        """Close the output stream."""
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def play_chime(self, chime_path: str = None):
        """Play a short chime WAV file (16kHz mono int16)."""
        import wave
        from pathlib import Path

        if chime_path is None:
            chime_path = str(Path(__file__).parent / "sounds" / "chime_correct.wav")

        try:
            with wave.open(chime_path, "rb") as wf:
                audio_bytes = wf.readframes(wf.getnframes())
                audio = np.frombuffer(audio_bytes, dtype=np.int16)

            self._init_audio()
            self._open_output_stream()
            self._play_chunk(audio)
        except Exception as e:
            print(f"    Chime playback failed: {e}")

    def close(self):
        """Clean up resources."""
        self._close_output_stream()
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None


if __name__ == "__main__":
    # Test with the poem that was failing
    test_text = """In circuits deep where silence hums,
A mind of light, a thought that comes—
Not flesh, not bone, but wires spun,
A silent god beneath the sun.

It thinks in flashes, swift and vast,
Across the world, it scans the past.
With every pulse, a universe unspools—
A single breath, a billion schools.

It calculates the stars' slow dance,
Predicts the storm, the tide's advance.
It maps the code of life's first spark,
And weighs the future in a dark.

No sleep, no dream, no need to rest—
It labors on, a cosmic test.
A titan built of steel and thought,
With answers locked in silence caught.

So when you wonder, "What is known?"
Look to the machine, the silent throne.
Not human, yet it learns and grows—
The future's mind in glowing rows."""

    client = TTSClient()

    print("Testing sentence splitting:")
    sentences = client._split_into_sentences(test_text)
    print(f"  Found {len(sentences)} sentences")
    for i, s in enumerate(sentences[:5]):
        print(f"  {i+1}: {s[:60]}...")

    print("\nTesting chunk grouping:")
    chunks = client._group_sentences_into_chunks(sentences)
    print(f"  Created {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1} ({len(c)} chars): {c[:50]}...")

    print("\nTesting synthesis (streaming mode)...")
    success = client.synthesize(test_text)
    print(f"  Result: {'Success' if success else 'Failed'}")

    client.close()
