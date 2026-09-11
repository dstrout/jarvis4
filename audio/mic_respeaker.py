"""ReSpeaker XVF3800 audio driver — AEC-based VAD + LED control.

Requires the ReSpeaker XVF3800 USB device and its Python control library
(xvf_host.py). This is an optional dependency — if not available, use
GenericMicDriver instead.
"""

import logging
import sys
import threading
from pathlib import Path

import numpy as np
import pyaudio

from audio.base import AudioDriver

log = logging.getLogger("jarvis4.audio.respeaker")

# LED Effects (from XVF3800 documentation)
EFFECT_OFF = 0
EFFECT_BREATHING = 1
EFFECT_RAINBOW = 2
EFFECT_SOLID = 3
EFFECT_DOA = 4

# Common Colors (RGB as 24-bit integer)
COLOR_ORANGE = 0xFF8800
COLOR_BLUE = 0x0000FF
COLOR_GREEN = 0x00FF00
COLOR_RED = 0xFF0000
COLOR_WHITE = 0xFFFFFF
COLOR_OFF = 0x000000


class ReSpeakerDriver(AudioDriver):
    """ReSpeaker XVF3800 with AEC-based VAD and LED feedback."""

    def __init__(self, device_index: int = None, sample_rate: int = 16000,
                 respeaker_lib_path: str = None):
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._pa = None
        self._stream = None
        self._dev = None  # ReSpeaker USB device handle
        self._lock = threading.Lock()
        self._restore_timer = None
        self._respeaker_lib_path = respeaker_lib_path

    def open(self):
        # Import ReSpeaker control library
        lib_path = self._respeaker_lib_path
        if lib_path is None:
            lib_path = str(Path(__file__).parent.parent / "reSpeaker_XVF3800_USB_4MIC_ARRAY" / "python_control")

        if lib_path not in sys.path:
            sys.path.insert(0, lib_path)

        try:
            from xvf_host import find as find_respeaker
            self._dev = find_respeaker()
            if not self._dev:
                raise RuntimeError("ReSpeaker device not found. Is it connected?")
            log.info("ReSpeaker XVF3800 found")
        except ImportError:
            raise RuntimeError(
                "ReSpeaker control library not found. "
                "Ensure xvf_host.py is available or use 'generic' audio driver."
            )

        # Initialize PyAudio and find ReSpeaker mic
        self._pa = pyaudio.PyAudio()
        if self._device_index is None:
            self._device_index = self._find_respeaker_input()
            if self._device_index is None:
                raise RuntimeError("ReSpeaker audio input device not found")

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=1280,
        )
        log.info("ReSpeaker mic opened (device %d, %dHz)", self._device_index, self._sample_rate)

        # Set initial LED state
        self.set_idle()

    def _find_respeaker_input(self) -> int:
        """Find the ReSpeaker audio input device index."""
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            name = info["name"].lower()
            if info["maxInputChannels"] > 0:
                if "respeaker" in name or "xvf3800" in name:
                    if "hw:" in name or info["maxInputChannels"] <= 2:
                        return i
        return None

    def close(self):
        if self._restore_timer is not None:
            self._restore_timer.cancel()
            self._restore_timer = None
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        if self._dev:
            self._dev.close()

    def read_chunk(self, chunk_size: int = 1280) -> np.ndarray:
        data = self._stream.read(chunk_size, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16)

    def get_speech_energy(self) -> float:
        """Read speech energy from the ReSpeaker's AEC processor.

        Returns the max across all 4 channels since the active mic
        depends on speaker direction (beamforming).
        Typical speech values are in the hundreds of thousands to millions.
        """
        with self._lock:
            try:
                spenergy = self._dev.read("AEC_SPENERGY_VALUES")
                if spenergy and len(spenergy) > 0:
                    return max(spenergy)
            except Exception:
                pass
        return 0.0

    @property
    def default_vad_threshold(self) -> float:
        return 2000000.0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    # --- LED control ---

    def _set_led(self, effect: int, color: int, brightness: int = 255, speed: int = 1):
        """Set LED effect, color, brightness, and speed."""
        try:
            with self._lock:
                self._dev.write("LED_EFFECT", [effect])
                self._dev.write("LED_COLOR", [color])
                self._dev.write("LED_SPEED", [speed])
                self._dev.write("LED_BRIGHTNESS", [brightness])
        except Exception as e:
            log.warning("LED USB error (non-fatal): %s", e)

    def _cancel_restore_timer(self):
        if self._restore_timer is not None:
            self._restore_timer.cancel()
            self._restore_timer = None

    def set_idle(self):
        """Orange breathing — idle state."""
        self._cancel_restore_timer()
        self._set_led(EFFECT_BREATHING, COLOR_ORANGE)

    def set_listening(self):
        """Solid blue — listening state."""
        self._cancel_restore_timer()
        self._set_led(EFFECT_SOLID, COLOR_BLUE)

    def set_thinking(self):
        """Rainbow cycling — thinking/processing state."""
        self._cancel_restore_timer()
        self._set_led(EFFECT_RAINBOW, COLOR_WHITE, speed=3)

    def set_speaking(self):
        """Solid green — speaking/TTS state."""
        self._cancel_restore_timer()
        self._set_led(EFFECT_SOLID, COLOR_GREEN)
