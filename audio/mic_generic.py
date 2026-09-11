"""Generic audio driver — works with any PyAudio-compatible microphone."""

import logging
import numpy as np
import pyaudio

from audio.base import AudioDriver

log = logging.getLogger("jarvis4.audio.generic")


class GenericMicDriver(AudioDriver):
    """PyAudio microphone with RMS-based VAD. No LEDs."""

    def __init__(self, device_index: int = None, sample_rate: int = 16000):
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._pa = None
        self._stream = None
        self._last_energy = 0.0

    def open(self):
        self._pa = pyaudio.PyAudio()
        if self._device_index is None:
            self._device_index = self._pa.get_default_input_device_info()["index"]
            log.info("Using default input device: %d", self._device_index)
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=1280,
        )
        log.info("Generic mic opened (device %d, %dHz)", self._device_index, self._sample_rate)

    def close(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    def read_chunk(self, chunk_size: int = 1280) -> np.ndarray:
        data = self._stream.read(chunk_size, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)
        # Compute RMS energy for VAD
        self._last_energy = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        return audio

    def get_speech_energy(self) -> float:
        return self._last_energy

    @property
    def default_vad_threshold(self) -> float:
        return 500.0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
