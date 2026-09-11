"""Abstract audio driver interface."""

from abc import ABC, abstractmethod
import numpy as np


class AudioDriver(ABC):
    """Interface for audio input with VAD and optional LED feedback."""

    @abstractmethod
    def open(self) -> None:
        """Initialize audio device and start streaming."""

    @abstractmethod
    def close(self) -> None:
        """Release audio device."""

    @abstractmethod
    def read_chunk(self, chunk_size: int) -> np.ndarray:
        """Read a chunk of audio as int16 numpy array."""

    @abstractmethod
    def get_speech_energy(self) -> float:
        """Return current speech energy for VAD. Scale is driver-specific."""

    @property
    @abstractmethod
    def default_vad_threshold(self) -> float:
        """Default VAD threshold for this driver."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""

    # LED feedback — optional, default is no-op
    def set_idle(self) -> None: pass
    def set_listening(self) -> None: pass
    def set_thinking(self) -> None: pass
    def set_speaking(self) -> None: pass
