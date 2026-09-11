"""Audio driver factory — selects driver based on config."""

from audio.base import AudioDriver


def create_driver(driver_name: str, **kwargs) -> AudioDriver:
    """Create an audio driver by name.

    Args:
        driver_name: "respeaker" or "generic" (default)
        **kwargs: Passed to driver constructor (device_index, sample_rate, etc.)
    """
    if driver_name == "respeaker":
        from audio.mic_respeaker import ReSpeakerDriver
        return ReSpeakerDriver(**kwargs)
    else:
        from audio.mic_generic import GenericMicDriver
        return GenericMicDriver(**kwargs)
