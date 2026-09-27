import sounddevice as sd

from config.config import FORMAT

def open_input_stream(
        device_index, *, rate, channels, record_seconds,
        sliding_window_amount, callback=None):
    """Open a microphone stream, for recording, listening or probing a device.

    The stream comes back stopped. Every caller starts its own, once whatever
    consumes the frames is ready for them.
    """
    return sd.InputStream(
        dtype=FORMAT,
        channels=channels,
        samplerate=rate,
        device=device_index,
        blocksize=round(rate * record_seconds / sliding_window_amount),
        callback=callback)
