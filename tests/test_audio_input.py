"""Tests
- the record and listen callbacks, and what they return
- validate_microphone_index and validate_microphone_input
- what open_input_stream asks the library for

Not tested, needs a real device
- that the library really calls back with the shape assumed here
- the listen loop
- stream_controls

Everything the audio library owns sits in the helpers above the test classes.
Swapping the library rewrites those, and some of them stop being injectable and
become patch targets. No test body names the library, so none should change.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from queue import Queue

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import sounddevice as sd
from unittest import mock

import lib.listen
# The shipped defaults, not config.config: that one execs the user's own
# config.py over the top, and setting your own RATE should not fail a test about
# what parrot ships recording at. INPUT_DEVICE_INDEX is merged on purpose, since
# it is the value the probes actually read.
from config.config import INPUT_DEVICE_INDEX
from lib.audio_input import open_input_stream
from lib.default_config import (CHANNELS, RATE, RECORD_SECONDS,
                                SLIDING_WINDOW_AMOUNT)
from lib.listen import nonblocking_record, validate_microphone_input
from lib.record_data import multithreaded_record, validate_microphone_index

FRAME_SAMPLES = round(RATE * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT)
# Counts up, so a resliced or reordered frame cannot still compare equal
PCM = bytes(bytearray(index % 256 for index in range(FRAME_SAMPLES * 2)))

MIC_INDEX = 7
MISSING = object()


def library_frame(pcm):
    """16 bit PCM as the library hands it to a callback."""
    return np.frombuffer(pcm, dtype=np.int16).reshape(-1, 1)


def keep_going(frame):
    """What a callback returns to leave the stream running."""
    return None


def device(name, input_channels):
    """A device row as the library reports it."""
    return {"name": name, "max_input_channels": input_channels, "hostapi": 0}


def stream_settings(kwargs):
    """An open call in plain values, whatever the library names them."""
    return {
        "rate": kwargs["samplerate"],
        "channels": kwargs["channels"],
        "sample_width": np.dtype(kwargs["dtype"]).itemsize,
        "frame_samples": kwargs["blocksize"],
        "device": kwargs["device"],
    }


# Raised for a device index that is not there. PortAudioError is not an
# OSError, so `except IOError` alone would miss it.
NO_SUCH_DEVICE = sd.PortAudioError

MIC = device("Fake mic", 1)
SPEAKERS = device("Fake speakers", 0)
HOST_API = {"name": "Fake host API"}


class DeviceTable:
    """The library's device list, as far as the probes read it."""

    def __init__(self, devices):
        self.devices = devices

    def query_devices(self, index):
        if index not in self.devices:
            raise NO_SUCH_DEVICE("Invalid device index")
        return self.devices[index]

    def query_hostapis(self, index):
        return HOST_API


def listing(devices):
    """The probes read the library's module level table, so it is patched."""
    table = DeviceTable(devices)
    return mock.patch.multiple(sd, query_devices=table.query_devices,
                               query_hostapis=table.query_hostapis)


class OpenedStream:
    def __init__(self):
        self.running = False

    def start(self):
        self.running = True


class RecordingLibrary:
    """Remembers what a stream was opened for instead of opening one."""

    def __init__(self):
        self.opened = {}

    def __call__(self, **kwargs):
        self.opened.update(kwargs)
        return OpenedStream()


def quietly(function, *arguments):
    """The probes report to the terminal. Not what is under test."""
    with redirect_stdout(io.StringIO()):
        return function(*arguments)


def probe_index(devices, index):
    with listing(devices):
        return quietly(validate_microphone_index, index)


def probe_configured_input(devices):
    with listing(devices):
        return quietly(validate_microphone_input)


def open_a_stream(rate=RATE, channels=CHANNELS, seconds=RECORD_SECONDS,
                  sliding_window=SLIDING_WINDOW_AMOUNT):
    """What the library was asked for, and the stream it gave back."""
    library = RecordingLibrary()
    with mock.patch.object(sd, "InputStream", library):
        stream = open_input_stream(MIC_INDEX, rate=rate, channels=channels,
                                   record_seconds=seconds, sliding_window_amount=sliding_window)
    return stream_settings(library.opened), stream


def listening_with(queue):
    """listen.py creates listening_state only once the listen loop starts."""
    previous = getattr(lib.listen, "listening_state", MISSING)
    lib.listen.listening_state = {"audioQueue": queue}

    def restore():
        if previous is MISSING:
            del lib.listen.listening_state
        else:
            lib.listen.listening_state = previous
    return restore


class RecordCallbackTest(unittest.TestCase):

    def test_a_frame_reaches_the_recording_queue_intact(self):
        queue = Queue()
        multithreaded_record(library_frame(PCM), FRAME_SAMPLES, None, None, queue)

        self.assertEqual(queue.get_nowait(), PCM)
        self.assertTrue(queue.empty())

    def test_frames_queue_up_in_the_order_they_arrived(self):
        queue = Queue()
        first, second = PCM, PCM[::-1]
        multithreaded_record(library_frame(first), FRAME_SAMPLES, None, None, queue)
        multithreaded_record(library_frame(second), FRAME_SAMPLES, None, None, queue)

        self.assertEqual([queue.get_nowait(), queue.get_nowait()], [first, second])

    def test_the_callbacks_leave_the_stream_running(self):
        frame = library_frame(PCM)
        self.addCleanup(listening_with(Queue()))

        self.assertEqual(
            multithreaded_record(frame, FRAME_SAMPLES, None, None, Queue()),
            keep_going(frame))
        self.assertEqual(
            nonblocking_record(frame, FRAME_SAMPLES, None, None),
            keep_going(frame))

    def test_a_frame_reaches_the_listening_queue_intact(self):
        queue = Queue()
        self.addCleanup(listening_with(queue))
        nonblocking_record(library_frame(PCM), FRAME_SAMPLES, None, None)

        self.assertEqual(queue.get_nowait(), PCM)
        self.assertTrue(queue.empty())


class MicrophoneProbeTest(unittest.TestCase):

    def test_a_microphone_is_accepted(self):
        self.assertIs(probe_index({MIC_INDEX: MIC}, MIC_INDEX), True)

    def test_a_device_with_no_input_channels_is_rejected(self):
        self.assertIs(probe_index({MIC_INDEX: SPEAKERS}, MIC_INDEX), False)

    def test_a_missing_device_is_rejected(self):
        self.assertIs(probe_index({}, MIC_INDEX), False)

    def test_the_configured_input_is_accepted(self):
        self.assertIs(probe_configured_input({INPUT_DEVICE_INDEX: MIC}), True)

    def test_a_configured_input_with_no_channels_is_rejected(self):
        self.assertIs(probe_configured_input({INPUT_DEVICE_INDEX: SPEAKERS}), False)

    def test_a_missing_configured_input_is_rejected(self):
        self.assertIs(probe_configured_input({}), False)


class StreamSettingsTest(unittest.TestCase):
    """Every recording and every trained model assumes 16 bit mono at RATE. A
    stream opened at a library's own defaults writes a playable wav full of the
    wrong numbers, and nothing downstream notices."""

    def test_a_stream_is_opened_for_the_audio_parrot_expects(self):
        # Written out rather than read back from config, so that changing the
        # audio parrot records has to be a decision and not a side effect
        settings, _ = open_a_stream()

        self.assertEqual(settings, {
            "rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "frame_samples": 240,
            "device": MIC_INDEX,
        })

    def test_a_models_own_settings_win(self):
        settings, _ = open_a_stream(rate=48000, seconds=0.06, sliding_window=3)

        self.assertEqual(settings["rate"], 48000)
        self.assertEqual(settings["frame_samples"], 960)

    def test_the_helper_does_not_start_the_stream(self):
        # Every caller starts its own. Starting here would move when frames
        # begin for the record and listen paths, which wait for their consumer
        # threads first.
        _, stream = open_a_stream()

        self.assertFalse(stream.running)


if __name__ == "__main__":
    unittest.main()
