"""The recorder patches the wav header itself while streaming, since the wave
module only writes one on close. Both write sites, append and truncate."""
import math
import os
import shutil
import struct
import sys
import tempfile
import types
import unittest
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import config.config
from lib.stream_recorder import StreamRecorder
from lib.typing import DetectionState, DetectionLabel

# One callback frame: RATE * RECORD_SECONDS / SLIDING_WINDOW_AMOUNT samples, 2 bytes each
FRAME_BYTES = round(config.config.RATE * config.config.RECORD_SECONDS
                    / config.config.SLIDING_WINDOW_AMOUNT) * 2
MS_PER_FRAME = math.floor(config.config.RECORD_SECONDS
                          / config.config.SLIDING_WINDOW_AMOUNT * 1000)
FRAMES_WRITTEN = 15
WAV_HEADER_SIZE = 44


class StoppableStream:
    def start(self):
        pass

    def stop(self):
        pass


def declared_frames(path):
    with wave.open(path, "rb") as handle:
        return handle.getnframes()


def declared_riff_size(path):
    # The wave module never reads this one back against the file length
    with open(path, "rb") as handle:
        handle.seek(4)
        return struct.unpack("<I", handle.read(4))[0]


class WavHeaderTest(unittest.TestCase):
    def setUp(self):
        workdir = tempfile.mkdtemp(prefix="parrot_header_")
        self.addCleanup(shutil.rmtree, workdir, True)
        self.wav_file = os.path.join(workdir, "total.wav")

        label = DetectionLabel("pop", 0, 0, "", 0, 0, 0, 0, 0)
        state = DetectionState("strategy", "recording", MS_PER_FRAME, 0, False,
                               0, 0, 0, 0, [label])
        self.recorder = StreamRecorder(
            StoppableStream(), self.wav_file,
            os.path.join(workdir, "total.v%d.srt" % config.config.CURRENT_VERSION),
            state)
        self.recorder.total_audio_frames = [b"\0" * FRAME_BYTES] * FRAMES_WRITTEN
        self.recorder.index = FRAMES_WRITTEN
        self.recorder.length_per_frame = FRAME_BYTES
        self.recorder.detection_frames = [types.SimpleNamespace(label="silence")
                                          for _ in range(FRAMES_WRITTEN)]
        self.recorder.persist_total_wav_file()

        self.written_bytes = FRAME_BYTES * FRAMES_WRITTEN

    def test_header_counts_the_samples_the_file_holds(self):
        self.assertEqual(declared_frames(self.wav_file), self.written_bytes // 2)

    def test_file_is_a_canonical_header_and_its_data(self):
        self.assertEqual(os.path.getsize(self.wav_file),
                         WAV_HEADER_SIZE + self.written_bytes)

    def test_riff_size_counts_everything_after_it(self):
        self.assertEqual(declared_riff_size(self.wav_file),
                         os.path.getsize(self.wav_file) - 8)

    def test_header_still_matches_after_clearing_the_last_seconds(self):
        removed = 5
        self.recorder.clear(removed * MS_PER_FRAME / 1000)

        kept_bytes = FRAME_BYTES * (FRAMES_WRITTEN - removed)
        self.assertEqual(os.path.getsize(self.wav_file),
                         WAV_HEADER_SIZE + kept_bytes)
        self.assertEqual(declared_frames(self.wav_file), kept_bytes // 2)
        self.assertEqual(declared_riff_size(self.wav_file),
                         os.path.getsize(self.wav_file) - 8)


if __name__ == "__main__":
    unittest.main()
