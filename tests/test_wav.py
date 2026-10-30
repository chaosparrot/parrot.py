import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from lib.wav import get_frame_count, read_wav, wav_duration, UnreadableWav


def write_wav(path, samples, rate=16000, channels=1, width=2):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(samples)


def raw_wav(path, fmt=1, channels=1, rate=16000, bits=16, data=b"", declared_size=None,
            before_data=b"", after_data=b""):
    """A wav written by hand, for headers the wave module won't write."""
    block = channels * bits // 8
    header = (b"RIFF" + struct.pack("<I", 36 + len(before_data) + len(data) + len(after_data))
              + b"WAVE"
              + b"fmt " + struct.pack("<IHHIIHH", 16, fmt, channels, rate, rate * block, block, bits)
              + before_data
              + b"data" + struct.pack("<I", len(data) if declared_size is None else declared_size))
    with open(path, "wb") as f:
        f.write(header + data + after_data)


def chunk(name, payload):
    return name + struct.pack("<I", len(payload)) + payload


def get_frame_count_of(path):
    with wave.open(path, "rb") as wf:
        return get_frame_count(wf, path)


class TestGetFrameCount(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_from_the_header(self):
        path = os.path.join(self.dir, "a.wav")
        write_wav(path, bytes(2 * 8000))
        self.assertEqual(get_frame_count_of(path), 8000)

    def test_empty(self):
        path = os.path.join(self.dir, "a.wav")
        write_wav(path, b"")
        self.assertEqual(get_frame_count_of(path), 0)

    def test_header_that_overcounts_is_capped_to_the_file(self):
        path = os.path.join(self.dir, "a.wav")
        raw_wav(path, data=bytes(2 * 8000), declared_size=4 * 8000)
        self.assertEqual(get_frame_count_of(path), 8000)

    def test_extra_chunks_keep_the_header_count(self):
        path = os.path.join(self.dir, "a.wav")
        info = chunk(b"LIST", b"INFO")
        for where in ("before_data", "after_data"):
            with self.subTest(where=where):
                raw_wav(path, data=bytes(2 * 8000), **{where: info})
                self.assertEqual(get_frame_count_of(path), 8000)

    def test_stereo(self):
        path = os.path.join(self.dir, "a.wav")
        write_wav(path, bytes(4 * 8000), channels=2)
        self.assertEqual(get_frame_count_of(path), 8000)


class TestReadWav(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def path(self, name):
        return os.path.join(self.dir, name)

    def assertUnreadable(self, path, reason):
        with self.assertRaises(UnreadableWav) as ctx:
            read_wav(path)
        self.assertEqual(ctx.exception.reason, reason)
        return ctx.exception

    def test_mono_16_bit(self):
        write_wav(self.path("a.wav"), np.array([0, 16384, -32768], dtype=np.int16).tobytes())
        samples, rate = read_wav(self.path("a.wav"))
        self.assertEqual(rate, 16000)
        self.assertEqual(samples.dtype, np.float32)
        np.testing.assert_array_equal(samples, [0.0, 0.5, -1.0])

    def test_stereo_is_averaged_to_mono(self):
        frames = np.array([16384, 0, -16384, -16384], dtype=np.int16).tobytes()
        write_wav(self.path("a.wav"), frames, channels=2)
        samples, _ = read_wav(self.path("a.wav"))
        np.testing.assert_array_equal(samples, [0.25, -0.5])

    def test_overcounting_header_reads_what_is_there(self):
        data = np.array([100, 200, 300], dtype=np.int16).tobytes()
        raw_wav(self.path("a.wav"), data=data, declared_size=2 * len(data))
        samples, _ = read_wav(self.path("a.wav"))
        self.assertEqual(len(samples), 3)

    def test_cut_off_mid_sample_keeps_the_whole_samples(self):
        raw_wav(self.path("a.wav"), data=np.array([100, 200], dtype=np.int16).tobytes() + b"\x01",
                declared_size=10, before_data=chunk(b"LIST", b"INFO"))
        samples, _ = read_wav(self.path("a.wav"))
        self.assertEqual(len(samples), 2)

    def test_missing_file(self):
        self.assertUnreadable(self.path("gone.wav"), UnreadableWav.CANT_OPEN)

    def test_not_a_wav(self):
        with open(self.path("a.wav"), "w") as f:
            f.write("not audio")
        self.assertUnreadable(self.path("a.wav"), UnreadableWav.DAMAGED)

    def test_header_cut_short(self):
        with open(self.path("a.wav"), "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        self.assertUnreadable(self.path("a.wav"), UnreadableWav.DAMAGED)

    def test_other_bit_depths(self):
        for width in (1, 3, 4):
            with self.subTest(bits=width * 8):
                write_wav(self.path("a.wav"), bytes(width * 10), width=width)
                exc = self.assertUnreadable(self.path("a.wav"), UnreadableWav.NOT_16_BIT)
                self.assertEqual(exc.bits, width * 8)

    def test_float_format(self):
        raw_wav(self.path("a.wav"), fmt=3, bits=32, data=bytes(40))
        self.assertUnreadable(self.path("a.wav"), UnreadableWav.NOT_PCM)

    def test_extensible_float_format(self):
        # ffmpeg's 32-bit float export
        float_guid = struct.pack("<H", 3) + bytes.fromhex("000000001000800000aa00389b71")
        fmt = (struct.pack("<HHIIHH", 0xFFFE, 1, 16000, 64000, 4, 32)
               + struct.pack("<HHI", 22, 32, 4) + float_guid)
        with open(self.path("a.wav"), "wb") as f:
            body = b"WAVE" + chunk(b"fmt ", fmt) + chunk(b"data", bytes(40))
            f.write(b"RIFF" + struct.pack("<I", len(body)) + body)
        self.assertUnreadable(self.path("a.wav"), UnreadableWav.NOT_PCM)

    def test_empty(self):
        write_wav(self.path("a.wav"), b"")
        self.assertUnreadable(self.path("a.wav"), UnreadableWav.EMPTY)

    def test_sample_rate_zero(self):
        raw_wav(self.path("a.wav"), rate=0, data=bytes(20))
        self.assertUnreadable(self.path("a.wav"), UnreadableWav.DAMAGED)


class TestWavDuration(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_from_the_header(self):
        path = os.path.join(self.dir, "a.wav")
        write_wav(path, bytes(2 * 8000))
        self.assertEqual(wav_duration(path), 0.5)

    def test_overcounting_header_is_capped_to_the_file(self):
        path = os.path.join(self.dir, "a.wav")
        raw_wav(path, data=bytes(2 * 8000), declared_size=4 * 8000)
        self.assertEqual(wav_duration(path), 0.5)

    def test_unreadable_is_zero(self):
        self.assertEqual(wav_duration(os.path.join(self.dir, "gone.wav")), 0.0)


class TestDetectWavFrames(unittest.TestCase):
    """Old headers used to stall segmenting progress at half."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def last_progress(self, path):
        from config.config import MS_PER_FRAME, CURRENT_DETECTION_STRATEGY
        from lib.stream_processing import detect_wav_frames
        from lib.typing import DetectionState, DetectionLabel
        label = DetectionLabel("pop", 0, 0, "", 0, 0, 0, 0, 0)
        # Like process_wav_file, without logging
        state = DetectionState(CURRENT_DETECTION_STRATEGY, "recording", MS_PER_FRAME,
                               0, False, 0, 0, 0, 0, [label], None, [])
        reported = []
        detect_wav_frames(path, state, lambda progress, _state: reported.append(progress))
        return reported[-1]

    def test_overcounting_header_reaches_the_end(self):
        rng = np.random.default_rng(0)
        data = (rng.normal(0, 3000, 16000)).astype(np.int16).tobytes()
        honest = os.path.join(self.dir, "honest.wav")
        old = os.path.join(self.dir, "old.wav")
        raw_wav(honest, data=data)
        raw_wav(old, data=data, declared_size=2 * len(data))
        self.assertAlmostEqual(self.last_progress(old), self.last_progress(honest))


FIXTURES = os.path.join(ROOT, "tests", "fixtures")


def old_header_copy(source, folder):
    """The same audio, with the data size doubled as before the header fix."""
    path = os.path.join(folder, "old.wav")
    shutil.copy(source, path)
    with open(path, "r+b") as f:
        f.seek(40)
        size = struct.unpack("<I", f.read(4))[0]
        f.seek(40)
        f.write(struct.pack("<I", 2 * size))
    return path


class TestLoadWavDataFromSrt(unittest.TestCase):
    """Old headers never changed training data. Keeps it that way."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_overcounting_header_gives_the_same_data(self):
        from lib.wav import load_wav_data_from_srt
        source = os.path.join(FIXTURES, "recordings", "pop", "source", "pop.wav")
        srt = os.path.join(FIXTURES, "segmentation", "pop.srt")
        old = old_header_copy(source, self.dir)
        for with_offset in (False, True):
            for background in (False, True):
                with self.subTest(with_offset=with_offset, background=background):
                    honest = load_wav_data_from_srt(srt, source, with_offset=with_offset,
                                                    background=background)
                    self.assertTrue(honest)
                    np.testing.assert_array_equal(
                        load_wav_data_from_srt(srt, old, with_offset=with_offset,
                                               background=background), honest)


class TestImportCost(unittest.TestCase):

    def test_importing_leaves_the_training_stack_unloaded(self):
        # The GUI imports both at launch
        for module in ("lib.wav", "lib.recordings"):
            with self.subTest(module=module):
                code = (f"import sys, {module}; print(','.join(m for m in "
                        "('lib.machinelearning', 'sklearn', 'audiomentations', 'scipy', 'matplotlib')"
                        " if m in sys.modules))")
                result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                        capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
