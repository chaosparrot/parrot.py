import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from lib import recordings
from lib.srt import current_v_ending, manual_ending


class TestRecordings(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        patch = mock.patch.object(recordings, "RECORDINGS_FOLDER", self.dir)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def touch(self, *parts):
        path = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        return path

    def test_labels_are_folders_in_order(self):
        self.touch("pop", "source", "a.wav")
        os.makedirs(os.path.join(self.dir, "click"))
        os.makedirs(os.path.join(self.dir, ".hidden"))
        self.touch("notes.txt")
        self.assertEqual(recordings.get_labels(), ["click", "pop"])

    def test_no_recordings_folder(self):
        with mock.patch.object(recordings, "RECORDINGS_FOLDER", os.path.join(self.dir, "gone")):
            self.assertEqual(recordings.get_labels(), [])

    def test_each_wav_with_its_srt(self):
        wav = self.touch("pop", "source", "mici_1__100.wav")
        unsegmented = self.touch("pop", "source", "mici_1__200.wav")
        self.touch("pop", "source", "notes.txt")
        srt = self.touch("pop", "segments", "mici_1__100" + current_v_ending)
        self.assertEqual(recordings.get_recordings("pop"), [
            {"filename": "mici_1__100.wav", "wav_path": wav, "srt_path": srt},
            {"filename": "mici_1__200.wav", "wav_path": unsegmented, "srt_path": None},
        ])

    def test_manual_srt_wins(self):
        self.touch("pop", "source", "a.wav")
        self.touch("pop", "segments", "a" + current_v_ending)
        manual = self.touch("pop", "segments", "a" + manual_ending)
        self.assertEqual(recordings.get_recordings("pop")[0]["srt_path"], manual)

    def test_no_source_folder(self):
        os.makedirs(os.path.join(self.dir, "pop"))
        self.assertEqual(recordings.get_recordings("pop"), [])

    def test_detected_ms_skips_an_srt_it_cant_parse(self):
        good = self.touch("pop", "segments", "a" + current_v_ending)
        with open(good, "w") as f:
            f.write("1\n00:00:01,000 --> 00:00:01,600\npop\n\n")
        bad = self.touch("pop", "segments", "b" + current_v_ending)
        with open(bad, "w") as f:
            f.write("1\nnot a time --> also not\npop\n\n")
        self.assertEqual(recordings.get_detected_ms("pop"), 600)

    def test_detected_ms_without_segments(self):
        os.makedirs(os.path.join(self.dir, "pop"))
        self.assertEqual(recordings.get_detected_ms("pop"), 0)


class TestRecordedAt(unittest.TestCase):

    def test_from_the_name(self):
        self.assertEqual(recordings.get_recorded_at("mici_1__1693668038.wav"),
                         datetime.fromtimestamp(1693668038))

    def test_without_extension(self):
        self.assertEqual(recordings.get_recorded_at("mici_1__1693668038"),
                         datetime.fromtimestamp(1693668038))

    def test_no_time_in_the_name(self):
        for name in ("pop.wav", "mici_1__take2.wav", "mici_1__99999999999999999999.wav"):
            with self.subTest(name=name):
                self.assertIsNone(recordings.get_recorded_at(name))


if __name__ == "__main__":
    unittest.main()
