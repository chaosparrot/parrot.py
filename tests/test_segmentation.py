"""Runs process_wav_file on the fixture recordings and compares the output to the
saved srt and thresholds files in tests/fixtures/segmentation.
If you changed detection on purpose, run tests/regenerate_segmentation_fixtures.py.
"""
import glob
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from lib.stream_processing import process_wav_file

FIXTURES = os.path.join(ROOT, "tests", "fixtures", "recordings")
EXPECTED = os.path.join(ROOT, "tests", "fixtures", "segmentation")
LABELS = ["pop", "ss"]


def read(path):
    with open(path) as handle:
        return handle.read()


def segment(label):
    """The srt and thresholds output for a fixture, by expected file name."""
    workdir = tempfile.mkdtemp(prefix="parrot_segmentation_")
    try:
        source = os.path.join(FIXTURES, label, "source", label + ".wav")
        thresholds = os.path.join(workdir, label + "_thresholds.txt")
        process_wav_file(source, os.path.join(workdir, label),
                         os.path.join(workdir, label + "_segmented.wav"),
                         thresholds, [label])
        srt = glob.glob(os.path.join(workdir, label + "*.srt"))
        assert len(srt) == 1, srt
        return {label + ".srt": read(srt[0]),
                label + "_thresholds.txt": read(thresholds)}
    finally:
        shutil.rmtree(workdir, True)


class SegmentationTest(unittest.TestCase):
    def test_output_unchanged(self):
        for label in LABELS:
            for name, actual in segment(label).items():
                with self.subTest(name):
                    self.assertEqual(actual, read(os.path.join(EXPECTED, name)))


if __name__ == "__main__":
    unittest.main()
