"""Regression tests for the srt count functions."""
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from lib.srt import (count_label_ms_in_srt, count_frames_in_srt, count_total_label_ms,
                     count_total_frames, count_total_silence_frames,
                     current_v_ending, manual_ending)


def srt_text(*entries):
    r"""Example:
    input:  srt_text((900, 1800, "pop"))
    output: "1\n00:00:00,900 --> 00:00:01,800\npop\n\n"
    """
    def time(ms):
        return f"00:{ms // 60000:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    return "".join(f"{i}\n{time(start)} --> {time(end)}\n{label}\n\n"
                   for i, (start, end, label) in enumerate(entries, 1))


SRT_CASES = {
    "one_entry": srt_text((900, 1800, "pop")),
    "several": srt_text((900, 1800, "pop"), (3000, 3600, "pop"), (61500, 62100, "pop")),
    "one_frame_gap": srt_text((900, 1800, "pop"), (1815, 2400, "pop")),
    "starts_at_zero": srt_text((0, 600, "pop")),
    "empty": srt_text(),
}

ROUNDING_MS = 15

# srt case: {label: (count_label_ms_in_srt, count_frames_in_srt)}
EXPECTED_COUNTS = {
    "one_entry": {"pop": (900, 59), "silence": (900, 59)},
    "several": {"pop": (2100, 137), "silence": (60000, 3997)},
    "one_frame_gap": {"pop": (1485, 97), "silence": (915, 59)},
    "starts_at_zero": {"pop": (600, 39), "silence": (0, 0)},
    "empty": {"pop": (0, 0), "silence": (0, 0)},
}


class SrtCountsTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def write(self, path, content):
        with open(path, "w") as f:
            f.write(content)

    def test_counts_in_one_srt(self):
        for name, content in SRT_CASES.items():
            srt_path = os.path.join(self.dir, name + ".srt")
            self.write(srt_path, content)
            for label, (ms, frames) in EXPECTED_COUNTS[name].items():
                with self.subTest(srt=name, label=label):
                    self.assertEqual(count_label_ms_in_srt(label, srt_path, ROUNDING_MS), ms)
                    self.assertEqual(count_frames_in_srt(label, srt_path, ROUNDING_MS), frames)

    def test_totals_use_manual_over_generated(self):
        segments = os.path.join(self.dir, "segments")
        os.mkdir(segments)
        for name, content in {
            "both" + current_v_ending: srt_text((900, 1800, "pop")),       # not used
            "both" + manual_ending: srt_text((900, 1200, "pop")),          # 300 ms
            "generated" + current_v_ending: srt_text((900, 1500, "pop")),  # 600 ms
            "manual" + manual_ending: srt_text((900, 1050, "pop")),        # 150 ms
            "old.v1.srt": srt_text((900, 1800, "pop")),                    # not used
        }.items():
            self.write(os.path.join(segments, name), content)
        self.assertEqual(count_total_label_ms("pop", self.dir, ROUNDING_MS), 1050)
        self.assertEqual(count_total_frames("pop", self.dir, ROUNDING_MS), 67)
        self.assertEqual(count_total_silence_frames(self.dir, ROUNDING_MS), 177)


if __name__ == "__main__":
    unittest.main()
