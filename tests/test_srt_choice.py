"""Tests which srt each recording uses: the .MANUAL.srt if there is one, else
the current version's. And that the counts use only those.
"""
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from lib.srt import (get_srt_path_for_source, get_srt_paths_for_sources, count_total_label_ms,
                     current_v_ending, manual_ending)


def srt(end):
    """One pop from 00:00:00,900 to end."""
    return f"1\n00:00:00,900 --> {end}\npop\n\n"


class TestSrtChoice(unittest.TestCase):

    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.dir = os.path.join(self.base, "segments")
        os.mkdir(self.dir)
        files = {
            "both" + current_v_ending: srt("00:00:01,800"),     # 900 ms, not used
            "both" + manual_ending: srt("00:00:01,200"),        # 300 ms
            "generated" + current_v_ending: srt("00:00:01,500"),  # 600 ms
            "manual" + manual_ending: srt("00:00:01,050"),      # 150 ms
            "old.v1.srt": srt("00:00:01,800"),                  # not used
            "folder" + current_v_ending: srt("00:00:01,200"),   # 300 ms
        }
        for name, content in files.items():
            with open(os.path.join(self.dir, name), "w") as f:
                f.write(content)
        # A folder with an srt's name is not an srt
        os.mkdir(os.path.join(self.dir, "folder" + manual_ending))

    def tearDown(self):
        shutil.rmtree(self.base)

    def path(self, name):
        return os.path.join(self.dir, name)

    def pick(self, source_name):
        return get_srt_path_for_source(get_srt_paths_for_sources(self.dir), source_name)

    def test_manual_wins(self):
        self.assertEqual(self.pick("both"), self.path("both" + manual_ending))

    def test_generated_when_no_manual(self):
        self.assertEqual(self.pick("generated"),
                         self.path("generated" + current_v_ending))

    def test_manual_only(self):
        self.assertEqual(self.pick("manual"), self.path("manual" + manual_ending))

    def test_old_version_is_not_used(self):
        self.assertIsNone(self.pick("old"))

    def test_none_when_not_segmented(self):
        self.assertIsNone(self.pick("missing"))

    def test_one_srt_per_recording(self):
        self.assertEqual(sorted(get_srt_paths_for_sources(self.dir)), sorted([
            self.path("both" + manual_ending),
            self.path("generated" + current_v_ending),
            self.path("manual" + manual_ending),
            self.path("folder" + current_v_ending),
        ]))

    def test_no_segments_folder(self):
        self.assertEqual(get_srt_paths_for_sources(self.path("nope")), [])

    def test_count_uses_only_those(self):
        self.assertEqual(count_total_label_ms("pop", self.base, 15), 300 + 600 + 150 + 300)


if __name__ == "__main__":
    unittest.main()
