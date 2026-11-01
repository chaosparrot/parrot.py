import os
import subprocess
import sys
import unittest
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gui.formatters import duration_text, spoken_duration, recorded_at


class TestDurationText(unittest.TestCase):
    def test_under_a_minute(self):
        self.assertEqual(duration_text(0), "0.0s")
        self.assertEqual(duration_text(7.44), "7.4s")

    def test_never_reads_sixty_seconds(self):
        self.assertEqual(duration_text(59.96), "1m 00s")

    def test_minutes_and_hours(self):
        self.assertEqual(duration_text(64), "1m 04s")
        self.assertEqual(duration_text(3 * 3600 + 5 * 60 + 9), "3h 05m")


class TestSpokenDuration(unittest.TestCase):
    def test_words(self):
        self.assertEqual(spoken_duration(0), "0 seconds")
        self.assertEqual(spoken_duration(1), "1 second")
        self.assertEqual(spoken_duration(7.9), "8 seconds")
        self.assertEqual(spoken_duration(60), "1 minute")
        self.assertEqual(spoken_duration(124.6), "2 minutes 5 seconds")
        self.assertEqual(spoken_duration(3785), "1 hour 3 minutes 5 seconds")

    def test_under_a_second_isnt_zero(self):
        self.assertEqual(spoken_duration(0.4), "under a second")

    def test_halves_round_up(self):
        self.assertEqual(spoken_duration(0.5), "1 second")
        self.assertEqual(spoken_duration(2.5), "3 seconds")


class TestRecordedAt(unittest.TestCase):
    def test_hour_has_no_leading_zero(self):
        self.assertEqual(recorded_at(datetime(2023, 9, 2, 9, 5)), "Sep 02, 2023, 9:05 AM")

    def test_midnight_and_noon_read_twelve(self):
        self.assertEqual(recorded_at(datetime(2023, 9, 2, 0, 30)), "Sep 02, 2023, 12:30 AM")
        self.assertEqual(recorded_at(datetime(2023, 9, 2, 12, 30)), "Sep 02, 2023, 12:30 PM")

    def test_afternoon(self):
        self.assertEqual(recorded_at(datetime(2026, 12, 31, 23, 59)), "Dec 31, 2026, 11:59 PM")


class TestImportCost(unittest.TestCase):
    def test_imports_no_qt(self):
        out = subprocess.run(
            [sys.executable, "-c", "import sys, gui.formatters; print('PyQt6' in sys.modules)"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "False")


if __name__ == "__main__":
    unittest.main()
