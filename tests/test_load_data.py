import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config.config import CURRENT_VERSION, TYPE_FEATURE_ENGINEERING_NORM_MFSC
from lib.load_data import generate_data_balance_strategy_map, sample_data_from_label

FIXTURES = os.path.join(ROOT, "tests", "fixtures")


class TestSampleDataFromLabel(unittest.TestCase):
    """AudioDataset reads augmented[i] as the augmented copy of data[i]."""

    @classmethod
    def setUpClass(cls):
        folder = tempfile.mkdtemp()
        label_dir = os.path.join(folder, "pop")
        os.makedirs(os.path.join(label_dir, "source"))
        os.makedirs(os.path.join(label_dir, "segments"))
        shutil.copy(os.path.join(FIXTURES, "recordings", "pop", "source", "pop.wav"),
                    os.path.join(label_dir, "source", "pop.wav"))
        shutil.copy(os.path.join(FIXTURES, "segmentation", "pop.srt"),
                    os.path.join(label_dir, "segments", f"pop.v{CURRENT_VERSION}.srt"))
        grouped = {"pop": [label_dir]}
        try:
            strategies = generate_data_balance_strategy_map(grouped, "all", True)
            cls.data = sample_data_from_label("pop", grouped, strategies,
                                              TYPE_FEATURE_ENGINEERING_NORM_MFSC, "all")
        finally:
            shutil.rmtree(folder)

    def test_every_sample_has_an_augmented_copy(self):
        self.assertTrue(self.data["label"])
        self.assertTrue(self.data["background"])
        self.assertEqual(len(self.data["augmented"]), len(self.data["label"]))
        self.assertEqual(len(self.data["background_augmented"]), len(self.data["background"]))

    def test_background_copies_are_augmented(self):
        plain = np.array([x[1] for x in self.data["background"]])
        augmented = np.array([x[1] for x in self.data["background_augmented"]])
        self.assertEqual(augmented.shape, plain.shape)
        self.assertFalse(np.allclose(plain, augmented))


if __name__ == "__main__":
    unittest.main()
