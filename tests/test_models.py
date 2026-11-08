import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import joblib
import numpy as np
import torch
from lib import models
from sklearn.ensemble import RandomForestClassifier
from lib.audio_model import AudioModel

AUDIO_SETTINGS = {"version": 1, "RATE": 16000, "CHANNELS": 1, "RECORD_SECONDS": 1,
                  "SLIDING_WINDOW_AMOUNT": 2, "FEATURE_ENGINEERING_TYPE": 1}


class ModelsFolder(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.dir, "models")
        self.replays_dir = os.path.join(self.dir, "replays")
        os.makedirs(self.models_dir)
        os.makedirs(self.replays_dir)
        for name, path in (("CLASSIFIER_FOLDER", self.models_dir),
                           ("REPLAYS_FOLDER", self.replays_dir)):
            patch = mock.patch.object(models, name, path)
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def touch(self, folder, filename):
        path = os.path.join(folder, filename)
        open(path, "w").close()
        return path

    # As lib/learn_data.py fit_sklearn_classifier saves one
    def save_model(self, name):
        classifier = RandomForestClassifier(n_estimators=2).fit([[0, 0], [1, 1]], ["pop", "silence"])
        path = os.path.join(self.models_dir, name + ".pkl")
        joblib.dump(AudioModel(AUDIO_SETTINGS, classifier), path)
        return path

    # As lib/audio_net.py saves them
    def save_checkpoint(self, filename, **fields):
        checkpoint = {"state_dict": {"fc1.weight": torch.zeros(2, 2)}, "input_size": 120,
                      "labels": ["pop", "silence"], "accuracy": 0.9,
                      "last_row": {"epoch": 3, "loss": np.sum([0.1]),
                                   "avg_validation_accuracy": np.average([0.9]),
                                   "pop": 0.9, "silence": 0.9},
                      "loss": 0.1, "epoch": 3, "random_seed": 1,
                      "label_accuracy": {"pop": 0.9, "silence": 0.9}, "combined_accuracy": 0.9,
                      "label_frames": {"pop": 10, "silence": 10}, "trained_at": 1786747776,
                      "run_settings": {}}
        checkpoint.update(fields)
        path = os.path.join(self.models_dir, filename)
        torch.save(checkpoint, path)
        return path


class TestGetModelNames(ModelsFolder):

    def test_pkls_in_order(self):
        self.touch(self.models_dir, "model_a.pkl")
        self.touch(self.models_dir, "model_b.pkl")
        self.touch(self.models_dir, "model_a.pkl_1-BEST-weights.pth.tar")
        self.touch(self.models_dir, "notes.txt")
        self.touch(self.models_dir, "._model_a.pkl")
        self.assertEqual(models.get_model_names(), ["model_a", "model_b"])

    def test_no_models_folder(self):
        shutil.rmtree(self.models_dir)
        self.assertEqual(models.get_model_names(), [])


class TestGetModelFiles(ModelsFolder):

    def test_pkl_then_its_checkpoints(self):
        pkl = self.touch(self.models_dir, "model_a.pkl")
        best = self.touch(self.models_dir, "model_a.pkl_1-BEST-weights.pth.tar")
        last = self.touch(self.models_dir, "model_a.pkl_1-weights.pth.tar")
        self.touch(self.models_dir, "model_a2.pkl")
        self.touch(self.models_dir, "model_a2.pkl_1-BEST-weights.pth.tar")
        self.touch(self.models_dir, "model_a.pkl_2.pkl")
        self.touch(self.models_dir, "model_a.pkl_2.pkl_1-BEST-weights.pth.tar")
        self.assertEqual(models.get_model_files("model_a"), [pkl, best, last])

    def test_no_files(self):
        self.assertEqual(models.get_model_files("model_a"), [])


class TestGetBestCheckpointPaths(ModelsFolder):

    def test_one_per_net(self):
        self.touch(self.models_dir, "model_a.pkl")
        best_1 = self.touch(self.models_dir, "model_a.pkl_1-BEST-weights.pth.tar")
        best_2 = self.touch(self.models_dir, "model_a.pkl_2-BEST-weights.pth.tar")
        self.touch(self.models_dir, "model_a.pkl_1-weights.pth.tar")
        self.assertEqual(models.get_best_checkpoint_paths("model_a"), [best_1, best_2])

    def test_combined_model_has_none(self):
        self.touch(self.models_dir, "model_a.pkl")
        self.assertEqual(models.get_best_checkpoint_paths("model_a"), [])


class TestReadCheckpoint(ModelsFolder):

    def test_fields_without_weights(self):
        checkpoint = models.read_checkpoint(self.save_checkpoint("model_a.pkl_1-BEST-weights.pth.tar"))
        self.assertEqual(checkpoint["labels"], ["pop", "silence"])
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["trained_at"], 1786747776)
        self.assertNotIn("state_dict", checkpoint)

    def test_damaged(self):
        path = os.path.join(self.models_dir, "model_a.pkl_1-BEST-weights.pth.tar")
        with open(path, "wb") as f:
            f.write(b"not a checkpoint")
        self.assertIsNone(models.read_checkpoint(path))


class TestReadModelLabels(ModelsFolder):

    def test_labels_from_pkl(self):
        self.save_model("model_a")
        self.assertEqual(models.read_model_labels("model_a"), ["pop", "silence"])

    def test_missing(self):
        self.assertEqual(models.read_model_labels("model_a"), [])


class TestGetTrainingRunTimes(ModelsFolder):

    def test_newest_run_per_model(self):
        # Newest isn't last alphabetically
        self.touch(self.replays_dir, "model_training_model_a.pkl1000.csv")
        self.touch(self.replays_dir, "model_training_model_a.pkl900.csv")
        self.touch(self.replays_dir, "model_training_model_b.pkl200.csv")
        self.touch(self.replays_dir, "replay_400.csv")
        self.touch(self.replays_dir, "run.csv")
        self.assertEqual(models.get_training_run_times(), {"model_a": 1000, "model_b": 200})

    def test_no_replays_folder(self):
        shutil.rmtree(self.replays_dir)
        self.assertEqual(models.get_training_run_times(), {})


class TestGetTrainedAt(ModelsFolder):

    def test_checkpoint_first(self):
        self.touch(self.models_dir, "model_a.pkl")
        self.assertEqual(models.get_trained_at("model_a", {"trained_at": 500}, {"model_a": 300}),
                         (500, "checkpoint"))

    def test_checkpoint_from_before_trained_at(self):
        self.touch(self.models_dir, "model_a.pkl")
        self.touch(self.replays_dir, "model_training_model_a.pkl300.csv")
        self.assertEqual(models.get_trained_at("model_a", {"epoch": 3}), (300, "replay"))

    def test_then_training_run(self):
        self.touch(self.models_dir, "model_a.pkl")
        self.touch(self.replays_dir, "model_training_model_a.pkl300.csv")
        self.assertEqual(models.get_trained_at("model_a"), (300, "replay"))

    def test_then_pkl_mtime(self):
        os.utime(self.touch(self.models_dir, "model_a.pkl"), (200, 200))
        self.assertEqual(models.get_trained_at("model_a"), (200, "mtime"))

    def test_no_model(self):
        self.assertEqual(models.get_trained_at("model_a"), (None, None))


class TestGetNewestFirstKey(ModelsFolder):

    def test_newest_first(self):
        trained_at = {"old": (100, "checkpoint"), "new": (200, "checkpoint")}
        self.assertEqual(sorted(trained_at, key=lambda name: models.get_newest_first_key(name, trained_at[name])),
                         ["new", "old"])

    def test_mtime_counts_to_the_day(self):
        day = 86400 * 20000
        trained_at = {"b": (day + 9, "mtime"), "a": (day + 5, "mtime"), "c": (day + 1, "checkpoint")}
        self.assertEqual(sorted(trained_at, key=lambda name: models.get_newest_first_key(name, trained_at[name])),
                         ["c", "a", "b"])


if __name__ == "__main__":
    unittest.main()
