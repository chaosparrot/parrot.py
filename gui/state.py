import os
from config.config import CLASSIFIER_FOLDER
from lib import recordings


class AppState:
    """What the pages show, read off data/. Never writes. Detected time is
    cached per label, so a recording made while the app is open counts toward
    it after a restart."""

    def __init__(self):
        self._duration_cache = {}

    def get_sound_labels(self):
        return recordings.get_labels()

    def get_recordings_for_label(self, label):
        return recordings.get_recordings(label)

    def get_label_duration_ms(self, label):
        """Cached: it parses every srt the label has."""
        if label not in self._duration_cache:
            self._duration_cache[label] = recordings.get_detected_ms(label)
        return self._duration_cache[label]

    def get_model_names(self):
        """Model names in data/models/, without the .pkl extension."""
        if not os.path.isdir(CLASSIFIER_FOLDER):
            return []
        return sorted(f[:-len(".pkl")] for f in os.listdir(CLASSIFIER_FOLDER)
                      if f.endswith(".pkl"))
