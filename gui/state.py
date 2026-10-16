import os
from config.config import RECORDINGS_FOLDER, CLASSIFIER_FOLDER


class AppState:
    """What the pages show, read off data/. Never writes."""

    def get_sound_labels(self):
        """Sound label directory names in data/recordings/."""
        if not os.path.isdir(RECORDINGS_FOLDER):
            return []
        return [entry for entry in sorted(os.listdir(RECORDINGS_FOLDER))
                if os.path.isdir(os.path.join(RECORDINGS_FOLDER, entry))]

    def get_model_names(self):
        """Model names in data/models/, without the .pkl extension."""
        if not os.path.isdir(CLASSIFIER_FOLDER):
            return []
        return sorted(f[:-len(".pkl")] for f in os.listdir(CLASSIFIER_FOLDER)
                      if f.endswith(".pkl"))
