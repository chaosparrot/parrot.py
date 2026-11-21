import contextlib
import os
from dataclasses import dataclass
from typing import List
from lib import models, recordings
from lib.typing import Checkpoint, TrainedAt


@dataclass
class ModelDetails:
    """What a model's files say about it. checkpoints holds one per net that
    could be read, so it can be shorter than net_count."""
    labels: List[str]
    checkpoints: List[Checkpoint]
    net_count: int
    trained_at: TrainedAt
    pkl_bytes: int


class AppState:
    """What the pages show, read off data/. Never writes. Detected time and
    model details are cached, so a recording made or a model retrained while
    the app is open shows after a restart."""

    def __init__(self):
        self._duration_cache = {}
        self._model_details_cache = {}

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
        return models.get_model_names()

    def get_models_newest_first(self):
        """[(name, trained_at)]. A model whose details are read is dated by its
        checkpoint, the others by replay csv or file date."""
        run_times = models.get_training_run_times()
        dated = [(name, self._model_trained_at(name, run_times))
                 for name in models.get_model_names()]
        return sorted(dated, key=lambda pair: models.get_newest_first_key(*pair))

    def _model_trained_at(self, name, run_times):
        details = self._model_details_cache.get(name)
        if details:
            return details.trained_at
        return models.get_trained_at(name, None, run_times)

    def get_net_count(self, name):
        return len(models.get_best_checkpoint_paths(name))

    def get_model_details(self, name):
        """Slow: reads every net's checkpoint, ~4 MB each. Off the UI thread.
        Cached."""
        if name not in self._model_details_cache:
            try:
                details = self._model_details_load(name)
            except Exception:
                # e.g. torch won't load. Empty labels show as unreadable
                details = ModelDetails(labels=[], checkpoints=[], net_count=0,
                                       trained_at=(None, None), pkl_bytes=0)
                # Runs on the reader thread, where an exception aborts the app
                with contextlib.suppress(OSError):
                    details.net_count = self.get_net_count(name)
                    details.trained_at = models.get_trained_at(name)
            self._model_details_cache[name] = details
        return self._model_details_cache[name]

    def get_model_details_if_read(self, name):
        """None until get_model_details has read it."""
        return self._model_details_cache.get(name)

    def _model_details_load(self, name):
        paths = models.get_best_checkpoint_paths(name)
        checkpoints = [c for c in map(models.read_checkpoint, paths) if c is not None]
        labels = next((list(c["labels"]) for c in checkpoints if c.get("labels")), None)
        if labels is None:
            # An sklearn model, one combined from other pkls, or damaged checkpoints
            labels = models.read_model_labels(name)
        pkl_path = models.get_model_path(name)
        return ModelDetails(
            labels=labels,
            checkpoints=checkpoints,
            net_count=len(paths),
            trained_at=models.get_trained_at(
                name, next((c for c in checkpoints if c.get("trained_at")), None)),
            pkl_bytes=os.path.getsize(pkl_path) if os.path.isfile(pkl_path) else 0,
        )
