"""Trained models: their files, checkpoints and when they were trained."""
import os
import re
from typing import Dict, List, Optional, Tuple
from config.config import CLASSIFIER_FOLDER, REPLAYS_FOLDER
from lib.typing import Checkpoint, TrainedAt

def get_model_path(name: str) -> str:
    return os.path.join(CLASSIFIER_FOLDER, name + ".pkl")

def get_model_names() -> List[str]:
    """Model names in data/models, without .pkl e.g. ["model_a", "model_b"]"""
    if not os.path.isdir(CLASSIFIER_FOLDER):
        return []
    return sorted(f[:-len(".pkl")] for f in os.listdir(CLASSIFIER_FOLDER)
                  if f.endswith(".pkl") and not f.startswith("."))

def get_model_files(name: str) -> List[str]:
    """A model's pkl and its checkpoints

    e.g. "model_a" -> ["data/models/model_a.pkl",
                       "data/models/model_a.pkl_1-BEST-weights.pth.tar",
                       "data/models/model_a.pkl_1-weights.pth.tar"]
    """
    files = []
    if os.path.isfile(get_model_path(name)):
        files.append(get_model_path(name))
    if os.path.isdir(CLASSIFIER_FOLDER):
        checkpoint = re.compile(re.escape(name) + r"\.pkl_\d+(-BEST)?-weights\.pth\.tar")
        files += [os.path.join(CLASSIFIER_FOLDER, f) for f in sorted(os.listdir(CLASSIFIER_FOLDER))
                  if checkpoint.fullmatch(f)]
    return files

def get_best_checkpoint_paths(name: str) -> List[str]:
    return [path for path in get_model_files(name) if path.endswith("-BEST-weights.pth.tar")]

def read_checkpoint(path: str) -> Optional[Checkpoint]:
    """Checkpoint data without state_dict e.g. labels, accuracy, epoch, trained_at.
    None if unreadable"""
    import torch
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        # A pickle: a damaged one can raise anything
        return None
    if not isinstance(checkpoint, dict):
        return None
    checkpoint.pop("state_dict", None)
    return checkpoint

def read_model_labels(name: str) -> List[str]:
    """Labels from the pkl, for a model without checkpoints. Slow: unpickles the
    whole model e.g. ["pop", "silence"]. [] if unreadable"""
    import joblib
    try:
        model = joblib.load(get_model_path(name))
    except Exception:
        return []
    return list(getattr(model, "classes_", []))

def get_training_run_times() -> Dict[str, int]:
    """get_trained_at's fallback when a checkpoint has no trained_at. Newest
    training run per model, from replay csv names

    e.g. "model_training_model_a.pkl1786747776.csv" -> {"model_a": 1786747776}
    """
    times = {}
    if not os.path.isdir(REPLAYS_FOLDER):
        return times
    for filename in os.listdir(REPLAYS_FOLDER):
        if not (filename.startswith("model_training_") and filename.endswith(".csv")):
            continue
        name, pkl, stamp = filename[len("model_training_"):-len(".csv")].rpartition(".pkl")
        if pkl and stamp.isdigit():
            times[name] = max(times.get(name, 0), int(stamp))
    return times

def get_trained_at(name: str, checkpoint: Optional[Checkpoint] = None,
                   run_times: Optional[Dict[str, int]] = None) -> TrainedAt:
    """When a model was trained. Most trustworthy first: checkpoint, replay csv, pkl mtime

    Returns (when, source) e.g. (1786747776, "checkpoint"). (None, None) if no model
    """
    if checkpoint and checkpoint.get("trained_at"):
        return checkpoint["trained_at"], "checkpoint"
    if run_times is None:
        run_times = get_training_run_times()
    if name in run_times:
        return run_times[name], "replay"
    if os.path.isfile(get_model_path(name)):
        return os.path.getmtime(get_model_path(name)), "mtime"
    return None, None

def get_newest_first_key(name: str, trained_at: TrainedAt) -> Tuple[float, str]:
    """Sort key, newest first. An mtime counts only to the (UTC) day: copying a
    data folder stamps every pkl at once"""
    when, source = trained_at
    when = when or 0
    if source == "mtime":
        when -= when % 86400
    return (-when, name)
