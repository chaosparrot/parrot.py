"""Recorded sounds: their labels, recordings and detected time."""
import os
from datetime import datetime
from config.config import RECORDINGS_FOLDER, MS_PER_FRAME
from lib.srt import get_srt_paths, get_srt_path_for_wav, count_label_ms_in_srt

def get_label_dir(label):
    return os.path.join(RECORDINGS_FOLDER, label)

# Example: -> ["hiss", "pop", ...]
def get_labels():
    if not os.path.isdir(RECORDINGS_FOLDER):
        return []
    return [entry for entry in sorted(os.listdir(RECORDINGS_FOLDER))
            if not entry.startswith(".") and os.path.isdir(os.path.join(RECORDINGS_FOLDER, entry))]

# Example: "pop" -> [{"filename": "mici_0__1786747776.wav",
#                     "wav_path": "data/recordings/pop/source/mici_0__1786747776.wav",
#                     "srt_path": "data/recordings/pop/segments/mici_0__1786747776.v3.srt"}]
def get_recordings(label):
    source_dir = os.path.join(get_label_dir(label), "source")
    if not os.path.isdir(source_dir):
        return []
    srt_paths = get_srt_paths(os.path.join(get_label_dir(label), "segments"))
    return [{"filename": wav_file,
             "wav_path": os.path.join(source_dir, wav_file),
             "srt_path": get_srt_path_for_wav(srt_paths, wav_file[:-len(".wav")])}
            for wav_file in sorted(f for f in os.listdir(source_dir) if f.endswith(".wav"))]

# Detected ms across a label's srts. One that can't be parsed (hand-edited,
# cut short) counts as 0 rather than failing the whole label.
def get_detected_ms(label):
    total = 0
    for srt_path in get_srt_paths(os.path.join(get_label_dir(label), "segments")):
        try:
            total += count_label_ms_in_srt(label, srt_path, MS_PER_FRAME)
        except (OSError, ValueError, IndexError):
            pass
    return total

# Example: "mici_0__1786747776.wav" -> datetime(2026, 8, 14, 17, 49, 36), local time
def get_recorded_at(filename):
    tail = os.path.splitext(filename)[0].rsplit("__", 1)[-1]
    if not tail.isdigit():
        return None
    try:
        return datetime.fromtimestamp(int(tail))
    except (ValueError, OSError, OverflowError):
        return None
