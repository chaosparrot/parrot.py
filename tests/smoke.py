"""Segments the fixture recordings, then trains a model and loads it back.

The fixtures are only the wav files. The segments come from running the
segmenter on them, so the test always trains on what it produces today.

    python tests/smoke.py

Exits 0 when the happy path works, 1 when it does not. Needs no microphone and
no terminal.

Nothing here checks quality, only that each step produces something the next
step can use. Event counts are bounded loosely because tuning a threshold is
supposed to move them, and that should not fail a build. Accuracy is not
checked at all, since load_data seeds its sampling with the clock and three
epochs on eight seconds of audio predicts nothing.
"""
import atexit
import contextlib
import glob
import io
import os
import re
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

FIXTURES = os.path.join(ROOT, "tests", "fixtures", "recordings")
# A pop is one short burst, an ss is held. Duration type decides how the audio
# gets cut up, so the two should not be heard the same way.
DURATION_TYPES = {"pop": "discrete", "ss": "continuous"}
LABELS = list(DURATION_TYPES)
EXPECTED_CLASSES = ["silence"] + LABELS
NET_COUNT = 3
EPOCHS = 3

# Nothing means a broken segmenter, one means the whole clip was heard as a
# single sound. Eight seconds is only 533 frames at 15ms, so the upper bound is
# a runaway guard rather than a tight limit; real counts here are 27 and 15.
MIN_EVENTS = 3
MAX_EVENTS = 200

failures = []

def check(description, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + description + (" " + detail if detail else ""))
    if not condition:
        failures.append(description)

def stage(name):
    print("\n" + name)
    return time.time()

def count_events(srt_path):
    with open(srt_path) as handle:
        blocks = re.split(r"\n\s*\n", handle.read().strip())
    return len([block for block in blocks if block.strip()])

started = time.time()

t = stage("Importing the configuration")
import config.config
from lib.combine_models import get_current_default_settings
import lib.combine_models
import lib.load_data
print("  took %.1fs, with no input device attached" % (time.time() - t))

# Rebound so that a local run cannot touch your own recordings or models.
workdir = tempfile.mkdtemp(prefix="parrot_smoke_")
atexit.register(shutil.rmtree, workdir, ignore_errors=True)
models_dir = os.path.join(workdir, "models")
replays_dir = os.path.join(workdir, "replays")
os.makedirs(models_dir)
os.makedirs(replays_dir)
lib.combine_models.CLASSIFIER_FOLDER = models_dir

settings = get_current_default_settings()

t = stage("Segmenting the fixture recordings")
from lib.stream_processing import process_wav_file

segmented = os.path.join(workdir, "segmented")
for label in LABELS:
    source = os.path.join(FIXTURES, label, "source", label + ".wav")
    source_dir = os.path.join(segmented, label, "source")
    segments_dir = os.path.join(segmented, label, "segments")
    os.makedirs(source_dir)
    os.makedirs(segments_dir)
    shutil.copy(source, source_dir)

    process_wav_file(source, os.path.join(segments_dir, label),
                     os.path.join(workdir, label + "_segmented.wav"),
                     os.path.join(workdir, label + "_thresholds.txt"), [label])

    written = glob.glob(os.path.join(segments_dir, label + "*.srt"))
    check("%s was segmented into an srt" % label, len(written) == 1,
          str([os.path.basename(path) for path in written]))
    if written:
        events = count_events(written[0])
        check("%s has a workable number of events" % label,
              MIN_EVENTS <= events <= MAX_EVENTS, "(%d)" % events)

    thresholds_file = os.path.join(workdir, label + "_thresholds.txt")
    if os.path.exists(thresholds_file):
        with open(thresholds_file) as handle:
            detected = dict(line.strip().split("=", 1) for line in handle if "=" in line)
        heard = detected.get(label + "_duration_type", "?")
        # TODO check this against DURATION_TYPES once discrete detection is
        # reliable. Today it depends on which eight seconds you feed it: across
        # five discrete sounds and thirty four windows it answered discrete
        # twelve times. Continuous sounds were never called discrete.
        print("       %s heard as %s at %s dBFS, wanted %s" % (
            label, heard, detected.get(label + "_min_dbfs", "?"), DURATION_TYPES[label]))
print("  took %.1fs" % (time.time() - t))

t = stage("Clamping the discrete threshold at the noise floor")
from lib.typing import DetectionFrame, DetectionLabel, DetectionState
from lib.stream_processing import post_processing
# Only a discrete take drops the threshold, and duration type is measured, not
# declared: equal length events with widely varying loudness read as discrete
FLOOR, BOUND, MARGIN = -60.0, -58.0, 3.0
spiky, frame_index = [], 0
for loudness in (-20.0, -32.0, -44.0, -56.0, -68.0):
    for _ in range(3):
        frame_index += 1
        spiky.append(DetectionFrame(frame_index, 15, True, False, 1.0,
                                    loudness, [], 0.0, LABELS[0]))
    for _ in range(3):
        frame_index += 1
        spiky.append(DetectionFrame(frame_index, 15, False, False, 1.0, -90.0,
                                    [], 0.0, config.config.BACKGROUND_LABEL))
spiky_state = DetectionState(
    config.config.CURRENT_DETECTION_STRATEGY, "recording", 15, 900, False,
    -20.0, 0.0, 30.0, FLOOR,
    [DetectionLabel(LABELS[0], 0, 0, "", 0, -96, -96, 0, 0)], [])
spiky_state.upper_bound_dBFS_threshold = BOUND
spiky_state.current_dBFS_threshold = BOUND
spiky_state.dBFS_error_margin = MARGIN
with contextlib.redirect_stdout(io.StringIO()):
    post_processing(spiky, spiky_state, os.path.join(workdir, "spiky"),
                    os.path.join(workdir, "spiky_thresholds.txt"))
check("a discrete drop stops at the noise floor",
      spiky_state.current_dBFS_threshold == FLOOR,
      "(%.2f, unclamped would be %.2f)" % (spiky_state.current_dBFS_threshold,
                                           BOUND - MARGIN * 3))

t = stage("Post processing frames stamped under a lower threshold")
# The live pass marked these detected against a lower bar. Nothing
# matches the settled one, so there is no label to write them under
SETTLED = 0.63
clipped = [DetectionFrame(index + 1, 15, True, False, 1.0, -20.0, [], 0.0, LABELS[0])
           for index in range(60)]
clipped_state = DetectionState(
    config.config.CURRENT_DETECTION_STRATEGY, "recording", 15, 900, False,
    -20.0, 0.0, 30.0, -60.0,
    [DetectionLabel(LABELS[0], 0, 0, "", 0, -96, -96, 0, 0)], [])
clipped_state.upper_bound_dBFS_threshold = SETTLED
clipped_state.current_dBFS_threshold = SETTLED
clipped_state.dBFS_error_margin = 1.5
try:
    with contextlib.redirect_stdout(io.StringIO()):
        post_processing(clipped, clipped_state, os.path.join(workdir, "stamped"),
                        os.path.join(workdir, "stamped_thresholds.txt"))
    check("a stamp with nothing to label it does not crash",
          not any(frame.positive for frame in clipped))
    check("and it counts what it silenced", clipped_state.unlabeled_frames == 60,
          "(%d frames)" % clipped_state.unlabeled_frames)
except Exception as error:
    check("a stamp with nothing to label it does not crash", False,
          "%s: %s" % (type(error).__name__, error))
print("  took %.1fs" % (time.time() - t))

t = stage("Loading that segmentation as training data")
lib.load_data.DATASET_FOLDER = segmented
data_x, data_y, _ = lib.load_data.load_sklearn_data(LABELS, settings["FEATURE_ENGINEERING_TYPE"])
print("  took %.1fs" % (time.time() - t))
check("samples were loaded", len(data_x) > 0, "(%d)" % len(data_x))
check("every label is there, not just silence", sorted(set(data_y)) == sorted(EXPECTED_CLASSES), str(sorted(set(data_y))))

t = stage("Loading a label that segmented to nothing")
empty_label = "quiet"
empty_source = os.path.join(segmented, empty_label, "source")
empty_segments = os.path.join(segmented, empty_label, "segments")
os.makedirs(empty_source)
os.makedirs(empty_segments)
shutil.copy(os.path.join(FIXTURES, LABELS[0], "source", LABELS[0] + ".wav"), empty_source)
open(os.path.join(empty_segments, empty_label + ".v3.srt"), "w").close()
try:
    with_empty_x, with_empty_y, _ = lib.load_data.load_sklearn_data(LABELS + [empty_label], settings["FEATURE_ENGINEERING_TYPE"])
    check("an empty label does not stop the rest loading", len(with_empty_x) > 0, "(%d samples)" % len(with_empty_x))
    check("and it contributes nothing", empty_label not in set(with_empty_y), str(sorted(set(with_empty_y))))
except Exception as error:
    check("an empty label does not stop the rest loading", False, "%s: %s" % (type(error).__name__, error))
shutil.rmtree(os.path.join(segmented, empty_label), ignore_errors=True)
print("  took %.1fs" % (time.time() - t))

t = stage("Training a random forest")
from sklearn.ensemble import RandomForestClassifier
from lib.audio_model import AudioModel
import joblib

forest = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=123)
forest.fit(data_x, data_y)
forest_file = os.path.join(models_dir, "smoke_forest.pkl")
joblib.dump(AudioModel(settings, forest), forest_file)
print("  took %.1fs" % (time.time() - t))

reloaded = joblib.load(forest_file)
check("the random forest model saved", os.path.exists(forest_file))
check("it loads back with the fixture labels", sorted(reloaded.classes_) == sorted(EXPECTED_CLASSES), str(list(reloaded.classes_)))
check("it predicts", len(reloaded.predict_proba(data_x[:4])) == 4)

t = stage("Training an audio net of %d, %d epochs" % (NET_COUNT, EPOCHS))
import lib.audio_net
lib.audio_net.CLASSIFIER_FOLDER = models_dir
lib.audio_net.REPLAYS_FOLDER = replays_dir
from lib.audio_dataset import AudioDataset
from lib.audio_net import AudioNetTrainer

dataset = AudioDataset(lib.load_data.load_pytorch_data(LABELS, settings["FEATURE_ENGINEERING_TYPE"]))
check("the dataset holds the fixture labels", sorted(dataset.get_labels()) == sorted(EXPECTED_CLASSES), str(dataset.get_labels()))

trainer = AudioNetTrainer(dataset, NET_COUNT, settings)
trainer.max_epochs = EPOCHS
trainer.train("smoke_net")
print("  took %.1fs" % (time.time() - t))

net_file = os.path.join(models_dir, "smoke_net")
check("the audio net model saved", os.path.exists(net_file))

import torch
for index in range(NET_COUNT):
    weights_file = os.path.join(models_dir, "smoke_net_%d-BEST-weights.pth.tar" % (index + 1))
    if not os.path.exists(weights_file):
        check("net %d saved its best weights" % (index + 1), False)
        continue
    weights = torch.load(weights_file, weights_only=False)
    missing = [key for key in ["state_dict", "labels", "input_size", "accuracy"] if key not in weights]
    check("net %d saved its best weights" % (index + 1), not missing, "missing %s" % missing if missing else "")
    check("net %d recorded the fixture labels" % (index + 1), sorted(weights["labels"]) == sorted(EXPECTED_CLASSES))

if os.path.exists(net_file):
    ensemble = joblib.load(net_file)
    check("the audio net loads back with the fixture labels", sorted(ensemble.classes_) == sorted(EXPECTED_CLASSES), str(list(ensemble.classes_)))
    probabilities = ensemble.predict_proba(data_x[:4])
    check("it predicts one row per sample", len(probabilities) == 4)
    check("each row covers every label", all(len(row) == len(EXPECTED_CLASSES) for row in probabilities))
    check("each row is a probability distribution", all(abs(sum(row) - 1) < 0.01 for row in probabilities))

print("\n" + "-" * 60)
if failures:
    print("FAILED after %.1fs" % (time.time() - started))
    for failure in failures:
        print("  " + failure)
    sys.exit(1)
print("Segmented, trained and loaded a model in %.1fs" % (time.time() - started))
