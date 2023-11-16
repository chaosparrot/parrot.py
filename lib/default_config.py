from importlib.util import find_spec
import sys

import pyaudio

if sys.platform == "darwin":
    # This is necessary to import before pyautogui
    # See https://github.com/asweigart/pyautogui/issues/495#issuecomment-778241850
    import AppKit

import pyautogui

pyautogui.FAILSAFE = False

default_audio = pyaudio.PyAudio().get_default_input_device_info()
REPEAT_DELAY = 0.5
REPEAT_RATE = 33
SPEECHREC_ENABLED = False

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 0.03
TEMP_FILE_NAME = "play.wav"
PREDICTION_LENGTH = 10
SILENCE_INTENSITY_THRESHOLD = 400
INPUT_DEVICE_INDEX = 1
if (default_audio is not None):
    INPUT_DEVICE_INDEX = default_audio['index']

SLIDING_WINDOW_AMOUNT = 2
INPUT_TESTING_MODE = False
USE_COORDINATE_FILE = False

TYPE_FEATURE_ENGINEERING_RAW_WAVE = 1
TYPE_FEATURE_ENGINEERING_OLD_MFCC = 2
TYPE_FEATURE_ENGINEERING_NORM_MFCC = 3
TYPE_FEATURE_ENGINEERING_NORM_MFSC = 4
FEATURE_ENGINEERING_TYPE = TYPE_FEATURE_ENGINEERING_NORM_MFSC

DATASET_FOLDER = "data/recordings"
RECORDINGS_FOLDER = "data/recordings"
REPLAYS_FOLDER = "data/replays"
REPLAYS_AUDIO_FOLDER = "data/replays/audio"
REPLAYS_FILE = REPLAYS_FOLDER + "/run.csv"
CLASSIFIER_FOLDER = "data/models"
OVERLAY_FOLDER = "data/overlays"
COORDINATE_FILEPATH = "config/current-coordinate.txt"
CONVERSION_OUTPUT_FOLDER = "data/output"
PATH_TO_FFMPEG = "ffmpeg/bin/ffmpeg"

DEFAULT_CLF_FILE = ""
STARTING_MODE = ""
MICROPHONE_SEPARATOR = None

SAVE_REPLAY_DURING_PLAY = True
SAVE_FILES_DURING_PLAY = False
EYETRACKING_TOGGLE = "f4"
OVERLAY_ENABLED = False

pytorch_spec = find_spec("torch")
PYTORCH_AVAILABLE = pytorch_spec is not None
IS_WINDOWS = sys.platform == 'win32'

dragonfly_spec = find_spec("dragonfly")
if( SPEECHREC_ENABLED == True ):
    SPEECHREC_ENABLED = dragonfly_spec is not None
    
BACKGROUND_LABEL = "silence"

# Detection strategies
CURRENT_VERSION = 1
CURRENT_DETECTION_STRATEGY = "auto_dBFS_mend_dBFS_30ms_secondary_dBFS_reject_cont_45ms_repair"
