import pyaudio
import pyautogui
import importlib
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

TYPE_FEATURE_ENGINEERING_RAW_WAVE = 1
TYPE_FEATURE_ENGINEERING_OLD_MFCC = 2
TYPE_FEATURE_ENGINEERING_NORM_MFCC = 3
FEATURE_ENGINEERING_TYPE = TYPE_FEATURE_ENGINEERING_NORM_MFCC

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

DEFAULT_CLF_FILE = "dummy"
STARTING_MODE = "mode_tutorial_a"

SAVE_REPLAY_DURING_PLAY = True
SAVE_FILES_DURING_PLAY = False
EYETRACKING_TOGGLE = "f4"
OVERLAY_ENABLED = False

pytorch_spec = importlib.util.find_spec("torch")
PYTORCH_AVAILABLE = pytorch_spec is not None
IS_WINDOWS = sys.platform == 'win32'

dragonfly_spec = importlib.util.find_spec("dragonfly")
if( SPEECHREC_ENABLED == True ):
    SPEECHREC_ENABLED = dragonfly_spec is not None
