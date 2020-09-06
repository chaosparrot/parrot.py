import pyaudio
import pyautogui
import importlib
pyautogui.FAILSAFE = False
REPEAT_DELAY = 0.5
REPEAT_RATE = 33

SPEECHREC_ENABLED = True
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0.03
TEMP_FILE_NAME = "play.wav"
PREDICTION_LENGTH = 10
SILENCE_INTENSITY_THRESHOLD = 400
INPUT_DEVICE_INDEX = 1
SLIDING_WINDOW_AMOUNT = 2
INPUT_TESTING_MODE = False

DATASET_FOLDER = "data/recordings/30mscold"
RECORDINGS_FOLDER = "data/recordings/30mscold"
REPLAYS_FOLDER = "data/replays"
REPLAYS_AUDIO_FOLDER = "data/replays/audio"
REPLAYS_FILE = REPLAYS_FOLDER + "/run.csv"
CLASSIFIER_FOLDER = "data/models"
OVERLAY_FOLDER = "data/overlays"
OVERLAY_FILE = "config/current-overlay-image.txt"
COMMAND_FILE = "config/current-log.txt"


DEFAULT_CLF_FILE = "tiny_gold_league_trio"
STARTING_MODE = "heroes"

SAVE_REPLAY_DURING_PLAY = True
SAVE_FILES_DURING_PLAY = False
EYETRACKING_TOGGLE = "f4"
OVERLAY_ENABLED = True

pytorch_spec = importlib.util.find_spec("torch")
PYTORCH_AVAILABLE = pytorch_spec is not None

dragonfly_spec = importlib.util.find_spec("dragonfly")
if( SPEECHREC_ENABLED == True ):
    SPEECHREC_ENABLED = dragonfly_spec is not None
