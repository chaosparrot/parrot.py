import pyaudio
import pyautogui
pyautogui.FAILSAFE = False

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 0.1
TEMP_FILE_NAME = "play.wav"
PREDICTION_LENGTH = 10

DATASET_FOLDER = "data/dataset"
RECORDINGS_FOLDER = "data/recordings"
REPLAYS_FOLDER = "data/replays"
REPLAYS_AUDIO_FOLDER = "data/replays/audio"
REPLAYS_FILE = REPLAYS_FOLDER + "/run.csv"
CLASSIFIER_FOLDER = "data/models"
DEFAULT_CLF_FILE = "train"

STARTING_MODE = "browse"

LEARN_VISUALISATION = False
LEARN_CONFUSION_MATRIX = True
LEARN_CROSS_ENTHROPY_CHECK = True