import pyaudio
import pyautogui
pyautogui.FAILSAFE = False

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0.05
TEMP_FILE_NAME = "play.wav"
PREDICTION_LENGTH = 10

DATASET_FOLDER = "data/recordings"
RECORDINGS_FOLDER = "data/recordings"
REPLAYS_FOLDER = "data/replays"
REPLAYS_AUDIO_FOLDER = "data/replays/audio"
REPLAYS_FILE = REPLAYS_FOLDER + "/run.csv"
CLASSIFIER_FOLDER = "data/models"
DEFAULT_CLF_FILE = "train"

STARTING_MODE = "browse"

SAVE_REPLAY_DURING_PLAY = True
SAVE_FILES_DURING_PLAY = False