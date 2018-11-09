import numpy as np
from config.config import *
from lib.machinelearning import feature_engineering, get_label_for_directory
import pyaudio
import wave
import time
import scipy
import scipy.io.wavfile
from scipy.fftpack import fft, rfft, fft2
from sklearn.externals import joblib
import hashlib
import os
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position
from scipy.fftpack import fft, rfft, fft2, dct, fftfreq
from python_speech_features import mfcc
import pyautogui
import winsound
import random
import operator
import audioop
import math
import time
import csv
import threading
import pythoncom
from lib.mode_switcher import ModeSwitcher
from time import sleep
from lib.listen import start_listen_loop
centerXPos, centerYPos = position()

# Load the trained classifier
classifier = joblib.load( CLASSIFIER_FOLDER + "/" + DEFAULT_CLF_FILE + ".pkl" )
print( "Loaded classifier " + CLASSIFIER_FOLDER + "/" + DEFAULT_CLF_FILE + ".pkl" )
	
# Generate the label mapping
mode_switcher = ModeSwitcher()
mode_switcher.switchMode( STARTING_MODE )
start_listen_loop( classifier, mode_switcher, True )