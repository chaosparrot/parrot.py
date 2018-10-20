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
centerXPos, centerYPos = position()

def getWavSegment( stream, frames ):
	range_length = int(RATE / CHUNK * RECORD_SECONDS)
	frames = frames[5:]
	frame_length = len( frames )
	
	intensity = []
	for i in range( frame_length, range_length):
		data = stream.read(CHUNK)
		peak = audioop.maxpp( data, 4 ) / 32767
		intensity.append( peak )
		frames.append(data)

	highestintensity = np.amax( intensity )
	return frames, highestintensity

# Generate the label mapping

labelDict = {}
dir_path = os.path.join( os.path.dirname(os.path.realpath(__file__)), DATASET_FOLDER)
data_directory_names = os.listdir( dir_path )
seperator = ", "
	
# Load the trained classifier
classifier = joblib.load( CLASSIFIER_FOLDER + "/" + DEFAULT_CLF_FILE + ".pkl" )
print( "Loaded classifier " + CLASSIFIER_FOLDER + "/" + DEFAULT_CLF_FILE + ".pkl" )

# Start streaming microphone data
label_array = classifier.classes_
print ( "Listening..." )
frames = []
winsound.PlaySound('config/responses/awaitingorders.wav', winsound.SND_FILENAME)
			
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
	rate=RATE, input=True,
	frames_per_buffer=CHUNK)

#stream.stop_stream()
total_frames = []
last_five_probabilities = []
action = ["", 0]
previousProbabilityDict = {}
frames = []
previousIntensity = 0
previousCluckIntensity = 0

# Get a minimum of these elements of data dictionaries
dataDictsLength = PREDICTION_LENGTH
dataDicts = []

starttime = int(time.time())
if( os.path.isfile( REPLAYS_FILE ) ):
	os.rename( REPLAYS_FILE, REPLAYS_FOLDER + '/previous_run_' + str(starttime) + ".csv")
		
# Write a replay for the percentages
with open(REPLAYS_FILE, 'a', newline='') as csvfile:
	mode_switcher = ModeSwitcher()
	mode_switcher.switchMode( STARTING_MODE )

	headers = ['time', 'winner', 'intensity']
	headers.extend( data_directory_names )
	writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
	writer.writeheader()

	for i in range( 0, dataDictsLength ):
		dataDict = {}
		for directoryname in data_directory_names:
			dataDict[ directoryname ] = {'percent': 0, 'intensity': 0}
		dataDicts.append( dataDict )
	
	while( True ):		
		frames, intensity = getWavSegment( stream, frames )
			
		tempFile = wave.open(TEMP_FILE_NAME, 'wb')
		tempFile.setnchannels(CHANNELS)
		tempFile.setsampwidth(audio.get_sample_size(FORMAT))
		tempFile.setframerate(RATE)
		tempFile.writeframes(b''.join(frames))
		tempFile.close()

		# FEATURE ENGINEERING
		data_row = feature_engineering( TEMP_FILE_NAME )		
		data = [ data_row ]
		
		# Predict the outcome - Only use the result if the probability of being correct is over 50 percent
		probabilities = classifier.predict_proba( data ) * 100
		probabilities = probabilities.astype(int)
		print( probabilities[0] )
		
		# Get the predicted winner
		predicted = np.argmax( probabilities[0] )
		if( isinstance(predicted, list) ):
			predicted = predicted[0]
		replay_row = { 'time': int((time.time() - starttime ) * 1000) / 1000, 'winner': predicted, 'intensity': int(intensity) }
		
		probabilityDict = {}
		for index, percent in enumerate( probabilities[0] ):
			label = str( label_array[ index ] )
			percentage = 0
			if( index == predicted ):
				percentage = 100
			probabilityDict[ label ] = { 'percent': percent, 'intensity': int(intensity), 'winner': index == predicted }
			replay_row[ label ] = percent
			
		writer.writerow( replay_row )
		csvfile.flush()
				
		dataDicts.append( probabilityDict )
		if( len(dataDicts) > dataDictsLength ):
			dataDicts.pop(0)
			
		mode_switcher.getMode().handle_input( dataDicts )