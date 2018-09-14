import numpy as np
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
pyautogui.FAILSAFE = False
import random
import operator
import audioop
import math
import time
import csv
import threading
import pythoncom
import mode_browse
import mode_switch
centerXPos, centerYPos = position()

TEMP_FILE_NAME = "play.wav"
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0.1

def hash_directory_to_number( setdir ):
	return float( int(hashlib.sha256( setdir.encode('utf-8')).hexdigest(), 16) % 10**8 )

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

def rotateMouse( radians, radius ):
	theta = np.radians( radians )
	c, s = np.cos(theta), np.sin(theta)
	R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
	
	mousePos = np.array([radius, radius])
	relPos = np.dot( mousePos, R )
	moveTo( centerXPos + relPos.flat[0], centerYPos + relPos.flat[1] )

# Generate the label mapping
labelDict = {}
dir_path = os.path.join( os.path.dirname(os.path.realpath(__file__)), "dataset")
data_directory_names = os.listdir( dir_path )
seperator = ", "
print( "Predicting sounds using machine learning with the following categories: " )
print( seperator.join(data_directory_names) )
for directoryname in data_directory_names:
	labelDict[ str( hash_directory_to_number( os.path.join( dir_path, directoryname ).lower() ) ) ] = directoryname

# Load the trained classifier
classifier = joblib.load( "train.pkl" )	

# Start streaming microphone data
label_array = classifier.classes_
print ( "Listening..." )
frames = []

def throttled_press_detection( currentDict, previousDict, label ):
	currentProbability = currentDict[ label ]
	previousProbability = previousDict[ label ]
	
	if( currentProbability > 70 and previousProbability < 70 ):
		return True
	elif( previousProbability < 70 and ( previousProbability + currentProbability ) / 2 > 50 ):
		return True
	else:
		return False

def continuous_detection( currentDict, previousDict, label ):
	currentProbability = currentDict[ label ]
	previousProbability = previousDict[ label ]
	
	if( currentProbability > 80 ):
		return True
	else:
		return False
		
def game_label( label ):
	print( label )
	if( label == "cluck" ):
		press('q')
		#click(button='left')
	elif( label == "finger_snap" ):
		click()
	elif( label == "sound_uh" or label == "sound_a" ):
		press('q')
	elif( label == 'sound_s' ):
		press('w')
	elif( label == 'sound_f' ):
		press('e')
	elif( label == "sound_ax" ):
		press('d')
	elif( label == "sound_lol" ):
		press('z')		
	elif( label == "sound_whistle" ):
		press('r')
	elif( label == "sound_oe" ):
		press('1')

def press_label( label ):
	if( label == "cluck" ):
		click()
	elif( label == "finger_snap" ):
		click(button='right')
	elif( label == "sound_s" ):
		scroll( -250 )
	elif( label == "sound_whistle" ):
		scroll( 250 )
		

winsound.PlaySound('responses/awaitingorders.wav', winsound.SND_FILENAME)
			
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
	rate=RATE, input=True,
	frames_per_buffer=CHUNK)

#stream.stop_stream()	
total_frames = []
last_five_probabilities = []
action = ["", 0]
previousProbabilityDict = {}
strategy = "browser"
frames = []
previousIntensity = 0
previousCluckIntensity = 0

# Get a minimum of these elements of data dictionaries
dataDictsLength = 10
dataDicts = []

starttime = int(time.time())
if( os.path.isfile('run.csv') ):
	os.rename('run.csv', 'previous_run_' + str(starttime) + ".csv")
	
# Write a replay for the percentages
with open('run.csv', 'a', newline='') as csvfile:
	currentMode = mode_browse.BrowseMode()
	currentMode.start()

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
		#stream.start_stream()
		frames, intensity = getWavSegment( stream, frames )
		#total_frames.extend( frames )
		#stream.stop_stream()
			
		tempFile = wave.open(TEMP_FILE_NAME, 'wb')
		tempFile.setnchannels(CHANNELS)
		tempFile.setsampwidth(audio.get_sample_size(FORMAT))
		tempFile.setframerate(RATE)
		tempFile.writeframes(b''.join(frames))
		tempFile.close()

		# FEATURE ENGINEERING
		fs, rawWav = scipy.io.wavfile.read( TEMP_FILE_NAME )
		chan1 = rawWav[:,0]
		chan2 = rawWav[:,1]
											
		# FFT is symmetrical - Only need one half of it to preserve memory
		ft = fft( chan1 )
		powerspectrum = np.abs( rfft( chan1 ) ) ** 2
		mfcc_result1 = mfcc( chan1, samplerate=fs, nfft=1103 )
		mfcc_result2 = mfcc( chan2, samplerate=fs, nfft=1103 )
		data_row = []
		data_row.extend( mfcc_result1.ravel() )
		data_row.extend( mfcc_result2.ravel() )
		
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
			label = labelDict[ str( label_array[ index ] ) ]
			probabilityDict[ label ] = { 'percent': percent, 'intensity': int(intensity), 'winner': index == predicted }
			replay_row[ label ] = percent
			
		writer.writerow( replay_row )
		csvfile.flush()
				
		dataDicts.append( probabilityDict )
		if( len(dataDicts) > dataDictsLength ):
			dataDicts.pop(0)
			
		currentMode.handle_input( dataDicts )

		pythoncom.PumpWaitingMessages()
				
		save_total_file = False
		if( save_total_file and len( total_frames ) > 500 ):
			tempFile = wave.open("audiotest-" + TEMP_FILE_NAME, 'wb')
			tempFile.setnchannels(CHANNELS)
			tempFile.setsampwidth(audio.get_sample_size(FORMAT))
			tempFile.setframerate(RATE)
			tempFile.writeframes(b''.join(total_frames))
			tempFile.close()
	
	