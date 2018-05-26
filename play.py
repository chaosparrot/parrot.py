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
from pyautogui import press, hotkey, click, scroll, typewrite

TEMP_FILE_NAME = "play.wav"
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0.5

def hash_directory_to_number( setdir ):
	return float( int(hashlib.sha256( setdir.encode('utf-8')).hexdigest(), 16) % 10**8 )

def getWavSegment( stream ):
	frames = []
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)
	return frames


# Generate the label mapping
labelDict = {}
dir_path = os.path.join( os.path.dirname(os.path.realpath(__file__)), "dataset")
data_directory_names = os.listdir( dir_path )
seperator = ", "
print( "Predicting sounds using machine learning with the following categories: " + seperator.join(data_directory_names) )
for directoryname in data_directory_names:
	labelDict[ str( hash_directory_to_number( os.path.join( dir_path, directoryname ) ) ) ] = directoryname

# Load the trained classifier
classifier = joblib.load( "train.pkl" )	

# Start streaming microphone data
label_array = classifier.classes_
print ( "Listening..." )
while( True ):
	audio = pyaudio.PyAudio()
	stream = audio.open(format=FORMAT, channels=CHANNELS,
				rate=RATE, input=True,
				frames_per_buffer=CHUNK)

	frames = getWavSegment( stream )
	
	stream.stop_stream()
	stream.close()
	audio.terminate()
	
	tempFile = wave.open(TEMP_FILE_NAME, 'wb')
	tempFile.setnchannels(CHANNELS)
	tempFile.setsampwidth(audio.get_sample_size(FORMAT))
	tempFile.setframerate(RATE)
	tempFile.writeframes(b''.join(frames))
	tempFile.close()

	# FEATURE ENGINEERING
	rawWav = scipy.io.wavfile.read( TEMP_FILE_NAME )[ 1 ]
	chan1 = rawWav[:,0]
					
	# FFT is symmetrical - Only need one half of it to preserve memory
	ft = fft( chan1 )
	powerspectrum = np.abs( ft ) ** 2
	data = [ powerspectrum ]
	
	# Predict the outcome - Only use the result if the probability of being correct is over 75 percent
	probabilities = classifier.predict_proba( data ) * 100
	probabilities = probabilities.astype(int)
	print( probabilities )
	for index, percent in enumerate( probabilities[0] ):
		if( percent > 75 ):
			label = labelDict[ str( label_array[ index ] ) ]
			if( label != "new_silence" ):
				print( label + " - " + str( percent ) + "%" )
				if( label == "bell" ):
					typewrite( 'Robbe' )
				elif( label == "tongue_cluck" ):
					click()
				elif( label == "knock" ):
					scroll( -300 )
				elif( label == "finger_snap" ):
					click(button='right')
				elif( label == "clap" ):
					press('enter')
				elif( label == "voice_humm" ):
					hotkey('ctrl','v')
			else:
				print( "" )
			break