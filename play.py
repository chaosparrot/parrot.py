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
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel
from scipy.fftpack import fft, rfft, fft2, dct
from python_speech_features import mfcc

TEMP_FILE_NAME = "play.wav"
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0.5

def hash_directory_to_number( setdir ):
	return float( int(hashlib.sha256( setdir.encode('utf-8')).hexdigest(), 16) % 10**8 )

def getWavSegment( stream, frames ):	
	range_length = int(RATE / CHUNK * RECORD_SECONDS)
	frames = frames[5:]
	frame_length = len( frames )
	
	#print( frame_length, range_length )
	for i in range( frame_length, range_length):
		data = stream.read(CHUNK)
		frames.append(data)
	return frames


# Generate the label mapping
labelDict = {}
dir_path = os.path.join( os.path.dirname(os.path.realpath(__file__)), "dataset")
data_directory_names = os.listdir( dir_path )
seperator = ", "
print( "Predicting sounds using machine learning with the following categories: " )
print( seperator.join(data_directory_names) )
for directoryname in data_directory_names:
	labelDict[ str( hash_directory_to_number( os.path.join( dir_path, directoryname ) ) ) ] = directoryname

# Load the trained classifier
classifier = joblib.load( "train.pkl" )	

# Start streaming microphone data
label_array = classifier.classes_
print ( "Listening..." )
frames = []

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
	rate=RATE, input=True,
	frames_per_buffer=CHUNK)

#stream.stop_stream()	
total_frames = []
last_five_probabilities = []
while( True ):
	#stream.start_stream()
	frames = getWavSegment( stream, frames )
	total_frames.extend( frames )
	#stream.stop_stream()
	
	#print( len( frames ) )
	
	tempFile = wave.open(TEMP_FILE_NAME, 'wb')
	tempFile.setnchannels(CHANNELS)
	tempFile.setsampwidth(audio.get_sample_size(FORMAT))
	tempFile.setframerate(RATE)
	tempFile.writeframes(b''.join(frames))
	tempFile.close()

	# FEATURE ENGINEERING
	#rawWav = scipy.io.wavfile.read( TEMP_FILE_NAME )[ 1 ]
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
	if( len( last_five_probabilities ) > 5 ):
		last_five_probabilities = last_five_probabilities[1:]
		
	last_five_probabilities.append( probabilities )
	probability_window = np.sum( last_five_probabilities, axis=0 ) / 5
	for index, percent in enumerate( probability_window[0] ):
		if( percent > 50 ):
			label = labelDict[ str( label_array[ index ] ) ]
			if( label != "new_silence" ):
				print( label + " - " + str( percent ) + "%" )
				i = True
				if( label == "bell" ):
					typewrite( 'Robbe' )
				elif( label == "tongue_cluck" ):
					click()
				elif( label == "flute_smallest" ):
					moveRel( 0, 50 )
				elif( label == "finger_snap" ):
					click(button='right')
				elif( label == "flute_largest" ):
					moveRel( 0, -50 )
				elif( label == "voice_bll" ):
					scroll( -300 )
				elif( label == "voice_humm" ):
					scroll( 300 )
				elif( label == "voice_roll(incomplete)" ):
					moveRel( -50, 0 )
					
			break
			
	save_total_file = False
	if( save_total_file and len( total_frames ) > 500 ):
		tempFile = wave.open("audiotest-" + TEMP_FILE_NAME, 'wb')
		tempFile.setnchannels(CHANNELS)
		tempFile.setsampwidth(audio.get_sample_size(FORMAT))
		tempFile.setframerate(RATE)
		tempFile.writeframes(b''.join(total_frames))
		tempFile.close()
	
	