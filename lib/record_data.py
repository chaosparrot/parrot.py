from config.config import *
import pyaudio
import wave
import time
from time import sleep
import scipy.io.wavfile
import audioop
import math
import numpy as np
from numpy.fft import rfft
from scipy.signal import blackmanharris
import os

# Countdown from seconds to 0
def countdown( seconds ):
	for i in range( -seconds, 0 ):
		print("recording in... " + str(abs(i)), end="\r")
		sleep( 1 )
	print("                          ", end="\r")

def record_sound():	
	print( "-------------------------" )
	print( "Let's record some sounds!")
	print( "This script is going to listen to your microphone input" )
	print( "And record tiny audio files to be used for learning later" )
	print( "-------------------------" )

	directory = input("Whats the name of the sound are you recording? ")
	if not os.path.exists(RECORDINGS_FOLDER + "/" + directory):
		os.makedirs(RECORDINGS_FOLDER + "/"  + directory)
	threshold = int( input("What loudness threshold do you need? " ) )
	print("")

	WAVE_OUTPUT_FILENAME = RECORDINGS_FOLDER + "/" + directory + "/" + str(int(time.time() ) ) + "file";
	WAVE_OUTPUT_FILE_EXTENSION = ".wav";

	countdown( 5 )
	
	for j in range( 0, int(300.0 * ( 1.0 / RECORD_SECONDS ))):
		audio = pyaudio.PyAudio()
		 
		# start listening
		stream = audio.open(format=FORMAT, channels=CHANNELS,
						rate=RATE, input=True,
						frames_per_buffer=CHUNK)
		frames = []
		intensity = []
		
		for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			data = stream.read(CHUNK)
			frames.append(data)
			peak = audioop.maxpp( data, 4 ) / 32767
			intensity.append( peak )
			
		highestintensity = np.amax( intensity )
		print( "%0d" % highestintensity )
		
		fileid = "%0.2f" % ((j + 1) * RECORD_SECONDS )
		 
		# stop Recording
		stream.stop_stream()
		stream.close()
		audio.terminate()
		 
		if( highestintensity > threshold ):
			print ("Saving recording " + fileid )
			waveFile = wave.open(WAVE_OUTPUT_FILENAME + fileid + WAVE_OUTPUT_FILE_EXTENSION, 'wb')
			waveFile.setnchannels(CHANNELS)
			waveFile.setsampwidth(audio.get_sample_size(FORMAT))
			waveFile.setframerate(RATE)
			waveFile.writeframes(b''.join(frames))
			waveFile.close()

