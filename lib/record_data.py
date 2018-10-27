from config.config import *
import pyaudio
import wave
import time
from time import sleep
import scipy.io.wavfile
import audioop
import math
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
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
		frequencies = [0]
		
		start_time = time.time()
		for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			data = stream.read(CHUNK)
			frames.append(data)
			peak = audioop.maxpp( data, 4 ) / 32767
			intensity.append( peak )
			
		highestintensity = np.amax( intensity )
			
		byteString = b''.join(frames)
		fftData = np.frombuffer( data, dtype=np.int16 )
		fft_result = fft( fftData )
		positiveFreqs = np.abs( fft_result[ 0:round( len(fft_result)/2 ) ] )
		highestFreq = 0
		loudestPeak = 1000
		for freq in range( 0, len( positiveFreqs ) ):
			if( positiveFreqs[ freq ] > loudestPeak ):
				loudestPeak = positiveFreqs[ freq ]
				highestFreq = freq
					
		if( loudestPeak > 1000 ):
			frequencies.append( highestFreq )
		
		if( RECORD_SECONDS < 1 ):
			# Considering our sound sample is, for example, 100 ms, our lowest frequency we can find is 10Hz ( I think )
			# So add that as a base to our found frequency
			freqInHz = ( 1 / RECORD_SECONDS ) + np.amax( frequencies )
		else:
			# I have no clue how to calculate Hz for fft frames longer than a second
			freqInHz = np.amax( frequencies )
		print( "Intensity: %0d - Freq: %0d" % ( highestintensity, freqInHz ) )
				
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
			waveFile.writeframes(byteString)
			waveFile.close()

