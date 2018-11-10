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
from lib.machinelearning import get_loudest_freq
import os
import msvcrt

# Countdown from seconds to 0
def countdown( seconds ):
	for i in range( -seconds, 0 ):
		print("recording in... " + str(abs(i)), end="\r")
		sleep( 1 )
		if( record_controls() == False ):
			return False;
	print("                          ", end="\r")
	return True

def record_controls():
	ESCAPEKEY = b'\x1b'
	SPACEBAR = b' '
	
	if( msvcrt.kbhit() ):
		character = msvcrt.getch()
		if( character == SPACEBAR ):
			print( "Recording paused!" )
			
			# Pause the recording by looping until we get a new keypress
			while( True ):
				if( msvcrt.kbhit() ):
					character = msvcrt.getch()
					if( character == SPACEBAR ):
						print( "Recording resumed!" )
						return True
					elif( character == ESCAPEKEY ):
						print( "Recording stopped" )
						return False
		elif( character == ESCAPEKEY ):
			print( "Recording stopped" )
			return False
			
		print( character )
	return True	
	
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
	frequency_threshold = int( input("What frequency threshold do you need? " ) )
	print("")
	print("You can pause/resume the recording session using the [SPACE] key, and stop the recording using the [ESC] key" )

	WAVE_OUTPUT_FILENAME = RECORDINGS_FOLDER + "/" + directory + "/" + str(int(time.time() ) ) + "file";
	WAVE_OUTPUT_FILE_EXTENSION = ".wav";

	if( countdown( 5 ) == False ):
		return;
	
	files_recorded = 0
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
		fftData = np.frombuffer( byteString, dtype=np.int16 )
		frequency = get_loudest_freq( fftData, RECORD_SECONDS )
				
		fileid = "%0.2f" % ((j + 1) * RECORD_SECONDS )
		
		if( record_controls() == False ):
			break;
		
		# stop Recording
		stream.stop_stream()
		stream.close()
		audio.terminate()
		 
		if( frequency > frequency_threshold and highestintensity > threshold ):
			files_recorded += 1
			print( "Files recorded: %0d - Intensity: %0d - Freq: %0d - Saving %s" % ( files_recorded, highestintensity, frequency, fileid ) )
			waveFile = wave.open(WAVE_OUTPUT_FILENAME + fileid + WAVE_OUTPUT_FILE_EXTENSION, 'wb')
			waveFile.setnchannels(CHANNELS)
			waveFile.setsampwidth(audio.get_sample_size(FORMAT))
			waveFile.setframerate(RATE)
			waveFile.writeframes(byteString)
			waveFile.close()
		else:
			print( "Files recorded: %0d - Intensity: %0d - Freq: %0d" % ( files_recorded, highestintensity, frequency ) )


