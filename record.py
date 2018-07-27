import pyaudio
import wave
from time import sleep
import time
import scipy.io.wavfile
import audioop
import math
import numpy as np
from numpy.fft import rfft
from scipy.signal import blackmanharris
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0.1
WAVE_OUTPUT_FILENAME = "recordings/" + str(int(time.time() ) ) + "file";
WAVE_OUTPUT_FILE_EXTENSION = ".wav";

print("recording in... 5")
sleep( 1 )
print("recording in... 4")
sleep( 1 )
print("recording in... 3")
sleep( 1 )
print("recording in... 2")
sleep( 1 )
print("recording in... 1")
sleep( 1 )

for j in range( 0, int(300.0 * ( 1.0 / RECORD_SECONDS ))):
	audio = pyaudio.PyAudio()
	 
	# start Recording
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
	 
	if( highestintensity > 1000 ):
		print ("Saving recording " + fileid )
		waveFile = wave.open(WAVE_OUTPUT_FILENAME + fileid + WAVE_OUTPUT_FILE_EXTENSION, 'wb')
		waveFile.setnchannels(CHANNELS)
		waveFile.setsampwidth(audio.get_sample_size(FORMAT))
		waveFile.setframerate(RATE)
		waveFile.writeframes(b''.join(frames))
		waveFile.close()

