import pyaudio
import wave
from time import sleep
import time
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0.5
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
	 
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)
	fileid = str( (j + 1) * RECORD_SECONDS )
	print ("finished recording " + fileid )
	 
	# stop Recording
	stream.stop_stream()
	stream.close()
	audio.terminate()
	 
	waveFile = wave.open(WAVE_OUTPUT_FILENAME + fileid + WAVE_OUTPUT_FILE_EXTENSION, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(frames))
	waveFile.close()
