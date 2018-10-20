import numpy as np
from config.config import *
from lib.machinelearning import feature_engineering, get_label_for_directory
import pyaudio
import wave
import time
import scipy
import scipy.io.wavfile
import hashlib
import os
import operator
import audioop
import math
import time
import csv

def start_listen_loop( classifier, persist_replay, persist_files, amount_of_seconds ):		
	# Get a minimum of these elements of data dictionaries
	dataDicts = []
	audio_frames = []
	for i in range( 0, PREDICTION_LENGTH ):
		dataDict = {}
		for directoryname in classifier.classes_:
			dataDict[ directoryname ] = {'percent': 0, 'intensity': 0}
		dataDicts.append( dataDict )
	
	audio = pyaudio.PyAudio()
	stream = audio.open(format=FORMAT, channels=CHANNELS,
		rate=RATE, input=True,
		frames_per_buffer=CHUNK)
		
	continue_loop = True
	starttime = int(time.time())
	replay_file = REPLAYS_FOLDER + "/replay_" + str(starttime) + ".csv"
	if( persist_replay ):
		with open(replay_file, 'a', newline='') as csvfile:
			headers = ['time', 'winner', 'intensity']
			headers.extend( classifier.classes_ )
			writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
			writer.writeheader()
			
			print ( "Listening for " + str( amount_of_seconds ) + " seconds..." )
			print ( "" )
			
			starttime = int(time.time())
			while( continue_loop ):
				seconds_playing = time.time() - starttime			
				if( seconds_playing > amount_of_seconds ):
					continue_loop = False
			
				probabilityDict, predicted, audio_frames, intensity = listen_loop( audio, stream, classifier, dataDicts, audio_frames )
				winner = classifier.classes_[ predicted ]
				dataDicts.append( probabilityDict )
				if( len(dataDicts) > PREDICTION_LENGTH ):
					dataDicts.pop(0)
				
				print( "Time: %0.2f - Winner: %s - Percentage: %0.2f                                         " % (seconds_playing, winner, probabilityDict[winner]['percent']), end="\r" )
				replay_row = { 'time': int(seconds_playing * 1000) / 1000, 'winner': winner, 'intensity': int(intensity) }
				for label, labelDict in probabilityDict.items():
					replay_row[ label ] = labelDict['percent']
				writer.writerow( replay_row )
				csvfile.flush()
				
				if( persist_files ):
					os.rename( TEMP_FILE_NAME, REPLAYS_AUDIO_FOLDER + "/%0.3f.wav" % (seconds_playing))
					
			print("Finished listening!                                                                                   ")
			stream.close()
	else:
		starttime = int(time.time())

		while( continue_loop ):
			probabilityDict, predicted, audio_frames = listen_loop( audio, stream, classifier, dataDicts, audio_frames )
			dataDicts.append( probabilityDict )
			if( len(dataDicts) > PREDICTION_LENGTH ):
				dataDicts.pop(0)
				
			seconds_playing = time.time() - starttime;					
			
			if( persist_files ):
				seconds = int(seconds_playing )
				milliseconds = int( seconds_playing * 1000 ) % 1000
				os.rename( TEMP_FILE_NAME, REPLAYS_AUDIO_FOLDER + '/' + str(seconds) + "." + str(milliseconds) + ".wav")

			if( seconds_playing > amount_of_seconds ):
				continue_loop = False

		stream.close()
		
	return replay_file

def listen_loop( audio, stream, classifier, dataDicts, audio_frames ):
	audio_frames, intensity = get_stream_wav_segment( stream, audio_frames )

	tempFile = wave.open(TEMP_FILE_NAME, 'wb')
	tempFile.setnchannels(CHANNELS)
	tempFile.setsampwidth(audio.get_sample_size(FORMAT))
	tempFile.setframerate(RATE)
	tempFile.writeframes(b''.join(audio_frames))
	tempFile.close()
	
	probabilityDict, predicted = predict_wav_file( TEMP_FILE_NAME, classifier, intensity )
	return probabilityDict, predicted, audio_frames, intensity
	
def get_stream_wav_segment( stream, frames ):
	range_length = int(RATE / CHUNK * RECORD_SECONDS)
	
	remove_half = int( range_length / 2 )
	
	frames = frames[remove_half:]
	frame_length = len( frames )
	
	intensity = []
	for i in range( frame_length, range_length):
		data = stream.read(CHUNK)
		peak = audioop.maxpp( data, 4 ) / 32767
		intensity.append( peak )
		frames.append(data)

	highestintensity = np.amax( intensity )
	return frames, highestintensity

def predict_wav_files( classifier, wav_files ):
	dataDicts = []
	audio_frames = []
	print ( "Analyzing " + str( len( wav_files) ) + " audio files..." )
	print ( "" )
	
	for i in range( 0, PREDICTION_LENGTH ):
		dataDict = {}
		for directoryname in classifier.classes_:
			dataDict[ directoryname ] = {'percent': 0, 'intensity': 0}
		dataDicts.append( dataDict )

	probabilities = []
	for index, wav_file in enumerate( wav_files ):
		probabilityDict, predicted = predict_wav_file( wav_file, classifier, 0 )
		winner = classifier.classes_[predicted]
		print( "Analyzing file " + str( index + 1 ) + " - Winner: %s - Percentage: %0.2f" % (winner, probabilityDict[winner]['percent']) , end="\r")
		probabilities.append( probabilityDict )

	print( "                                                                          ", end="\r" )
	
	return probabilities
	
def predict_wav_file( wav_file, classifier, intensity ):
	# FEATURE ENGINEERING
	data_row = feature_engineering( wav_file )		
	data = [ data_row ]
		
	# Predict the outcome of the audio file
	probabilities = classifier.predict_proba( data ) * 100
	probabilities = probabilities.astype(int)
		
	# Get the predicted winner
	predicted = np.argmax( probabilities[0] )
	if( isinstance(predicted, list) ):
		predicted = predicted[0]
		
	probabilityDict = {}
	for index, percent in enumerate( probabilities[0] ):
		label = classifier.classes_[ index ]
		probabilityDict[ label ] = { 'percent': percent, 'intensity': int(intensity), 'winner': index == predicted }
				
	return probabilityDict, predicted

	