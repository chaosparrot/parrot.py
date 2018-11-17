import numpy as np
from config.config import *
from lib.machinelearning import feature_engineering, feature_engineering_raw, get_label_for_directory, get_highest_intensity_of_wav_file
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
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
import msvcrt

def break_loop_controls():
	ESCAPEKEY = b'\x1b'
	SPACEBAR = b' '
	
	if( msvcrt.kbhit() ):
		character = msvcrt.getch()
		if( character == SPACEBAR ):
			print( "Listening paused                                                          " )
			
			# Pause the recording by looping until we get a new keypress
			while( True ):
				if( msvcrt.kbhit() ):
					character = msvcrt.getch()
					if( character == SPACEBAR ):
						print( "Listening resumed!                                                   " )
						return True
					elif( character == ESCAPEKEY ):
						print( "Listening stopped                                                    " )
						return False
		elif( character == ESCAPEKEY ):
			print( "Listening stopped                                                         " )
			return False			
	return True	

def start_listen_loop( classifier, mode_switcher = False, persist_replay = False, persist_files = False, amount_of_seconds=-1 ):
	# Get a minimum of these elements of data dictionaries
	dataDicts = []
	audio_frames = []
	for i in range( 0, PREDICTION_LENGTH ):
		dataDict = {}
		for directoryname in classifier.classes_:
			dataDict[ directoryname ] = {'percent': 0, 'intensity': 0, 'frequency': 0, 'winner': False}
		dataDicts.append( dataDict )
	
	audio = pyaudio.PyAudio()
	stream = audio.open(format=FORMAT, channels=CHANNELS,
		rate=RATE, input=True,
		frames_per_buffer=CHUNK)
		
	continue_loop = True
	starttime = int(time.time())
	replay_file = REPLAYS_FOLDER + "/replay_" + str(starttime) + ".csv"
	
	infinite_duration = amount_of_seconds == -1
	if( infinite_duration ):
		print( "Listening..." )
	else:
		print ( "Listening for " + str( amount_of_seconds ) + " seconds..." )
	print ( "" )
	
	if( persist_replay ):
		with open(replay_file, 'a', newline='') as csvfile:
			headers = ['time', 'winner', 'intensity', 'frequency', 'actions']
			headers.extend( classifier.classes_ )
			writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
			writer.writeheader()
			
			starttime = int(time.time())
			while( continue_loop ):			
				seconds_playing = time.time() - starttime			
			
				probabilityDict, predicted, audio_frames, intensity, frequency, wavData = listen_loop( audio, stream, classifier, dataDicts, audio_frames )
				winner = classifier.classes_[ predicted ]
				dataDicts.append( probabilityDict )
				if( len(dataDicts) > PREDICTION_LENGTH ):
					dataDicts.pop(0)
					
				prediction_time = time.time() - starttime - seconds_playing
				
				print( "Time: %0.2f - Prediction in: %0.2f - Winner: %s - Percentage: %0d - Frequency %0d                                        " % (seconds_playing, prediction_time, winner, probabilityDict[winner]['percent'], probabilityDict[winner]['frequency']), end="\r" )				
				if( ( infinite_duration == False and seconds_playing > amount_of_seconds ) or break_loop_controls() == False ):
					continue_loop = False

				actions = []
				if( mode_switcher ):
					actions = mode_switcher.getMode().handle_input( dataDicts )
					if( isinstance( actions, list ) == False ):
						actions = []
						
				replay_row = { 'time': int(seconds_playing * 1000) / 1000, 'winner': winner, 'intensity': int(intensity), 'frequency': frequency, 'actions': ':'.join(actions) }
				for label, labelDict in probabilityDict.items():
					replay_row[ label ] = labelDict['percent']
				writer.writerow( replay_row )
				csvfile.flush()					
					
				if( persist_files ):
					audioFile = wave.open(REPLAYS_AUDIO_FOLDER + "/%0.3f.wav" % (seconds_playing), 'wb')
					audioFile.setnchannels(CHANNELS)
					audioFile.setsampwidth(audio.get_sample_size(FORMAT))
					audioFile.setframerate(RATE)
					audioFile.writeframes(wavData)
					audioFile.close()
					
			print("Finished listening!                                                                                   ")
			stream.close()
	else:
		starttime = int(time.time())

		while( continue_loop ):
			probabilityDict, predicted, audio_frames, intensity, frequency, wavData = listen_loop( audio, stream, classifier, dataDicts, audio_frames )
			dataDicts.append( probabilityDict )
			if( len(dataDicts) > PREDICTION_LENGTH ):
				dataDicts.pop(0)
			
			seconds_playing = time.time() - starttime;
			if( ( infinite_duration == False and seconds_playing > amount_of_seconds ) or break_loop_controls() == False ):
				continue_loop = False

			if( mode_switcher ):
				mode_switcher.getMode().handle_input( dataDicts )
			
			if( persist_files ):
				audioFile = wave.open(REPLAYS_AUDIO_FOLDER + "/%0.3f.wav" % (seconds_playing), 'wb')
				audioFile.setnchannels(CHANNELS)
				audioFile.setsampwidth(audio.get_sample_size(FORMAT))
				audioFile.setframerate(RATE)
				audioFile.writeframes(wavData)
				audioFile.close()
			
		stream.close()
		
	return replay_file

def listen_loop( audio, stream, classifier, dataDicts, audio_frames ):
	audio_frames, intensity = get_stream_wav_segment( stream, audio_frames )	
	wavData = b''.join(audio_frames)
	
	probabilityDict, predicted, frequency = predict_raw_data( wavData, classifier, intensity )
	return probabilityDict, predicted, audio_frames, intensity, frequency, wavData
	
def get_stream_wav_segment( stream, frames ):
	stream.start_stream()
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
	stream.stop_stream()
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
		highestintensity = get_highest_intensity_of_wav_file( wav_file )
		probabilityDict, predicted, frequency = predict_wav_file( wav_file, classifier, highestintensity )
		
		winner = classifier.classes_[predicted]
		print( "Analyzing file " + str( index + 1 ) + " - Winner: %s - Percentage: %0d - Frequency: %0d           " % (winner, probabilityDict[winner]['percent'], probabilityDict[winner]['frequency']) , end="\r")
		probabilities.append( probabilityDict )

	print( "                                                                                           ", end="\r" )
	
	return probabilities
		
def predict_raw_data( wavData, classifier, intensity ):
	# FEATURE ENGINEERING
	first_channel_data = np.frombuffer( wavData, dtype=np.int16 )[::2]
	data_row, frequency = feature_engineering_raw( first_channel_data, RATE, intensity )
	data = [ data_row ]

	return create_probability_dict( classifier, data, frequency, intensity )
		
def predict_wav_file( wav_file, classifier, intensity ):
	# FEATURE ENGINEERING
	data_row, frequency = feature_engineering( wav_file )
	data = [ data_row ]
	
	return create_probability_dict( classifier, data, frequency, intensity )

def create_probability_dict( classifier, data, frequency, intensity ):
	if( intensity > 400 ):
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
			probabilityDict[ label ] = { 'percent': percent, 'intensity': int(intensity), 'winner': index == predicted, 'frequency': frequency }
	else:
		# SKIP PREDICTION - MOST CERTAINLY SILENCE
		probabilityDict = {}
		index = 0
		for label in classifier.classes_:
			winner = False
			percent = 0
			if( label == 'silence' ):
				predicted = index
				percent = 100
				winner = True
				
			probabilityDict[ label ] = { 'percent': percent, 'intensity': int(intensity), 'winner': winner, 'frequency': frequency }
			index += 1
			
	return probabilityDict, predicted, frequency

	