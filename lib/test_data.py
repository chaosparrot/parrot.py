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
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from lib.listen import start_nonblocking_listen_loop, predict_wav_files

def test_data( with_intro ):
	available_models = []
	for fileindex, file in enumerate(os.listdir( CLASSIFIER_FOLDER )):
		if ( file.endswith(".pkl") ):
			available_models.append( file )

	available_replays = []
	for fileindex, file in enumerate(os.listdir( REPLAYS_FOLDER )):
		if ( file.endswith(".csv") ):
			available_replays.append( file )
			
	if( len( available_models ) == 0 ):
		print( "It looks like you haven't trained any models yet..." )
		print( "Please train an algorithm first using the [L] option in the main menu" )
		return
	elif( with_intro ):
		print("-------------------------")
		print("We can analyze our performance in two different ways")
		print(" - [A] for analyzing an audio stream ( useful for comparing models )")
		print(" - [R] for analyzing an existing replay file")
		print(" - [X] for exiting analysis mode")
	
	analyze_replay_or_audio( available_models, available_replays )
		
def replay( available_replays ):
	if( len( available_replays ) == 1 ):
		replay_index = 0
	else:
		print( "-------------------------" )	
		print( "Select a replay to analyze:" )
		print( "( empty continues the list of replays, [X] exit this mode )" )
		for index, replay in enumerate( available_replays ):
			filesize = os.path.getsize( REPLAYS_FOLDER + "/" + replay )

			print( "- [" + str( index + 1 ) + "] - " + replay + " (" + str( filesize / 1000 ) + "kb)" )
			replay_prompt = ( index + 1 ) % 10 == 0 or index + 1 == len( available_replays )
			if( replay_prompt ):
				replay_index = input("")
				if( replay_index == "" ):
					continue
				elif( replay_index == "x" ):
					return
				else:
					replay_index = int( replay_index ) - 1
					break

	while( replay_index == "" ):
		replay_index = input("")
		if( replay_index == "" ):
			continue
		if( replay_index == "x" ):
			return
		else:
			replay_index = int( replay_index ) - 1
					
	replay_file = available_replays[ replay_index ]
	print( "Analyzing " + replay_file )
	plot_replay( pd.read_csv( REPLAYS_FOLDER + "/" + replay_file, skiprows=0, header=0) )
	
	# Go back to main menu afterwards
	test_data( True )
		
def audio_analysis( available_models ):
	print( "-------------------------" )
	print( "Putting our algorithms to the test!")
	print( "This script is going to load in your model" )
	print( "And test it against audio files" )
	print( "-------------------------" )
	
	if( len( available_models ) == 1 ):
		classifier_file_index = 0
	else:			
		print( "Available models:" )
		for modelindex, available_model in enumerate(available_models):
			print( " - [" + str( modelindex + 1 ) + "] " + available_model )
		classifier_file_index = input("Type the number of the model that you want to test: ")
		if( classifier_file_index == "" ):
			classifier_file_index = 0
		elif( int( classifier_file_index ) > 0 ):
			classifier_file_index = int( classifier_file_index ) - 1
		
	classifier_file = CLASSIFIER_FOLDER + "/" + available_models[ classifier_file_index ]
	print( "Loading model " + classifier_file )
	classifier = joblib.load( classifier_file )
	print( "This model can detect the following classes: " + ", ".join( classifier.classes_ ) )
		
	if not os.path.exists(REPLAYS_AUDIO_FOLDER ):
		os.makedirs(REPLAYS_AUDIO_FOLDER)	

	wav_files = os.listdir(REPLAYS_AUDIO_FOLDER)

	print( "Should we analyse the existing audio files or record a new set?" )
	print( " - [E] for using the existing files" )
	print( " - [N] for clearing the files and recording new ones" )
	new_or_existing = ""
	while( new_or_existing == "" ):
		new_or_existing = input("")
	
	if( new_or_existing.lower() == "n" ):
		for wav_file in wav_files:
			file_path = os.path.join(REPLAYS_AUDIO_FOLDER, wav_file)
			if( file_path.endswith(".wav") ):
				try:
					if os.path.isfile(file_path):
						os.unlink(file_path)
				except Exception as e:
					print("Could not delete " + wav_file)
		
		print( "-------------------------" )
		seconds = input("How many seconds of audio should we record? ( default is 15 )" )
		if( seconds == "" ):
			seconds = 15
		else:
			seconds = int( seconds )
			
		# Give the user a delay of 5 seconds before audio recording begins
		for i in range( -5, 0 ):
			print("Recording in... " + str(abs(i)), end="\r")
			sleep( 1 )
			
		print( "Recording new audio files" )
		replay_file = start_nonblocking_listen_loop( classifier, False, True, True, seconds )
		classifier = None
		print( "-------------------------" )
		print( "Analyzing file " + replay_file )
		plot_replay( pd.read_csv( replay_file, skiprows=0, header=0) )
	elif( new_or_existing.lower() == "e" ):
		print( "-------------------------" )
		print( "Analysing existing audio files" )
		full_wav_files = []
		
		# First sort the wav files by time
		raw_wav_filenames = []
		for wav_file in wav_files:
			if( wav_file.endswith(".wav") ):
				raw_wav_filenames.append( float( wav_file.replace(".wav", "") ) )
				
		raw_wav_filenames.sort()
		
		for float_wav_file in raw_wav_filenames:
			wav_file_name = '%0.3f.wav' % ( float_wav_file )
			file_path = os.path.join(REPLAYS_AUDIO_FOLDER, wav_file_name )
			full_wav_files.append( file_path )
				
		predictions = predict_wav_files( classifier, full_wav_files )
		
		dataRows = []
		for index, prediction in enumerate( predictions ):
			timeString = full_wav_files[ index ].replace( REPLAYS_AUDIO_FOLDER + os.sep, "" ).replace( ".wav", "" )
			dataRow = {'time': int(float(timeString) * 1000) / 1000, 'intensity': 0, 'actions': [], 'buffer': 0 }
			for column in prediction:
				dataRow[column] = prediction[ column ]['percent']
				if( prediction[ column ]['winner'] ):
					dataRow['winner'] = column
					dataRow['frequency'] = prediction[column]['frequency']
					dataRow['intensity'] = prediction[column]['intensity']					

			dataRows.append( dataRow )

		classifier = None
		print( "-------------------------" )
		print( "Analyzing replay!" )
		plot_replay( pd.DataFrame(data=dataRows) )
			
	# Go back to main menu afterwards
	test_data( True )

def analyze_replay_or_audio( available_models, available_replays ):
	replay_or_audio = input( "" )
	if( replay_or_audio.lower() == "r" ):
		if( len(available_replays ) == 0 ):
			print( "No replays to be analyzed yet - Make sure to do a practice run using the play mode first" )
			analyze_replay_or_audio( available_models, available_replays )
		else:
			replay( available_replays )
	elif( replay_or_audio.lower() == "a" ):
		audio_analysis( available_models )
	elif( replay_or_audio.lower() == "x" ):
		print("")
		return
	else:
		analyze_replay_or_audio( available_models, available_replays )

def plot_replay( replay_data ):
	plt.style.use('seaborn-darkgrid')
	num = 0
	bottom=0

	colors = ['darkviolet','red', 'gold', 'green', 'deepskyblue', 'navy', 'gray', 'black', 'pink',
		'firebrick', 'orange', 'lawngreen', 'darkturquoise', 'khaki', 'indigo', 'blue', 'teal',
		'cyan', 'seagreen', 'silver', 'saddlebrown', 'tomato', 'steelblue', 'lavenderblush', 'orangered']
	
	# Add percentage plot
	plt.subplot(2, 1, 1)
	plt.title("Percentage distribution of predicted sounds", loc='left', fontsize=12, fontweight=0, color='black')
	plt.ylabel("Percentage")

	for column in replay_data.drop(['winner', 'intensity', 'time', 'frequency', 'actions', 'buffer'], axis=1):
		if( column != "silence" ):
			color = colors[num]		
			num+=1
			plt.bar(np.arange(replay_data['time'].size), replay_data[column], color=color, linewidth=1, alpha=0.9, label=column, bottom=bottom)
			bottom += np.array( replay_data[column] )
			
	plt.legend(loc=1, bbox_to_anchor=(1, 1.3), ncol=4)

	ax1 = plt.subplot(2, 1, 2)

	# Add audio subplot
	plt.title('Audio', loc='left', fontsize=12, fontweight=0, color='black')
	ax1.set_ylabel('Loudness', color='green')
	ax1.set_xlabel("Time( in files )")
	ax1.set_ylim(ymax=40000)
	ax1.tick_params('y', colors='black')
	ax1.bar(np.arange(replay_data['time'].size), np.array( replay_data['intensity'] ), color='green', linewidth=1)
	
	frequencyAxis = ax1.twinx()
	frequencyAxis.plot(np.arange(replay_data['time'].size), replay_data['frequency'], '-', color='red')
	frequencyAxis.set_ylabel('Frequency', color='red')
	frequencyAxis.set_ylim(ymax=800)

	plt.show()