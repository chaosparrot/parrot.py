from config.config import *

def setup_config( with_intro ):
	print("-------------------------")
	print("We can set the configuration in multiple ways")
	print(" - [G] for changing the general settings")
	print(" - [M] for changing the settings for specific modes")
	print(" - [X] for exiting configuration mode")
	
	select_config_mode()
	
def select_config_mode():
	config_mode = input("").lower()
	if( config_mode == "g"):
		change_general_settings()
	elif( config_mode == "m" ):
		print( "SETTING UP MODES" )
	elif( config_mode == "x" ):
		print("")
		return
	else:
		select_config_mode()
		
def change_general_settings():
	# 'CHANNELS': {'explanation': "The number of sound channels we are using for data retrieval", 'value': 2, "type": "int"},
	# 'RATE': {'explanation': "The send rate of the microphone in hertz", 'value': 44100, "type": "int"},
	# 'CHUNK': {'explanation': "The chunk size retrieved per frame in bytes", 'value': 1024, "type": "int"},

	print("-------------------------")
	print("Lets set up settings for sound")

	simple_sound_settings = {
		'RECORD_MILLISECONDS': {'explanation': "The amount of milliseconds we are using for recording \nNOTE! - Changing this value will make you have to record new sound files and train new models for recognition", 'value': 50, "type": "int"},
		'PREDICTION_LENGTH': {'explanation': "The amount of predictions we keep in our memory \nThis can be useful for detecting complicated patterns", 'value': 10, "type": "int"}
	};
	
	map = update_configuration( simple_sound_settings )
	
	print("-------------------------")
	print("Now changing settings for play mode")

	play_settings = {
		'STARTING_MODE': {'explanation': "What should be our starting mode when we run play?", 'value': 'browse', "type": "string"},
		'SAVE_REPLAY_DURING_PLAY': {'explanation': "Should we save a simple replay of recognitions and actions during play mode? ( Useful for later analysis - Turning this off will slightly increase performance )", 'value': 'y', "type": "bool"},
		'SAVE_FILES_DURING_PLAY': {'explanation': "Should we save all the audio files during play mode? ( Very useful for later analysis of multiple models - Turning this on will decrease performance and use lots of disk space)", 'value': 'n', "type": "bool"},		
	}
	
	map = update_configuration( play_settings )

	print("-------------------------")
	print("Now changing file locations")
		
	file_settings = {
		'DATASET_FOLDER': {'explanation': "The directory from which we will retrieve our data used in our model training", "value": "data/recordings", "type": "string"},
		'RECORDINGS_FOLDER': {'explanation': "The directory in which we will place all our recordings", "value": "data/recordings", "type": "string"},
		'REPLAYS_FOLDER': {'explanation': "The directory in which we will place all our replays", "value": "data/replays", "type": "string"},
		'REPLAYS_AUDIO_FOLDER': {'explanation': "The directory in which we will place all our audio files used for analysis", "value": "data/replays/audio", "type": "string"},
		'CLASSIFIER_FOLDER': {'explanation': "The directory in which we will place all our trained machine learning models", "value": "data/models", "type": "string"},
	 	'TEMP_FILE_NAME': {'explanation': "The temporary file we use for recognition - This file will constantly be overwritten during play", 'value': "play.wav", "type": "string"},
	 	'DEFAULT_CLF_FILE': {'explanation': "The default model we will use for sound recognition", 'value': "train", "type": "string"},
	}
	
	map = update_configuration( file_settings )

	
def update_configuration( map ):
	result_map = {}
	for key in map:
		print("Changing the value for " + key + "( Current value is " + str(map[key]['value']) + ", this will be used if your input is empty )" )

		print( map[key]['explanation'] )
		setting_input = input( key + ":" )
		
		if( setting_input == "" ):
			setting_input = map[key]['value']
		else:
			if( map[key]['type'] == "string" ):
				setting_input = setting_input
			elif( map[key]['type'] == "int" ):
				setting_input = int( setting_input )
			elif( map[key]['type'] == "bool" ):
				setting_input = setting_input.lower() == "y"

		result_map[ key ] = setting_input

	return result_map
	