from config.config import *
import scipy
import scipy.io.wavfile
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.externals import joblib
import time
import warnings
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.fftpack import fft, rfft, fft2, dct
from python_speech_features import mfcc
from sklearn.manifold import TSNE
from sklearn import preprocessing
from lib.machinelearning import *
from sklearn.ensemble import ExtraTreesClassifier

def learn_data():
	print( "-------------------------" )
	print( "Time to learn the audio files")
	print( "This script is going to analyze all your generated audio snippets" )
	print( "And attempt to learn the sounds to their folder names" )
	print( "-------------------------" )

	clf_filename = input("Insert the model name ( empty is 'train' ) ")
	if( clf_filename == "" ):
		clf_filename = DEFAULT_CLF_FILE
	clf_filename += '.pkl'
		
	max_files_per_category = input("How many files should we analyze per category? ( empty is all )" )
	if( max_files_per_category == "" ):
		max_files_per_category = 1000000
	else:
		max_files_per_category = int( max_files_per_category )
	
	print( "--------------------------" )
	dataX, dataY, directory_names, total_feature_engineering_time = load_data( max_files_per_category )
	print( "--------------------------" )
	print( "Learning the data...", end="\r" )
	classifier = get_classifier()
	
	classifier.fit( dataX, dataY )
	print( "Data analyzed!               " )
	
	print( "Testing algorithm speed... ", end="\r" )
	feature_engineering_speed_ms = int(total_feature_engineering_time) / len( dataX )
	prediction_speed_ms = average_prediction_speed( classifier, dataX )
	print( "Worst case reaction speed: +- %0.4f milliseconds " % ( feature_engineering_speed_ms + prediction_speed_ms + ( RECORD_SECONDS * 1000 ) ) )
	print( "- Preparing data speed %0.4f ms " % feature_engineering_speed_ms )
	print( "- Predicting label speed %0.4f ms" % prediction_speed_ms )
	print( "- Recording length %0.4f ms" % ( RECORD_SECONDS * 1000 ) )
	
	print( "Saving the model to " + CLASSIFIER_FOLDER + "/" + clf_filename )
	joblib.dump( classifier, CLASSIFIER_FOLDER + "/" + clf_filename )
	print( "--------------------------" )

	accuracy_analysis = input("Should we analyze the accuracy of the model? Y/n" ).lower() == 'y'	
	if( accuracy_analysis ):
		print( "Predicting recognition accuracy using cross validation...", end="\r" )
		scores = cross_validation( get_classifier(), dataX, dataY )
		print( "Accuracy: %0.4f (+/- %0.4f)                               " % (scores.mean(), scores.std() * 2))
	
	detailed_analysis = input("Should we do a detailed analysis of the model? Y/n" ).lower() == 'y'
	if( detailed_analysis ):
		create_confusion_matrix( get_classifier(), dataX, dataY, directory_names )
		print( "--------------------------" )
	

def load_wav_files( directory, label, int_label, start, end ):
	category_dataset_x = []
	category_dataset_labels = []
	first_file = False
	
	totalFeatureEngineeringTime = 0
	for fileindex, file in enumerate(os.listdir(directory)):
		if ( file.endswith(".wav") and fileindex >= start and fileindex < end ):
			full_filename = os.path.join(directory, file)
			print( "Loading " + str(fileindex) + " files for " + label + "... ", end="\r" )
			
			# Load the WAV file and turn it into a onedimensional array of numbers
			feature_engineering_start = time.time() * 1000
			data_row, frequency = feature_engineering( full_filename )
			category_dataset_x.append( data_row )
			category_dataset_labels.append( label )
			totalFeatureEngineeringTime += time.time() * 1000 - feature_engineering_start

	print( "Loaded " + str( len( category_dataset_labels ) ) + " .wav files for category " + label + " (id: " + str(int_label) + ")" )
	return category_dataset_x, category_dataset_labels, totalFeatureEngineeringTime

def get_classifier():
	#return ExtraTreesClassifier(n_estimators=500, max_depth=20, random_state=123 )
	return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=123)
		
def load_data( max_files ):
	# Get the full directories for the dataset
	dir_path = os.path.join( os.path.dirname( os.path.dirname( os.path.realpath(__file__)) ), DATASET_FOLDER)
	data_directory_names =  [directory for directory in os.listdir( dir_path ) if directory != ".gitkeep"]
	
	print( "Selecting categories to train on... ( Y / N )" )
	filtered_data_directory_names = []
	for directory_name in data_directory_names:
		add = input(" - " + directory_name)
		if( add == "" or add.lower() == "y" ):
			filtered_data_directory_names.append( directory_name )
		else:
			print( "Disabled " + directory_name )

	data_directories = list( map( lambda n: (DATASET_FOLDER + "/" + n).lower(), filtered_data_directory_names) )

	# Add a label used for classifying the sounds
	data_directories_label = list( map( get_label_for_directory, data_directories ) )
	warnings.filterwarnings(action='ignore', category=DeprecationWarning)

	# Generate the training set and labels with them
	dataset = []
	dataset_x = []
	dataset_labels = []
	
	totalFeatureEngineeringTime = 0
	for index, directory in enumerate( data_directories ):
		id_label = data_directories_label[ index ]
		str_label = filtered_data_directory_names[ index ]
		cat_dataset_x, cat_dataset_labels, featureEngineeringTime = load_wav_files( directory, str_label, id_label, 0, max_files )
		totalFeatureEngineeringTime += featureEngineeringTime
		dataset_x.extend( cat_dataset_x )
		dataset_labels.extend( cat_dataset_labels )

	return dataset_x, dataset_labels, filtered_data_directory_names, totalFeatureEngineeringTime

