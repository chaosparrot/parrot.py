from config.config import *
from sklearn.externals import joblib
from lib.hierarchial_classifier import *
import os

def combine_models():
	print( "-------------------------" )
	print( "This script is going to join multiple models into a single model" )
	print( "To increase the likelihood of being good at sound prediction" )
	print( "-------------------------" )

	available_models = []
	for fileindex, file in enumerate(os.listdir( CLASSIFIER_FOLDER )):
		if ( file.endswith(".pkl") ):
			available_models.append( file )
			
	if( len( available_models ) < 2 ):
		print( "It looks like you haven't trained more than one model yet..." )
		print( "Please train an algorithm first using the [L] option in the main menu" )
		return
	
	clf_filename = input("Insert the combined model name ( empty is '" + DEFAULT_CLF_FILE + "_combined' ) ")
	if( clf_filename == "" ):
		clf_filename = DEFAULT_CLF_FILE + "_combined"
	clf_filename += '.pkl'

	print_available_models( available_models )
	classifier_file_index = input("Type the number of the model that you want as the base model: ")
	while( int( classifier_file_index ) <= 0 ):
		classifier_file_index = input("")
		if( classifier_file_index.lower() == "x" ):
			return
	classifier_file_index = int( classifier_file_index ) - 1

	main_classifier_file = CLASSIFIER_FOLDER + "/" + available_models[ classifier_file_index ]
	main_classifier = joblib.load( main_classifier_file )
	classifier_map = {
		'main': main_classifier
	}
	labels = main_classifier.classes_
	
	print( "-------------------------" )
	print( "Determining decision layers... " )
	for index, label in enumerate( labels ):
		answer = input( "Should '" + label + "' connect to another model? Y/N ( Empty is no ) " )
		if( answer.lower() == "y" ):
			print_available_models( available_models )
			
			leaf_index = input("Type the number of the model that you want to run when '" + label + "' is detected: ")
			while( int( leaf_index ) <= 0 ):
				leaf_index = input("")
				if( leaf_index.lower() == "x" ):
					return
			leaf_index = int( leaf_index ) - 1
			classifier_file = CLASSIFIER_FOLDER + "/" + available_models[ leaf_index ]
			classifier_map[ label ] = joblib.load( classifier_file )
	
	connect_model( clf_filename, classifier_map )
	
def print_available_models( available_models ):
	print( "Available models:" )
	for modelindex, available_model in enumerate(available_models):
		print( " - [" + str( modelindex + 1 ) + "] " + available_model )

	
def connect_model( clf_filename, classifier_map ):
	classifier = HierarchialClassifier( classifier_map )	
	classifier_filename = CLASSIFIER_FOLDER + "/" + clf_filename
	joblib.dump( classifier, classifier_filename )
	print( "-------------------------" )
	print( "Model combined!" )
	print( "The combined model contains the following " + str(len( classifier.classes_ )) + " possible predictions: " )
	print( ", ".join( classifier.classes_ ) )
	print( "-------------------------" )

