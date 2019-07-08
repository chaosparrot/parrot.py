import time
import pyautogui
from config.config import *
import math
pyautogui.FAILSAFE = False
import joblib
import numpy as np
import copy

# NOTE - This classifier is only meant to be used when trained
# There is no way to 'fit' this classifier - It needs fitted classifiers to generate the state changes
class MarkovChainClassifier:
	
	# A map of all the classifiers to switch between
	current_classifier = "main"
	classifiers = {}
	
	# The latest prediction made
	prediction = []

	# Previous states
	previous_states = ["main"]
	
	# A list of all the available classes which will be used as a starting point
	# When a prediction is made without this map having the key, it will not be added
	classes_ = []
		
	# Initialize the classifiers and their leaf classes
	def __init__( self, classifier_map ):
		self.classes_ = []
		self.classifiers = classifier_map
		
		silence_prediction = []
		for index,classifier_label in enumerate( self.classifiers ):
			for label in self.classifiers[ classifier_label ].classes_:
				if( label not in self.classifiers ):
					self.classes_.append( label )
					
	# Predict the probabilities of the given data array
	def predict_proba( self, data ):
		predictions = []
		for data_row in data:
			predictions.append( self.predict_single_proba(data_row) )
		
		return np.asarray( predictions )
		
	# Generate a silence prediction where everything but the silence category is 0
	def generate_silence_prediction( self ):
		silence_prediction = []
		for label in self.classes_:
			percent = 0
			if( label == 'silence' ):
				percent = 1
			silence_prediction.append( percent )
	
		return np.asarray( silence_prediction, dtype=np.float64 )
		
	def inside_state_range( self, classifier, intensity, frequency ):
		if( classifier == "cat_click" and intensity < 3000 ):
			return False
		
		return True
		
	# Detect on what state we currently are
	def detect_state_change( self, data_row ):
	
		# Snap back to the main classifier if the intensity is zero
		prediction_weights = None
		if( data_row[ len( data_row ) - 1 ] < SILENCE_INTENSITY_THRESHOLD ):
			self.prediction = self.generate_silence_prediction()
			return "silence"
		
		# Determine the next state given the main classifier
		elif( self.current_classifier == "main" ):
			prediction_weights = True
		# Leaf state
		elif( self.current_classifier != "main" ):
		
			# Determine if we should still be in the same classifier
			if( not self.inside_state_range( self.current_classifier, data_row[ len( data_row ) - 1 ], data_row[ len( data_row ) - 2 ] ) ):
				prediction_weights = True
				
		# Determine a new state
		if( prediction_weights != None ):
			probabilities = self.classifiers[ self.current_classifier ].predict_proba( [data_row] )[0]
		
			predicted = np.argmax( probabilities )
			if( isinstance(predicted, list) ):
				predicted = predicted[0]
		
			predicted_label = self.classifiers[ self.current_classifier ].classes_[ predicted ]
		
			# Check if the winner is actually another state
			if( predicted_label in self.classifiers.keys() ):
				return predicted_label			
				
		# Just return the current state if no changes were detected
		return self.current_classifier

			
	# Predict a single data row
	# This will go through the tree structure using state change detections
	# This classifier assumes that when a state is entered, only one prediction can be made in that state
	def predict_single_proba( self, data_row, type=None ):
		# Only change type once during a prediction
		if( type == None ):
			type = self.detect_state_change( data_row )
			
			# Special case - Revert to main classifier and return empty prediction
			if( type == "silence" ):
				self.current_classifier = "main"
				return self.prediction
			
			# If a state change is predicted - Make sure to return the previous prediction
			# To give the classifier time for the next state to be solidified
			if( type != self.current_classifier ):
				print( "STATE CHANGE! " + self.current_classifier + " -> " + type )
				self.current_classifier = type
			
				# Add the new state change
				self.previous_states = self.previous_states[-2:]
				self.previous_states.append( type )
				
				return self.prediction
			
		probabilityList = []
		probabilities = self.classifiers[ type ].predict_proba( [data_row] )[0]
		
		predicted = np.argmax( probabilities )
		if( isinstance(predicted, list) ):
			predicted = predicted[0]
		
		predicted_label = self.classifiers[type].classes_[ predicted ]
		
		# Check if the winner is actually another category that needs to be classified
		if( predicted_label in self.classifiers.keys() ):
			return self.predict_single_proba( data_row, predicted_label, type )
		
		# Leaf node classifier
		else:
			probabilityDict = {}
			for index, percent in enumerate( probabilities ):
				label = self.classifiers[ type ].classes_[ index ]
				probabilityDict[ label ] = percent

			# Make the complete list of probabilities in the order of the classes array
			for label in self.classes_:
				if( label in probabilityDict ):
					probabilityList.append( probabilityDict[label] )
				else:
					probabilityList.append( 0 )
					
		self.prediction = np.asarray( probabilityList, dtype=np.float64 )
		return self.prediction