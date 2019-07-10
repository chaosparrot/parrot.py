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
	previous_data = None
	previous_states = ["main"]
	
	# A list of all the available classes which will be used as a starting point
	# When a prediction is made without this map having the key, it will not be added
	classes_ = []
		
	# Initialize the classifiers and their leaf classes
	def __init__( self, classifier_map ):
		self.classes_ = []
		self.classifiers = classifier_map['main'].classifiers
		self.classifiers['main']
				
		silence_prediction = []
		for index,classifier_label in enumerate( self.classifiers ):
			for label in self.classifiers[ classifier_label ].classes_:
				if( label not in self.classifiers ):
					self.classes_.append( label )
					
		self.prediction = self.generate_silence_prediction()
					
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
		
	def calculate_prediction_weights( self, data_row ):
		intensity = data_row[ len( data_row ) - 1]
		frequency = data_row[ len( data_row ) - 2]
		weights = []
		intensity_diff = intensity - self.previous_data[ len( self.previous_data ) - 1 ]
		
		for type in self.classifiers['main'].classes_:
			if( intensity < 600 and intensity_diff < 0 ):
				if( type == "silence" ):
					weights.append( 1 )
				else:
					weights.append( 0 )
					
			# Stops can only transition into vowels
			elif( self.current_classifier == "cat_stop" ):
				if( type == "cat_vowel" and intensity_diff > 1000 and frequency < 100 ):
					weights.append( 1 )
				else:
					weights.append( 0 )
					
			# Sibilants can transition into all other three categories ( vowels, stops, soronants )
			elif( self.current_classifier == "cat_sibilant" ):
				intensity_diff = intensity - self.previous_data[ len( self.previous_data ) - 1 ]
				if( type == "cat_vowel" and frequency < 100 ):
					weights.append( 1 )
				elif( type == "cat_stop" and intensity_diff < -500 and intensity < 1200 and self.previous_states[ len( self.previous_states ) - 2 ] != "cat_vowel" ):
					weights.append( 1 )
				else:
					weights.append( 0 )
					
			# Vowels can only be followed by sibilants or silence
			elif( self.current_classifier == "cat_vowel" ):
				if( type == "cat_sibilant" and frequency > 100 and intensity_diff > 500 ):
					weights.append( 0 )
				elif( type == "silence" and intensity < 1000 ):
					weights.append( 1 )
				else:
					weights.append( 0 )
			else:			
				if( type in self.classifiers.keys() ):
					weights.append( 1 if self.inside_state_range( type, data_row, intensity, frequency ) == True else 0 )
				else:
					weights.append( 1 )
				
		return weights
		
	def inside_state_range( self, classifier, data_row, intensity, frequency ):
		if( intensity < 500 ):
			return False		
		elif( classifier == "cat_stop" ):
			intensity_diff = intensity - self.previous_data[ len( self.previous_data ) - 1 ]
			
			max_sibilant_certainty = 1
			
			# Entry point!
			if( self.current_classifier == "main" ):
				probabilities = self.classifiers['cat_sibilant'].predict_proba( [data_row] )[0]
				max_sibilant_certainty = max( probabilities )
			
			return max_sibilant_certainty < 0.8 or ( intensity < 5000 or abs( intensity_diff ) < 500 )

		elif( classifier == "cat_vowel" ):
			previous_frequency = self.previous_data[ len( self.previous_data ) - 2 ]
			frequency_diff = frequency - previous_frequency
			
			print( "STAY IN VOWEL?", intensity, frequency, frequency_diff, )
			return intensity > 1000 and ( ( frequency < 100 ) or ( self.current_classifier == "cat_vowel" and frequency_diff < 50 ) )
			
		# The sibilant category is fairly accurate even with wrong inputs
		# So it is valid if the certainty is high
		elif( classifier == "cat_sibilant" ):
			probabilities = self.classifiers['cat_sibilant'].predict_proba( [data_row] )[0]

			return max( probabilities ) > 0.8
		elif( classifier == "cat_soronant" ):
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
			if( not self.inside_state_range( self.current_classifier, data_row, data_row[ len( data_row ) - 1 ], data_row[ len( data_row ) - 2 ] ) ):
				prediction_weights = True
			else:
				print( self.current_classifier + " IS INSIDE RANGE!" )
		# Determine a new state
		if( prediction_weights != None ):
			probabilities = self.classifiers['main'].predict_proba( [data_row] )[0]
			
			# Calculate and apply the probability weights
			weights = self.calculate_prediction_weights( data_row )
			for index, probability in enumerate( probabilities ):
				probabilities[ index ] = probability * weights[ index ]
		
			predicted = np.argmax( probabilities )
			if( isinstance(predicted, list) ):
				predicted = predicted[0]
			
			predicted_label = self.classifiers['main'].classes_[ predicted ]
		
			# Check if the winner is actually another state
			if( predicted_label in self.classifiers.keys() ):
				return predicted_label
			elif( predicted_label == "silence" ):
				return "silence"
			else:
				return 'main'
				
		# Just return the current state if no changes were detected
		return self.current_classifier

			
	# Predict a single data row
	# This will go through the tree structure using state change detections
	# This classifier assumes that when a state is entered, only one prediction can be made in that state
	def predict_single_proba( self, data_row, type=None ):
		# Only change type once during a prediction
		if( self.previous_data == None ):
			self.previous_data = data_row
		
		if( type == None ):
			type = self.detect_state_change( data_row )
			
			# Special case - Revert to main classifier and return empty prediction
			if( len( self.prediction ) > 0 and type == "silence" ):
				if( self.current_classifier != "main" ):
					print( "" )
					print( "STATE CHANGE! " + self.current_classifier + " -> main" )
					print( "!!!!!!!!! MAIN MAIN MAIN !!!!!!!!!!!!!!!!!!!!!" )

					self.current_classifier = "main"
					self.previous_states = self.previous_states[-3:]
					self.previous_states.append( "main" )
					
				
				return self.prediction
			
			# If a state change is predicted - Make sure to se the previous prediction
			if( type != self.current_classifier ):
				print( "" )
				print( "STATE CHANGE! " + self.current_classifier + " -> " + type )
				self.current_classifier = type
			
				# Add the new state change
				self.previous_states = self.previous_states[-3:]
				self.previous_states.append( type )
							
		probabilityList = []
		probabilities = self.classifiers[ type ].predict_proba( [data_row] )[0]
		
		predicted = np.argmax( probabilities )
		if( isinstance(predicted, list) ):
			predicted = predicted[0]
		
		predicted_label = self.classifiers[type].classes_[ predicted ]
		
		# Check if the winner is actually another category that needs to be classified
		if( predicted_label in self.classifiers.keys() ):
			return self.predict_single_proba( data_row, predicted_label )
		
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
					
		self.previous_data = data_row
		self.prediction = np.asarray( probabilityList, dtype=np.float64 )
		return self.prediction