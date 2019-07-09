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
class ExpertClassifier:
	
	classifiers = {}
		
	# A list of all the available classes which will be used as a starting point
	# When a prediction is made without this map having the key, it will not be added
	classes_ = []
		
	# Initialize the classifiers and their leaf classes
	def __init__( self, classifier_map ):
		self.classes_ = []
		self.classifiers = classifier_map
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
			
	# Predict a single data row
	# This will ask all the classifiers for a prediction
	# The one with the highest prediction wins
	def predict_single_proba( self, data_row ):
		probabilityList = []
		
		highest_probabilities = []
		highest_probability = 0
		type = None
		for index in self.classifiers.keys():
			probabilities = self.classifiers[ index ].predict_proba( [data_row] )[0]
			
			certainty = max( probabilities )
			if( certainty > highest_probability ):
				type = index
				highest_probability = certainty
				highest_probabilities = probabilities
				
		# Make the dictionary of percentages
		probabilityDict = {}
		for index, percent in enumerate( highest_probabilities ):
			label = self.classifiers[ type ].classes_[ index ]
			probabilityDict[ label ] = percent

		# Make the complete list of probabilities in the order of the classes array
		for label in self.classes_:
			if( label in probabilityDict ):
				probabilityList.append( probabilityDict[label] )
			else:
				probabilityList.append( 0 )
					
		return np.asarray( probabilityList, dtype=np.float64 )