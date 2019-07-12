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
class EnsembleClassifier:
	
	classifiers = {}
		
	# A list of all the available classes which will be used as a starting point
	# When a prediction is made without this map having the key, it will not be added
	classes_ = []
		
	# Initialize the classifiers and their leaf classes
	def __init__( self, classifier_map ):
		self.classes_ = []
		self.classifiers = classifier_map
		for index,classifier_label in enumerate( self.classifiers ):
			if( index == 0 ):
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
		totalProbabilities = []
		
		type = None
		for index in self.classifiers.keys():
			probabilities = self.classifiers[ index ].predict_proba( [data_row] )[0]
			
			if( len( totalProbabilities ) == 0 ):
				totalProbabilities = probabilities
			else:
				for probindex, probability in enumerate( probabilities ):
					totalProbabilities[ probindex ] = totalProbabilities[ probindex ] + probability

		# Normalize the model
		for probindex, probability in enumerate( totalProbabilities ):
			totalProbabilities[ probindex ] = totalProbabilities[ probindex ] * ( 1 / len( self.classifiers.keys() ) )
												
		return np.asarray( totalProbabilities, dtype=np.float64 )