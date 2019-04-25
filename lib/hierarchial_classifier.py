import time
import pyautogui
from config.config import *
import math
pyautogui.FAILSAFE = False
from sklearn.externals import joblib
import copy

# NOTE - This classifier is only meant to be used when trained
# There is no way to 'fit' this classifier - It needs fitted classifiers to generate the tree
class HierarchialClassifier:

	# A map of all the classifiers inside of this classifier
	classifiers = {}

	# A list of all the available classes which will be used as a starting point
	# When a prediction is made without this map having the key, it will not be added
	classes_ = []
	
	# An empty prediction dict for performance purposes
	empty_prediction_dict = {}
	
	# Initialize the classifiers and their leaf classes
	def _init_( self, classifier_map ):
		self.classes_ = []
		self.classifiers = classifier_map
		
		for label, classifier in self.classifiers:
			if( label not in self.classifiers ):
				self.classes_.append( label )
				self.empty_prediction_dict[ label ] = { 'percent':  0, 'intensity': 0, 'winner': False, 'frequency': 0 }

	# Get an empty prediction map
	def get_empty_prediction( self ):
		return copy.deepcopy( self.empty_prediction_dict )
					
	# Predict the probabilities of the given data array
	def predict_proba( self, data ):
		predictions = []
		for data_row in data:
			predictions.append( self.predict_single_proba )
		
		return predictions
		
	# Predict a single data row
	# This will recursively go through the tree structure of classifiers until it has reached an end
	# When it reaches the end, it will only retain the end nodes probabilities since that classifier is deemed to be the most knowledgeable
	def predict_single_proba( self, data_row, type="main" ):
		probabilityDict = {}
		probabilities = self.classifiers[ type ].predict_proba( [data_row] )[0]
		
		predicted = np.argmax( probabilities )
		if( isinstance(predicted, list) ):
			predicted = predicted[0]
		
		predicted_label = self.classifiers[type].classes_
		
		# Check if the winner is actually another category that needs to be classified
		if( predicted_label in self.classifiers ):
			probabilityDict = self.predict_single_proba( data_row, self.classifiers[ predicted_label ] )
		
		# Leaf node classifier
		else:
			probabilityDict = get_empty_prediction()
			for index, percent in enumerate( probabilities[0] ):
				label = classifier.classes_[ index ]
				if( label in probabilityDict ):
					probabilityDict[ label ] = { 'percent': percent, 'intensity': int(intensity), 'winner': index == predicted, 'frequency': frequency }
				
		return probabilityDict